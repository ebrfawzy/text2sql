"""Tests for text2sql.db — DatabaseConnection, schema introspection, and execution.

Covers context management, dialect detection, schema dict structure,
DDL generation, execute/execute_safe with edge cases, and URI masking.
"""

from __future__ import annotations

import time

import pytest

from text2sql import db
from text2sql.db import DatabaseConnection, split_statements, sqlglot_name

# ── Connection lifecycle ─────────────────────────────────────────


class TestDatabaseConnection:
    """Connection, dialect, and lifecycle management."""

    def test_connect_sqlite(self, sample_db):
        with DatabaseConnection(sample_db) as db:
            assert db.dialect_name == "sqlite"
            assert db.uri == sample_db

    def test_close_is_idempotent(self, sample_db):
        db = DatabaseConnection(sample_db)
        db.__enter__()
        db.__exit__(None, None, None)
        db.close()  # closing an already-closed connection must not raise
        assert db._engine is not None


# ── URI masking ──────────────────────────────────────────────────


class TestSafeUri:
    """Password masking in _safe_uri."""

    def test_uri_without_credentials_is_unchanged(self):
        db = DatabaseConnection("sqlite:///test.db")
        assert db._safe_uri() == "sqlite:///test.db"
        db.close()

    def test_password_masked(self):
        # Not a real connection, just test the string logic
        db = DatabaseConnection.__new__(DatabaseConnection)
        db.uri = "postgresql://user:secret@localhost:5432/mydb"
        assert "secret" not in db._safe_uri()
        assert "***" in db._safe_uri()
        assert "localhost:5432/mydb" in db._safe_uri()


# ── Schema introspection ─────────────────────────────────────────


class TestSchemaIntrospection:
    """get_table_names, get_schema, get_schema_ddl."""

    def test_get_table_names(self, db_conn):
        tables = db_conn.get_table_names()
        assert "users" in tables
        assert "orders" in tables
        assert len(tables) == 2

    def test_get_schema_is_cached(self, db_conn):
        """Schema is memoized: repeated calls return the same object."""
        assert db_conn.get_schema() is db_conn.get_schema()

    def test_quote_identifier_default_dialect(self, db_conn):
        assert db_conn.quote_identifier("name") == '"name"'
        # Embedded double quotes are doubled.
        assert db_conn.quote_identifier('we"ird') == '"we""ird"'

    def test_quote_identifier_mysql(self, db_conn, monkeypatch):
        monkeypatch.setattr(type(db_conn), "dialect_name", property(lambda self: "mysql"))
        assert db_conn.quote_identifier("name") == "`name`"
        assert db_conn.quote_identifier("we`ird") == "`we``ird`"

    def test_get_schema_structure(self, db_schema):
        assert "tables" in db_schema and db_schema["dialect"] == "sqlite"

    def test_get_schema_columns_carry_type_and_nullability(self, db_schema):
        users = db_schema["tables"]["users"]
        cols = {c["name"]: c for c in users["columns"]}
        assert set(cols) == {"id", "name", "email", "age", "city"}
        assert "INTEGER" in cols["id"]["type"].upper()
        assert cols["name"]["nullable"] is False  # declared NOT NULL
        assert "id" in users["primary_keys"]

    def test_get_schema_foreign_keys(self, db_schema):
        fks = db_schema["tables"]["orders"]["foreign_keys"]
        assert fks[0] == {"column": "user_id", "referred_table": "users",
                          "referred_column": "id"}

    def test_get_schema_ddl_sqlite(self, db_conn):
        ddl = db_conn.get_schema_ddl()
        assert "CREATE TABLE" in ddl
        assert "users" in ddl
        assert "orders" in ddl

    def test_generic_ddl_carries_constraints(self, db_conn):
        """The non-sqlite path rebuilds the DDL from introspection."""
        ddl = db_conn._get_generic_ddl()
        assert "CREATE TABLE" in ddl and "users" in ddl and "orders" in ddl
        assert "NOT NULL" in ddl and "PRIMARY KEY" in ddl and "FOREIGN KEY" in ddl

    def test_row_count_is_none_in_schema(self, db_schema):
        """Schema introspection doesn't fill row_count (profiler does)."""
        for table_info in db_schema["tables"].values():
            assert table_info["row_count"] is None


# ── SQL execution ────────────────────────────────────────────────


class TestExecute:
    """execute() for various query types."""

    def test_select_count(self, db_conn):
        results = db_conn.execute("SELECT COUNT(*) AS cnt FROM users")
        assert len(results) == 1
        assert results[0]["cnt"] == 5

    def test_select_with_where(self, db_conn):
        results = db_conn.execute("SELECT * FROM users WHERE city = 'London'")
        assert len(results) == 2

    def test_select_with_join(self, db_conn):
        results = db_conn.execute(
            "SELECT u.name, o.amount FROM users u "
            "JOIN orders o ON u.id = o.user_id "
            "ORDER BY o.amount DESC"
        )
        assert len(results) == 5
        assert results[0]["amount"] == 200.00

    def test_select_empty_result(self, db_conn):
        results = db_conn.execute("SELECT * FROM users WHERE age > 999")
        assert results == []

    def test_execute_with_params(self, db_conn):
        results = db_conn.execute(
            "SELECT name FROM users WHERE city = :city",
            params={"city": "Paris"},
        )
        assert len(results) == 1
        assert results[0]["name"] == "Eve"

    def test_aggregate_functions(self, db_conn):
        results = db_conn.execute(
            "SELECT AVG(age) AS avg_age, MIN(age) AS min_age, MAX(age) AS max_age FROM users"
        )
        assert results[0]["min_age"] == 25
        assert results[0]["max_age"] == 35

    def test_null_handling(self, db_conn):
        results = db_conn.execute(
            "SELECT name FROM users WHERE email IS NULL"
        )
        assert len(results) == 1
        assert results[0]["name"] == "Charlie"


class TestExecuteSafe:
    """execute_safe() returns (results, error) tuple."""

    def test_success_returns_results_and_none_error(self, db_conn):
        results, error = db_conn.execute_safe("SELECT COUNT(*) AS cnt FROM users")
        assert results is not None
        assert error is None
        assert results[0]["cnt"] == 5

    def test_error_returns_none_and_message(self, db_conn):
        results, error = db_conn.execute_safe("SELECT * FROM nonexistent_table")
        assert results is None
        assert error is not None
        assert len(error) > 0

    def test_syntax_error(self, db_conn):
        results, error = db_conn.execute_safe("SELECTX * FROMM users")
        assert results is None
        assert error is not None

    def test_safe_with_params(self, db_conn):
        results, error = db_conn.execute_safe(
            "SELECT name FROM users WHERE age = :age",
            params={"age": 30},
        )
        assert error is None
        assert results[0]["name"] == "Alice"


# ── Session context manager ─────────────────────────────────────


class TestSession:
    """session() context manager."""

    def test_session_yields(self, db_conn):
        with db_conn.session() as s:
            assert s is not None


class TestMultipleResultSets:
    def test_a_runaway_query_is_stopped_at_the_deadline(self, db_conn, monkeypatch):
        """`asyncio.wait_for` abandons the await, not the thread: one agent's correlated
        subquery held a core at 100% for the full 300s budget and kept burning it after the
        tool had already given up. SQLite has to be interrupted, not just waited on."""
        monkeypatch.setattr(db, "EXEC_TIMEOUT_S", 0.3)
        started = time.monotonic()
        rows, error = db_conn.execute_safe(
            "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r) "
            "SELECT COUNT(*) FROM r")
        assert rows is None and "interrupted" in error
        assert time.monotonic() - started < 5

    def test_every_row_producing_statement_comes_back(self, db_conn):
        """`execute` returns the last set, which silently discarded batched probes — one
        measured run spent 22 turns on one probe each because batching looked broken."""
        sql = "SELECT 1 AS a; SELECT 2 AS b; SELECT 3 AS c"
        assert db_conn.execute(sql) == [{"c": 3}]
        assert db_conn.execute_sets(sql) == [[{"a": 1}], [{"b": 2}], [{"c": 3}]]
        assert db_conn.execute_safe(sql, sets=True)[0] == db_conn.execute_sets(sql)
        assert db_conn.execute_safe(sql)[0] == [{"c": 3}]   # the shapes share one execution


class TestSplitStatements:
    def test_trigger_body_stays_one_statement(self):
        """The naive `;` split shipped `END;` as its own statement, and the evaluator
        rejected every trigger prediction with `incomplete input`."""
        trigger = ("CREATE TRIGGER t AFTER INSERT ON x FOR EACH ROW BEGIN "
                   "INSERT INTO y (a) VALUES (NEW.a); END;")
        assert split_statements(trigger) == [trigger]
        assert split_statements(f"DELETE FROM y; {trigger}") == ["DELETE FROM y;", trigger]

    def test_terminator_is_kept_and_blanks_dropped(self):
        assert split_statements("SELECT 1; ; SELECT 2") == ["SELECT 1;", "SELECT 2;"]
        assert split_statements("") == []


class TestSqlglotDialect:
    @pytest.mark.parametrize(("sqlalchemy", "expected"), [
        ("postgresql", "postgres"), ("awsathena", "athena"), ("mariadb", "mysql"),
        ("sqlite", "sqlite"), ("mysql", "mysql"),
    ])
    def test_the_names_sqlglot_spells_differently_are_translated(self, sqlalchemy, expected):
        """sqlglot *raises* on SQLAlchemy's spelling, and every caller swallowed it: an
        unbounded SELECT went uncapped and every prose answer was dropped on those backends."""
        import sqlglot

        assert sqlglot_name(sqlalchemy) == expected
        assert sqlglot.parse_one("SELECT 1 FROM t", dialect=expected) is not None
