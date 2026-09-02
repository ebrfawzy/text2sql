"""SQLAlchemy-based database connection manager.

Works with any database that has a SQLAlchemy driver:
  - SQLite (built-in)
  - PostgreSQL (psycopg2)
  - MySQL (pymysql)
  - Snowflake (snowflake-sqlalchemy)
  - DuckDB, ClickHouse, etc.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import time
import warnings
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine, Inspector
from sqlalchemy.exc import SAWarning
from sqlalchemy.orm import Session, sessionmaker

warnings.filterwarnings("ignore", category=SAWarning)

logger = logging.getLogger(__name__)

EXEC_TIMEOUT_S = 300.0

# `_abort_after` firing surfaces as a bare interrupt; a real fault always names something.
TIMED_OUT = re.compile(r"\)\s*interrupted\b")


@contextmanager
def _abort_after(dbapi: Any, seconds: float) -> Generator[None, None, None]:
    """Interrupt a running SQLite statement at a deadline; waiting on it does not stop it.

    Args:
        dbapi: The raw sqlite3 connection.
        seconds: Budget from now.

    Yields:
        None, with the handler installed.
    """
    deadline = time.monotonic() + seconds
    dbapi.set_progress_handler(lambda: time.monotonic() > deadline, 10_000)
    try:
        yield
    finally:
        dbapi.set_progress_handler(None, 0)


# SQLAlchemy dialect names that sqlglot spells differently.
_SQLGLOT_NAMES = {"postgresql": "postgres",
                  "awsathena": "athena", "mariadb": "mysql"}


def sqlglot_name(dialect: str) -> str:
    """Translate a SQLAlchemy dialect name to sqlglot's spelling.

    Args:
        dialect: The SQLAlchemy dialect name.

    Returns:
        The sqlglot name; sqlglot raises on SQLAlchemy's spelling of postgres, athena and
        mariadb.
    """
    return _SQLGLOT_NAMES.get(dialect, dialect)


def split_statements(sql: str) -> list[str]:
    """Split SQL on statement boundaries, keeping each terminator.

    SQLite-shaped: it knows a ``CREATE TRIGGER ... BEGIN ...; END`` body holds its own
    ``;``, but nothing of Postgres ``$$`` or MySQL DELIMITER.

    Args:
        sql: One or more statements.

    Returns:
        The statements, each terminated; an unterminated tail is kept for the driver to
        report.
    """
    out, buf = [], ""
    for part in (sql or "").split(";"):
        buf += part + ";"
        if sqlite3.complete_statement(buf):
            if stmt := buf.strip().strip(";").strip():
                out.append(f"{stmt};")
            buf = ""
    if stmt := buf.strip().strip(";").strip():  # unterminated tail: let the driver report it
        out.append(f"{stmt};")
    return out


class DatabaseConnection:
    """Manages a single database connection via SQLAlchemy.

    Usage::

        db = DatabaseConnection("sqlite:///example.db")
        schema = db.get_schema()
        rows = db.execute("SELECT COUNT(*) FROM users")
        db.close()
    """

    def __init__(self, uri: str, **engine_kwargs: Any) -> None:
        """Initialize a database connection.

        Args:
            uri: SQLAlchemy connection URI.
            **engine_kwargs: Extra kwargs for ``create_engine`` (e.g. pool_size, echo).
        """
        self.uri = uri
        self._engine: Engine = create_engine(uri, **engine_kwargs)
        self._session_factory = sessionmaker(bind=self._engine)
        self._inspector: Inspector | None = None
        self._schema_cache: dict[str, Any] | None = None
        # Per-request memo of read-only executions: {param-free SQL: (result sets, error)}.
        self._exec_cache: dict[str,
                               tuple[list[list[dict[str, Any]]] | None, str | None]] = {}
        logger.info("Connected to database: %s", self._safe_uri())

    def reset_cache(self) -> None:
        """Forget memoized executions. Called once per request so results never go stale."""
        self._exec_cache.clear()

    def quote_identifier(self, identifier: str) -> str:
        """Quote a table or column identifier for the current SQL dialect.

        Args:
            identifier: The name to quote.

        Returns:
            The name in backticks for MySQL/MariaDB and double quotes elsewhere, with any
            embedded quote characters escaped.
        """
        if self.dialect_name in ("mysql", "mariadb"):
            return "`" + identifier.replace("`", "``") + "`"
        return '"' + identifier.replace('"', '""') + '"'

    def cast_to_text(self, expr_sql: str) -> str:
        """Render a dialect-correct ``CAST(<expr> AS <string type>)`` fragment.

        Args:
            expr_sql: The expression to cast.

        Returns:
            The fragment, with the target type taken from SQLAlchemy's type compiler, since
            a hardcoded ``TEXT`` is rejected by Trino/Athena/MySQL.
        """
        from sqlalchemy import String, cast, literal_column

        return str(
            cast(literal_column(expr_sql), String).compile(
                dialect=self._engine.dialect,
                compile_kwargs={"literal_binds": True},
            )
        )

    def approx_count_distinct(self, expr_sql: str) -> str:
        """Render a distinct-count fragment, approximate where the dialect supports it.

        Approximate cardinality is far cheaper than an exact ``COUNT(DISTINCT ...)`` on
        warehouses.

        Args:
            expr_sql: The expression to count.

        Returns:
            The fragment, falling back to the exact, portable form.
        """
        if self.dialect_name == "awsathena":  # Trino/Presto
            return f"approx_distinct({expr_sql})"
        if self.dialect_name == "snowflake":
            return f"APPROX_COUNT_DISTINCT({expr_sql})"
        if self.dialect_name == "redshift":
            return f"APPROXIMATE COUNT(DISTINCT {expr_sql})"
        # sqlite / postgresql / mysql / fallback
        return f"COUNT(DISTINCT {expr_sql})"

    def _safe_uri(self) -> str:
        """Mask the password in the connection URI.

        Returns:
            The URI, safe to log.
        """
        if "@" in self.uri:
            pre, post = self.uri.split("@", 1)
            if ":" in pre:
                scheme_user = pre.rsplit(":", 1)[0]
                return f"{scheme_user}:***@{post}"
        return self.uri

    @property
    def dialect_name(self) -> str:
        """The SQLAlchemy dialect name, e.g. ``sqlite``, ``postgresql``, ``mysql``."""
        return self._engine.dialect.name

    @property
    def sqlglot_dialect(self) -> str:
        """``dialect_name`` under sqlglot's spelling."""
        return sqlglot_name(self.dialect_name)

    @property
    def inspector(self) -> Inspector:
        """Lazy-load and cache the SQLAlchemy Inspector."""
        if self._inspector is None:
            self._inspector = inspect(self._engine)
        return self._inspector

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """Yield a managed SQLAlchemy session."""
        s = self._session_factory()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    def execute(self, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Execute SQL and return the last result set.

        Args:
            sql: One or more statements.
            params: Bind parameters.

        Returns:
            Rows from the last row-producing statement.
        """
        sets = self.execute_sets(sql, params)
        return sets[-1] if sets else []

    def execute_sets(self, sql: str,
                     params: dict[str, Any] | None = None) -> list[list[dict[str, Any]]]:
        """Execute SQL and return every result set, so a batched probe keeps them all.

        Args:
            sql: One or more statements.
            params: Bind parameters.

        Returns:
            One result set per row-producing statement. Every statement runs in a
            transaction that is always rolled back, DDL included.
        """
        stmts = split_statements(sql)
        with self._engine.connect() as conn:
            if self.dialect_name == "sqlite":
                dbapi = conn.connection.dbapi_connection
                dbapi.isolation_level = None
                conn.exec_driver_sql("BEGIN")
                try:
                    with _abort_after(dbapi, EXEC_TIMEOUT_S):
                        return self._run_statements(conn, stmts, params)
                finally:
                    conn.exec_driver_sql("ROLLBACK")
                    logger.debug("Transaction rolled back.")
            trans = conn.begin()
            try:
                return self._run_statements(conn, stmts, params)
            finally:
                trans.rollback()
                logger.debug("Transaction rolled back.")

    def _run_statements(
        self, conn: Any, stmts: list[str], params: dict[str, Any] | None
    ) -> list[list[dict[str, Any]]]:
        """Execute each statement on an open connection.

        Args:
            conn: The open connection.
            stmts: Statements to run.
            params: Bind parameters.

        Returns:
            One result set per row-producing statement.
        """
        sets: list[list[dict[str, Any]]] = []
        for stmt in stmts:
            logger.debug("Executing SQL: %s", stmt)
            result = conn.execute(text(stmt), params or {})
            if result.returns_rows:
                columns = list(result.keys())
                sets.append([dict(zip(columns, row))
                            for row in result.fetchall()])
        return sets

    def execute_safe(
        self, sql: str, params: dict[str, Any] | None = None, *, sets: bool = False
    ) -> tuple[Any, str | None]:
        """Execute SQL without raising.

        Args:
            sql: One or more statements.
            params: Bind parameters.
            sets: Return every result set rather than the last.

        Returns:
            ``(rows, error)``. Identical parameter-free SQL is cached within one request,
            and the cache holds every set, so the two shapes never cost two executions.
        """
        if params is None and sql in self._exec_cache:
            outcome = self._exec_cache[sql]
        else:
            try:
                outcome = (self.execute_sets(sql, params), None)
            except Exception as e:
                logger.warning("SQL execution error: %s", e)
                outcome = (None, str(e))
            if params is None:
                self._exec_cache[sql] = outcome
        rows, error = outcome
        if rows is None or sets:
            return rows, error
        return (rows[-1] if rows else []), error

    def get_table_names(self) -> list[str]:
        """List every table in the database.

        Returns:
            Table names.
        """
        return self.inspector.get_table_names()

    def list_columns(self) -> dict[str, list[str]]:
        """Map every table to its column names, which feeds the profiling-selection UIs.

        Returns:
            ``{table: [columns]}``.
        """
        return {t: [c["name"] for c in info["columns"]]
                for t, info in self.get_schema()["tables"].items()}

    def get_schema(self) -> dict[str, Any]:
        """Introspect the full schema, memoized for the life of the connection.

        Callers must treat the returned dict as read-only.

        Returns::

            {
                "tables": {
                    "users": {
                        "columns": [
                            {"name": "id", "type": "INTEGER", "nullable": False, "primary_key": True},
                            {"name": "email", "type": "VARCHAR(255)", "nullable": False, "primary_key": False},
                            ...
                        ],
                        "primary_keys": ["id"],
                        "foreign_keys": [
                            {"column": "org_id", "referred_table": "orgs", "referred_column": "id"}
                        ],
                        "row_count": None  # Populated by profiler
                    },
                    ...
                },
                "dialect": "sqlite"
            }
        """
        if self._schema_cache is None:
            self._schema_cache = self._build_schema()
        return self._schema_cache

    def _build_schema(self) -> dict[str, Any]:
        """Introspect the database and build the structured schema dict.

        Returns:
            ``{"tables": {...}, "dialect": name}``.
        """
        tables: dict[str, Any] = {}

        for table_name in self.get_table_names():
            columns = []
            pk_cols = set(self.inspector.get_pk_constraint(
                table_name).get("constrained_columns", []))

            for col in self.inspector.get_columns(table_name):
                columns.append(
                    {
                        "name": col["name"],
                        "type": str(col["type"]),
                        "nullable": col.get("nullable", True),
                        "primary_key": col["name"] in pk_cols,
                        "default": str(col.get("default")) if col.get("default") else None,
                    }
                )

            foreign_keys = []
            for fk in self.inspector.get_foreign_keys(table_name):
                for i, col in enumerate(fk.get("constrained_columns", [])):
                    referred_cols = fk.get("referred_columns", [])
                    foreign_keys.append(
                        {
                            "column": col,
                            "referred_table": fk.get("referred_table", ""),
                            "referred_column": referred_cols[i] if i < len(referred_cols) else "",
                        }
                    )

            tables[table_name] = {
                "columns": columns,
                "primary_keys": list(pk_cols),
                "foreign_keys": foreign_keys,
                "row_count": None,  # Filled by profiler
            }

        return {"tables": tables, "dialect": self.dialect_name}

    def get_schema_ddl(self) -> str:
        """Render CREATE TABLE statements for every table.

        Returns:
            The DDL, read from ``sqlite_master`` on SQLite and reconstructed from Inspector
            metadata elsewhere.
        """
        if self.dialect_name == "sqlite":
            return self._get_sqlite_ddl()
        return self._get_generic_ddl()

    def _get_sqlite_ddl(self) -> str:
        """Read the DDL directly from ``sqlite_master``.

        Returns:
            The CREATE TABLE statements.
        """
        rows = self.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL ORDER BY name")
        return "\n\n".join(row["sql"] + ";" for row in rows)

    def _get_generic_ddl(self) -> str:
        """Reconstruct the DDL from Inspector metadata.

        Returns:
            The CREATE TABLE statements.
        """
        schema = self.get_schema()
        ddl_parts = []

        for table_name, table_info in schema["tables"].items():
            col_defs = []
            for col in table_info["columns"]:
                parts = [
                    f'  {self.quote_identifier(col["name"])} {col["type"]}']
                if col["primary_key"]:
                    parts.append("PRIMARY KEY")
                if not col["nullable"]:
                    parts.append("NOT NULL")
                col_defs.append(" ".join(parts))

            for fk in table_info["foreign_keys"]:
                col_defs.append(
                    f'  FOREIGN KEY ({self.quote_identifier(fk["column"])}) '
                    f'REFERENCES {self.quote_identifier(fk["referred_table"])} '
                    f'({self.quote_identifier(fk["referred_column"])})'
                )

            ddl = f'CREATE TABLE {self.quote_identifier(table_name)} (\n' + \
                ",\n".join(col_defs) + "\n);"
            ddl_parts.append(ddl)

        return "\n\n".join(ddl_parts)

    def close(self) -> None:
        """Dispose of the engine and close all connections."""
        self._engine.dispose()
        logger.info("Database connection closed: %s", self._safe_uri())

    def __enter__(self) -> DatabaseConnection:
        """Enter the context manager.

        Returns:
            This connection.
        """
        return self

    def __exit__(self, *args: Any) -> None:
        """Close the connection on exit."""
        self.close()
