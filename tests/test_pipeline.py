"""Tests for pipeline modules — Repair checkers, Selector, ExampleStore, Generator, Tracer.

Covers the 8-checker cascade, candidate selection modes, example store
search, schema order randomization, and pipeline tracing lifecycle.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from text2sql.pipeline.repair import (
    CHECKERS,
    Issue,
    actionable,
    check_as_stored,
    check_division,
    check_dry_run,
    check_join,
    check_json_compare,
    check_null,
    check_order_by,
    check_rebuild,
    check_rename,
    check_result,
    check_returning,
    check_rowid_pk,
    check_syntax,
    check_time,
)
from text2sql.prompts.manager import PromptManager

# ── Checker cascade metadata ────────────────────────────────────


class TestCheckerCascade:
    def test_cascade_order(self):
        """Syntax before logic before quality: the driver repairs on the most severe finding,
        and `result` runs last because a 0-row read is the weakest evidence of a fault."""
        assert [name for name, _ in CHECKERS] == [
            "syntax", "dry_run", "join", "order_by", "time", "null", "division",
            "json_compare", "as_stored", "returning", "rename", "rebuild", "rowid_pk",
            "precision", "result"]


class TestSchemaTier:
    """Which schema a finding's fix actually needs — the repair prompt used to dump one
    for every finding, including `add IS NOT NULL`, which invites a wholesale rewrite."""

    @pytest.mark.parametrize(("sql", "check", "tier"), [
        ("SELECT * FROM t WHERE x -> '$.k' = 'v'", check_json_compare, "none"),
        ("DELETE FROM users RETURNING id", check_returning, "none"),
        ("SELECT * FROM users WHERE d > '2024-01'", check_time, "none"),
        ("SELECT * FROM users JOIN orders", check_join, "touched"),
    ])
    def test_tier_per_checker(self, sql, check, tier):
        found = check(sql)
        assert found is not None and found.schema == tier

    def test_integer_division_needs_no_schema(self, db_conn):
        """It needs the column *types*, which the checker already read — not the prompt."""
        assert check_division("SELECT id / age FROM users", db_conn).schema == "none"

    def test_dry_run_tier_follows_the_error(self, db_conn):
        """45 of 59 measured dry_run findings were dialect errors (`no such function: LEAST`),
        where the schema says nothing; only a missing name needs it, and needs it un-narrowed
        because the column lives in a table the query does not touch."""
        assert check_dry_run("SELECT nope FROM users", db=db_conn).schema == "full"
        assert check_dry_run("SELECT LEAST(1, 2) FROM users", db=db_conn).schema == "none"


class TestIssue:
    def test_defaults(self):
        issue = Issue("bad", "fix it")
        assert (issue.severity, issue.schema) == ("warning", "touched")
        assert issue.message == "bad"
        assert issue.directive == "fix it"


# ── Individual checkers (return Issue on failure, None on pass) ───


class TestSyntaxChecker:
    def test_valid_sql(self):
        assert check_syntax("SELECT COUNT(*) FROM users") is None

    def test_complex_valid_sql(self):
        sql = "SELECT u.name, COUNT(o.id) FROM users u JOIN orders o ON u.id = o.user_id GROUP BY u.name"
        assert check_syntax(sql) is None

    def test_invalid_sql(self):
        # sqlglot is lenient; just assert the call returns Issue | None.
        assert check_syntax("SELECTX * FROMM users") is None or isinstance(check_syntax("x ("), Issue)


class TestDryRunChecker:
    def test_without_db_always_passes(self):
        assert check_dry_run("SELECT 1", db=None) is None

    def test_valid_query_passes(self, db_conn):
        assert check_dry_run("SELECT 1", db=db_conn) is None

    def test_invalid_table_fails(self, db_conn):
        result = check_dry_run("SELECT * FROM nonexistent", db=db_conn)
        assert result is not None
        assert "fail" in result.message.lower() or "error" in result.message.lower()
        assert result.severity == "error"

    def test_the_execution_deadline_never_asks_for_a_rewrite(self, db_conn, monkeypatch):
        """`museum_8` submitted a passing 53s query; the 50s interrupt read as a fault and
        repair rewrote it into a wrong one."""
        monkeypatch.setattr(db_conn, "execute_safe",
                            lambda *a, **k: (None, "(sqlite3.OperationalError) interrupted"))
        result = check_dry_run("SELECT 1", db=db_conn)
        assert result is not None and result.severity == "info"
        assert not actionable([("dry_run", result)])


class TestJoinChecker:
    def test_proper_join(self):
        sql = "SELECT * FROM users JOIN orders ON users.id = orders.user_id"
        assert check_join(sql) is None

    def test_cartesian_product(self):
        result = check_join("SELECT * FROM users, orders")
        assert result is not None
        assert "Cartesian" in result.message

    def test_comma_join_with_where(self):
        sql = "SELECT * FROM users, orders WHERE users.id = orders.user_id"
        assert check_join(sql) is None

    def test_cross_join_allowed(self):
        assert check_join("SELECT * FROM users CROSS JOIN orders") is None

    def test_single_table(self):
        assert check_join("SELECT * FROM users WHERE age > 20") is None

    def test_an_identifier_ending_in_join_is_not_a_join(self):
        """`insider_3` named a CTE `trader_compliance_join`; with no word boundary before
        JOIN the checker counted it twice, and the agent spent three review rounds and
        double the tokens chasing a join that was never missing."""
        sql = ("WITH trader_compliance_join AS (SELECT 1 AS x) "
               "SELECT * FROM trader_compliance_join WHERE x > 0")
        assert check_join(sql) is None

    @pytest.mark.parametrize("sql", [
        "SELECT * FROM users JOIN orders\nON users.id = orders.user_id",   # wrapped line
        "SELECT * FROM users JOIN orders USING (id)",                      # no ON at all
    ])
    def test_a_join_condition_counts_however_it_is_written(self, sql):
        """Counting " ON " called two gold queries unjoinable, at error severity."""
        assert check_join(sql) is None


class TestConstraintNaming:
    """`museum_M_4` built the constraint correctly but anonymously; its test reads the name
    back out of `sqlite_master`, so a working CHECK still failed."""

    @staticmethod
    def _fires(question, sql):
        from text2sql.pipeline.repair import _naming_conflict
        return _naming_conflict(sql, question) is not None

    def test_fires_when_the_named_constraint_is_anonymous(self):
        q = "Add a data integrity constraint 'hist_sign_rating_check' to the table."
        assert self._fires(q, "CREATE TABLE t (a INTEGER CHECK (a >= 1 AND a <= 10))")

    def test_fires_when_a_ctas_discards_what_the_question_declares(self):
        """`museum_M_3` answered "add an integer column, then create three partitions" with
        CREATE TABLE ... AS SELECT, which carries no type, key or CHECK — and its test inserts
        rows that omit the primary key."""
        q = "Add an integer column read_year, then create the partitioned table."
        found = self._issue(q, "CREATE TABLE part_2023 AS SELECT * FROM readings;")
        assert found is not None and found.severity == "error"
        assert "VIEW" in found.directive  # SQLite has no partitioning

    @pytest.mark.parametrize(("question", "sql"), [
        # Gold's shape: both instances that name a constraint declare it by name.
        ("Add a constraint 'hist_sign_rating_check'.",
         "CREATE TABLE t (a INTEGER, CONSTRAINT hist_sign_rating_check CHECK (a >= 1))"),
        ("List the artifacts by rating.", "SELECT * FROM t"),          # names nothing
        ("Add a check so values stay in range.", "CREATE TABLE t (a INTEGER CHECK (a >= 1))"),
        # Gold's shape for the CTAS rule: the table is declared, then filled.
        ("Add an integer column read_year.",
         "CREATE TABLE part (read_year INTEGER NOT NULL); INSERT INTO part SELECT y FROM r;"),
        ("Summarise the readings per year.", "CREATE TABLE s AS SELECT y FROM r;"),
    ])
    def test_silent_otherwise(self, question, sql):
        assert not self._fires(question, sql)

    @staticmethod
    def _issue(question, sql):
        from text2sql.pipeline.repair import _naming_conflict
        return _naming_conflict(sql, question)


class TestRebuildChecker:
    def test_fires_when_a_rebuild_drops_the_constraints(self):
        """SQLite has no ADD CONSTRAINT, so a table is rebuilt — and `CREATE TABLE ... AS
        SELECT` copies none, which is how one agent run lost the CHECK it was asked to add."""
        found = check_rebuild("CREATE TABLE t_new AS SELECT * FROM t; DROP TABLE t;")
        assert found is not None and found.severity == "error"

    @pytest.mark.parametrize("sql", [
        # Gold's shape: the replacement is declared with its constraints, then filled.
        "CREATE TABLE t_new (a INTEGER CHECK (a > 0)); INSERT INTO t_new SELECT * FROM t; DROP TABLE t;",
        "CREATE TABLE t_copy AS SELECT * FROM t;",       # a copy that replaces nothing
        "CREATE TEMP TABLE tmp AS SELECT * FROM t; DROP TABLE tmp;",   # scratch, no constraints
        "DROP VIEW IF EXISTS v; CREATE VIEW v AS SELECT * FROM t;",    # views carry none
        "DROP TABLE t;",
    ])
    def test_silent_otherwise(self, sql):
        assert check_rebuild(sql) is None


class TestPrecisionChecker:
    """The scorer rounds floats to 2 places, so rounding a *stored* value below that discards
    a difference a write test can see: `polar_M_3`'s test asserts the updated values are not
    equal to their own 2-place rounding."""

    @pytest.mark.parametrize(("sql", "flagged"), [
        ("UPDATE t SET x = ROUND(x * 1.15, 1)", True),
        ("UPDATE t SET x = ROUND(x * 1.15)", True),
        # Gold's shape: it rounds writes to 2 and 4 places, both invisible to the scorer.
        ("UPDATE t SET x = ROUND(x * 1.15, 2)", False),
        ("UPDATE t SET x = ROUND(x, 4)", False),
        ("UPDATE t SET x = x * 1.15", False),
        ("SELECT ROUND(x, 1) FROM t", False),        # reads are stripped of ROUND anyway
    ])
    def test_only_a_write_that_loses_precision_is_flagged(self, sql, flagged):
        from text2sql.pipeline.repair import check_precision
        assert bool(check_precision(sql)) is flagged


class TestAsStoredChecker:
    """The grader compares values, and the corpus returns them as stored: `alien_3`, `alien_7`,
    `crypto_1`, `fake_4` and `museum_8` each differ from gold only by a trimmed padded value."""

    @pytest.mark.parametrize("sql", [
        "SELECT TRIM(a) FROM t",
        "SELECT LOWER(a) AS a FROM t",
        "SELECT TRIM(json_extract(a, '$.k')) FROM t",
    ])
    def test_fires_on_a_stored_value_the_projection_changes(self, sql):
        found = check_as_stored(sql)
        assert found is not None and found.severity == "warning"

    @pytest.mark.parametrize("sql", [
        "SELECT a FROM t WHERE TRIM(a) = 'x'",                      # a predicate may trim
        "SELECT a FROM t JOIN u ON TRIM(t.a) = TRIM(u.b)",          # so may a join key
        "SELECT TRIM(CASE WHEN a THEN 'x, ' ELSE '' END, ', ') FROM t",  # its own string
        "SELECT TRIM(a || b) FROM t",
        "SELECT a FROM (SELECT TRIM(a) AS a FROM t)",               # only the outermost counts
        "UPDATE t SET a = TRIM(a)",                                 # not a read
        "SELECT a FROM t",
    ])
    def test_silent_otherwise(self, sql):
        assert check_as_stored(sql) is None


class TestRowidPkChecker:
    """Only the bare type `INTEGER` makes a primary key a rowid alias."""

    @pytest.mark.parametrize("sql", [
        'CREATE TABLE t ("a" integer(64) NOT NULL PRIMARY KEY, b TEXT)',
        # `museum_M_4` declared the key this way and every test insert died on NOT NULL.
        'CREATE TABLE t ("a" integer(64) NOT NULL, b TEXT, CONSTRAINT p PRIMARY KEY ("a"))',
    ])
    def test_fires_on_a_sized_integer_primary_key(self, sql):
        found = check_rowid_pk(sql)
        assert found is not None and found.severity == "error"
        assert "a INTEGER PRIMARY KEY" in found.directive

    @pytest.mark.parametrize("sql", [
        "CREATE TABLE t (a INTEGER PRIMARY KEY, b TEXT)",              # the rowid alias
        "CREATE TABLE t (a INTEGER NOT NULL PRIMARY KEY, b TEXT)",     # still one
        "CREATE TABLE t (a text(10) PRIMARY KEY, b TEXT)",             # no rowid alias to lose
        "CREATE TABLE t (a integer(64), b TEXT)",                      # not a key
        "CREATE TABLE t AS SELECT * FROM u",
        "SELECT * FROM t",
    ])
    def test_silent_otherwise(self, sql):
        assert check_rowid_pk(sql) is None


class TestRenameChecker:
    """A CREATE's column names are graded: `crypto_M_4` aliased `o.recordvault AS ecordvault`,
    honouring a typo in the question, and its view test asserts the real name."""

    def test_fires_on_a_near_miss_of_the_source_column(self):
        found = check_rename("CREATE VIEW v AS SELECT o.recordvault AS ecordvault FROM orders o")
        assert found is not None and found.severity == "warning" and found.schema == "none"
        assert "recordvault" in found.message

    @pytest.mark.parametrize("sql", [
        # Gold renames passthrough columns deliberately three times; none resembles its source.
        "CREATE VIEW v AS SELECT d.devregistry AS device_id FROM d",
        "CREATE VIEW v AS SELECT t.trdref AS trader_id FROM t",
        "CREATE VIEW v AS SELECT e.effectivenessrobot AS robot_id FROM e",
        # A computed column takes its name from the question, and gold expects that.
        "CREATE VIEW v AS SELECT CASE WHEN a THEN b ELSE c END AS available_liquidity FROM t",
        "CREATE VIEW v AS SELECT o.recordvault FROM orders o",     # no alias at all
        "SELECT o.recordvault AS ecordvault FROM orders o",        # a read: names are stripped
    ])
    def test_silent_otherwise(self, sql):
        assert check_rename(sql) is None


class TestReturningChecker:
    @pytest.mark.parametrize("sql", [
        "DELETE FROM users WHERE id = 1 RETURNING id",
        "UPDATE users SET age = 1 RETURNING id",
        "CREATE TABLE t(a); DELETE FROM users RETURNING id",   # not the first statement
    ])
    def test_fires_when_the_scorer_would_commit_mid_cursor(self, sql):
        """The scorer commits a statement not opening WITH/SELECT before fetching, so
        RETURNING dies there — while it executes fine here, so `dry_run` cannot catch it."""
        found = check_returning(sql)
        assert found is not None and found.severity == "error"

    @pytest.mark.parametrize("sql", [
        "WITH old AS (SELECT id FROM users) DELETE FROM users WHERE id IN (SELECT id FROM old) RETURNING id",
        "DELETE FROM users WHERE id = 1",
        "SELECT * FROM users",
        "SELECT 'RETURNING' AS label FROM users",   # the word, not the clause
    ])
    def test_silent_otherwise(self, sql):
        """A leading CTE takes the scorer's fetch path, so RETURNING survives — and one gold
        test asserts the RETURNING clause returns three columns, so it must not be dropped."""
        assert check_returning(sql) is None


class TestOrderByChecker:
    def test_limit_without_order_by(self):
        result = check_order_by("SELECT * FROM users LIMIT 5")
        assert result is not None
        assert "LIMIT without ORDER BY" in result.message

    def test_limit_with_order_by(self):
        assert check_order_by("SELECT * FROM users ORDER BY name LIMIT 5") is None

    def test_no_limit(self):
        assert check_order_by("SELECT * FROM users") is None

    def test_order_by_without_limit(self):
        assert check_order_by("SELECT * FROM users ORDER BY name") is None

    def test_order_by_unknown_column_flagged_with_db(self, db_conn):
        result = check_order_by("SELECT * FROM users ORDER BY nope", db=db_conn)
        assert result is not None
        assert "nope" in result.message
        assert result.severity == "error"

    def test_order_by_real_column_passes_with_db(self, db_conn):
        assert check_order_by("SELECT * FROM users ORDER BY name", db=db_conn) is None

    def test_order_by_alias_passes_with_db(self, db_conn):
        sql = "SELECT COUNT(*) AS cnt FROM users ORDER BY cnt"
        assert check_order_by(sql, db=db_conn) is None


class TestTimeChecker:
    def test_no_dates(self):
        assert check_time("SELECT * FROM users") is None

    def test_date_function_passes(self):
        sql = "SELECT * FROM orders WHERE DATE(created_at) > '2025-01-01'"
        assert check_time(sql) is None

    def test_string_date_comparison_warns(self):
        sql = "SELECT * FROM orders WHERE created_at > '2025-01-01'"
        result = check_time(sql)
        assert result is not None
        assert "date" in result.message.lower()


class TestNullChecker:
    def test_order_by_asc_without_not_null(self, db_conn):
        result = check_null("SELECT * FROM users ORDER BY age ASC LIMIT 1", db_conn)
        assert result is not None
        assert "NULL" in result.message

    def test_order_by_asc_with_not_null(self, db_conn):
        sql = "SELECT * FROM users WHERE age IS NOT NULL ORDER BY age ASC LIMIT 1"
        assert check_null(sql, db_conn) is None

    def test_a_non_nullable_column_cannot_sort_nulls_first(self, db_conn):
        """`name` is NOT NULL and `id` is the PK: neither can put a NULL anywhere."""
        assert check_null("SELECT * FROM users ORDER BY name ASC", db_conn) is None
        assert check_null("SELECT * FROM users ORDER BY id LIMIT 3", db_conn) is None

    def test_a_bare_min_is_not_a_finding(self, db_conn):
        """Flagging MIN() without asking the data fired on a gold whose column holds no NULL,
        and over 540 measured predictions never changed a verdict."""
        assert check_null("SELECT MIN(age) FROM users", db_conn) is None

    def test_no_asc_ordering(self, db_conn):
        assert check_null("SELECT * FROM users", db_conn) is None


class TestResultChecker:
    def test_without_db(self):
        assert check_result("SELECT 1", db=None) is None

    def test_non_empty_result(self, db_conn):
        assert check_result("SELECT * FROM users", db=db_conn) is None

    def test_empty_result_warns(self, db_conn):
        result = check_result("SELECT * FROM users WHERE age > 999", db=db_conn)
        assert result is not None
        assert "0 rows" in result.message

    def test_an_all_null_row_is_the_same_non_answer(self, db_conn):
        """An aggregate over an empty set returns one NULL row, not none — the right shape and
        no answer. One measured run submitted exactly that; no gold of the 270 returns one."""
        result = check_result("SELECT AVG(age) AS a FROM users WHERE age > 999", db=db_conn)
        assert result is not None and "only NULLs" in result.message


class TestDivisionAndJsonCheckers:
    """The two rules that map to counted LiveSQLBench failure modes, now enforced."""

    def test_a_nullif_guard_is_wanted_only_where_the_data_holds_a_zero(self, db_conn):
        """The shape alone flagged 42 of 270 golds, and in all 58 resolvable cases the
        denominator held neither a zero nor a NULL — so ask the data, not the syntax."""
        from text2sql.pipeline.repair import _has_zero, check_division
        assert _has_zero(db_conn, "email") and not _has_zero(db_conn, "id")
        assert not _has_zero(db_conn, "nosuchcolumn")   # an alias or CTE column fails open
        assert check_division("SELECT amount / age FROM orders, users", db_conn) is None

    def test_integer_division_truncation_needs_the_schema(self, db_conn):
        """`id / age` are both INTEGER, so the remainder is discarded — only the schema says so."""
        from text2sql.pipeline.repair import check_division
        sql = "SELECT id / NULLIF(age, 0) FROM users"
        assert check_division(sql) is None                 # dbless: types unknown, stay quiet
        assert "truncates" in check_division(sql, db_conn).message

    @pytest.mark.parametrize("sql, flagged", [
        ("SELECT c -> '$.k' = 'High' FROM users", True),
        ("SELECT c ->> '$.k' = 'High' FROM users", False),
        # sqlglot maps json_extract() and -> to the same node, and only -> is wrong
        ("SELECT json_extract(c, '$.k') = 'High' FROM users", False),
    ])
    def test_json_arrow_compared_to_a_bare_string(self, sql, flagged):
        from text2sql.pipeline.repair import check_json_compare
        assert bool(check_json_compare(sql)) is flagged


class TestRepairSafety:
    """A repair that cannot run is worse than the query it replaced."""

    async def test_a_rewrite_that_does_not_run_is_discarded(self, db_conn, prompt_manager):
        """One measured run turned an executable diagnostic into three rounds of SQL naming
        invented tables, and shipped the last of them."""
        from text2sql.pipeline.repair import SQLRepair

        class LLM:
            async def chat_for_sql(self, *a, **kw):
                return "SELECT * FROM no_such_table"

        repair = SQLRepair(LLM(), db_conn, prompt_manager, max_retries=2)
        # A read returning no rows is a warning, so a rewrite is attempted.
        original = "SELECT * FROM users WHERE age > 999"
        sql, issues = await repair.repair(original, "q", "schema")
        assert sql == original
        assert any("discarded" in i for i in issues)



class TestFindings:
    """One policy and one rendering for both callers: the repair loop and the agent's submit
    gate report the same findings, and had drifted into different prefixes, separators,
    orderings and directive counts."""

    FOUND = [("noisy", Issue("advice", "ignore me", "info")),
             ("late", Issue("a warning", "fix the warning")),
             ("first", Issue("cannot run", "fix the error", "error"))]

    def test_the_audit_log_keeps_every_finding_without_its_fix(self):
        from text2sql.pipeline.repair import render
        assert render(self.FOUND) == ["[noisy] advice", "[late] a warning", "[first] cannot run"]

    def test_what_an_llm_reads_is_actionable_worst_first_and_carries_every_fix(self):
        from text2sql.pipeline.repair import actionable, render
        assert render(actionable(self.FOUND), fixes=True) == [
            "[first] cannot run fix the error", "[late] a warning fix the warning"]


class TestQuestionChecker:
    """The only checker that reads the question. Measured over the 270 LiveSQLBench gold
    queries it fires on zero of them; every guard below was bought by a real one, and 37-58%
    of golds degraded by dropping ORDER BY / ROUND / LIMIT are caught."""

    @staticmethod
    def _fires(question, sql):
        from text2sql.pipeline.repair import _question_conflict
        return _question_conflict(sql, question) is not None

    def test_what_the_grader_cannot_score_is_advisory_only(self):
        """`test_case_default` strips ROUND and DISTINCT from prediction *and* gold, and this
        checker is inert on the write answers where a stored value is asserted — so neither can
        change a verdict, and neither may spend a repair round or a submission refusal."""
        from text2sql.pipeline.repair import _question_conflict

        for question in ("Report the mean, rounded to two decimal places.",
                         "List all the different showcase IDs."):
            assert _question_conflict("SELECT AVG(x) AS a FROM t", question).severity == "info"
        # Paired with a scored rule it is actionable again, and carries both messages.
        both = _question_conflict("SELECT a FROM t",
                                  "Sorted by score descending, to two decimal places.")
        assert both.severity == "warning" and "decimal places" in both.message

    @pytest.mark.parametrize(("sql", "flagged"), [
        ("SELECT AVG(x) AS a FROM t", True),
        ("SELECT ROUND(AVG(x), 2) AS a FROM t", False),
        ("SELECT ROUND(AVG(x)) AS a FROM t", True),      # a bare ROUND states no precision
    ])
    def test_stated_decimal_places_need_that_precision(self, sql, flagged):
        assert self._fires("Report the mean, rounded to two decimal places.", sql) is flagged

    @pytest.mark.parametrize(("sql", "flagged"), [
        ("SELECT CASE WHEN x > 1 THEN 'True' ELSE 'False' END AS ok FROM t", True),
        ("SELECT CASE WHEN x > 1 THEN TRUE ELSE FALSE END AS ok FROM t", False),
        ("SELECT CASE WHEN x > 1 THEN 1 ELSE 0 END AS ok FROM t", False),
    ])
    def test_a_boolean_column_is_a_boolean_not_its_name(self, sql, flagged):
        """`alien_7` grouped on 'True'/'False'; gold's TRUE/FALSE store as 1/0, and the grader
        compares the values, so the strings can never match."""
        assert self._fires("Group them by whether it qualifies (bool:True or False).",
                           sql) is flagged

    @pytest.mark.parametrize(("sql", "flagged"), [
        ("SELECT c, json_array(kind) AS kinds FROM t", True),
        ("SELECT c, GROUP_CONCAT(kind) AS kinds FROM t GROUP BY c", False),
        # Four golds build a JSON array deliberately - from an aggregate, not a bare column.
        ("SELECT c, json_group_array(kind) AS kinds FROM t GROUP BY c", False),
        ("SELECT c, json_array(kind, other) AS kinds FROM t", False),
    ])
    def test_an_array_of_values_must_aggregate_them(self, sql, flagged):
        assert self._fires("Show the id and an array of alert types.", sql) is flagged

    @pytest.mark.parametrize(("sql", "flagged"), [
        ("SELECT a.x FROM a JOIN b ON b.k = a.k", True),
        ("SELECT a.x FROM a LEFT JOIN b ON b.k = a.k", False),
        ("SELECT a.x FROM a", False),  # nothing to keep on the other side
    ])
    def test_keeping_unmatched_entities_needs_an_outer_join(self, sql, flagged):
        """Gold LEFT JOINs on 28 of this corpus's instances precisely to keep the zero rows;
        this wording fires on none of the 270 golds."""
        assert self._fires("List them, even if some lack a record.", sql) is flagged

    @pytest.mark.parametrize(("sql", "flagged"), [
        ("SELECT a FROM t", True),
        ("SELECT a FROM t ORDER BY s DESC", False),
        ("WITH c AS (SELECT a FROM t ORDER BY a) SELECT a FROM c", True),  # a CTE's, not the answer's
    ])
    def test_a_question_that_asks_for_sorting_needs_an_outer_order_by(self, sql, flagged):
        assert self._fires("List them, sorted by score in descending order.", sql) is flagged

    def test_a_rank_window_is_its_own_ordering(self):
        """One gold sorts "by influence rank" with no outer ORDER BY at all."""
        assert not self._fires("List them sorted by influence rank.", """
            WITH r AS (SELECT a, DENSE_RANK() OVER (ORDER BY s DESC) AS influence_rank FROM t)
            SELECT a FROM r WHERE influence_rank <= 10""")

    @pytest.mark.parametrize(("question", "sql", "flagged"), [
        ("Sort from highest to lowest.", "SELECT a FROM t ORDER BY s", True),
        ("Sort from highest to lowest.", "SELECT a FROM t ORDER BY s DESC", False),
        # A gold sorts "from newest to oldest" with ORDER BY panel_age_years ASC: newest is
        # the smallest age, so date words carry no direction.
        ("Sort from newest to oldest panels.", "SELECT a FROM t ORDER BY age", False),
    ])
    def test_only_explicit_direction_wording_pins_the_order_by(self, question, sql, flagged):
        assert self._fires(question, sql) is flagged

    @pytest.mark.parametrize(("sql", "flagged"), [
        ("SELECT a FROM t ORDER BY s DESC", True),
        ("SELECT a FROM t ORDER BY s DESC LIMIT 5", False),
        # One gold caps inside a CTE, another with a rank filter, a third has LIMIT and no
        # ordering at all — so this asks for a cap only, never for an ORDER BY.
        ("WITH c AS (SELECT a FROM t LIMIT 5) SELECT a FROM c", False),
        ("WITH r AS (SELECT a, RANK() OVER (ORDER BY s) rk FROM t) SELECT a FROM r WHERE rk <= 5",
         False),
    ])
    def test_top_n_needs_a_row_cap_not_an_ordering(self, sql, flagged):
        assert self._fires("Find the top 5 flows.", sql) is flagged

    @pytest.mark.parametrize(("question", "sql", "flagged"), [
        ("For each region, show the average score.", "SELECT AVG(s) AS a FROM t", True),
        ("For each region, show the average score.",
         "SELECT region, AVG(s) AS a FROM t GROUP BY region", False),
        ("For each region, show the average score.",
         "SELECT region, AVG(s) OVER (PARTITION BY region) AS a FROM t", False),
        # One gold computes "for each artifact" with a correlated subquery, and a third of
        # these questions say "for each" to mean one row per row.
        ("For each artifact, show the average sensitivity.",
         "SELECT id, (SELECT AVG(s) FROM m WHERE m.id = t.id) AS a FROM t", False),
        ("For each customer, show the ID, net worth and total assets.",
         "SELECT id, a - b AS nw, c FROM t", False),
    ])
    def test_only_a_bare_single_row_aggregate_is_a_missing_group_by(self, question, sql, flagged):
        assert self._fires(question, sql) is flagged

    @pytest.mark.parametrize(("sql", "flagged"), [
        ("SELECT SUM(s) AS a FROM t", True),
        ("SELECT AVG(s) AS a FROM t", False),
        ("SELECT SUM(s) / COUNT(*) AS a FROM t", False),  # a hand-rolled mean
    ])
    def test_average_wording_needs_an_average(self, sql, flagged):
        assert self._fires("Give the average score.", sql) is flagged

    @pytest.mark.parametrize(("question", "sql", "flagged"), [
        ("List all the different showcase IDs.", "SELECT id FROM t", True),
        ("List all the different showcase IDs.", "SELECT DISTINCT id FROM t", False),
        ("Count all the different showcase IDs.", "SELECT COUNT(DISTINCT id) AS n FROM t", False),
        # Bare "different"/"unique" is noise: one means grouping, another names a column.
        ("Show the mean score across different weather conditions.", "SELECT AVG(s) FROM t GROUP BY w",
         False),
        ("Show the inverter's unique identifier.", "SELECT id FROM t", False),
    ])
    def test_only_explicit_distinctness_wording_demands_distinct(self, question, sql, flagged):
        assert self._fires(question, sql) is flagged

    def test_definitions_are_not_requirements(self, prompt_manager):
        """Definition formulas add "average" to 13 questions and "for each" to 6. The marker
        must be the one `question_knowledge` emits: written by hand as `HINTS:` this passed
        while the template wrote `HINTS`, and the checker read every formula as the question."""
        entry = SimpleNamespace(knowledge="FSI", description="a score",
                                definition="the average of a, b and c, sorted descending")
        question = prompt_manager.render("question_knowledge", question="List the IDs.",
                                         entries=[entry])
        assert not self._fires(question, "SELECT id FROM t")

    def test_non_select_statements_are_never_checked(self):
        """28% of gold is DDL/DML, where "2 decimal places" is a NUMERIC(6,2) column spec."""
        assert not self._fires(
            "Add a score column rounded to 2 decimal places, then fill it.",
            "ALTER TABLE t ADD COLUMN s NUMERIC(6, 2); UPDATE t SET s = a + b")

    def test_the_cascade_reports_it_only_with_a_question(self, db_conn):
        """The default keeps every caller that predates the argument working."""
        from text2sql.pipeline.repair import run_checkers
        sql = "SELECT name FROM users"
        assert not [n for n, _ in run_checkers(sql, db_conn) if n == "question"]
        assert [n for n, _ in run_checkers(sql, db_conn, question="Sort by name, descending.")
                if n == "question"]


# ── SQLRepair: what actually reaches the repair prompt ───────────


class TestRepairPromptSize:
    """One repair prompt measured 26k chars: the query appeared three times (itself, SQLite's
    `near "<statement>"`, sqlglot's context) and the schema re-listed every linked table."""

    @staticmethod
    def _repairer(db_conn, schema_loader, prompt_manager):
        from unittest.mock import AsyncMock, MagicMock

        from text2sql.pipeline.repair import SQLRepair

        llm = MagicMock()
        llm.chat_for_sql = AsyncMock(return_value="SELECT 1")
        return SQLRepair(llm, db_conn, prompt_manager, max_retries=1,
                         schema_loader=schema_loader)

    def test_a_driver_never_echoes_the_query_back_into_the_message(self, db_conn):
        from text2sql.pipeline.repair import check_dry_run, check_syntax

        sql = "```SQL SELECT " + ", ".join(f"c{i}" for i in range(200)) + " FROM users```"
        for issue in (check_dry_run(sql, db_conn), check_syntax(sql)):
            assert issue is not None
            assert "c150" not in issue.message, issue.message
            assert len(issue.message) < 300
            assert "\x1b" not in issue.message  # sqlglot colour codes

    def test_the_schema_covers_only_the_tables_the_query_touches(
            self, db_conn, schema_loader, prompt_manager):
        """Tables, not the query's exact columns: the named column may be the mistake."""
        repairer = self._repairer(db_conn, schema_loader, prompt_manager)
        narrowed = repairer._schema_for("SELECT nope FROM users u", "FULL SCHEMA TEXT", [])
        assert "Table: users" in narrowed
        assert "Table: orders" not in narrowed
        assert "email" in narrowed  # the other columns of a touched table stay reachable

    def test_an_unparseable_query_keeps_the_full_schema(
            self, db_conn, schema_loader, prompt_manager):
        """Nothing to narrow to, and a schema-less repair prompt is unanswerable."""
        repairer = self._repairer(db_conn, schema_loader, prompt_manager)
        assert repairer._schema_for("%%%", "FULL SCHEMA TEXT", []) == "FULL SCHEMA TEXT"
        assert repairer._schema_for("SELECT 1 FROM ghost_table", "FULL", []) == "FULL"

    def test_an_answerless_query_keeps_the_full_schema(
            self, db_conn, schema_loader, prompt_manager):
        """The table a 0-row query is missing is the likely bug, and narrowing hides it: one run
        was left with 2 of 7 tables and invented a JSON path for the column it could not find."""
        from text2sql.pipeline.repair import Issue
        repairer = self._repairer(db_conn, schema_loader, prompt_manager)
        found = [("result", Issue("Query returned 0 rows", "", schema="full"))]
        assert repairer._schema_for("SELECT nope FROM users", "FULL", found) == "FULL"

    def test_a_mechanical_fix_gets_no_schema_at_all(
            self, db_conn, schema_loader, prompt_manager):
        """`add IS NOT NULL` rewrites what the query already says; a schema beside it is an
        invitation to rewrite the whole query, which one measured round did."""
        from text2sql.pipeline.repair import Issue
        repairer = self._repairer(db_conn, schema_loader, prompt_manager)
        found = [("null", Issue("MIN() may return NULL", "", schema="none"))]
        assert repairer._schema_for("SELECT MIN(age) FROM users", "FULL", found) == ""
        # The widest finding wins when several fire together.
        found.append(("join", Issue("JOIN without ON", "")))
        assert "Table: users" in repairer._schema_for("SELECT MIN(age) FROM users", "FULL", found)

    async def test_the_prompt_shows_the_query_exactly_once(
            self, db_conn, schema_loader, prompt_manager):
        """`nope` is a missing name, so the schema stays un-narrowed — the table holding it is
        by definition one the query does not touch."""
        repairer = self._repairer(db_conn, schema_loader, prompt_manager)
        await repairer.repair("SELECT nope FROM users", "How many?", "FULL SCHEMA TEXT")
        prompt = repairer.llm.chat_for_sql.await_args[0][0]
        assert prompt.count("SELECT nope") == 1
        assert "FULL SCHEMA TEXT" in prompt


# ── CandidateSelector ───────────────────────────────────────────


class TestCandidateSelector:
    async def test_single_mode(self, db_conn):
        from text2sql.pipeline.selector import CandidateSelector
        selector = CandidateSelector(db=db_conn, mode="single")
        sql, results, meta = await selector.select(["SELECT COUNT(*) AS cnt FROM users"])
        assert sql == "SELECT COUNT(*) AS cnt FROM users"
        assert meta["method"] == "single_candidate"

    async def test_majority_vote(self, db_conn):
        from text2sql.pipeline.selector import CandidateSelector
        selector = CandidateSelector(db=db_conn, mode="majority")
        candidates = [
            "SELECT COUNT(*) AS cnt FROM users",
            "SELECT COUNT(*) AS cnt FROM users",
            "SELECT 999 AS cnt",
        ]
        sql, results, meta = await selector.select(candidates)
        assert results is not None
        assert meta.get("agreement", 0) >= 2

    async def test_empty_candidates(self, db_conn):
        from text2sql.pipeline.selector import CandidateSelector
        selector = CandidateSelector(db=db_conn)
        sql, results, meta = await selector.select([])
        assert sql == ""
        assert "error" in meta

    async def test_single_candidate_auto_single_mode(self, db_conn):
        from text2sql.pipeline.selector import CandidateSelector
        selector = CandidateSelector(db=db_conn, mode="majority")
        sql, results, meta = await selector.select(["SELECT 1 AS x"])
        assert meta["method"] == "single_candidate"

    async def test_all_failed(self, db_conn):
        from text2sql.pipeline.selector import CandidateSelector
        selector = CandidateSelector(db=db_conn, mode="majority")
        sql, results, meta = await selector.select([
            "SELECT * FROM no_table_1",
            "SELECT * FROM no_table_2",
        ])
        assert meta["method"] == "all_failed"

    def test_hash_results_empty(self):
        from text2sql.pipeline.selector import CandidateSelector
        assert CandidateSelector._hash_results(None) == "empty"
        assert CandidateSelector._hash_results([]) == "empty"

    def test_hash_results_deterministic(self):
        from text2sql.pipeline.selector import CandidateSelector
        rows = [{"a": 1, "b": 2}]
        h1 = CandidateSelector._hash_results(rows)
        h2 = CandidateSelector._hash_results(rows)
        assert h1 == h2

    def test_hash_results_ignores_column_names(self):
        """Candidates differing only by alias must land in the same cluster."""
        from text2sql.pipeline.selector import CandidateSelector
        h1 = CandidateSelector._hash_results([{"total": 7}])
        h2 = CandidateSelector._hash_results([{"cnt": 7}])
        assert h1 == h2

    def test_hash_results_ignores_row_order(self):
        from text2sql.pipeline.selector import CandidateSelector
        h1 = CandidateSelector._hash_results([{"a": 1}, {"a": 2}])
        h2 = CandidateSelector._hash_results([{"a": 2}, {"a": 1}])
        assert h1 == h2

    def test_hash_results_distinguishes_values(self):
        from text2sql.pipeline.selector import CandidateSelector
        h1 = CandidateSelector._hash_results([{"a": 1}])
        h2 = CandidateSelector._hash_results([{"a": 2}])
        assert h1 != h2

    async def test_majority_tie_prefers_non_empty(self, db_conn):
        """A 2-2 tie between an empty and a non-empty cluster picks the non-empty one."""
        from text2sql.pipeline.selector import CandidateSelector
        selector = CandidateSelector(db=db_conn, mode="majority")
        sql, results, meta = await selector.select([
            "SELECT id FROM users WHERE city = 'Atlantis'",  # empty
            "SELECT id FROM users WHERE city = 'Narnia'",    # empty
            "SELECT id FROM users",                          # non-empty
            "SELECT id FROM users ORDER BY id",              # same rows -> same cluster
        ])
        assert meta["method"] == "majority_vote"
        assert results  # the empty cluster must not win the tie

    def test_parse_json_response_plain(self):
        from text2sql.llm import parse_llm_json
        result = parse_llm_json('{"selected": 1}')
        assert result["selected"] == 1

    def test_parse_json_response_fenced(self):
        from text2sql.llm import parse_llm_json
        text = '```json\n{"selected": 2}\n```'
        result = parse_llm_json(text)
        assert result["selected"] == 2

    def test_parse_json_literal_newline_in_string(self):
        # LLMs emit un-escaped newlines inside long string values; strict JSON rejects them.
        from text2sql.llm import parse_llm_json
        result = parse_llm_json('{"long": "line one\nline two"}')
        assert result["long"] == "line one\nline two"

    def test_parse_json_preserves_double_slash_urls(self):
        # // comment-stripping must not corrupt s3:// URLs inside valid JSON.
        from text2sql.llm import parse_llm_json
        result = parse_llm_json('{"long": "path is s3://bucket/a/b.json.gz"}')
        assert result["long"] == "path is s3://bucket/a/b.json.gz"

    def test_parse_json_wrapped_in_prose(self):
        from text2sql.llm import parse_llm_json
        result = parse_llm_json('Sure, here it is:\n{"selected": 3}\nHope that helps!')
        assert result["selected"] == 3

    def test_parse_json_comments_and_trailing_commas_fallback(self):
        from text2sql.llm import parse_llm_json
        result = parse_llm_json('{\n  "a": 1, // note\n  "b": 2,\n}')
        assert result == {"a": 1, "b": 2}


# ── ExampleStore ─────────────────────────────────────────────────


class TestExampleStore:
    def test_empty_store(self):
        from text2sql.pipeline.examples import ExampleStore
        store = ExampleStore()
        assert store.headings == []
        assert store.search("test") == []

    def test_load_sections(self, example_store):
        assert len(example_store.headings) == 3
        assert "Revenue" in example_store.headings
        assert "Churn" in example_store.headings
        assert "Active Users" in example_store.headings

    def test_search_match(self, example_store):
        results = example_store.search("revenue")
        assert len(results) >= 1
        assert "Revenue" in results[0]

    def test_search_no_match(self, example_store):
        results = example_store.search("xyznonexistent")
        assert results == []

    def test_file_not_found_no_crash(self, tmp_path):
        from text2sql.pipeline.examples import ExampleStore
        store = ExampleStore(str(tmp_path / "missing.md"))
        assert store.headings == []

    def test_search_top_k(self, example_store):
        results = example_store.search("revenue", top_k=1)
        assert len(results) <= 1


# ── SQLGenerator ─────────────────────────────────────────────────


class TestGenerationStrategies:
    """Candidates should differ structurally, not just by sampling noise."""

    def test_named_strategy_is_constant_across_candidates(self):
        from text2sql.pipeline.generator import strategy_for
        assert strategy_for("decompose", 0) == strategy_for("decompose", 3) == "decompose"

    def test_diverse_cycles_through_every_style(self):
        from text2sql.pipeline.generator import STRATEGIES, strategy_for
        got = [strategy_for("diverse", i) for i in range(len(STRATEGIES))]
        assert len(set(got)) == len(STRATEGIES)
        assert strategy_for("diverse", len(STRATEGIES)) == got[0]  # wraps

    @pytest.mark.parametrize("strategy, expected", [
        ("direct", False), ("decompose", True), ("query_plan", True)])
    def test_only_a_named_style_adds_an_approach_block(self, prompt_manager, strategy, expected):
        """`direct` is the absence of an instruction, so it must not print an empty one."""
        rendered = prompt_manager.render("generate_sql", schema="S", question="Q",
                                         dialect="sqlite", strategy=strategy)
        assert ("<procedure>" in rendered) is expected


class TestGeneratorErrorSurfacing:
    """A rate limit that kills every candidate must not look like a model that declined:
    that scored 0 on five benchmark instances with `llm_calls: 0` and no visible cause."""

    async def test_total_failure_reraises_the_cause(self, db_conn, prompt_manager):
        from unittest.mock import MagicMock

        from text2sql.pipeline.generator import SQLGenerator

        llm = MagicMock()

        async def boom(*a, **kw):
            raise RuntimeError("rate limit exceeded")
            yield  # pragma: no cover — makes this an async generator

        llm.stream_chat_messages = boom
        gen = SQLGenerator(llm, db_conn, prompt_manager)
        with pytest.raises(RuntimeError, match="rate limit"):
            async for _ in gen.generate("Q", "S", num_candidates=2):
                pass

    async def test_one_survivor_is_enough(self, db_conn, prompt_manager):
        """A partial outage still yields whatever candidates did come back."""
        from unittest.mock import MagicMock

        from text2sql.pipeline.generator import SQLGenerator

        llm, calls = MagicMock(), []

        async def flaky(*a, **kw):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("rate limit exceeded")
            yield ("SELECT 1", False)

        llm.stream_chat_messages = flaky
        gen = SQLGenerator(llm, db_conn, prompt_manager)
        out = [i async for i in gen.generate("Q", "S", num_candidates=2)]
        assert out[-1] == ["SELECT 1"]


class TestPromptManager:
    """Template loading, rendering, and the override hooks."""

    def test_render_from_a_custom_template_dir(self, tmp_path):
        v1 = tmp_path / "v1"
        v1.mkdir()
        (v1 / "test.j2").write_text("Hello {{ name }}, your value is {{ value }}.")
        result = PromptManager(template_dir=tmp_path, version="v1").render(
            "test", name="World", value="42")
        assert "World" in result and "42" in result

    def test_render_shipped_templates(self, prompt_manager):
        sql = prompt_manager.render("generate_sql", schema="Table: users\n  id INTEGER",
                                    question="How many users?", profile_context="")
        assert "users" in sql and "How many users?" in sql
        repair = prompt_manager.render(
            "repair_sql", sql="SELECT * FROM users", schema="Table: users",
            findings=["[dry_run] column 'foo' not found Check column names"],
            question="How many users?")
        assert "foo" in repair and "SELECT * FROM users" in repair
        # No schema means no heading: an empty placeholder is something a small model fills in.
        bare = prompt_manager.render(
            "repair_sql", sql="SELECT * FROM users", schema="",
            findings=["[null] MIN() may return NULL Filter the MIN() column IS NOT NULL."],
            question="How many users?")
        assert "Schema:" not in bare and "How many users?" in bare

    def test_template_not_found(self, prompt_manager):
        from jinja2 import TemplateNotFound
        with pytest.raises(TemplateNotFound):
            prompt_manager.render("nonexistent_template_xyz")

    def test_env_var_overrides_a_single_template(self, tmp_path, monkeypatch):
        custom = tmp_path / "custom_test.j2"
        custom.write_text("Custom: {{ name }}")
        monkeypatch.setenv("TEXT2SQL_PROMPT_TEST_PATH", str(custom))
        assert PromptManager(version="v1").render("test", name="Override") == "Custom: Override"


class TestSharedSQLRules:
    """Both generation prompts carry the same rules, each a counted LiveSQLBench failure
    mode. Duplicated rather than included, so these guard against drift."""

    #: Rules the direct prompt must carry; the agent's subset keeps only the first.
    MARKERS = ["NULLIF", "json_extract", "IF NOT EXISTS", "LIMIT"]
    AGENT_MARKERS = ["IF NOT EXISTS", "CREATE/ALTER/UPDATE"]

    @staticmethod
    def _rules(name):
        from text2sql.prompts.manager import PromptManager
        text = (PromptManager._BUNDLED_DIR / "v1" / f"{name}.j2").read_text()
        return text[text.index("<rules>"):text.index("</rules>")].splitlines()

    def test_agent_rules_are_a_subset_of_direct(self):
        """No include means no shared source, so drift is guarded line-by-line instead. The
        agent carries fewer: the rules it drops are the ones the cascade enforces at submit,
        and a small model pays for every line it reads."""
        agent, direct = self._rules("agent_system"), self._rules("generate_sql")
        assert set(agent) <= set(direct)
        assert len(agent) < len(direct)  # otherwise the trim silently regressed

    def test_no_template_includes_another(self):
        """Partials glued lines across include boundaries: `no SELECT *.Return only the SQL`
        shipped in every direct-generation prompt. Keep every template standalone."""
        for path in (PromptManager._BUNDLED_DIR / "v1").glob("*.j2"):
            assert "{% include" not in path.read_text(), path.name
            assert not path.name.startswith("_"), path.name

    @staticmethod
    def _agent_system(prompt_manager, **kw):
        return prompt_manager.render("agent_system", schema="S", dialect="sqlite",
                                     strategy="direct", tools=[], **kw)

    def test_each_prompt_carries_the_rules_it_needs(self, prompt_manager):
        """The direct path has one shot, so it keeps every rule; the agent's dropped rules are
        the ones `division`, `json_compare`, `join` and `question` check for it."""
        direct = prompt_manager.render("generate_sql", schema="S", question="Q",
                                       dialect="sqlite", strategy="direct")
        agent = self._agent_system(prompt_manager)
        for marker in self.MARKERS:
            assert marker in direct, marker
        for marker in self.AGENT_MARKERS:
            assert marker in agent, marker
        assert "NULLIF" not in agent and "json_extract" not in agent

    @pytest.mark.parametrize("mode,marker", [
        ("retrieval", "down to columns you have seen"),
        ("schema_preloaded", "to columns in the schema below"),
    ])
    def test_each_agent_mode_grounds_the_question_differently(self, prompt_manager, mode, marker):
        """Step 1 used to say "identify the tables and columns", impossible under `retrieval`,
        whose prompt carries only a table list."""
        prompt = self._agent_system(prompt_manager, mode=mode, max_turns=12)
        assert marker in prompt
        # The budget is not stated up front: it read as an allowance to spend, and the
        # per-result `[turn N/M]` carries it without anchoring the whole run on the number.
        assert "12 turns" not in prompt

    @pytest.mark.parametrize("mode", ["retrieval", "schema_preloaded", None])
    @pytest.mark.parametrize("strategy", ["direct", "decompose", "query_plan", ""])
    def test_the_procedure_numbers_every_step_it_renders(self, prompt_manager, mode, strategy):
        """A branch that renders nothing leaves a numbered list with a hole, which reads as a
        missing instruction: an unset `strategy` skipped step 2 outright."""
        prompt = prompt_manager.render("agent_system", schema="S", dialect="sqlite",
                                       strategy=strategy, tools=[],
                                       **({"mode": mode} if mode else {}))
        block = prompt[prompt.index("<procedure>"):prompt.index("</procedure>")]
        steps = [int(line.strip()[0]) for line in block.splitlines()
                 if line.strip()[:1].isdigit()]
        assert sorted(set(steps)) == list(range(1, max(steps) + 1))

    def test_a_render_without_a_mode_keeps_every_shared_instruction(self, prompt_manager):
        """`mode` and `max_turns` are undefined for any caller that predates them; the whole
        procedure must still render, grounding against the schema the template always carries."""
        prompt = self._agent_system(prompt_manager)
        assert "SELECT is not a substitute" in prompt
        assert "<procedure>" in prompt and "<schema>" in prompt
        # The budget reaches the model through `[turn N/M]`; stated up front it reads as an
        # allowance to spend, and 30 turns produced 26 probes where 15 produced 11.
        assert "turns" not in prompt

    def test_ddl_is_not_treated_as_a_failed_query(self, prompt_manager):
        """28% of golds are non-SELECT; the loop used to push the agent to substitute a SELECT."""
        assert "SELECT is not a substitute" in self._agent_system(prompt_manager)

    def test_context_headings_appear_only_when_they_have_content(self, prompt_manager):
        """Each block is announced by a heading; an empty one used to print a bare label."""
        from text2sql.profiler.knowledge import KnowledgeEntry

        bare = self._agent_system(prompt_manager)
        # The closing tag used to sit outside the `if`, so every knowledge-less prompt ended
        # with a bare `</domain_knowledge>`.
        assert "domain_knowledge" not in bare

        full = self._agent_system(
            prompt_manager,
            knowledge=[KnowledgeEntry(0, "Repeat Buyer", "buys twice", "orders >= 2")],
            knowledge_full=True)
        assert "<domain_knowledge>\n- Repeat Buyer: buys twice\n  Definition: orders >= 2" in full

    def test_each_knowledge_level_renders_its_own_shape(self, prompt_manager):
        """`off` passes no entries at all, so the heading cannot survive it either."""
        from text2sql.profiler.knowledge import KnowledgeEntry

        entries = [KnowledgeEntry(0, "Repeat Buyer", "buys twice", "orders >= 2")]
        terms = self._agent_system(prompt_manager, knowledge=entries, knowledge_full=False)
        assert "- Repeat Buyer: buys twice" in terms and "Definition:" not in terms

        off = self._agent_system(prompt_manager, knowledge=[], knowledge_full=False)
        assert "<domain_knowledge>" not in off and "Repeat Buyer" not in off


class TestSQLGenerator:
    def test_randomize_schema_order(self):
        from text2sql.pipeline.generator import SQLGenerator
        schema = "Table: users\n  id INTEGER\n  name TEXT\n  email TEXT\n\nTable: orders\n  id INTEGER\n  amount REAL"
        result = SQLGenerator._randomize_schema_order(schema)
        assert "Table: users" in result
        assert "Table: orders" in result

    def test_randomize_preserves_headers(self):
        from text2sql.pipeline.generator import SQLGenerator
        schema = "Table: users\n  id INTEGER\n  name TEXT\n  email TEXT"
        result = SQLGenerator._randomize_schema_order(schema)
        lines = result.split("\n")
        assert lines[0] == "Table: users"

    def test_randomize_shuffles_the_agents_retrieval_table_list(self):
        """The one-line `Tables:` form has no field lines, so the names shuffle instead."""
        from text2sql.pipeline.generator import randomize_schema_order
        schema = "Tables: " + ", ".join(f"t{i}" for i in range(20))
        result = randomize_schema_order(schema)
        assert result.startswith("Tables: ")
        assert sorted(result[len("Tables: "):].split(", ")) == sorted(f"t{i}" for i in range(20))
        assert result != schema  # 20! orderings — a stable result means it did not shuffle

    def test_the_join_map_survives_the_shuffle(self):
        """Splitting the whole remainder on commas ate the join line it now carries."""
        from text2sql.pipeline.generator import randomize_schema_order
        result = randomize_schema_order("Tables: a, b\nJoins: b.x -> a.y; b.z -> a.y")
        assert result.endswith("\nJoins: b.x -> a.y; b.z -> a.y")
        assert sorted(result.splitlines()[0][len("Tables: "):].split(", ")) == ["a", "b"]


# ── PipelineTracer ───────────────────────────────────────────────


class TestPipelineTracer:
    def test_full_lifecycle(self):
        import time

        from text2sql.pipeline.tracer import PipelineTracer
        tracer = PipelineTracer()
        tracer.start_pipeline("How many users?", "sqlite:///t.db", "gpt-4o")

        step = tracer.start_step("profiling", tables=5)
        tracer.end_step(step, tables_profiled=5)

        step2 = tracer.start_step("generation")
        tracer.end_step(step2, candidates=3)

        time.sleep(0.01)  # ensure measurable time.time() delta
        tracer.end_pipeline(sql="SELECT 1", results=[{"x": 1}])

        assert tracer.trace.question == "How many users?"
        assert len(tracer.trace.steps) == 2
        assert tracer.trace.final_sql == "SELECT 1"
        assert tracer.trace.duration_seconds > 0

    def test_to_dict_structure(self):
        from text2sql.pipeline.tracer import PipelineTracer
        tracer = PipelineTracer()
        tracer.start_pipeline("Q", "uri", "model")
        tracer.end_pipeline(sql="S")
        d = tracer.trace.to_dict()
        assert "question" in d
        assert "duration_seconds" in d
        assert "steps" in d
        assert "llm_usage" in d

    def test_step_duration(self):
        import time

        from text2sql.pipeline.tracer import StepTrace
        step = StepTrace(step_name="test", started_at=time.time())
        time.sleep(0.01)
        step.completed_at = time.time()
        assert step.duration_seconds > 0

    def test_error_in_pipeline(self):
        from text2sql.pipeline.tracer import PipelineTracer
        tracer = PipelineTracer()
        tracer.start_pipeline("Q", "uri", "model")
        tracer.end_pipeline(error="something broke")
        assert tracer.trace.error == "something broke"


# ── Selector confidence modes ────────────────────────────────────


class TestCandidateSelectorConfidence:
    """Tests for confidence-aware and edge-case selection paths."""

    async def test_single_success_path(self, db_conn):
        """When only 1 of N candidates executes successfully -> single_success."""
        from text2sql.pipeline.selector import CandidateSelector
        selector = CandidateSelector(db=db_conn, mode="majority")
        candidates = [
            "SELECT COUNT(*) AS cnt FROM users",
            "SELECT * FROM nonexistent_table",
            "SELECT * FROM another_bad_table",
        ]
        sql, results, meta = await selector.select(candidates)
        assert meta["method"] == "single_success"
        assert results is not None

    async def test_no_agreement_picks_at_random_without_adjudication(self, db_conn):
        """`majority` has nothing to vote on when every candidate disagrees."""
        from text2sql.pipeline.selector import CandidateSelector
        selector = CandidateSelector(db=db_conn, mode="majority")
        _, _, meta = await selector.select(["SELECT 1 AS x", "SELECT 2 AS x", "SELECT 3 AS x"])
        assert meta["method"] == "random_selection" and meta["reason"] == "no_agreement"

    @pytest.mark.parametrize("candidates, method, uncertain", [
        (["SELECT 1 AS x"] * 3, "unanimous", None),
        (["SELECT 1 AS x", "SELECT 1 AS x", "SELECT 2 AS x"], "majority_vote", True),
    ])
    async def test_the_ladder_is_counts_not_thresholds(
            self, db_conn, candidates, method, uncertain):
        """At the 3-candidate ceiling only unanimous / plurality / none-agree are expressible:
        2 of 3 is 0.667, which the old 0.67 high threshold could never clear."""
        from text2sql.pipeline.selector import CandidateSelector
        selector = CandidateSelector(db=db_conn, mode="confidence")
        _, _, meta = await selector.select(candidates)
        assert meta["method"] == method
        assert meta.get("uncertain") is uncertain

    async def test_only_confidence_mode_adjudicates(self, db_conn, mock_llm, prompt_manager):
        from text2sql.pipeline.selector import CandidateSelector
        mock_llm.chat.return_value = '{"selected": 1, "reasoning": "first is best"}'
        candidates = ["SELECT 1 AS x", "SELECT 2 AS x", "SELECT 3 AS x"]
        for mode, method in (("confidence", "llm_adjudication"), ("majority", "random_selection")):
            selector = CandidateSelector(db=db_conn, mode=mode, llm=mock_llm,
                                         prompt_manager=prompt_manager)
            _, _, meta = await selector.select(candidates, question="Q", schema_text="S")
            assert meta["method"] == method, mode

    async def test_adjudication_prompt_reports_the_true_row_count(
            self, db_conn, mock_llm, prompt_manager):
        """Results were truncated to 5 before rendering, so the template's row count and its
        "... N total" line could never exceed 5 — a 900-row read looked like a 5-row one."""
        from text2sql.pipeline.selector import CandidateSelector
        mock_llm.chat.return_value = '{"selected": 1, "reasoning": "r"}'
        selector = CandidateSelector(db=db_conn, mode="confidence", llm=mock_llm,
                                     prompt_manager=prompt_manager)
        rows = "SELECT 1 AS x UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4 " \
               "UNION ALL SELECT 5 UNION ALL SELECT 6 UNION ALL SELECT 7"
        await selector.select([rows, "SELECT 8 AS x"], question="Q", schema_text="S")
        prompt = mock_llm.chat.call_args[0][0]
        assert "Results (7 rows)" in prompt and "... (7 total)" in prompt

    async def test_llm_adjudication_fallback(self, db_conn, mock_llm, prompt_manager):
        """LLM returns garbage JSON -> adjudication_fallback."""
        from text2sql.pipeline.selector import CandidateSelector
        mock_llm.chat.return_value = "not valid json at all"
        selector = CandidateSelector(db=db_conn, mode="confidence", llm=mock_llm,
                                     prompt_manager=prompt_manager)
        candidates = ["SELECT 1 AS x", "SELECT 2 AS x"]
        sql, results, meta = await selector.select(candidates, question="Q", schema_text="S")
        assert meta["method"] == "adjudication_fallback"


# ── SQLRepair full cascade ───────────────────────────────────────


class TestSQLRepairCascade:
    """Tests for the SQLRepair.repair() method with mocked LLM."""

    async def test_valid_sql_no_repair(self, db_conn, mock_llm, prompt_manager):
        from text2sql.pipeline.repair import SQLRepair
        repair = SQLRepair(mock_llm, db_conn, prompt_manager, max_retries=2)
        sql, issues = await repair.repair(
            "SELECT COUNT(*) AS cnt FROM users",
            question="How many users?",
            schema_text="Table: users",
        )
        assert sql == "SELECT COUNT(*) AS cnt FROM users"
        assert issues == []

    async def test_repair_fixes_bad_sql(self, db_conn, mock_llm, prompt_manager):
        from text2sql.pipeline.repair import SQLRepair
        # LLM returns fixed SQL when asked to repair
        mock_llm.chat_for_sql.return_value = "SELECT COUNT(*) AS cnt FROM users"
        repair = SQLRepair(mock_llm, db_conn, prompt_manager, max_retries=2)
        sql, issues = await repair.repair(
            "SELECT * FROM nonexistent_table",
            question="How many users?",
            schema_text="Table: users",
        )
        # Should have found issues and attempted repair
        assert len(issues) > 0
        # Final SQL should be the repaired version
        assert sql == "SELECT COUNT(*) AS cnt FROM users"

    async def test_repair_max_retries(self, db_conn, mock_llm, prompt_manager):
        from text2sql.pipeline.repair import SQLRepair
        # LLM keeps returning bad SQL
        mock_llm.chat_for_sql.return_value = "SELECT * FROM still_bad"
        repair = SQLRepair(mock_llm, db_conn, prompt_manager, max_retries=1)
        sql, issues = await repair.repair(
            "SELECT * FROM bad_table",
            question="Q",
            schema_text="S",
        )
        assert len(issues) > 0
        # After max retries, returns whatever it has

    async def test_repair_issues_contain_checker_name(self, db_conn, mock_llm, prompt_manager):
        from text2sql.pipeline.repair import SQLRepair
        mock_llm.chat_for_sql.return_value = "SELECT * FROM users"
        repair = SQLRepair(mock_llm, db_conn, prompt_manager, max_retries=1)
        _, issues = await repair.repair(
            "SELECT * FROM nonexistent",
            question="Q",
            schema_text="S",
        )
        # Issues should be tagged with checker name
        assert any("[dry_run]" in issue for issue in issues)

    async def test_repair_stops_when_llm_returns_unchanged_sql(self, db_conn, mock_llm, prompt_manager):
        from text2sql.pipeline.repair import SQLRepair
        # LLM echoes back the same broken SQL (modulo whitespace/case/`;`) every time.
        mock_llm.chat_for_sql.return_value = "select  *  from   nonexistent ;"
        repair = SQLRepair(mock_llm, db_conn, prompt_manager, max_retries=5)
        await repair.repair("SELECT * FROM nonexistent", question="Q", schema_text="S")
        # Should bail after the first no-op repair, not burn all 5 retries.
        assert mock_llm.chat_for_sql.await_count == 1

    async def test_empty_select_triggers_repair(self, db_conn, mock_llm, prompt_manager):
        """An empty read is a wrong predicate far more often than a true answer.

        Benchmarked: the agent submitted `json_extract(...) = '"High"'` (quoted JSON scalar)
        on two instances, matched nothing, and nothing caught it while `result` was info-only.
        """
        from text2sql.pipeline.repair import SQLRepair
        repair = SQLRepair(mock_llm, db_conn, prompt_manager, max_retries=3)
        _, issues = await repair.repair(
            "SELECT id FROM users WHERE city = 'Atlantis'",
            question="Who lives in Atlantis?",
            schema_text="Table: users",
        )
        assert any("[result]" in issue for issue in issues)
        assert mock_llm.chat_for_sql.await_count > 0

    @pytest.mark.parametrize("write", [
        "CREATE INDEX IF NOT EXISTS ix_users_name ON users(name)",
        # A CTE-led write read as a query, and its correct empty result as a defect: one
        # measured run spent a whole repair budget rewriting every join key as TRIM(x)=TRIM(y).
        "WITH old AS (SELECT id FROM users) DELETE FROM users WHERE id IN (SELECT id FROM old)",
        "PRAGMA foreign_keys=OFF; CREATE TABLE t2 (a INTEGER)",
    ])
    async def test_a_write_returning_nothing_is_not_repaired(
            self, db_conn, mock_llm, prompt_manager, write):
        """A write returns no rows by nature, not by mistake, wherever the verb sits."""
        from text2sql.pipeline.repair import SQLRepair
        repair = SQLRepair(mock_llm, db_conn, prompt_manager, max_retries=3)
        sql, issues = await repair.repair(write, question="Do it", schema_text="")
        assert sql == write
        assert not any("[result]" in issue for issue in issues)
        assert mock_llm.chat_for_sql.await_count == 0

    async def test_info_severity_still_never_triggers_repair(self, db_conn, mock_llm,
                                                             prompt_manager, monkeypatch):
        """No checker reports `info` today, but the filter that skips them must keep working."""
        from text2sql.pipeline import repair as repair_mod

        monkeypatch.setattr(repair_mod, "CHECKERS", [
            ("noisy", lambda sql, db=None: repair_mod.Issue("fyi", "do nothing", "info")),
        ])
        repair = repair_mod.SQLRepair(mock_llm, db_conn, prompt_manager, max_retries=3)
        sql, issues = await repair.repair("SELECT 1", question="q", schema_text="")
        assert sql == "SELECT 1"
        assert any("[noisy]" in issue for issue in issues)
        assert mock_llm.chat_for_sql.await_count == 0


# ── JoinChecker JOIN without ON ──────────────────────────────────


class TestJoinCheckerEdge:
    def test_join_without_on(self):
        """JOIN without ON clause should fail."""
        result = check_join("SELECT * FROM users JOIN orders")
        assert result is not None
        assert "ON" in result.message

