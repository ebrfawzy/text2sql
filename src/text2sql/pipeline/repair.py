"""SQL validation and LLM-based repair.

Each checker returns an :class:`Issue` with a fix directive, and a failing query goes to the
LLM for up to ``max_retries`` rounds. Groups: syntax (syntax, dry_run), logic (join, order_by,
time), quality (null, division, json_compare, result), plus naming and question outside the
registry, which need the question the others do not.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

import sqlglot
from sqlglot import exp

from text2sql.db import EXEC_TIMEOUT_S, TIMED_OUT, DatabaseConnection, split_statements
from text2sql.llm import LLMClient
from text2sql.profiler.knowledge import DatabaseKnowledge
from text2sql.prompts.manager import PromptManager
from text2sql.schema.linker import SchemaLinker
from text2sql.schema.loader import SchemaLoader

logger = logging.getLogger(__name__)

_READS = (exp.Select, exp.Union, exp.Except, exp.Intersect)
_DATE_FUNCS = ("DATE(", "DATETIME(", "STRFTIME(", "DATE_FORMAT(", "TO_DATE(")

# Similarity above which an alias reads as a misspelling rather than a rename.
_NEAR_RATIO = 0.9

# SQLite quotes the whole statement back when it cannot tokenise the start.
_ECHO_RE = re.compile(r'near "[^"]{60,}"')
_MAX_MESSAGE = 300

# A name the query got wrong lives in a table it does not touch, so narrowing hides the fix.
_MISSING_NAME_RE = re.compile(r"no such column|no such table|ambiguous column", re.IGNORECASE)

# A constraint the question names by quoting it - the write tests read the name back.
_CONSTRAINT_NAME_RE = re.compile(r"constraint\s+['\"`]([A-Za-z_]\w*)['\"`]", re.IGNORECASE)

# Declarations a question can ask for that CREATE TABLE ... AS SELECT does not carry.
_DECLARED_RE = re.compile(r"\b(?:integer|text|numeric|real|decimal|primary key|constraint"
                          r"|not null|auto[- ]?increment)\b", re.IGNORECASE)
_CTAS_RE = re.compile(r"CREATE\s+TABLE\b[^;]*?\bAS\s+SELECT\b", re.IGNORECASE | re.DOTALL)
_PARTITION_RE = re.compile(r"\bpartition", re.IGNORECASE)

# A string a query builds is its own; TRIM there is formatting, not a stored value.
_BUILT = (exp.Case, exp.Concat, exp.DPipe)

_SEVERITY_RANK = {"error": 0, "warning": 1, "info": 2}
_SCHEMA_RANK = {"none": 0, "touched": 1, "full": 2}


@dataclass
class Issue:
    """A problem found by a checker, with a directive for LLM repair.

    Attributes:
        message: What is wrong.
        directive: The fix the repair prompt asks for.
        severity: ``error``, ``warning`` or ``info``; ``info`` never triggers a rewrite.
        schema: How much schema the fix needs - ``none``, ``touched`` or ``full``.
    """

    message: str
    directive: str
    severity: str = "warning"
    schema: str = "touched"


Checker = Callable[[str, "DatabaseConnection | None"], "Issue | None"]
Finding = tuple[str, Issue]

# Ordered cascade of (name, checker) pairs, populated by @checker.
CHECKERS: list[tuple[str, Checker]] = []


def checker(name: str) -> Callable[[Checker], Checker]:
    """Register a checker function in the cascade.

    Args:
        name: Name the finding is reported under.

    Returns:
        The decorator that registers and returns the checker unchanged.
    """

    def register(fn: Checker) -> Checker:
        """Append the checker to the cascade.

        Args:
            fn: The checker.

        Returns:
            The checker, unchanged.
        """
        CHECKERS.append((name, fn))
        return fn

    return register


# ---- Shared helpers -----------------------------------------------------


def parse_sql(sql: str, dialect: str = "") -> Any | None:
    """Parse one statement; the pipeline's only sqlglot entry point.

    Args:
        sql: The SQL to parse.
        dialect: Must be ``db.sqlglot_dialect`` - sqlglot raises on SQLAlchemy's spelling
            of postgres/athena/mariadb.

    Returns:
        The parsed expression, or None when it does not parse.
    """
    try:
        return sqlglot.parse_one(sql, dialect=dialect or None)
    except Exception:
        return None


def _normalize(sql: str) -> str:
    """Reduce SQL to a canonical form for detecting a no-op repair.

    Args:
        sql: The SQL to normalize.

    Returns:
        The SQL, whitespace-, case- and ``;``-insensitive.
    """
    return re.sub(r"\s+", " ", sql).strip().rstrip("; ").lower()


def clean_error(error: str) -> str:
    """Strip every echo of the query from a driver error, leaving the diagnosis.

    Args:
        error: The raw driver message.

    Returns:
        The message, truncated at ``_MAX_MESSAGE``.
    """
    error = re.split(r"\s*\[SQL:", error, maxsplit=1)[0]
    error = re.split(r"\s*\(Background on this error", error, maxsplit=1)[0]
    error = _ECHO_RE.sub('near "<the query above>"', error).strip()
    return error if len(error) <= _MAX_MESSAGE else error[:_MAX_MESSAGE].rstrip() + " ..."


def reads(sql: str, dialect: str = "") -> list[Any]:
    """The statements of a query that only reads, else ``[]``.

    Args:
        sql: The SQL to classify.
        dialect: sqlglot dialect name.

    Returns:
        Every parsed statement when they are all SELECT or a set operation over them -
        a CTE-led write and a script opening with PRAGMA are writes - else ``[]``.
    """
    try:
        stmts = [s for s in sqlglot.parse(sql, dialect=dialect or None) if s]
    except Exception:
        return []
    return stmts if stmts and all(isinstance(s, _READS) for s in stmts) else []


def _dialect(db: DatabaseConnection | None) -> str:
    """Resolve the sqlglot dialect for a connection.

    Args:
        db: The connection, or None.

    Returns:
        The dialect name, or ``""`` when there is no connection.
    """
    return db.sqlglot_dialect if db is not None else ""


def _columns(db: DatabaseConnection) -> dict[str, dict[str, Any]]:
    """Index every column of the database by name.

    Args:
        db: The connection to read the schema from.

    Returns:
        ``{lowercased column name: column info}``, first name winning.
    """
    columns: dict[str, dict[str, Any]] = {}
    try:
        schema = db.get_schema()
    except Exception:
        return columns
    for table_info in schema.get("tables", {}).values():
        for col in table_info.get("columns", []):
            name = str(col.get("name", "")).lower()
            if name and name not in columns:
                columns[name] = col
    return columns


def _bare(node: Any) -> Any | None:
    """Reduce one side of a division to the column it reads.

    Args:
        node: The expression on that side.

    Returns:
        The column reached through NULLIF/COALESCE, or None - reading past anything else
        calls ``a * 10.0`` an integer.
    """
    while isinstance(node, exp.Nullif | exp.Coalesce):
        node = node.this
    return node if isinstance(node, exp.Column) else None


def _has_zero(db: DatabaseConnection | None, column: str) -> bool:
    """Ask the data whether a column ever holds a zero or NULL.

    Args:
        db: The connection to probe, or None.
        column: Column name.

    Returns:
        True when such a row exists; an unresolvable name counts as safe, like every other
        checker's fail-open.
    """
    if db is None:
        return False
    table = next((t for t, cols in db.list_columns().items() if column in cols), "")
    if not table:
        return False
    rows, error = db.execute_safe(
        f'SELECT 1 AS hit FROM {db.quote_identifier(table)} WHERE {db.quote_identifier(column)} = 0 '
        f'OR {db.quote_identifier(column)} IS NULL LIMIT 1')
    return not error and bool(rows)


def _select_aliases(parsed: Any) -> set[str]:
    """Collect the output names of every SELECT, CTEs included.

    Args:
        parsed: The parsed statement.

    Returns:
        Lowercased aliases; counting only the outermost calls an ORDER BY over a CTE's own
        alias an unknown column.
    """
    return {p.alias_or_name.lower() for select in parsed.find_all(exp.Select)
            for p in select.expressions if p.alias_or_name}


def _unknown_order_columns(sql: str, db: DatabaseConnection) -> set[str]:
    """Find ORDER BY references that are neither real columns nor aliases.

    Args:
        sql: The SQL to inspect.
        db: The connection supplying the real column names.

    Returns:
        The unknown names; functions, expressions and ordinals are skipped.
    """
    parsed = parse_sql(sql, _dialect(db))
    order = parsed.find(exp.Order) if parsed is not None else None
    if order is None:
        return set()
    valid = set(_columns(db)) | _select_aliases(parsed)
    return {
        col.name
        for ordered in order.find_all(exp.Ordered)
        if isinstance(col := ordered.this, exp.Column) and col.name and col.name.lower() not in valid
    }



# ---- Checkers (syntax -> logic -> quality) ------------------------------


@checker("syntax")
def check_syntax(sql: str, db: DatabaseConnection | None = None) -> Issue | None:
    """Check that the SQL parses.

    Args:
        sql: The candidate SQL.
        db: Connection, for the dialect.

    Returns:
        An error Issue when sqlglot cannot parse it, else None.
    """
    try:
        sqlglot.parse(sql, dialect=_dialect(db) or None)
    except Exception as e:
        # First line only; the second is sqlglot's coloured echo of the query.
        return Issue(
            f"Syntax error: {clean_error(str(e)).splitlines()[0]}",
            "Balance parentheses and quotes and use valid SQL keywords.",
            "error",
            schema="none",
        )
    return None


@checker("dry_run")
def check_dry_run(sql: str, db: DatabaseConnection | None = None) -> Issue | None:
    """Execute the SQL and report any driver error.

    Args:
        sql: The candidate SQL.
        db: Connection to run against; None skips the check.

    Returns:
        An error Issue carrying the cleaned driver message, else None. A missing name asks
        for the full schema, since the column lives in a table the query does not touch;
        a dialect or function error is answered by no schema at all. The execution deadline
        firing is reported as ``info``: it says nothing about the query.
    """
    if db is None:
        return None
    _, error = db.execute_safe(sql)
    if error and TIMED_OUT.search(error):
        return Issue(f"Execution exceeded {EXEC_TIMEOUT_S:.0f}s and was interrupted.",
                     "", "info", schema="none")
    if error:
        return Issue(
            f"Execution failed: {clean_error(error)}",
            "Use only schema table/column names, keep JOIN keys valid, and CAST "
            "VARCHAR columns before comparing them to dates or numbers.",
            "error",
            schema="full" if _MISSING_NAME_RE.search(error) else "none",
        )
    return None


@checker("join")
def check_join(sql: str, db: DatabaseConnection | None = None) -> Issue | None:
    """Check that every join is expressed and conditioned.

    Args:
        sql: The candidate SQL.
        db: Unused.

    Returns:
        An Issue for a comma cartesian product or a JOIN with no ON/USING, else None.
    """
    up = sql.upper()
    if re.search(r"FROM\s+\w+\s*,\s*\w+", up) and "WHERE" not in up:
        return Issue(
            "Cartesian product: comma-separated tables without a WHERE clause.",
            "Replace comma-separated tables with explicit JOIN ... ON syntax.",
        )
    joins = re.findall(r"(?:LEFT|RIGHT|INNER|OUTER|CROSS|FULL)?\s*\bJOIN\s+\w+", up)
    if len(joins) - up.count("CROSS JOIN") > len(re.findall(r"\bON\b|\bUSING\b", up)):
        return Issue(
            "JOIN without an ON condition.",
            "Give every non-CROSS JOIN an explicit ON condition.",
            "error",
        )
    return None


@checker("order_by")
def check_order_by(sql: str, db: DatabaseConnection | None = None) -> Issue | None:
    """Check that a capped query orders, and that it orders by something real.

    Args:
        sql: The candidate SQL.
        db: Connection supplying the real column names; None skips the second half.

    Returns:
        An info Issue for LIMIT without ORDER BY - ordering advice is no licence to rewrite
        a working query - an error Issue for an unknown ORDER BY column, else None.
    """
    up = sql.upper()
    if "LIMIT" in up and "ORDER BY" not in up:
        return Issue(
            "LIMIT without ORDER BY: results may be non-deterministic.",
            "Add an ORDER BY before LIMIT (e.g. the primary key or a relevant column).",
            "info",
            schema="none",
        )
    if db is not None and "ORDER BY" in up and (unknown := _unknown_order_columns(sql, db)):
        cols = ", ".join(sorted(unknown))
        return Issue(
            f"ORDER BY references unknown column(s): {cols}.",
            f"Order by a real column or a SELECT alias; {cols} is neither.",
            "error",
        )
    return None


@checker("time")
def check_time(sql: str, db: DatabaseConnection | None = None) -> Issue | None:
    """Check that date comparisons go through a date function.

    Args:
        sql: The candidate SQL.
        db: Unused.

    Returns:
        An Issue when a date-shaped literal is compared raw, else None.
    """
    if re.search(r"[<>]=?\s*'?\d{4}[-/]\d{1,2}", sql) and not any(f in sql.upper() for f in _DATE_FUNCS):
        return Issue(
            "Date compared against a raw string literal.",
            "Wrap date/time comparisons in the dialect's date function (DATE(), STRFTIME(), ...).",
            schema="none",
        )
    return None


@checker("null")
def check_null(sql: str, db: DatabaseConnection | None = None) -> Issue | None:
    """Check for NULLs that can silently reorder or empty the answer.

    Args:
        sql: The candidate SQL.
        db: Connection supplying column nullability.

    Returns:
        An info Issue for an ascending ORDER BY on a nullable column, else None.
    """
    if "IS NOT NULL" in sql.upper() or (parsed := parse_sql(sql, _dialect(db))) is None:
        return None
    # Only a column the schema says is nullable can sort NULLs first.
    nullable = {n for n, c in _columns(db).items()
                if c.get("nullable") and not c.get("primary_key")} if db else set()
    for ordered in parsed.find_all(exp.Ordered):
        col = ordered.this
        if not ordered.args.get("desc") and isinstance(col, exp.Column) \
                and col.name.lower() in nullable:
            return Issue(
                f"ORDER BY {col.name} ASC without a NOT NULL filter: NULLs may sort first.",
                "Filter the ordered column IS NOT NULL, or use NULLS LAST if the dialect "
                "supports it.",
                # Info: a voluntary ORDER BY is not a wrong answer, and as a warning this
                # blocked a submit to buy a spurious IS NOT NULL predicate.
                "info",
                schema="none",
            )
    return None


@checker("division")
def check_division(sql: str, db: DatabaseConnection | None = None) -> Issue | None:
    """Check both halves of the division rule: silent NULL, and silent truncation.

    Args:
        sql: The candidate SQL.
        db: Connection, probed for whether a denominator really holds a zero or NULL.

    Returns:
        An Issue for an unguarded denominator or an integer-only division, else None.
    """
    if (parsed := parse_sql(sql, _dialect(db))) is None:
        return None
    columns = _columns(db) if db is not None else {}
    for div in parsed.find_all(exp.Div):
        if isinstance(div.expression, exp.Column) and _has_zero(db, div.expression.name):
            return Issue(
                f"Division by column '{div.expression.name}', which holds a zero or NULL.",
                "Wrap the denominator: CAST(a AS REAL) / NULLIF(b, 0).",
                schema="none",
            )
        types = [str(columns.get(c.name.lower(), {}).get("type", ""))
                 for side in (div.this, div.expression) if (c := _bare(side)) is not None]
        if len(types) == 2 and all(re.search(r"INT", t, re.I) for t in types):
            return Issue(
                f"Integer division truncates: {div.sql()} discards the remainder.",
                "CAST the numerator to REAL before dividing.",
                schema="none",
            )
    return None


@checker("json_compare")
def check_json_compare(sql: str, db: DatabaseConnection | None = None) -> Issue | None:
    """Check that a JSON extraction is compared to a literal of the matching quoting.

    Args:
        sql: The candidate SQL.
        db: Connection, for the dialect.

    Returns:
        An Issue when the comparison can never match, else None.
    """
    # sqlglot maps `->` and `json_extract()` to the same node, so the operator is read from
    # the text and the comparison shape from the tree.
    if (parsed := parse_sql(sql, _dialect(db))) is None:
        return None
    quoted = "->" in sql.replace("->>", "")
    for eq in parsed.find_all(exp.EQ, exp.NEQ):
        sides = (eq.this, eq.expression)
        literals = [x for x in sides if isinstance(x, exp.Literal) and x.is_string]
        if not any(isinstance(x, exp.JSONExtract) for x in sides) or not literals:
            continue
        # A `"..."` literal is a JSON string, so it only matches what `->` returns.
        if quoted != literals[0].name.startswith('"'):
            return Issue(
                "`->` returns quoted JSON and json_extract/->> return bare scalars, so this "
                "comparison never matches.",
                "Use ->> (or json_extract) and compare to the bare value.",
                schema="none",
            )
    return None


@checker("as_stored")
def check_as_stored(sql: str, db: DatabaseConnection | None = None) -> Issue | None:
    """Check that a read returns its columns as the database stores them.

    Args:
        sql: The candidate SQL.
        db: Connection, for the dialect.

    Returns:
        An Issue when the outermost projection wraps a value in TRIM/LOWER/UPPER without
        building a string of its own, else None.
    """
    if not (stmts := reads(sql, _dialect(db))):
        return None
    for e in (stmts[-1].find(exp.Select) or stmts[-1]).expressions:
        for node in e.find_all(exp.Trim, exp.Lower, exp.Upper):
            if not node.find(*_BUILT):
                return Issue(f"{node.sql()} changes a stored value in the result.",
                             "Return the column as stored; TRIM or LOWER only in a predicate.",
                             schema="none")
    return None


@checker("returning")
def check_returning(sql: str, db: DatabaseConnection | None = None) -> Issue | None:
    """Check for a RETURNING clause the official scorer cannot execute.

    The scorer commits any statement not opening WITH/SELECT before fetching its cursor, so
    RETURNING there dies as ``cannot commit transaction``. Leading with a CTE, which every
    gold using RETURNING does, takes the fetch path instead.

    Args:
        sql: The candidate SQL.
        db: Connection, for the dialect.

    Returns:
        An error Issue on that shape only, else None.
    """
    dialect = db.sqlglot_dialect if db else ""
    for stmt in split_statements(sql):
        if stmt.lstrip().upper().startswith(("WITH", "SELECT")):
            continue
        if (tree := parse_sql(stmt, dialect)) and tree.find(exp.Returning):
            return Issue(
                "RETURNING on a statement that does not open with WITH.",
                "Lead the write with a CTE, keeping RETURNING.",
                "error", schema="none")
    return None


@checker("rename")
def check_rename(sql: str, db: DatabaseConnection | None = None) -> Issue | None:
    """Check a CREATE for an alias that misspells the column it passes through.

    A CREATE's column names are graded, and an alias one edit from its source is a typo
    carried over from the question, never a deliberate rename.

    Args:
        sql: The candidate SQL.
        db: Connection, for the dialect.

    Returns:
        An Issue for a near-miss alias, else None.
    """
    for stmt in split_statements(sql):
        tree = parse_sql(stmt, _dialect(db))
        if tree is None or not isinstance(tree, exp.Create):
            continue
        for select in tree.find_all(exp.Select):
            for e in select.expressions:
                if not isinstance(e, exp.Alias) or not isinstance(e.this, exp.Column):
                    continue
                alias, source = e.alias.lower(), e.this.name.lower()
                if alias != source and SequenceMatcher(None, alias, source).ratio() >= _NEAR_RATIO:
                    return Issue(
                        f"Column aliased to '{e.alias}', a near-miss of '{e.this.name}'.",
                        "Name a passthrough column as the schema names it.",
                        schema="none")
    return None


@checker("rebuild")
def check_rebuild(sql: str, db: DatabaseConnection | None = None) -> Issue | None:
    """Check for a table rebuild that silently drops the original's constraints.

    SQLite cannot ALTER a constraint in, so the table is rebuilt, and
    ``CREATE TABLE ... AS SELECT`` copies none.

    Args:
        sql: The candidate SQL.
        db: Connection, for the dialect.

    Returns:
        An error Issue when a non-temp TABLE is created from a source the script then drops,
        else None.
    """
    trees = [t for stmt in split_statements(sql) if (t := parse_sql(stmt, _dialect(db)))]
    dropped = {t.this.name.lower() for t in trees
               if isinstance(t, exp.Drop) and isinstance(t.this, exp.Table)}
    for t in trees:
        # A view carries no constraints and a TEMP table is scratch - gold uses both freely.
        if not (isinstance(t, exp.Create) and isinstance(t.expression, exp.Select)
                and t.args.get("kind", "").upper() == "TABLE"
                and not (t.args.get("properties") or exp.Properties()).find(exp.TemporaryProperty)):
            continue
        if {x.name.lower() for x in t.expression.find_all(exp.Table)} & dropped:
            return Issue(
                "CREATE TABLE ... AS SELECT copies no constraints, so the rebuild loses them.",
                "Declare the replacement table with its columns and constraints, then copy the rows in.",
                "error")
    return None


@checker("rowid_pk")
def check_rowid_pk(sql: str, db: DatabaseConnection | None = None) -> Issue | None:
    """Check that an integer primary key is declared so SQLite fills it in.

    Only the bare type ``INTEGER`` makes a primary key a rowid alias; ``integer(64)`` does
    not, so an insert omitting it stores NULL, or fails outright when the column is NOT NULL.
    The key is read from the column constraint or from a table-level PRIMARY KEY naming it.

    Args:
        sql: The candidate SQL.
        db: Connection, for the dialect.

    Returns:
        An error Issue when a created table declares a parameterised integer primary key,
        else None.
    """
    for stmt in split_statements(sql):
        tree = parse_sql(stmt, _dialect(db))
        if not (isinstance(tree, exp.Create) and isinstance(tree.this, exp.Schema)):
            continue
        named = {e.name.lower() for pk in tree.this.find_all(exp.PrimaryKey)
                 for e in pk.expressions}
        for col in tree.this.find_all(exp.ColumnDef):
            kind = col.args.get("kind")
            if ((col.find(exp.PrimaryKeyColumnConstraint) or col.name.lower() in named)
                    and kind and kind.this in exp.DataType.INTEGER_TYPES and kind.expressions):
                return Issue(
                    f"Primary key {col.name} has a sized integer type, so SQLite does not fill it in.",
                    f"Declare it as {col.name} INTEGER PRIMARY KEY.",
                    "error", schema="none")
    return None


@checker("precision")
def check_precision(sql: str, db: DatabaseConnection | None = None) -> Issue | None:
    """Check that a write does not round a stored value below the scorer's precision.

    Rows are compared with floats rounded to 2 decimal places, so rounding to 2 or more is
    invisible while rounding below 2 discards a difference the test can see.

    Args:
        sql: The candidate SQL.
        db: Connection, for the dialect.

    Returns:
        A warning Issue when an UPDATE stores a value rounded to fewer than 2 places,
        else None.
    """
    for stmt in split_statements(sql):
        tree = parse_sql(stmt, _dialect(db))
        if not isinstance(tree, exp.Update):
            continue
        for rnd in tree.find_all(exp.Round):
            places = rnd.args.get("decimals")
            if places is None or (places.is_int and int(places.name) < 2):
                return Issue(
                    f"{rnd.sql()} stores a value rounded below 2 decimal places.",
                    "Store the value unrounded, or round to at least 2 places.")
    return None


@checker("result")
def check_result(sql: str, db: DatabaseConnection | None = None) -> Issue | None:
    """Check that a read actually answers something.

    Args:
        sql: The candidate SQL.
        db: Connection to run against; None skips the check.

    Returns:
        A warning Issue when the query returns no rows or a single all-NULL row - an
        aggregate over an empty set is the same non-answer in the shape the question asked
        for - else None. Only a read is checked; ``dry_run`` already reports errors.
    """
    if db is None or not reads(sql, _dialect(db)):
        return None
    results, error = db.execute_safe(sql)
    if error or results is None \
            or any(v is not None for row in results for v in row.values()):
        return None
    # Warning, not info: an empty read is a wrong predicate far more often than a true
    # answer, and info findings never reach the LLM.
    return Issue(
        f"Query returned {'0 rows' if not results else 'only NULLs'}: a predicate is likely wrong.",
        "Check JSON quoting (json_extract returns bare scalars, -> returns quoted JSON), "
        "value spelling and join keys; loosen the filters if they cannot match.",
        schema="full",
    )


# ---- Checkers outside the registry (they need extra arguments) ----------



def _naming_conflict(sql: str, question: str) -> Issue | None:
    """Check that the DDL declares what the question names.

    A write test reads names out of ``sqlite_master`` and inserts rows of its own, so an
    anonymous CHECK, or a ``CREATE TABLE ... AS SELECT`` that carries no types or keys at
    all, is correct SQL that still fails.

    Args:
        sql: The candidate SQL.
        question: The user's question.

    Returns:
        An Issue when the question quotes a constraint name the DDL omits or names
        declarations a CTAS discards, else None.
    """
    if (m := _CONSTRAINT_NAME_RE.search(question)) \
            and not re.search(rf"CONSTRAINT\s+{re.escape(m.group(1))}\b", sql, re.IGNORECASE):
        return Issue(
            f"The question names the constraint '{m.group(1)}'; the DDL does not carry it.",
            f"Declare it as CONSTRAINT {m.group(1)} CHECK (...).", schema="none")
    asked = _DECLARED_RE.search(DEFINITIONS.split(question, maxsplit=1)[0])
    if not asked or not _CTAS_RE.search(sql):
        return None
    partition = "SQLite has no partitioning: make each range a table with its own CHECK and " \
        "the parent a VIEW that UNION ALLs them. " if _PARTITION_RE.search(question) else ""
    return Issue(
        f"The question names {asked.group(0).lower()}; CREATE TABLE ... AS SELECT carries none.",
        f"{partition}Declare the table with its columns, types and constraints, "
        "then copy the rows in.", "error", schema="none")


# ---- Question-stated constraints ----------------------------------------

DEFINITIONS = re.compile(r"<definitions>", re.IGNORECASE)
_RANK_RE = re.compile(r"\b(ROW_NUMBER|RANK|DENSE_RANK|NTILE|PERCENT_RANK)\s*\(", re.IGNORECASE)
_NUMBERS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten".split())}
_DEC_RE = re.compile(r"\b(\d+|one|two|three|four|five|six)\s+decimal\s+place", re.IGNORECASE)
_SORT_RE = re.compile(r"\b(?:sort|sorted|order|ordered)\s+(?:the\s+\w+\s+)?(?:by|in|from)\b"
                      r"|\b(?:descending|ascending)\b", re.IGNORECASE)
# `newest`/`oldest` are excluded: "from newest to oldest panels" is ORDER BY age ASC. Bare
# superlatives describe the metric, not the sort.
_DESC_RE = re.compile(r"descending|highest first|from highest|from largest|from most"
                      r"|largest first", re.IGNORECASE)
_ASC_RE = re.compile(r"ascending|lowest first|from lowest|from smallest|smallest first",
                     re.IGNORECASE)
_TOP_RE = re.compile(r"\b(?:top|bottom)[- ]?"
                     r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b", re.IGNORECASE)
_EACH_RE = re.compile(r"\b(?:for each|per)\s+[a-z]", re.IGNORECASE)
_AGG_RE = re.compile(r"\b(?:average|avg|mean|total|sum|count|number of|how many|median)\b",
                     re.IGNORECASE)
_AVG_RE = re.compile(r"\b(?:average|mean)\b", re.IGNORECASE)
# Bare `different`/`unique` are noise: "across different weather conditions" is grouping and
# "the inverter's unique identifier" is a column name.
_DIST_RE = re.compile(r"\b(?:distinct|how many different|number of different"
                      r"|number of unique|all the different|the unique)\b", re.IGNORECASE)
_BOOL_RE = re.compile(r"\bbool(?:ean)?\b", re.IGNORECASE)
_BOOL_LIT_RE = re.compile(r"\b(?:THEN|ELSE)\s+'(?:true|false)'", re.IGNORECASE)
_ARRAY_RE = re.compile(r"\b(?:an array of|a list of|array whose)\b", re.IGNORECASE)
# A bare column inside json_array() is one element per row, not an aggregate over a group.
_ARRAY_ONE_RE = re.compile(r"json_array\s*\(\s*[A-Za-z_][\w.]*\s*\)", re.IGNORECASE)
# Wording that asks for the entities with nothing on the other side of the join. Widening
# this list starts firing on gold; these phrases fire on none of the 270.
_INCLUSIVE_RE = re.compile(r"even if|even when|including those|include them|if any"
                           r"|regardless of whether|whether or not|including any"
                           r"|with no |lack |missing", re.IGNORECASE)

_Ask = tuple[str, str] | None  # (message, directive)

# Rules the official grader cannot score: it strips ROUND and DISTINCT from prediction and
# gold alike.
_UNSCORED = ("round", "distinct")


def _asks_round(q: str, sql: str, stmts: list[Any], final: Any) -> _Ask:
    """Check a stated decimal precision against the query's ROUND calls.

    Args:
        q: The question, truncated before its definitions block.
        sql: The candidate SQL.
        stmts: Every parsed statement.
        final: The last statement.

    Returns:
        ``(message, directive)`` when the precision is unmet, else None.
    """
    if not (want := {int(_NUMBERS.get(m.lower(), m)) for m in _DEC_RE.findall(q)}):
        return None
    # A bare ROUND(x) never satisfies a stated precision.
    got = {int(d.name) for r in final.find_all(exp.Round)
           if isinstance(d := r.args.get("decimals"), exp.Literal) and d.name.isdigit()}
    if want & got:
        return None
    n = min(want)
    return (f"Question asks for {n} decimal places; no ROUND(..., {n}) in the query.",
            f"Wrap the reported value in ROUND(..., {n}).")


def _asks_sort(q: str, sql: str, stmts: list[Any], final: Any) -> _Ask:
    """Check that a question asking for sorted output gets an ORDER BY.

    Args:
        q: The question, truncated before its definitions block.
        sql: The candidate SQL.
        stmts: Every parsed statement.
        final: The last statement.

    Returns:
        ``(message, directive)`` when the outer SELECT does not order, else None.
    """
    # A rank window is its own ordering: one gold sorts "by influence rank" with no ORDER BY.
    if not _SORT_RE.search(q) or _RANK_RE.search(sql) or final.args.get("order"):
        return None
    return ("Question asks for sorted output; the outer SELECT has no ORDER BY.",
            "Add an ORDER BY on the column the question names.")


def _asks_direction(q: str, sql: str, stmts: list[Any], final: Any) -> _Ask:
    """Check the ORDER BY direction against the one the question states.

    Args:
        q: The question, truncated before its definitions block.
        sql: The candidate SQL.
        stmts: Every parsed statement.
        final: The last statement.

    Returns:
        ``(message, directive)`` when the direction disagrees, else None.
    """
    desc = bool(_DESC_RE.search(q))
    if desc == bool(_ASC_RE.search(q)) or _RANK_RE.search(sql):
        return None
    order = final.args.get("order")
    terms = order.expressions if order else []
    if not terms or any("rank" in t.sql().lower() for t in terms):
        return None
    if any(bool(t.args.get("desc")) == desc for t in terms):
        return None
    want = "descending" if desc else "ascending"
    return (f"Question asks for {want} order; the ORDER BY is not {want}.",
            f"Order by the ranked column {'DESC' if desc else 'ASC'}.")


def _asks_top_n(q: str, sql: str, stmts: list[Any], final: Any) -> _Ask:
    """Check that a top/bottom-N question caps its rows.

    Args:
        q: The question, truncated before its definitions block.
        sql: The candidate SQL.
        stmts: Every parsed statement.
        final: The last statement.

    Returns:
        ``(message, directive)`` when nothing caps, else None.
    """
    # A cap only, never an ordering: `check_order_by` owns that, and one gold caps in a CTE.
    if not _TOP_RE.search(q) or _RANK_RE.search(sql):
        return None
    if any(s.find(exp.Limit) for s in stmts):
        return None
    return ("Question asks for the top/bottom N rows; the query caps nothing.",
            "Add LIMIT N, or filter a rank window on <= N.")


def _asks_group(q: str, sql: str, stmts: list[Any], final: Any) -> _Ask:
    """Check that a per-group question does not aggregate to a single row.

    Args:
        q: The question, truncated before its definitions block.
        sql: The candidate SQL.
        stmts: Every parsed statement.
        final: The last statement.

    Returns:
        ``(message, directive)`` for a bare single-row aggregate only - "for each X, show
        its id and metric" means one row per row in a third of these questions - else None.
    """
    if not (_EACH_RE.search(q) and _AGG_RE.search(q)):
        return None
    if final.find(exp.Group) or final.find(exp.Window):
        return None
    if not any((a := e.find(exp.Sum, exp.Avg, exp.Count)) is not None
               and a.parent_select is final for e in final.expressions):
        return None
    return ("Question asks for one row per group but the query aggregates to a single row.",
            "Add a GROUP BY on the grouping column the question names.")


def _asks_average(q: str, sql: str, stmts: list[Any], final: Any) -> _Ask:
    """Check that a question asking for an average computes one.

    Args:
        q: The question, truncated before its definitions block.
        sql: The candidate SQL.
        stmts: Every parsed statement.
        final: The last statement.

    Returns:
        ``(message, directive)`` when no average is computed, else None.
    """
    upper = sql.upper()
    if not _AVG_RE.search(q) or any(f in upper for f in ("AVG(", "MEDIAN(", "PERCENTILE")) \
            or ("SUM(" in upper and "COUNT(" in upper):
        return None
    return ("Question asks for an average; the query computes none.",
            "Use AVG(), or SUM()/COUNT() over the group.")


def _asks_distinct(q: str, sql: str, stmts: list[Any], final: Any) -> _Ask:
    """Check that a question asking for distinct values deduplicates.

    Args:
        q: The question, truncated before its definitions block.
        sql: The candidate SQL.
        stmts: Every parsed statement.
        final: The last statement.

    Returns:
        ``(message, directive)`` when the query neither DISTINCTs nor groups, else None.
    """
    if not _DIST_RE.search(q) or "DISTINCT" in sql.upper() or final.find(exp.Group):
        return None
    return ("Question asks for distinct values; the query neither DISTINCTs nor groups.",
            "Add DISTINCT, or GROUP BY the identifying column.")


def _asks_boolean(q: str, sql: str, stmts: list[Any], final: Any) -> _Ask:
    """Check that a boolean column is emitted as a boolean, not as its English name.

    Args:
        q: The question, truncated before its definitions block.
        sql: The candidate SQL.
        stmts: Every parsed statement.
        final: The last statement.

    Returns:
        ``(message, directive)`` when a CASE yields the strings, else None.
    """
    if not _BOOL_RE.search(q) or not _BOOL_LIT_RE.search(sql):
        return None
    return ("Question asks for a boolean; the query yields the strings 'True'/'False'.",
            "Yield TRUE/FALSE, which store as 1/0, not their names in quotes.")


def _asks_array(q: str, sql: str, stmts: list[Any], final: Any) -> _Ask:
    """Check that a question asking for an array aggregates rather than wrapping one value.

    Args:
        q: The question, truncated before its definitions block.
        sql: The candidate SQL.
        stmts: Every parsed statement.
        final: The last statement.

    Returns:
        ``(message, directive)`` when json_array() wraps a bare column, else None.
    """
    if not _ARRAY_RE.search(q) or not _ARRAY_ONE_RE.search(sql):
        return None
    return ("Question asks for an array of values; json_array(column) wraps one value per "
            "row.", "Aggregate the values over the group with GROUP_CONCAT.")


def _asks_inclusive(q: str, sql: str, stmts: list[Any], final: Any) -> _Ask:
    """Check that a question asking to keep unmatched entities does not inner-join them away.

    Args:
        q: The question, truncated before its definitions block.
        sql: The candidate SQL.
        stmts: Every parsed statement.
        final: The last statement.

    Returns:
        ``(message, directive)`` when every join is inner, else None.
    """
    joins = [j for s in stmts for j in s.find_all(exp.Join) if j.args.get("on")]
    if not _INCLUSIVE_RE.search(q) or not joins or any(j.side for j in joins):
        return None
    return ("Question asks to keep entities with nothing on the other side; every join is "
            "inner.", "LEFT JOIN the optional side so its rows survive with NULLs.")


_QUESTION_RULES = (_asks_round, _asks_sort, _asks_direction, _asks_top_n, _asks_group,
                   _asks_average, _asks_distinct, _asks_boolean, _asks_array, _asks_inclusive)


def _question_conflict(sql: str, question: str, dialect: str = "") -> Issue | None:
    """Collapse every unmet question-stated requirement into one Issue.

    Args:
        sql: The candidate SQL.
        question: The user's question.
        dialect: sqlglot dialect name.

    Returns:
        One Issue over every rule that fired - ``info`` when only the rules the grader
        strips did - else None. Inert on a write, and on everything inside the definitions
        block.
    """
    q = DEFINITIONS.split(question, maxsplit=1)[0]
    # `syntax` and `dry_run` own an unparseable query; a write is where "2 decimal places"
    # is a column spec rather than a rounding request.
    if not q.strip() or not (stmts := reads(sql, dialect)):
        return None
    asks = [(rule.__name__.removeprefix("_asks_"), a) for rule in _QUESTION_RULES
            if (a := rule(q, sql, stmts, stmts[-1]))]
    if not asks:
        return None
    scored = [a for name, a in asks if name not in _UNSCORED]
    return Issue("; ".join(m for _, (m, _d) in asks),
                 " ".join(d for _n, (_m, d) in asks),
                 "warning" if scored else "info", schema="none")


# ---- Cascade -------------------------------------------------------------


def run_checkers(sql: str, db: DatabaseConnection,
                 question: str = "") -> list[tuple[str, Issue]]:
    """Run the whole cascade over one candidate.

    Args:
        sql: The candidate SQL.
        db: Connection the data-backed checkers probe.
        question: The user's question, enabling the ``naming`` and ``question`` checkers.

    Returns:
        Every ``(checker name, finding)``, in cascade order.
    """
    found = [(name, issue) for name, check in CHECKERS if (issue := check(sql, db))]
    if question and (issue := _naming_conflict(sql, question)):
        found.append(("naming", issue))
    if question and (issue := _question_conflict(sql, question, _dialect(db))):
        found.append(("question", issue))
    return found


def actionable(found: list[Finding]) -> list[Finding]:
    """Select the findings worth an LLM rewrite.

    Args:
        found: Every finding the cascade reported.

    Returns:
        The non-``info`` findings, worst first and ``result`` ahead of its severity peers,
        since a cosmetic finding can otherwise spend the whole budget.
    """
    return sorted((f for f in found if f[1].severity != "info"),
                  key=lambda f: (_SEVERITY_RANK[f[1].severity], f[0] != "result"))


def render(found: Iterable[Finding], *, fixes: bool = False) -> list[str]:
    """Render findings as ``[checker] message`` lines; the one renderer both callers use.

    Args:
        found: Findings to render.
        fixes: Also append each fix directive, which is what an LLM reads.

    Returns:
        One line per finding.
    """
    return [f"[{n}] {i.message}{' ' + i.directive if fixes else ''}".rstrip() for n, i in found]


def _errors(found: list[Finding]) -> bool:
    """Whether a finding says the query cannot run as written.

    Args:
        found: Findings to inspect.

    Returns:
        True when any is an ``error``.
    """
    return any(issue.severity == "error" for _, issue in found)


# ---- Repair driver -------------------------------------------------------


class SQLRepair:
    """Validates SQL through the checker cascade and repairs failures via LLM."""

    def __init__(
        self,
        llm: LLMClient,
        db: DatabaseConnection,
        prompt_manager: PromptManager,
        max_retries: int = 3,
        schema_loader: SchemaLoader | None = None,
        knowledge: DatabaseKnowledge | None = None,
    ) -> None:
        self.llm = llm
        self.db = db
        self.prompt_manager = prompt_manager
        self.max_retries = max_retries
        self.schema_loader = schema_loader
        self.knowledge = knowledge

    async def repair(self, sql: str, question: str, schema_text: str) -> tuple[str, list[str]]:
        """Repair via LLM until the cascade is clean or the retries run out.

        Args:
            sql: The candidate SQL.
            question: The user's question.
            schema_text: The pipeline's schema, used when a finding asks for the full tier.

        Returns:
            ``(final SQL, every [name] message flagged across all attempts)``.
        """
        issues: list[str] = []
        current = sql
        rounds = 0
        failed = run_checkers(current, self.db, question)
        while failed:
            issues.extend(render(failed))  # the audit log keeps `info`; the prompt does not
            if not (acts := actionable(failed)) or rounds == self.max_retries:
                break
            rounds += 1
            repaired = await self._llm_repair(current, render(acts, fixes=True), question,
                                              self._schema_for(current, schema_text, acts))
            # Retrying unchanged SQL is futile: it yields the same failures.
            if _normalize(repaired) == _normalize(current):
                logger.info("Repair stalled: LLM returned unchanged SQL after %d round(s)", rounds)
                break
            fresh = run_checkers(repaired, self.db, question)
            # A rewrite that cannot run is worse than the query it replaced.
            if _errors(fresh) and not _errors(failed):
                issues.append(f"[repair] discarded a rewrite that does not run: {repaired[:80]}")
                logger.info("Repair rejected: rewrite fails where the input ran")
                break
            current, failed = repaired, fresh

        if issues:
            logger.info("Repair finished: %d issue(s) over %d LLM round(s)", len(issues), rounds)
        return current, issues

    def _schema_for(self, sql: str, fallback: str, found: list[Finding]) -> str:
        """Render as much schema as the findings' fixes need, and no more.

        Args:
            sql: The candidate SQL, naming the tables the ``touched`` tier renders.
            fallback: The pipeline's schema text, used for the ``full`` tier.
            found: The actionable findings.

        Returns:
            The schema text, or ``""`` when every directive only rewrites what the query
            already says - a schema beside it is an invitation to rewrite the whole query.
        """
        tier = max((i.schema for _n, i in found), key=lambda t: _SCHEMA_RANK[t], default="touched")
        if tier == "none":
            return ""
        if tier == "full" or self.schema_loader is None:
            return fallback
        known = set(self.schema_loader.get_table_names())
        touched = [t for t in SchemaLinker.extract_fields(sql) if t in known]
        if not touched:
            return fallback
        logger.debug("Repair schema narrowed to %d of %d tables: %s",
                     len(touched), len(known), ", ".join(touched))
        return self.schema_loader.format_schema(tables=touched)

    async def _llm_repair(self, sql: str, findings: list[str], question: str,
                          schema_text: str) -> str:
        """Ask the LLM for one rewrite.

        Args:
            sql: The SQL to fix.
            findings: Rendered findings with their fix directives.
            question: The user's question.
            schema_text: Schema to include, possibly empty.

        Returns:
            The rewritten SQL.
        """
        prompt = self.prompt_manager.render(
            "repair_sql",
            sql=sql,
            findings=findings,
            schema=schema_text,
            question=question,
            dialect=self.db.dialect_name,
        )
        repaired = await self.llm.chat_for_sql(prompt)
        logger.info("Repaired (%s) → %d chars", findings[0][:60], len(repaired))
        return repaired
