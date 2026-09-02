"""LLM-generated knowledge base: cross-column relations the per-column summaries miss.

Mirrors summarizer.py: one batched LLM call per table, bounded concurrency, JSON out.
Format matches the LiveSQLBench ``*_kb.jsonl`` files so either source is interchangeable.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Callable, Collection
from dataclasses import dataclass, field, replace
from itertools import combinations
from typing import Any

import sqlglot

from text2sql.llm import LLMClient, parse_llm_json
from text2sql.profiler.stats import DatabaseProfile, DictSerde, TableProfile, flat, group
from text2sql.prompts.manager import PromptManager

logger = logging.getLogger(__name__)

MAX_CONCURRENCY = 7
# Entries asked of one table, from its column count: a 3-column table cannot support 10
# relations and a 30-column one supports more. Advisory - nothing truncates the reply.
_MIN_ENTRIES, _MAX_ENTRIES = 5, 25
# Max entries injected into a generation prompt after schema linking.
MAX_SELECTED = 10


def _cell(text: Any) -> str:
    """Flatten a value into one markdown table cell.

    Args:
        text: The value to render.

    Returns:
        The text with newlines collapsed and pipes replaced, either of which breaks the row.
    """
    return re.sub(r"\s+", " ", str(text)).replace("|", "/").strip()


def _columns(table: TableProfile, describe: Any = None) -> list[dict[str, str]]:
    """Render one table's columns as prompt rows, with a description only when one is supplied.

    Args:
        table: The table profile.
        describe: ``describe(table, column)`` supplying each column's description, or None.

    Returns:
        One row per column.
    """
    return [
        {
            "column_name": c.column_name,
            "column_type": c.column_type + (" space-padded" if c.padded else ""),
            "description": _cell(describe(table.table_name, c.column_name)) if describe else "",
            # Not "values": Jinja resolves ``c.values`` to ``dict.values`` before the key.
            "top_values": _cell(", ".join(str(v["value"]) for v in c.top_k_values)),
        }
        for c in table.columns.values()
    ]


# The shipped KB writes formulas in LaTeX; prompts and the data check both need SQL, so the
# markup is rewritten to SQL operators, functions and CASE expressions.
_MATH_SYMBOLS = {
    "times": "*", "cdot": "*", "div": "/", "neq": "!=", "geq": ">=", "leq": "<=",
    "approx": "~=", "equiv": "=", "mid": "where", "sum": "SUM", "infty": "infinity",
}
_MATH_WRAPPERS = "text|textbf|textit|mathrm|mathbf|mathit|mathcal|mathbb|operatorname"
_MATH_FUNCS = "sqrt|log|exp|max|min|sum|abs"
# Spacing-only commands: dropped outright.
_MATH_SPACERS = {",", ";", ":", "!", " ", "\n"}
# Some KB lines write `\text` with one backslash, so JSON parses it as a TAB and the command
# loses its head. Restore the few that occur.
_MATH_TORN = {"\text": "\\text", "\times": "\\times", "\x0crac": "\\frac",
              "\x08egin": "\\begin", "\right": "\\right"}
# Between bars: a set or a bare relation name is counted, anything else is an absolute value.
_MATH_SET = ("\\{", "\\mathcal")
_MATH_UNICODE = str.maketrans({"×": "*", "·": "*", "÷": "/", "−": "-",
                               "≥": ">=", "≤": "<=", "≠": "!=", "∑": "SUM"})


# A name the formula sums over is a collection, so bars around it are a row count.
_ITERATED = re.compile(r"\\in\s*(?:\\[a-z]+\{)?\s*([A-Za-z_]\w*)")


def _case(body: str) -> str:
    r"""Rewrite a ``\begin{cases}`` body, ``value & condition`` per row, as a SQL CASE.

    Only inside this span does a lone backslash before whitespace mean a row break rather
    than an escaped space. The arms still carry their LaTeX, which the caller resolves.
    """
    arms = []
    for row in re.split(r"\\+(?=\s|$)", body):
        value, _, condition = row.partition("&")
        value, condition = value.strip().rstrip(","), condition.strip().rstrip(",")
        arms.append(f"WHEN {condition} THEN {value}" if condition else f"ELSE {value}")
    return f"CASE {' '.join(arms)} END"


def _bars(body: str, collections_: Collection[str]) -> str:
    """Rewrite a bar-delimited fragment as ``COUNT`` for a set or a collection, ``ABS``
    otherwise: ``|macdtrail|`` is a magnitude and ``|treatmentoutcomes|`` is a row count.
    """
    inner = body.strip()
    counted = any(k in body for k in _MATH_SET) or inner.lower() in collections_
    return f"{'COUNT' if counted else 'ABS'}({inner})"


def plain_text(text: str, tables: Collection[str] = ()) -> str:
    r"""Rewrite LaTeX maths as a SQL expression.

    Args:
        text: The definition or description to rewrite.
        tables: Lowercased table names, so bars around one read as a row count.

    Returns:
        The text with the markup resolved: ``\frac{a}{b}`` -> ``(a) / (b)``, ``|x|`` ->
        ``ABS(x)``, ``\begin{cases}`` -> ``CASE WHEN ... END``, ``a^b`` -> ``POWER(a, b)``.
    """
    text = text.translate(_MATH_UNICODE)
    for torn, command in _MATH_TORN.items():
        text = text.replace(torn, command)
    if "\\" not in text and "$" not in text:
        return text
    s = re.sub(r"(?s)\\begin\{cases\}(.*?)\\end\{cases\}", lambda m: _case(m.group(1)), text)
    for bar in (r"\left|", r"\right|", r"\lvert", r"\rvert"):
        s = s.replace(bar, "|")
    counted = {t.lower() for t in tables} | {n.lower() for n in _ITERATED.findall(s)}
    s = re.sub(r"\|([^|]{1,120}?)\|", lambda m: _bars(str(m.group(1)), counted), s)
    s = s.replace("\\\\", "; ").replace("\\{", "(").replace("\\}", ")")
    for _ in range(8):  # nested braces resolve inside-out, one layer per pass
        before = s
        s = re.sub(rf"\\(?:{_MATH_WRAPPERS})\s*\{{([^{{}}]*)\}}", r"\1", s)
        s = re.sub(r"([_^])\{([^{}]*)\}", r"\1\2", s)
        s = re.sub(rf"\\({_MATH_FUNCS})\s*\{{([^{{}}]*)\}}",
                   lambda m: f"{m.group(1).upper()}({m.group(2)})", s)
        s = re.sub(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"(\1) / (\2)", s)
        s = re.sub(r"\\binom\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"C(\1, \2)", s)
        s = re.sub(r"(?<=[0-9])\{([^{}]*)\}", r"\1", s)  # digit grouping: 1{,}000
        s = re.sub(r"\\(?:left|right|bigl?|Bigl?)\s*([([{|.\\])", r"\1", s)
        if s == before:
            break
    s = re.sub(r"\\(?:left|right)", "", s)
    s = re.sub(r"\\([A-Za-z]+)", lambda m: _MATH_SYMBOLS.get(m.group(1), m.group(1)), s)
    s = re.sub(r"\\(.)", lambda m: "" if m.group(1) in _MATH_SPACERS else m.group(1), s)
    s = s.replace("$", "").replace("{", "").replace("}", "")
    s = re.sub(r"\bWHEN\s+if\s+", "WHEN ", s)
    s = re.sub(r"\bWHEN\s+(?:otherwise|else)\s+THEN\b", "ELSE", s)
    s = re.sub(r"([\w.]{2,}|\d)\s*\^\s*(-?[\w.]+)", r"POWER(\1, \2)", s)  # a bare base only
    return re.sub(r"[ \t]{2,}", " ", s).strip()


WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _expression(text: str, columns: set[str]) -> bool:
    """Whether a fragment is a SQL expression over the given columns alone.

    Args:
        text: The fragment to test.
        columns: The column names it may refer to.

    Returns:
        True when every word is one of ``columns`` and the fragment parses as SQL.
    """
    if not (words := set(WORD.findall(text))) or not words <= columns:
        return False
    try:
        return sqlglot.parse_one(text, read="sqlite") is not None
    except Exception:
        return False  # a fragment like `tierstep <` passes the word test but is not SQL


def _aggregated(text: str) -> bool:
    """Whether an expression aggregates or windows, so no row-wise probe can run it.

    Args:
        text: The expression to test.

    Returns:
        True for a one-argument aggregate or an OVER clause. SQLite's scalar ``MAX(a, b)``
        takes more than one argument and stays measurable.
    """
    try:
        parsed = sqlglot.parse_one(text, read="sqlite")
    except Exception:
        return False
    return any(isinstance(n, sqlglot.exp.Window) or not n.args.get("expressions")
               for n in parsed.find_all(sqlglot.exp.AggFunc, sqlglot.exp.Window))


_BOOLEAN = re.compile(r"[<>]|!=|\bIN\b|\bLIKE\b|\bIS\b|\bBETWEEN\b|=", re.IGNORECASE)


def json_paths(definition: str, columns: set[str]) -> str:
    """Rewrite the profile's dotted pseudo-columns to ``json_extract`` calls.

    The dotted form is what the model copies and no database accepts it.

    Args:
        definition: The definition to rewrite.
        columns: The table's column names, dotted paths included.

    Returns:
        The definition, longest path substituted first.
    """
    for path in sorted((c for c in columns if "." in c), key=len, reverse=True):
        parent, _, rest = path.partition(".")
        definition = definition.replace(path, f"json_extract({parent}, '$.{rest}')")
    return definition


# {2,12}, not {2,6}: `(NETWORTH)` is 8 characters, and a short cap silently dropped both the
# child link and the column the term names.
_ACRONYM = re.compile(r"\s*\(([A-Z]{2,12})\)\s*$")
# Shortest acronym worth matching a question against: "MI" hits "minimum", "AE" hits "are".
_MIN_ACRONYM = 3


def _term(knowledge: str) -> tuple[str, str]:
    """Split a term into its name and acronym.

    Args:
        knowledge: The term as written, e.g. ``Net Worth (NETWORTH)``.

    Returns:
        ``(base name lowercased, trailing acronym or "")``.
    """
    m = _ACRONYM.search(knowledge)
    return _ACRONYM.sub("", knowledge).strip().lower(), m.group(1) if m else ""


def owners(definition: str, columns: dict[str, set[str]]) -> tuple[str, ...]:
    """The tables whose columns a definition names.

    Args:
        definition: The definition text, whose column names may be qualified.
        columns: ``{table: {lowercased columns}}``.

    Returns:
        The matching table names, sorted; empty when the definition names only other terms.
    """
    words = {w.lower() for w in WORD.findall(definition)}
    return tuple(sorted(t for t, cols in columns.items() if words & cols))


def relates(definition: str) -> bool:
    """Whether a definition says more than a column name does.

    Args:
        definition: The definition to test.

    Returns:
        False for a lone identifier, which is that column's own description and is already
        carried by ``*_meaning_base_long.json``.
    """
    return bool(d := definition.strip()) and not WORD.fullmatch(d)


def restates_a_join(definition: str) -> bool:
    """Whether a definition across tables only says that their rows relate.

    Args:
        definition: The definition to test.

    Returns:
        True when every comparison equates two columns or tests one for NULL, which is the
        foreign key map rather than domain knowledge.
    """
    try:
        parsed: sqlglot.exp.Expression = sqlglot.maybe_parse(definition)
    except Exception:
        return False
    tests = list(parsed.find_all(sqlglot.exp.EQ, sqlglot.exp.Is))
    return bool(tests) and len(tests) == len(list(parsed.find_all(sqlglot.exp.Predicate))) and all(
        isinstance(t.this, sqlglot.exp.Column)
        and isinstance(t.expression, sqlglot.exp.Column | sqlglot.exp.Null) for t in tests)


def unquoted(definition: str, known: set[str]) -> bool:
    """Whether a comparison in the definition reads an unquoted text literal as a column.

    ``sancresult = Fail`` is a filter that errors rather than one that selects.

    Args:
        definition: The definition to test.
        known: Real column names and the words of every term name, lowercased.

    Returns:
        True when the compared-against name does not exist.
    """
    try:
        parsed: sqlglot.exp.Expression = sqlglot.maybe_parse(definition)
    except Exception:
        return False
    return any(c.name.lower() not in known
               for p in parsed.find_all(sqlglot.exp.Predicate)
               if isinstance(right := p.args.get("expression"), sqlglot.exp.Expression)
               for c in right.find_all(sqlglot.exp.Column))


def asserted(definition: str, knowledge: str, columns: set[str]) -> str:
    """Turn a bare formula into the equality the data can check, when the term names a column.

    ``Net Worth (NETWORTH)`` defined as ``totassets - totliabs`` claims nothing testable;
    as ``networth = totassets - totliabs`` the data can refute it.

    Args:
        definition: The definition as written.
        knowledge: The term the definition belongs to.
        columns: The table's column names.

    Returns:
        The asserted definition, or the original when it already compares something.
    """
    if _BOOLEAN.search(definition):
        return definition
    base, acronym = _term(knowledge)
    column = next((c for c in (acronym.lower(), base.replace(" ", "")) if c in columns), "")
    return f"{column} = {definition}" if column and column not in definition else definition


def measured(definition: str, table: str, columns: set[str],
             count: Callable[[str, str], int | None]) -> tuple[str, bool]:
    """Measure a definition against the data.

    Args:
        definition: The definition to measure.
        table: The table that owns it.
        columns: That table's column names.
        count: Runs a predicate against a table and returns the matching row count, or None
            when it does not run.

    Returns:
        ``(prose verdict, keep the definition)``. A dropped definition gets no prose: the
        term's name survives, and saying why costs a line at every point of use.
    """
    if " != " in definition:
        return "", True  # `verified_definition` already said what the data thinks
    sides = [s for part in definition.split("=") if (s := part.strip())]
    exprs = [_expression(s, columns) for s in sides]
    if sum(exprs) >= 2:
        return "", True  # a formula between real columns: `verified_definition` owns it
    # `annualexpenses = mthexp * 12` names the derived value rather than claiming a column
    # holds it, which is the shape the shipped bases use.
    if len(sides) >= 2 and any(exprs) and all(
            WORD.fullmatch(s) for s, e in zip(sides, exprs, strict=True) if not e):
        return "No column stores this; compute the formula.", True
    if _aggregated(definition):
        return "", True  # nothing to probe row by row, and the formula still stands
    if not _BOOLEAN.search(definition):
        # A bare formula has no share, but it can still name a column its own table lacks.
        return ("", count(table, f"({definition}) IS NOT NULL") is not None)
    hits, misses = count(table, definition), count(table, f"NOT ({definition})")
    if hits is None or misses is None or not hits + misses:
        return "", False
    # A cohort filter, so a low share means precise rather than false; an identity claim goes
    # through `verified_definition`. Only the extremes say nothing. Rounded, so the label and
    # the verdict cannot disagree.
    share = round(100 * hits / (hits + misses))
    if share in (0, 100):
        return "", False
    return f"Selects {share}% of rows.", True


def verified_definition(
    definition: str, tables: dict[str, set[str]], mismatches: Callable[[str, str], int | None]
) -> str:
    """Flip an equality the data refutes to ``!=``, so the false claim does not stand.

    Args:
        definition: The definition to verify.
        tables: ``{table: {columns}}`` the sides may refer to.
        mismatches: Runs a predicate against a table and returns the matching row count.

    Returns:
        The definition, with the refuted equality flipped.
    """
    sides = [s for part in definition.split("=") if (s := part.strip())]
    for table, columns in tables.items():
        comparable = [(i, s) for i, s in enumerate(sides) if _expression(s, columns)]
        for (_, left), (j, right) in combinations(comparable, 2):
            if mismatches(table, f"({left}) IS NOT NULL AND ({right}) IS NOT NULL "
                                 f"AND ABS(({left}) - ({right})) > 0.001"):
                return " = ".join(sides[:j]) + " != " + " = ".join(sides[j:])
    return definition


@dataclass
class KnowledgeEntry(DictSerde):
    """One knowledge relation, in the LiveSQLBench ``*_kb.jsonl`` record shape."""

    id: int
    knowledge: str
    description: str = ""
    definition: str = ""
    type: str = "domain_knowledge"
    children_knowledge: list[int] | int = -1
    tables: tuple[str, ...] = ()  # the tables whose columns the definition names
    refuted: bool = False  # the data contradicted the equality, so it now reads ``!=``

    def __post_init__(self) -> None:
        """Normalise the owners, which JSON round-trips as a list."""
        self.tables = tuple(self.tables)

    @property
    def children(self) -> list[int]:
        """The child entry ids, or an empty list when there are none."""
        c = self.children_knowledge
        return c if isinstance(c, list) else []

    @property
    def refutes(self) -> tuple[str, str] | None:
        """The ``(column, expression)`` pair the data contradicted, or None if nothing was."""
        if not self.refuted or " != " not in self.definition:
            return None
        left, _, right = self.definition.rpartition(" != ")
        column, expression = ((right, left) if WORD.fullmatch(right.strip())
                              else (left.rsplit("=", 1)[-1], right))
        return column.strip(), expression.strip()

    def plain(self, tables: Collection[str] = ()) -> KnowledgeEntry:
        """Flatten LaTeX maths out of every text field.

        Args:
            tables: Lowercased table names, so bars around one read as a row count.

        Returns:
            A copy of the entry with plain-text fields.
        """
        return replace(self, knowledge=plain_text(self.knowledge, tables),
                       description=plain_text(self.description, tables),
                       definition=plain_text(self.definition, tables))

    def mentions(self, names: set[str]) -> int:
        """Count how many of the given names this entry's text refers to.

        Args:
            names: Names to look for; matched as whole words.

        Returns:
            The number of names mentioned.
        """
        text = f"{self.knowledge} {self.description} {self.definition}".lower()
        return sum(bool(re.search(rf"\b{re.escape(n)}\b", text)) for n in names)

    def named_in(self, text: str) -> bool:
        """Whether a text refers to this term by name or by acronym.

        Args:
            text: The text to search, typically the question.

        Returns:
            True when the term's name or an acronym of at least ``_MIN_ACRONYM`` characters
            appears as a whole word.
        """
        name, acronym = _term(self.knowledge)
        wanted = [name] + ([acronym.lower()] if len(acronym) >= _MIN_ACRONYM else [])
        return any(re.search(rf"\b{re.escape(w)}\b", text.lower()) for w in wanted if w)


@dataclass
class DatabaseKnowledge:
    """All knowledge entries for a database, keyed by id."""

    entries: dict[int, KnowledgeEntry] = field(default_factory=dict)

    def to_flat(self, prefix: str) -> dict[str, Any]:
        """Flatten the entries for caching.

        Args:
            prefix: Database prefix for the cache keys.

        Returns:
            ``{db|tables|term: entry}``, the owners comma-joined so one key stays one entry.
        """
        return {flat(prefix, ",".join(e.tables), e.knowledge): e.to_dict()
                for e in self.entries.values()}

    @classmethod
    def from_flat(cls, doc: dict[str, Any]) -> DatabaseKnowledge:
        """Rebuild the knowledge base from a cached flat document.

        Args:
            doc: The cache envelope.

        Returns:
            The knowledge base, entries keyed by id.
        """
        dk = cls()
        for entries in group(doc).values():
            for value in entries.values():
                entry = KnowledgeEntry.from_dict(value)
                dk.entries[entry.id] = entry
        return dk

    @classmethod
    def from_jsonl(cls, text: str, tables: Collection[str] = ()) -> DatabaseKnowledge:
        """Load the shipped ``*_kb.jsonl`` form, one record per line and no table column.

        Args:
            text: The file contents.
            tables: Lowercased table names, so bars around one read as a row count.

        Returns:
            The knowledge base, with LaTeX maths flattened.
        """
        dk = cls()
        for line in text.splitlines():
            if line.strip():
                entry = KnowledgeEntry.from_dict(json.loads(line)).plain(tables)
                dk.entries[entry.id] = entry
        return dk

    def verified(self, tables: dict[str, set[str]],
                 mismatches: Callable[[str, str], int | None]) -> DatabaseKnowledge:
        """Annotate every entry with the data's verdict.

        Args:
            tables: ``{table: {columns}}`` the definitions may refer to.
            mismatches: Runs a predicate against a table and returns the matching row count.

        Returns:
            A new knowledge base. Definitions the data cannot support are cleared, but the
            names stay, so the term is still findable.
        """
        entries = {}
        for i, e in self.entries.items():
            owner = e.tables[0] if len(e.tables) == 1 else ""
            columns = {c for t in e.tables for c in tables.get(t, set())}
            definition = verified_definition(
                asserted(json_paths(e.definition, columns), e.knowledge, columns),
                tables, mismatches)
            if refuted := " != " in definition and " != " not in e.definition:
                logger.warning("Knowledge '%s' contradicts the data: %s", e.knowledge, definition)
            note, usable = (measured(definition, owner, columns, mismatches)
                            if owner in tables else ("", True))
            entries[i] = replace(e, definition=definition if usable else "", refuted=refuted,
                                 description=f"{e.description} {note}".strip())
        return DatabaseKnowledge(entries)

    def linked(self) -> DatabaseKnowledge:
        """Derive the child links the model did not declare.

        Returns:
            A new knowledge base where an entry naming another, by name or acronym, carries
            it as a child.
        """
        base, acronym = {}, {}
        for i, e in self.entries.items():
            base[i], a = _term(e.knowledge)
            if a:
                acronym[i] = a
        entries = {}
        for i, e in self.entries.items():
            refs = {j for j, n in base.items() if j != i and n and e.mentions({n})}
            refs |= {j for j, a in acronym.items() if j != i
                     and re.search(rf"\b{a}\b", f"{e.definition} {e.description}")}
            entries[i] = replace(e, children_knowledge=sorted(set(e.children) | refs) or -1)
        return DatabaseKnowledge(entries)

    def merge(self, other: DatabaseKnowledge) -> DatabaseKnowledge:
        """Append another knowledge base after this one.

        Args:
            other: The base to append.

        Returns:
            The merged base, with the appended ids and child links remapped.
        """
        off = max(self.entries, default=-1) + 1
        merged = DatabaseKnowledge(dict(self.entries))
        for e in other.entries.values():
            merged.entries[e.id + off] = replace(
                e, id=e.id + off, children_knowledge=[c + off for c in e.children] or -1)
        return merged

    def _subset(self, kept: set[int]) -> DatabaseKnowledge:
        """Restrict the base to a set of entries.

        Args:
            kept: Entry ids to keep.

        Returns:
            Those entries, with child links to dropped entries pruned.
        """
        return DatabaseKnowledge(
            {i: replace(e, children_knowledge=[c for c in e.children if c in kept] or -1)
             for i, e in self.entries.items() if i in kept})

    def without(self, tables: set[str]) -> DatabaseKnowledge:
        """Drop what a re-profile of the given tables replaces.

        A single-table entry survives while its table is untouched. Everything relating two
        tables, or built on other terms, is database-wide and regenerated whole.

        Args:
            tables: Table names being re-profiled.

        Returns:
            The remaining entries.
        """
        return self._subset({i for i, e in self.entries.items()
                             if len(e.tables) == 1 and e.tables[0] not in tables})

    def ground(self, columns: set[str]) -> DatabaseKnowledge:
        """Drop entries that are not a relation over real columns.

        A small model's invented column can never be selected, and a one-column definition
        restates that column.

        Args:
            columns: The real column names, lowercased.

        Returns:
            The grounded entries, including those built on a grounded one.
        """
        known = columns | {w for e in self.entries.values()
                           for w in re.findall(r"[a-z_][a-z_0-9]*", e.knowledge.lower())}
        relational = [e for e in self.entries.values()
                      if relates(e.definition) and not unquoted(e.definition, known)
                      and not (len(e.tables) > 1 and restates_a_join(e.definition))]
        kept = {e.id for e in relational if e.mentions(columns)}
        while True:  # an entry built on a grounded one is itself grounded
            names = {self.entries[i].knowledge.lower() for i in kept}
            extra = {e.id for e in relational if e.id not in kept and e.mentions(names)}
            if not extra:
                break
            kept |= extra
        if dropped := [e.knowledge for e in self.entries.values() if e.id not in kept]:
            logger.warning("Dropped %d knowledge entries that are not a column relation: %s",
                           len(dropped), ", ".join(dropped))
        return self._subset(kept)

    def _close(self, scored: dict[int, int]) -> dict[int, int]:
        """Extend a scored set with every entry it depends on.

        Args:
            scored: ``{entry id: score}``, mutated in place.

        Returns:
            The same mapping, with dependencies added at score 0 so they sort after direct
            hits.
        """
        pending = list(scored)
        while pending:
            for child in self.entries[pending.pop()].children:
                if child in self.entries and child not in scored:
                    scored[child] = 0
                    pending.append(child)
        return scored

    def with_children(self, entries: list[KnowledgeEntry]) -> list[KnowledgeEntry]:
        """Close a list of entries over their dependencies.

        Args:
            entries: The entries to start from.

        Returns:
            Those entries and their dependencies, in the order the walk reaches them.
        """
        return [self.entries[i] for i in self._close({e.id: 1 for e in entries})]

    def select(self, names: set[str], question: str = "",
               limit: int = MAX_SELECTED) -> list[KnowledgeEntry]:
        """Pick the entries a linked schema needs.

        Args:
            names: Linked table and column names.
            question: The question, whose named terms outrank the mention count, so that a
                wider schema cannot cut a term the question asks about.
            limit: Maximum entries returned.

        Returns:
            The entries mentioning those names, plus their dependencies, best first.
        """
        if not names:
            return []
        scored = self._close({e.id: n for e in self.entries.values() if (n := e.mentions(names))})
        return [self.entries[i] for i in sorted(
            scored,
            key=lambda i: (not self.entries[i].named_in(question), -scored[i], i))[:limit]]


def _resolve(name: str, by_name: dict[str, int]) -> int | None:
    """Resolve a child reference to an entry index.

    Args:
        name: The name the model pointed at, which it often shortens.
        by_name: ``{lowercased name: index}``.

    Returns:
        The index by exact name, else a unique containment match, else None.
    """
    key = " ".join(name.split()).lower()
    if not key:
        return None
    if key in by_name:
        return by_name[key]
    hits = {i for n, i in by_name.items() if key in n or n in key}
    return hits.pop() if len(hits) == 1 else None


class KnowledgeGenerator:
    """Derives cross-column knowledge from a profile + its column summaries."""

    def __init__(
        self,
        llm: LLMClient,
        prompt_manager: PromptManager,
        *,
        max_concurrency: int = MAX_CONCURRENCY,
    ) -> None:
        self.llm = llm
        self.prompt_manager = prompt_manager
        self.max_concurrency = max_concurrency

    async def generate(
        self, profile: DatabaseProfile, describe: Any, only: list[str] | None = None,
        joins: str = "",
    ) -> DatabaseKnowledge:
        """Generate the knowledge base: one LLM call per table, then one for the database.

        Args:
            profile: The database profile.
            describe: ``describe(table, column)`` supplying each column's cached description.
            only: Table names to restrict the work to, or None for every table.
            joins: The declared foreign keys, one ``a.b -> c.d`` per line, for the pass that
                relates two tables.

        Returns:
            The grounded, linked knowledge base; ids and child links are assigned after
            every call has answered.
        """
        tables = [tp for name, tp in profile.tables.items() if only is None or name in only]
        columns = {t: {c.lower() for c in tp.columns} for t, tp in profile.tables.items()}
        sem = asyncio.Semaphore(self.max_concurrency)

        async def run(tp: TableProfile) -> tuple[tuple[str, ...], list[dict[str, Any]]]:
            """Generate one table's entries under the concurrency limit.

            Args:
                tp: The table profile.

            Returns:
                ``((table name,), raw entry rows)``.
            """
            async with sem:
                return (tp.table_name,), await self._generate_table(tp, describe)

        raw: list[tuple[tuple[str, ...], dict[str, Any]]] = []
        for owned, rows in await asyncio.gather(*(run(tp) for tp in tables)):
            raw += [(owned, {**row, "knowledge": name}) for row in rows
                    if (name := str(row.get("knowledge", "")).strip())]

        # One further call per database reaches what a per-table call cannot: a relation over
        # two tables, and a term defined over the terms the first pass named.
        if len(columns) > 1:
            linking = await self._generate_links(
                profile, describe, joins, [row["knowledge"] for _o, row in raw])
            raw += [(owners(str(row.get("definition", "")), columns), {**row, "knowledge": name})
                    for row in linking if (name := str(row.get("knowledge", "")).strip())]

        by_name = {row["knowledge"].lower(): i for i, (_o, row) in enumerate(raw)}
        dk = DatabaseKnowledge()
        for entry_id, (owned, row) in enumerate(raw):
            children = [i for n in row.get("children", [])
                        if (i := _resolve(str(n), by_name)) is not None and i != entry_id]
            dk.entries[entry_id] = KnowledgeEntry(
                id=entry_id,
                knowledge=row["knowledge"],
                description=str(row.get("description", "")).strip(),
                definition=str(row.get("definition", "")).strip(),
                type=str(row.get("type", "domain_knowledge")).strip(),
                children_knowledge=children or -1,
                tables=owned,
            )
        dk = dk.linked()
        logger.info("Knowledge generation complete: %d entries, %d child link(s)",
                    len(dk.entries), sum(len(e.children) for e in dk.entries.values()))
        return dk.ground({c for cols in columns.values() for c in cols})

    async def _generate_table(self, table: TableProfile, describe: Any) -> list[dict[str, Any]]:
        """Ask the LLM for relations over one table's columns.

        Args:
            table: The table profile.
            describe: ``describe(table, column)`` supplying each column's description.

        Returns:
            The raw entry rows; an unparseable reply yields none.
        """
        prompt = self.prompt_manager.render(
            "generate_knowledge",
            table_name=table.table_name,
            count=max(_MIN_ENTRIES, min(_MAX_ENTRIES, len(table.columns) // 2)),
            columns=_columns(table, describe),
        )
        return await self._ask(prompt, table.table_name)

    async def _generate_links(self, profile: DatabaseProfile, describe: Any, joins: str,
                              terms: list[str]) -> list[dict[str, Any]]:
        """Ask the LLM for relations that span tables, and for terms built on other terms.

        Args:
            profile: The database profile; every table's columns are offered with their types
                and frequent values, which is what keeps a literal in the reply real.
            describe: ``describe(table, column)`` supplying each column's description.
            joins: The declared foreign keys, one per line.
            terms: The term names the per-table pass produced.

        Returns:
            The raw entry rows; an unparseable reply yields none.
        """
        prompt = self.prompt_manager.render(
            "generate_links",
            tables=[{"name": t, "columns": _columns(tp, describe)}
                    for t, tp in profile.tables.items()],
            joins=joins,
            terms=", ".join(terms),
            count=max(_MIN_ENTRIES, min(_MAX_ENTRIES, len(profile.tables))),
        )
        return await self._ask(prompt, "the database")

    async def _ask(self, prompt: str, what: str) -> list[dict[str, Any]]:
        """Send one knowledge prompt under the shared rules and parse the reply.

        Args:
            prompt: The rendered user message.
            what: What is being described, for the log line.

        Returns:
            The raw entry rows; an unparseable reply yields none.
        """
        raw = await self.llm.chat(prompt, system=self.prompt_manager.render("knowledge_rules"))
        try:
            parsed = parse_llm_json(raw)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("Failed to parse knowledge JSON for %s (%s); skipping", what, e)
            return []
        rows = parsed.get("knowledge", parsed) if isinstance(parsed, dict) else parsed
        return [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []
