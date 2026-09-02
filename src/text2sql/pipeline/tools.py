"""The SQL agent's tools: string in, string out, so each JSON schema comes from ``params`` alone."""

from __future__ import annotations

import asyncio
import difflib
import logging
import re
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from functools import partial
from typing import Any

from text2sql.db import EXEC_TIMEOUT_S, DatabaseConnection
from text2sql.llm import extract_sql
from text2sql.pipeline.examples import ExampleStore
from text2sql.pipeline.repair import DEFINITIONS, actionable, clean_error, parse_sql, render, run_checkers
from text2sql.profiler.knowledge import DatabaseKnowledge
from text2sql.schema import lexical
from text2sql.schema.loader import SchemaLoader, type_text

logger = logging.getLogger(__name__)

_CREATES_RE = re.compile(r"(?:ADD\s+COLUMN|CREATE\s+(?:TEMP\s+)?(?:TABLE|VIEW|INDEX|TRIGGER))"
                         r"\s+(?:IF\s+NOT\s+EXISTS\s+)?[\"'`\[]?(\w+)", re.IGNORECASE)
_MISSING_RE = re.compile(r"no such (?:column|table):\s*([\w.]+)", re.IGNORECASE)


MAX_SHOWN_ROWS = 10   # rows rendered back to the model
MAX_TOOL_ROWS = MAX_SHOWN_ROWS + 1  # LIMIT injected: one past what renders, so a cap is visible
MAX_CELL = 200       # one JSON blob column can otherwise fill the context
MAX_HITS = 40        # ranked schema hits per search: names only, so ~28 chars each
MAX_TERMS = 3        # knowledge entries per lookup, before their dependencies
MAX_NEAR = 5         # names offered when nothing matched
MAX_VALUES = 8       # values listed for a JSON path
MAX_VALUE_CHARS = 40 # one nested blob would otherwise fill the line
CLEAN = "No static issues found."
REVIEW = "review_sql"
SUBMIT = "submit_sql"


@dataclass(frozen=True)
class Tool:
    """One callable exposed to the model.

    Attributes:
        name: Tool name the model calls.
        description: What the tool does, as the model sees it.
        params: ``{arg name: description}``; all required strings.
        run: The async implementation.
    """

    name: str
    description: str
    params: dict[str, str]
    run: Callable[..., Awaitable[str]]

    def spec(self) -> dict[str, Any]:
        """Build the function schema for this tool.

        Returns:
            The provider-agnostic schema LiteLLM forwards to any model.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {k: {"type": "string", "description": v}
                                   for k, v in self.params.items()},
                    "required": list(self.params),
                },
            },
        }


def _clip(text: str, limit: int = MAX_CELL) -> str:
    """Truncate one rendered cell.

    Args:
        text: Cell text.
        limit: Maximum characters kept.

    Returns:
        The text, ellipsised when longer than ``limit``.
    """
    return text if len(text) <= limit else text[:limit] + "..."


def _rows(results: list[dict[str, Any]], capped: bool = False) -> str:
    """Render query results as a compact table.

    Args:
        results: Result rows.
        capped: Whether an injected LIMIT may have truncated them.

    Returns:
        Header, up to ``MAX_SHOWN_ROWS`` rows, and a count line that says so when the cap
        was hit rather than reporting a total it cannot know.
    """
    if not results:
        return "0 rows."
    head = results[:MAX_SHOWN_ROWS]
    lines = [" | ".join(head[0]),
             *(" | ".join("NULL" if r.get(c) is None else _clip(str(r[c])) for c in head[0])
               for r in head)]
    if capped and len(results) == MAX_TOOL_ROWS:
        lines.append(
            f"{len(head)}+ rows (capped): COUNT(*) for the real total")
    else:
        lines.append(f"{len(results)} rows" + (f", first {len(head)} shown"
                                               if len(results) > len(head) else ""))
    return "\n".join(lines)


def build_tools(
    db: DatabaseConnection,
    loader: SchemaLoader,
    knowledge: DatabaseKnowledge,
    example_store: ExampleStore,
    *,
    tools: str = "retrieval",
    question: str = "",
    linked: dict[str, list[str]] | None = None,
    shown: Iterable[int] = (),
) -> list[Tool]:
    """Build the tool list for one agent run.

    Args:
        db: Database the tools query.
        loader: Schema loader supplying columns, descriptions and templates.
        knowledge: Domain terms ``search_knowledge`` looks up.
        example_store: Scenario store ``lookup_example`` searches.
        tools: ``schema_preloaded`` (execute/review/submit) or ``retrieval``, which
            prepends the retrieval tools in priority order.
        question: The user's question, so the checkers that need it can run.
        linked: The linked ``{table: [columns]}``, ordering ``describe_table``'s output.
        shown: Knowledge entry ids the prompt already inlined, so ``search_columns`` does
            not print them a second time.

    Returns:
        The enabled tools.
    """
    linked = linked or {}
    schema = db.get_schema()["tables"]

    # Objects a write in an earlier call created, then lost to that call's rollback.
    rolled_back: set[str] = set()
    # Ranking documents, per description level, built on first use.
    docs: dict[str, Any] = {}
    # Knowledge entries already shown - the agent prompt's own block, then every tool result.
    told: set[int] = set(shown)
    # Ranked hits already returned, against the term that first produced them.
    searched: dict[str, str] = {}

    def columns() -> dict[str, list[str]]:
        """List the columns of every table.

        Returns:
            The profiled columns when available, which unlike the live schema include JSON
            paths; otherwise the live columns.
        """
        profile = loader.profile
        return {t: list(tp.columns) for t, tp in profile.tables.items()} if profile \
            and profile.tables else db.list_columns()

    def _values(table: str, column: str) -> str:
        """Summarise a profiled JSON path's stored values; ordinary columns keep no stats.

        Args:
            table: Table name.
            column: Column name; only a dotted JSON path is summarised.

        Returns:
            ``values: ...`` for a small domain, ``range: lo-hi`` otherwise, else "".
        """
        tp = loader.profile.tables.get(table) if loader.profile else None
        cp = tp.columns.get(column) if tp and "." in column else None
        if not cp:
            return ""
        if cp.top_k_values and 0 < cp.distinct_count <= MAX_VALUES:
            return "values: " + ", ".join(
                f"'{str(v['value'])[:MAX_VALUE_CHARS]}'" for v in cp.top_k_values[:MAX_VALUES])
        lo, hi = cp.min_value, cp.max_value
        if lo is None or hi is None:
            return ""
        # An array or object leaf has no bounds to report; one value shows its shape.
        if str(lo).lstrip()[:1] in "[{":
            first = cp.top_k_values[0]["value"] if cp.top_k_values else lo
            return f"e.g. '{str(first)[:MAX_VALUE_CHARS]}'"
        return f"range: {str(lo)[:MAX_VALUE_CHARS]}-{str(hi)[:MAX_VALUE_CHARS]}"

    def _rank(term: str) -> list[tuple[str, str]]:
        """Rank fields against a search term.

        Args:
            term: Name or words to match.

        Returns:
            ``(table, column)`` pairs by BM25+RRF, then the shared-prefix hits BM25 cannot
            make (``delinquency`` and ``delinqcount`` meet only on six characters).
        """
        words = lexical.tokens(term)
        for d in lexical.DETAIL_LEVELS:  # over the profiled fields, so JSON paths stay reachable
            if d not in docs:
                docs[d] = lexical.documents({
                    (t, c): f"{t} {c} {loader.column_description(t, c, d)}"
                    for t, cols in columns().items() for c in cols})
        ranked = lexical.fuse([lexical.bm25(words, *docs[d], subword=True)
                               for d in lexical.DETAIL_LEVELS])
        prefix = [(t, c) for t, cols in columns().items() for c in cols
                  if any(w[:6] == x[:6] for w in words if len(w) >= 6
                         for x in lexical.token_list(c) if len(x) >= 6)]
        return list(dict.fromkeys(ranked + prefix))

    def _table(name: str) -> str:
        """Resolve a table name the agent copied out of prose.

        Args:
            name: Name as written, possibly quoted, mis-cased or a prefix.

        Returns:
            The real table name, or ``""`` when nothing resolves unambiguously.
        """
        want = name.strip().strip('"\'`[]').lower()
        names = list(columns())
        if exact := [t for t in names if t.lower() == want]:
            return exact[0]
        near = [t for t in names if want and (
            t.lower().startswith(want) or want in t.lower())]
        return near[0] if len(near) == 1 else ""

    def _terms(term: str) -> list[Any]:
        """Find the knowledge entries a term refers to.

        Args:
            term: Term or words to match.

        Returns:
            Entries whose name matches the term, at most ``MAX_TERMS``. A term the base does
            not name returns nothing, so the caller reports the miss instead of a guess.
        """
        if not (want := term.strip().lower()) or not any(c.isalnum() for c in want):
            return []
        return [e for e in knowledge.entries.values()
                if want in e.knowledge.lower() or e.knowledge.lower() in want][:MAX_TERMS]

    def _render(matched: list[Any], *, new_only: bool = False) -> str:
        """Render knowledge entries and their children, and mark them as shown.

        Args:
            matched: Entries to render.
            new_only: Drop entries already shown. Applied *after* the child expansion,
                which can otherwise re-introduce one (entry 21's child is entry 20).

        Returns:
            The ``knowledge_lookup`` template, or "" when nothing is left to say.
        """
        expanded = knowledge.with_children(matched)
        if new_only:
            expanded = [e for e in expanded if e.id not in told]
        told.update(e.id for e in expanded)
        return loader.prompts.render("knowledge_lookup", entries=expanded) if expanded else ""

    async def execute_sql(sql: str) -> str:
        """Run the statements off the event loop, time-boxed, one result block each.

        Args:
            sql: One or more semicolon-separated statements.

        Returns:
            A rendered result block per statement, or an ``ERROR:`` line.
        """
        parsed = parse_sql(sql, db.sqlglot_dialect)
        # Unparseable reads as a SELECT: the checkers, not this, report syntax. The cap is
        # appended as text, never re-serialized - re-serializing rewrites
        # `json_extract(x, p)` to `x -> p`, which returns quoted JSON.
        select = parsed is None or parsed.key == "select"
        capped = f"{sql.rstrip().rstrip(';').rstrip()}\nLIMIT {MAX_TOOL_ROWS}" \
            if select and parsed is not None and not parsed.args.get("limit") else sql
        try:
            sets, error = await asyncio.wait_for(asyncio.to_thread(
                partial(db.execute_safe, sets=True), capped), EXEC_TIMEOUT_S)
        except TimeoutError:
            return f"ERROR: timed out after {EXEC_TIMEOUT_S}s: narrow it or add LIMIT."
        rolled_back.update(m.group(1).lower() for m in _CREATES_RE.finditer(sql) if m.group(1))
        if error:
            # The driver quotes the whole statement back and the agent already has it.
            message = clean_error(error)
            # The driver's message alone reads as a failed ALTER, and the agent then spends
            # turns re-verifying a write the rollback discarded.
            if (m := _MISSING_RE.search(message)) and m.group(1).split(".")[-1] in rolled_back:
                message += (f" '{m.group(1)}' was rolled back with the call that created it: "
                            "re-run that write and this query in the same call.")
            return f"ERROR: {message}"
        # "0 rows" on rolled-back DDL reads as failure and has pushed the agent to replace a
        # correct CREATE INDEX with a SELECT.
        if not sets and not select:
            return "Statement executed successfully; no rows is expected."
        return "\n\n".join(_rows(rows, capped != sql) for rows in sets) if sets else _rows([])

    async def review_sql(sql: str) -> str:
        """Run the whole checker cascade over the query as it would ship.

        Args:
            sql: The SQL to review.

        Returns:
            Every actionable finding with its fix, plus the rows the query returns.
        """
        sql = extract_sql(sql)  # review the string the agent will submit
        found = actionable(await asyncio.to_thread(run_checkers, sql, db, question))
        logger.info("Review found (%s)", ",".join(
            n for n, _ in found) or "nothing")
        if not found:
            return CLEAN
        rows, _ = await asyncio.to_thread(db.execute_safe, sql)  # cached by the cascade
        return ("\n".join(render(found, fixes=True))
                + f"\nIt returned: {_rows(rows or [])}")

    async def submit_sql(sql: str) -> str:
        """End the run with this SQL as the answer.

        Args:
            sql: The final SQL.

        Returns:
            The normalized SQL.
        """
        return extract_sql(sql)

    async def describe_table(table: str) -> str:
        """Describe one table's shape: the question's columns with types, the rest by name.

        No meanings (``describe_columns`` is the only source) and no FKs, which the prompt's
        own join map already carries.

        Args:
            table: Table name; case and quotes ignored.

        Returns:
            The table line, its typed columns, a ``more:`` line for the remainder, or the
            closest table names when nothing resolves.
        """
        if not (name := _table(table)):
            close = list(dict.fromkeys(t for t, _ in _rank(table)))[
                :5] or list(columns())[:5]
            return f"No such table. Closest: {', '.join(close)}"
        tp = loader.profile.tables.get(name) if loader.profile else None
        cols = columns()[name]
        # `or cols` keeps a linking-off run, and a table linking never named, on the listing.
        typed = [c for c in cols if c in linked.get(name, ())] or cols
        lines = [name + (f" ({tp.row_count} rows)" if tp else "")]
        live = {x["name"]: x["type"] for x in schema.get(name, {}).get("columns", ())}
        for c in typed:
            kind = type_text(tp.columns.get(c) if tp else None, live.get(c, ""))
            lines.append(f"{c}[{kind}]" + (f": {v}" if (v := _values(name, c)) else ""))
        if more := [c for c in cols if c not in typed]:
            lines.append("more: " + ", ".join(more))
        return "\n".join(lines)

    async def describe_columns(fields: str) -> str:
        """Give the long meanings of the named columns.

        Args:
            fields: Comma-separated ``table.column`` names; a JSON path works.

        Returns:
            One line per column carrying both description levels, a JSON parent bringing its
            leaves along, plus any names that did not resolve; the closest fields when none did.
        """
        found, unknown = [], []
        for ref in fields.replace(";", ",").split(","):
            # First dot, not last: a profiled JSON path *is* a dotted column name.
            table, _, column = ref.strip().strip('"\'`[]').partition(".")
            name = _table(table)
            tp = loader.profile.tables.get(name) if loader.profile else None
            if match := [c for c in columns().get(name, ())
                         if c.lower() == column.strip().lower()]:
                found += [f"{name}.{c}"
                          + ("[space-padded]" if tp and getattr(tp.columns.get(c), "padded", False) else "")
                          # Short names the column in words, long gives its shape and stats.
                          + (f": {tail}" if (tail := loader.column_description(name, c, "full")
                                             or _values(name, c)) else "")
                          for c in columns()[name]
                          if c == match[0] or c.startswith(f"{match[0]}.")]
            elif ref.strip():
                unknown.append(ref.strip())
        if not found:
            close = [f"{t}.{c}" for t, c in _rank(fields)[:5]]
            return f"No such column. Closest: {', '.join(close) or 'none'}."
        return "\n".join(dict.fromkeys(found)) \
            + (f"\nUnknown: {', '.join(unknown)}" if unknown else "")

    async def search_columns(term: str) -> str:
        """Search field names by concept or question words.

        Args:
            term: Name or words to match.

        Returns:
            Up to ``MAX_HITS`` ranked ``table.column`` names, plus any unseen term the same
            words define - the definition rides along, since ``search_knowledge`` is asked
            on 0.4% of turns.
        """
        hits = _rank(term)
        lines = [f"{t}.{c}" for t, c in hits[:MAX_HITS]]
        if len(hits) > MAX_HITS:
            lines.append(f"... {len(hits)} matched; add words to narrow.")
        out = "\n".join(lines) or "No matching tables or columns."
        if (first := searched.setdefault(out, term)) != term:
            out = f"Same fields as '{first}'."
        block = _render(_terms(term), new_only=True)
        return f"{out}\n\n{block}" if block else out

    async def search_knowledge(term: str) -> str:
        """Look a domain term up in the knowledge base.

        Args:
            term: Term to define.

        Returns:
            The matched entries and their children, or the ``MAX_NEAR`` closest term names
            when nothing matched - never a definition the ranker doubts.
        """
        if not term.strip() or not any(c.isalnum() for c in term):
            return "Name the term to define."
        if not (matched := _terms(term)):
            close = difflib.get_close_matches(
                term, [e.knowledge for e in knowledge.entries.values()], n=MAX_NEAR, cutoff=0)
            return f"No definition matched '{term}'. Closest terms: {', '.join(close) or 'none'}"
        return _render(matched)

    async def lookup_example(query: str) -> str:
        """Search the scenario store for guidance on a business term.

        Args:
            query: Concept to look up.

        Returns:
            Up to three matching scenarios, separated by rules.
        """
        return "\n\n---\n\n".join(example_store.search(query, top_k=3)) or "No matching scenarios."

    built = [
        Tool("execute_sql",
             "Executes one or more semicolon-separated SQL statements and returns all result "
             "sets. Changes are rolled back. Cost: 1 turn",
             {"sql": "SQL to run."}, execute_sql),
        Tool(REVIEW, "Run static quality checks on your final SQL and report what they find, with "
             "the rows it returns. Cost: 1 turn", {"sql": "The SQL to review."}, review_sql),
        Tool(SUBMIT, "Submit your final SQL answer; nothing after it is used. Cost: 1 turn",
             {"sql": "The final SQL."}, submit_sql),
    ]
    if tools == "retrieval":
        # Retrieval tools first: they are what the agent needs before it can write anything.
        retrieval = [
            Tool("search_columns", "Find table.column names by concept or question words; adds "
                 "any domain term those words define. Cost: 1 turn",
                 {"term": "Name or words to match."}, search_columns),
            Tool("describe_columns", "Full meanings of named columns. Cost: 1 turn",
                 {"fields": "Comma-separated table.column names; a JSON path works."},
                 describe_columns),
            Tool("describe_table", "One table's column names with their types. Cost: 1 turn",
                 {"table": "Table name; case and quotes ignored."}, describe_table),
        ]
        # A question carrying its own definitions has nothing to look up.
        if knowledge.entries and not DEFINITIONS.search(question):
            retrieval.append(Tool("search_knowledge", "Find a domain term's definition. "
                                  "Cost: 1 turn", {"term": "Term to define."}, search_knowledge))
        if example_store.headings:
            retrieval.append(Tool("lookup_example", "Scenario guidance for a business term. "
                                  "Cost: 1 turn", {"query": "Concept to look up."}, lookup_example))
        built = retrieval + built
    return built
