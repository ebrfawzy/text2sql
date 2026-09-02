"""Multi-pass schema linking over the direct, reversed and value modes."""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from collections.abc import AsyncIterator, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import sqlglot

from text2sql.llm import LLMClient, parse_llm_json, retry_with_backoff
from text2sql.pipeline.events import EventEmitter, PipelineEvent, Stage, Status
from text2sql.profiler.knowledge import KnowledgeEntry
from text2sql.profiler.minhash import ValueIndex
from text2sql.prompts.manager import PromptManager
from text2sql.schema import lexical
from text2sql.schema.loader import SchemaLoader

logger = logging.getLogger(__name__)

# Description levels ranked and fused.
_LEXICAL_DETAIL = lexical.DETAIL_LEVELS

# The candidate budget as a share of the schema, so it tracks schema width; `top_k` is
# only the floor under it.
_TOP_K_RATIO = 0.45

# Shortest bare integer probed against the value index: "top 10" is a LIMIT, but 10 is a
# stored value in every count column.
_MIN_LITERAL_DIGITS = 4

# Regex fallback for unparseable SQL: CTE names, FROM/JOIN tables, their aliases, `t.col`.
_CTE_RE = re.compile(r"(?:WITH|,)\s*([A-Za-z_]\w*)\s+AS\s*\(", re.I)
_FROM_RE = re.compile(r"\b(?:FROM|JOIN)\s+([A-Za-z_]\w*)", re.I)
_ALIAS_RE = re.compile(r"\b(?:FROM|JOIN)\s+([A-Za-z_]\w*)(?:\s+AS)?\s+([A-Za-z_]\w*)", re.I)
_REF_RE = re.compile(r"\b([A-Za-z_]\w*)\.([A-Za-z_]\w*)")
_SQL_WORDS = frozenset(
    "as on and or where group order having limit join left right inner outer full cross "
    "union select using set values".split())

# Tokenised column documents and the token -> fields index built over them.
_Documents = tuple[dict[tuple[str, str], "Counter[str]"], dict[str, set[tuple[str, str]]]]


@dataclass(frozen=True)
class SchemaIndex:
    """The schema-derived maps the candidate set ranks and expands against, built in one walk.

    Attributes:
        keys: ``{table: (key fields,)}`` - primary keys then foreign keys, deduplicated.
        by_name: ``{column name: (fields carrying it,)}``, keyed both as written and with
            the separators removed, which is the form a question word is tokenised to.
        joins: ``{table: ((own column, other table, other column),)}`` per foreign key,
            recorded from both ends so one hop is a lookup either way.
    """
    keys: dict[str, tuple[tuple[str, str], ...]]
    by_name: dict[str, tuple[tuple[str, str], ...]]
    joins: dict[str, tuple[tuple[str, str, str], ...]]


@dataclass(frozen=True)
class VariantSpec:
    """One linking mode's prompt inputs: which schema, which descriptions, which knowledge.

    Attributes:
        scopes: ``full`` or ``focused`` schema, one pass each.
        descriptions: ``short`` or ``long`` column descriptions, one pass each.
        knowledge: Knowledge level rendered into the prompt - ``off``, ``terms`` or ``full``.
    """
    scopes: tuple[str, ...] = ("full",)
    descriptions: tuple[str, ...] = ("short",)
    knowledge: str = "terms"


class SchemaLinker:
    """Multi-pass schema linker over any combination of the direct, reversed and value modes.

    Usage::

        linker = SchemaLinker(schema_loader, llm, prompt_manager, modes=["reversed", "value"])
        linked = await linker.link("How many orders from Acme Corp?")
        # {"orders": ["id", "user_id", ...], "customers": ["name", ...]}
    """

    _bm25 = staticmethod(lexical.bm25)
    _fuse = staticmethod(lexical.fuse)
    _tokens = staticmethod(lexical.tokens)

    def __init__(
        self,
        schema_loader: SchemaLoader,
        llm: LLMClient,
        prompt_manager: PromptManager,
        modes: Sequence[str] | None = None,
        direct: VariantSpec | None = None,
        reversed_: VariantSpec | None = None,
        value_index: ValueIndex | None = None,
        knowledge: list[KnowledgeEntry] | None = None,
        top_k: int = 30,
        event_verbosity: str = "verbose",
    ) -> None:
        """Initialize the schema linker.

        Args:
            schema_loader: Provides schema formatting at various detail levels.
            llm: LLM client for schema linking prompts.
            prompt_manager: Jinja2 template manager.
            modes: Linkers to run and union - any of "direct", "reversed", "value".
            direct: The direct mode's schema scopes, descriptions and knowledge level.
            reversed_: The same three for the reversed mode.
            value_index: LSH over the profile's values, matching a question literal to
                the fields holding it. ``None`` leaves name matching to work alone.
            knowledge: The database's domain terms, injected per each mode's level.
            top_k: Floor on the candidate set the ranking is cut to.
            event_verbosity: Suppresses per-pass progress events below "verbose".
        """
        self.schema_loader = schema_loader
        self.llm = llm
        self.prompt_manager = prompt_manager
        self.knowledge = knowledge or []
        self.modes = list(modes) if modes is not None else ["reversed"]
        self.direct = direct or VariantSpec()
        self.reversed = reversed_ or VariantSpec()
        self.value_index = value_index
        self.top_k = top_k
        self.event_verbosity = event_verbosity
        # Column documents and their token index, per detail level, built on first use.
        self._documents: dict[str, _Documents] = {}
        self._index_: SchemaIndex | None = None

    # ---- Public API -------------------------------------------------------

    async def link(self, question: str) -> dict[str, list[str]]:
        """Find relevant tables and columns for a question.

        Args:
            question: The user's natural language question.

        Returns:
            ``{table: [columns]}``.
        """
        result: dict[str, list[str]] = {}
        async for item in self.link_stream(question):
            if not isinstance(item, PipelineEvent):
                result = item
        return result

    async def link_stream(
        self, question: str, emitter: EventEmitter | None = None
    ) -> AsyncIterator[PipelineEvent | dict[str, list[str]]]:
        """Run every selected mode, yielding per-pass progress events then the linked fields.

        Args:
            question: The user's natural language question.
            emitter: Event emitter to build progress events with.

        Yields:
            :class:`PipelineEvent` per pass, then the linked ``{table: [columns]}`` last.
        """
        # The candidate set is built once here and passed down: `value` mode *is* that set
        # and a `focused` scope trims the schema to it.
        emitter = emitter or EventEmitter()
        candidates = self._candidate_fields(question) if self._needs_candidates() else {}
        focused = {t: sorted(c) for t, c in candidates.items()} or None
        # The candidate set is reported on its own only when an LLM pass narrows it again;
        # as `value` mode's whole output it is reported once, below.
        if focused and self._focused_scope():
            columns = sum(len(c) for c in focused.values())
            logger.info("Candidate fields: %d tables, %d columns (budget=%d, detail=%s)",
                        len(focused), columns, self._budget(), "+".join(_LEXICAL_DETAIL))
            logger.debug("Candidates: %s", focused)
            if event := self._progress(
                    emitter, f"Focused on {columns} candidate columns across {len(focused)} tables",
                    candidates=focused):
                yield event

        all_fields: dict[str, set[str]] = {}
        for mode in self.modes:
            if mode == "direct":
                self._merge(all_fields, await self._link_direct(question, focused))
            elif mode == "reversed":
                self._merge(all_fields, await self._link_reversed(question, focused))
            else:
                self._merge(all_fields, candidates)
            if event := self._progress(
                    emitter, f"{mode} linking: {sum(len(c) for c in all_fields.values())} "
                    f"columns so far", mode=mode):
                yield event

        result = self._drop_unknown(all_fields)
        tables = self.schema_loader._schema["tables"]
        logger.info("Schema linking (%s): %d/%d tables, %d/%d columns",
                    "+".join(self.modes), len(result), len(tables),
                    sum(len(c) for c in result.values()),
                    sum(len(t.get("columns", [])) for t in tables.values()))
        yield result

    @staticmethod
    def extract_fields(sql: str) -> dict[str, list[str]]:
        """Extract the ``{table: [columns]}`` a SQL query references, via sqlglot.

        Resolves aliases, attributes unqualified columns to the sole FROM/JOIN table, and
        excludes CTE names and the statement's own SELECT aliases so neither looks like a
        schema column. Static so the benchmark can reuse it on gold SQL when scoring recall.

        Args:
            sql: The SQL to parse.

        Returns:
            ``{table: [columns]}``; falls back to a regex scan when parsing fails.
        """
        fields: dict[str, set[str]] = {}
        try:
            for statement in sqlglot.parse(sql):
                if statement is None:
                    continue
                cte_names = {c.alias for c in statement.find_all(sqlglot.exp.CTE) if c.alias}
                output_names = {a.alias for a in statement.find_all(sqlglot.exp.Alias) if a.alias}
                alias_map: dict[str, str] = {}
                tables: list[str] = []
                for t in statement.find_all(sqlglot.exp.Table):
                    if not t.name:
                        continue
                    # Map the alias *before* skipping a CTE, else `FROM ranked rq` leaves `rq`
                    # unresolved and `rq.col` is recorded as a table of its own.
                    if t.alias:
                        alias_map[t.alias] = t.name
                    if t.name in cte_names:
                        continue
                    tables.append(t.name)
                    fields.setdefault(t.name, set())  # keep FROM tables even without column refs
                sole_table = tables[0] if len(tables) == 1 else None
                for col in statement.find_all(sqlglot.exp.Column):
                    if not col.table and col.name in output_names:
                        continue  # the query's own output name, not a schema column
                    if col.table:
                        owner = alias_map.get(col.table, col.table)
                    elif sole_table:
                        owner = sole_table  # unqualified column in a single-table query
                    else:
                        continue  # unqualified + multiple tables: can't safely attribute
                    if owner not in cte_names:
                        fields.setdefault(owner, set()).add(col.name)
        except Exception as e:
            logger.debug("sqlglot parse failed, falling back to regex: %s", e)
            return SchemaLinker._extract_fields_regex(sql)
        return {t: sorted(cols) for t, cols in fields.items()}

    @staticmethod
    def _extract_fields_regex(sql: str) -> dict[str, list[str]]:
        """Fallback field extraction for SQL sqlglot cannot parse, mostly truncated replies.

        Args:
            sql: The SQL to scan.

        Returns:
            ``{table: [columns]}`` over real tables only; a qualifier that ties to no
            FROM/JOIN table is dropped.
        """
        cte = {m.group(1).lower() for m in _CTE_RE.finditer(sql)}
        tables = {m.group(1) for m in _FROM_RE.finditer(sql) if m.group(1).lower() not in cte}
        alias_map = {m.group(2): m.group(1) for m in _ALIAS_RE.finditer(sql)
                     if m.group(2).lower() not in _SQL_WORDS and m.group(1) in tables}
        fields: dict[str, set[str]] = {t: set() for t in tables}
        for qualifier, column in _REF_RE.findall(sql):
            if owner := alias_map.get(qualifier, qualifier if qualifier in tables else ""):
                fields[owner].add(column)
        return {t: sorted(cols) for t, cols in fields.items()}

    # ---- Orchestration helpers -------------------------------------------

    def _focused_scope(self) -> bool:
        """Whether an LLM mode renders the candidate set as its schema."""
        return (("direct" in self.modes and "focused" in self.direct.scopes)
                or ("reversed" in self.modes and "focused" in self.reversed.scopes))

    def _needs_candidates(self) -> bool:
        """Whether any selected mode consumes the candidate set.

        Returns:
            True when ``value`` is selected or either LLM mode runs a focused scope.
        """
        return "value" in self.modes or self._focused_scope()

    def _progress(self, emitter: EventEmitter, message: str, **data: Any) -> PipelineEvent | None:
        """Build a linking progress event.

        Args:
            emitter: Event emitter to build with.
            message: Human-readable progress line.
            **data: Extra event payload.

        Returns:
            The event, or None unless the verbosity asks for detail.
        """
        if self.event_verbosity != "verbose":
            return None
        return emitter.emit(Stage.SCHEMA_LINKING, Status.PROGRESS, message, **data)

    def _drop_unknown(self, fields: dict[str, set[str]]) -> dict[str, list[str]]:
        """Drop hallucinated table and column names, keeping everything real.

        Args:
            fields: The union of every mode's proposed ``{table: {columns}}``.

        Returns:
            ``{table: [columns]}`` in the schema's own spelling; matching is
            case-insensitive. An empty result makes ``core.py`` fall back to the full DDL.
        """
        schema_tables = self.schema_loader._schema["tables"]
        table_by_lower = {t.lower(): t for t in schema_tables}
        result: dict[str, list[str]] = {}
        for table, cols in fields.items():
            canonical = table_by_lower.get(table.lower())
            if canonical is None:
                logger.warning("Schema linking: dropping non-existent table '%s'", table)
                continue
            col_by_lower = {
                str(c["name"]).lower(): str(c["name"])
                for c in schema_tables[canonical].get("columns", [])
            }
            # A dotted name is a field inside a JSON column: keep it when the parent is real.
            kept, unknown = set(), set()
            for c in cols:
                parent, dot, path = c.partition(".")
                if (canonical_col := col_by_lower.get(parent.lower())) is not None:
                    kept.add(canonical_col + dot + path)
                else:
                    unknown.add(c)
            if unknown:
                logger.warning("Schema linking: dropping non-existent column(s) on '%s': %s",
                               canonical, ", ".join(sorted(unknown)))
            result.setdefault(canonical, [])
            result[canonical] = sorted(set(result[canonical]) | kept)
        return result

    @staticmethod
    def _merge(target: dict[str, set[str]], source: dict[str, set[str]]) -> None:
        """Merge source fields into target, in place.

        Args:
            target: Mapping mutated to hold the union.
            source: Fields to add.
        """
        for table, cols in source.items():
            if table not in target:
                target[table] = set()
            target[table].update(cols)

    @staticmethod
    def _collect(fields: Iterable[tuple[str, str]]) -> dict[str, set[str]]:
        """Group an iterable of fields by table.

        Args:
            fields: ``(table, column)`` pairs.

        Returns:
            ``{table: {columns}}``.
        """
        collected: dict[str, set[str]] = {}
        for table, column in fields:
            collected.setdefault(table, set()).add(column)
        return collected

    # ---- Variants (shared by direct and reversed) -------------------------

    def _variants(
        self, spec: VariantSpec, focused: dict[str, list[str]] | None
    ) -> Iterator[tuple[str, dict[str, list[str]] | None, str]]:
        """Enumerate a mode's passes as the cartesian product of its two axes.

        Args:
            spec: The mode's scopes, descriptions and knowledge level.
            focused: The candidate fields a ``focused`` scope renders.

        Yields:
            ``(label, fields, detail)``; ``fields=None`` means the full schema.
        """
        for scope in spec.scopes:
            for detail in spec.descriptions:
                yield f"{scope}_{detail}", focused if scope == "focused" else None, detail

    def _knowledge_for(self, level: str) -> dict[str, Any]:
        """Template arguments for a knowledge level.

        Args:
            level: ``off``, ``terms`` or ``full``.

        Returns:
            Template kwargs; ``off`` passes an empty list so the heading renders nothing.
        """
        return {"knowledge": [] if level == "off" else self.knowledge,
                "knowledge_full": level == "full"}

    # ---- Direct linking ---------------------------------------------------

    async def _link_direct(
        self, question: str, focused: dict[str, list[str]] | None = None
    ) -> dict[str, set[str]]:
        """Ask the LLM which tables and columns matter, once per schema variant.

        Args:
            question: The user's natural language question.
            focused: Candidate fields, rendered by a ``focused`` scope.

        Returns:
            The union of every variant's ``{table: {columns}}``; a failed pass is skipped.
        """
        fields: dict[str, set[str]] = {}
        for label, scope, detail in self._variants(self.direct, focused):
            logger.info("Direct linking pass: %s", label)
            prompt = self.prompt_manager.render(
                "schema_link_direct", question=question,
                schema=self.schema_loader.format_schema(
                    fields=scope, detail=detail),
                **self._knowledge_for(self.direct.knowledge))

            async def _attempt(prompt: str = prompt) -> dict[str, set[str]]:
                """Run one linking call and parse its reply.

                Args:
                    prompt: The rendered prompt.

                Returns:
                    ``{table: {columns}}``.
                """
                parsed = parse_llm_json(await self.llm.chat(prompt))
                found: dict[str, set[str]] = {}
                for item in parsed if isinstance(parsed, list) else []:
                    if table := item.get("table", ""):
                        found.setdefault(table, set()).update(item.get("columns", []))
                return found

            try:
                self._merge(fields, await retry_with_backoff(
                    _attempt, max_retries=self.llm.max_retries, idle_ms=self.llm.idle_ms,
                    retry_on=json.JSONDecodeError, label="Direct schema linking"))
            except Exception as e:
                logger.warning("Direct linking pass %s failed: %s", label, e)
        return fields

    # ---- Reversed linking -------------------------------------------------

    async def _link_reversed(
        self, question: str, focused: dict[str, list[str]] | None = None
    ) -> dict[str, set[str]]:
        """Generate SQL from each schema variant and extract the entities it references.

        Args:
            question: The user's natural language question.
            focused: Candidate fields, rendered by a ``focused`` scope.

        Returns:
            The union of every variant's ``{table: {columns}}``; a failed pass is skipped.
        """
        fields: dict[str, set[str]] = {}
        for label, scope, detail in self._variants(self.reversed, focused):
            logger.info("Reversed linking pass: %s", label)
            try:
                sql = await self.llm.chat_for_sql(self.prompt_manager.render(
                    "schema_link_reversed", question=question,
                    schema=self.schema_loader.format_schema(
                        fields=scope, detail=detail),
                    dialect=self.schema_loader.db.dialect_name,
                    **self._knowledge_for(self.reversed.knowledge)))
                self._merge(fields, {t: set(c) for t, c in self.extract_fields(sql).items()})
            except Exception as e:
                logger.warning("Reversed linking pass %s failed: %s", label, e)
        return fields

    # ---- Candidate fields (value mode + the focused schema) ---------------

    def _candidate_fields(self, question: str) -> dict[str, set[str]]:
        """Fields the question names, the keys they join through, and the values it quotes.

        Recall-first by design: it feeds ``value`` mode and the ``focused`` scope, both of
        which hand a smaller schema to a pass that filters it again, so a wrong extra column
        costs little and a missing one costs the query.

        Args:
            question: The user's natural language question.

        Returns:
            ``{table: {columns}}``.
        """
        words = self._tokens(question)
        rankings = [ranked for detail in _LEXICAL_DETAIL
                    if (ranked := self._lexical_ranking(words, detail))]
        fields = self._collect(self._promote_keys(self._fuse(rankings))[:self._budget()])
        if self.value_index is not None:
            self._merge(fields, self.value_index.fields_for(self._question_literals(question)))
        # Admitted *past* the budget: each is near-certain, and neither can be afforded a
        # slot the ranking needs.
        self._merge(fields, self._collect(self._promote_keys(self._named(words))))
        self._merge(fields, self._collect(self._join_closure(fields)))
        return fields

    def _named(self, words: set[str]) -> list[tuple[str, str]]:
        """Columns the question names outright, admitted whatever the ranking made of them.

        Args:
            words: Question tokens.

        Returns:
            ``(table, column)`` pairs whose column name is one of the words.
        """
        by_name = self._index().by_name
        return [field for word in sorted(words) for field in by_name.get(word, ())]

    def _join_closure(self, tables: Iterable[str]) -> Iterator[tuple[str, str]]:
        """Both columns of every foreign key one hop from a candidate table.

        A gold query joins a path, and the bridge table in the middle is named by neither
        end. Only the two columns an edge is declared on are added, never the neighbour's
        other columns.

        Args:
            tables: Candidate table names.

        Yields:
            ``(table, column)`` for each end of each edge.
        """
        joins = self._index().joins
        for table in tables:
            for column, other, other_column in joins.get(table, ()):
                yield table, column
                yield other, other_column

    def _budget(self) -> int:
        """The number of columns the candidate ranking is cut to.

        Returns:
            A share of the schema, so the budget tracks its width, with ``top_k`` as the
            floor under it.
        """
        columns = sum(len(info.get("columns", []))
                      for info in self.schema_loader._schema["tables"].values())
        return max(self.top_k, round(_TOP_K_RATIO * columns))

    def _lexical_ranking(self, words: set[str], detail: str) -> list[tuple[str, str]]:
        """Rank fields by BM25 over their column document at one description level.

        Args:
            words: Question tokens.
            detail: Description level.

        Returns:
            ``(table, column)`` pairs, best first.
        """
        ranked = self._bm25(words, *self._column_documents(detail), subword=True)
        logger.debug("Lexical ranking (%s): %d fields scored; best %s",
                     detail, len(ranked), [f"{t}.{c}" for t, c in ranked[:5]])
        return ranked

    def _column_documents(self, detail: str) -> _Documents:
        """Tokenised column documents for one description level, memoized.

        Args:
            detail: Description level.

        Returns:
            ``({field: term counts}, {token: fields})``.
        """
        if detail not in self._documents:
            self._documents[detail] = lexical.documents({
                (table, str(col["name"])):
                    f"{table} {col['name']} "
                    f"{self.schema_loader.column_description(table, str(col['name']), detail)}"
                for table, info in self.schema_loader._schema["tables"].items()
                for col in info.get("columns", [])
            })
            logger.debug("Column documents (%s): %d fields, %d distinct tokens", detail,
                         len(self._documents[detail][0]), len(self._documents[detail][1]))
        return self._documents[detail]

    def _promote_keys(self, ranked: list[tuple[str, str]]) -> list[tuple[str, str]]:
        """Splice a table's key columns in the moment the ranking first reaches that table.

        43% of the columns a gold query touches are opaque join keys the question never
        mentions. This widens the candidate set only, so no FK expansion reaches the
        generation prompt.

        Args:
            ranked: Fields, best first.

        Returns:
            The same ranking with each table's key fields inserted at its first appearance.
        """
        keys = self._index().keys
        promoted: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for field in ranked:
            for candidate in (field, *keys.get(field[0], ())):
                if candidate not in seen:
                    seen.add(candidate)
                    promoted.append(candidate)
        return promoted

    def _index(self) -> SchemaIndex:
        """The :class:`SchemaIndex`, built on first use.

        Returns:
            The keys, name index and join edges, all three from one walk of the schema.
        """
        if self._index_ is None:
            tables = self.schema_loader._schema["tables"]
            keys: dict[str, tuple[tuple[str, str], ...]] = {}
            by_name: dict[str, list[tuple[str, str]]] = {}
            joins: dict[str, list[tuple[str, str, str]]] = {}
            for table, info in tables.items():
                names = [str(col["name"]) for col in info.get("columns", [])]
                for name in names:
                    for key in {name.lower(), re.sub(r"[^a-z0-9]", "", name.lower())}:
                        by_name.setdefault(key, []).append((table, name))
                named = list(info.get("primary_keys", []))
                for fk in info.get("foreign_keys", []):
                    named.append(fk["column"])
                    other, column = fk.get("referred_table", ""), fk.get("referred_column", "")
                    if other in tables and column:
                        joins.setdefault(table, []).append((fk["column"], other, column))
                        joins.setdefault(other, []).append((column, table, fk["column"]))
                keys[table] = tuple((table, c) for c in dict.fromkeys(named) if c in set(names))
            self._index_ = SchemaIndex(
                keys,
                {n: tuple(dict.fromkeys(f)) for n, f in by_name.items()},
                {t: tuple(dict.fromkeys(e)) for t, e in joins.items()})
        return self._index_

    def _question_literals(self, question: str) -> list[str]:
        """Data values a question quotes: quoted text, dates, and long-enough numbers.

        Capitalized words are deliberately not probed - they are prose and domain-term
        names, and measured as pure false positives.

        Args:
            question: The user's natural language question.

        Returns:
            Literals to probe against the value index.
        """
        candidates: list[str] = re.findall(r"['\"]([^'\"]+)['\"]", question)
        candidates += re.findall(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b", question)
        candidates += [
            n for n in re.findall(r"\b\d+(?:\.\d+)?\b", question)
            if "." in n or len(n) >= _MIN_LITERAL_DIGITS
        ]
        return candidates
