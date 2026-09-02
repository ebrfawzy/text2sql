"""LLM summarization of raw profiles into short + long column descriptions."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from text2sql.llm import LLMClient, parse_llm_json
from text2sql.profiler.stats import ColumnProfile, DatabaseProfile, DictSerde, TableProfile, flat, group
from text2sql.prompts.manager import PromptManager

logger = logging.getLogger(__name__)

# Max batches summarized concurrently (bounded so we don't flood the provider).
MAX_CONCURRENCY = 8


@dataclass
class ColumnSummary(DictSerde):
    """LLM-generated summaries for a column."""

    table_name: str
    column_name: str
    short_summary: str = ""
    long_summary: str = ""


@dataclass
class DatabaseSummary(DictSerde):
    """LLM-generated summaries for all columns in a database."""

    columns: dict[str, dict[str, ColumnSummary]] = field(default_factory=dict)

    def describe(self, table: str, col: str, *, long: bool = True) -> str:
        """Read one column's description.

        Args:
            table: Table name.
            col: Column name.
            long: Prefer the long summary.

        Returns:
            The requested description, falling back to the other kind when only one was
            generated, else ``""``.
        """
        s = self.columns.get(table, {}).get(col)
        if not s:
            return ""
        return (s.long_summary or s.short_summary) if long else (s.short_summary or s.long_summary)

    def get_short(self, table: str, col: str) -> str:
        """The column's short description.

        Args:
            table: Table name.
            col: Column name.

        Returns:
            The short summary, or the long one when only that exists.
        """
        return self.describe(table, col, long=False)

    def get_long(self, table: str, col: str) -> str:
        """The column's long description.

        Args:
            table: Table name.
            col: Column name.

        Returns:
            The long summary, or the short one when only that exists.
        """
        return self.describe(table, col)

    def to_flat(self, prefix: str, *, long: bool) -> dict[str, Any]:
        """Flatten one meaning base for caching.

        Args:
            prefix: Database prefix for the cache keys.
            long: Write the long summaries rather than the short ones.

        Returns:
            ``{db|table|column: summary}``.
        """
        return {flat(prefix, t, c): (s.long_summary if long else s.short_summary)
                for t, cols in self.columns.items() for c, s in cols.items()}

    @classmethod
    def from_flat(cls, short: dict[str, Any], long: dict[str, Any]) -> DatabaseSummary:
        """Rebuild the summaries from the two cached meaning bases.

        Args:
            short: The short meaning base; may be empty.
            long: The long meaning base; may be empty.

        Returns:
            The summaries.
        """
        ds, s, ln = cls(), group(short), group(long)
        for t in s.keys() | ln.keys():
            ds.columns[t] = {
                c: ColumnSummary(t, c, str(s.get(t, {}).get(c, "")), str(ln.get(t, {}).get(c, "")))
                for c in s.get(t, {}).keys() | ln.get(t, {}).keys()
            }
        return ds


class ProfileSummarizer:
    """Uses an LLM to generate human-readable summaries from raw profiles."""

    def __init__(
        self,
        llm: LLMClient,
        prompt_manager: PromptManager,
        *,
        one_call_per_table: bool = True,
        max_concurrency: int = MAX_CONCURRENCY,
    ) -> None:
        self.llm = llm
        self.prompt_manager = prompt_manager
        self.one_call_per_table = one_call_per_table
        self.max_concurrency = max_concurrency

    async def summarize_database(
        self,
        profile: DatabaseProfile,
        *,
        generate_short: bool = True,
        generate_long: bool = True,
        only: dict[str, list[str]] | None = None,
    ) -> DatabaseSummary:
        """Summarize the profile's columns concurrently.

        Args:
            profile: The database profile.
            generate_short: Produce short summaries.
            generate_long: Produce long summaries.
            only: ``{table: [columns]}`` to restrict the work while still using each full
                table for sibling context, so re-profiling a few columns need not
                re-summarize a whole cached table. None summarizes everything.

        Returns:
            The summaries.
        """
        # Batch work-list: (table_profile, [target ColumnProfiles]).
        batches: list[tuple[TableProfile, list[ColumnProfile]]] = []
        for table_name, tp in profile.tables.items():
            targets = None if only is None else only.get(table_name)
            if only is not None and not targets:
                continue
            cols = [c for name, c in tp.columns.items() if targets is None or name in targets]
            if not cols:
                continue
            if self.one_call_per_table:
                batches.append((tp, cols))
            else:
                batches.extend((tp, [c]) for c in cols)

        sem = asyncio.Semaphore(self.max_concurrency)

        async def run(tp: TableProfile, cols: list[ColumnProfile]) -> tuple[str, dict[str, ColumnSummary]]:
            """Summarize one batch under the concurrency limit.

            Args:
                tp: The batch's table profile.
                cols: The columns to summarize.

            Returns:
                ``(table name, {column: summary})``.
            """
            async with sem:
                return tp.table_name, await self._summarize_batch(
                    tp, cols, generate_short=generate_short, generate_long=generate_long)

        db_summary = DatabaseSummary()
        for table_name, col_summaries in await asyncio.gather(*(run(tp, cols) for tp, cols in batches)):
            db_summary.columns.setdefault(table_name, {}).update(col_summaries)
        logger.info("Summarization complete. LLM usage: %s", self.llm.usage.summary())
        return db_summary

    async def _summarize_batch(
        self, table: TableProfile, cols: list[ColumnProfile], *, generate_short: bool, generate_long: bool
    ) -> dict[str, ColumnSummary]:
        """Summarize one batch of a table's columns in a single LLM call.

        Args:
            table: The table profile, whose column names give sibling context.
            cols: The columns to summarize.
            generate_short: Produce short summaries.
            generate_long: Produce long summaries.

        Returns:
            ``{column: summary}``; an unparseable reply yields empty summaries.
        """
        column_blocks = [
            {
                "column_name": c.column_name,
                "column_type": c.column_type,
                "profile_english": c.to_english(),
            }
            for c in cols
        ]
        logger.info("Summarizing %s: %s", table.table_name, ", ".join(c.column_name for c in cols))

        prompt = self.prompt_manager.render(
            "summarize_columns",
            table_name=table.table_name,
            other_columns=", ".join(table.columns.keys()),
            columns=column_blocks,
        )
        raw = await self.llm.chat(prompt, system=self.prompt_manager.render(
            "summarize_rules", generate_short=generate_short, generate_long=generate_long))

        try:
            parsed = parse_llm_json(raw)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("Failed to parse summary JSON for %s (%s); using empty summaries", table.table_name, e)
            parsed = {}

        result: dict[str, ColumnSummary] = {}
        for c in cols:
            entry = parsed.get(c.column_name, {}) if isinstance(parsed, dict) else {}
            result[c.column_name] = ColumnSummary(
                table_name=table.table_name,
                column_name=c.column_name,
                short_summary=str(entry.get("short", "")).strip() if generate_short else "",
                long_summary=str(entry.get("long", "")).strip() if generate_long else "",
            )
        return result
