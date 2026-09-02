"""Schema loading for prompt construction.

Collects the tables, columns and profile statistics each prompt needs; the layout itself
lives in ``schema.j2``.
"""

from __future__ import annotations

import logging
from typing import Any

from text2sql.db import DatabaseConnection
from text2sql.profiler.stats import DatabaseProfile
from text2sql.profiler.summarizer import DatabaseSummary
from text2sql.prompts.manager import PromptManager

logger = logging.getLogger(__name__)

# A domain small enough to list outright; past it the bounds say more than the members.
_MAX_LISTED = 50


def _clip(value: Any, limit: int = 40) -> str:
    """Flatten a value onto one line, ellipsised past ``limit``, so a JSON blob cannot
    bloat the prompt."""
    text = str(value).replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "…"


def type_text(cp: Any, fallback: str = "") -> str:
    """The column's declared type, marked when its stored values carry surrounding
    whitespace: no fixed-width type shows padding, and an equality join against an unpadded
    key returns nothing. ``fallback`` covers an unprofiled column."""
    if not cp:
        return fallback
    return str(cp.column_type) + (" space-padded" if cp.padded else "")


def _stats(cp: Any) -> list[str]:
    """The column's value statistics, or nothing when unprofiled.

    Returns:
        Either the value domain or its bounds with examples, then cardinality and
        nullability where they are informative.
    """
    if not cp:
        return []
    out = []
    if cp.top_k_values and 0 < cp.distinct_count <= _MAX_LISTED:
        out.append("values: " + ", ".join(f"'{_clip(v['value'])}'" for v in cp.top_k_values[:10]))
    elif cp.min_value is not None and cp.max_value is not None:
        examples = ", ".join(f"'{_clip(v['value'])}'" for v in cp.top_k_values[:3])
        out.append(f"range: {_clip(cp.min_value)}-{_clip(cp.max_value)}"
                   + (f" e.g. {examples}" if examples else ""))
    if cp.distinct_count:
        out.append(f"~{cp.distinct_count} distinct")
    if cp.row_count and cp.null_count / cp.row_count > 0.5:
        out.append("mostly NULL")
    return out


class SchemaLoader:
    """Builds prompt-ready schema representations at varying detail levels."""

    def __init__(
        self,
        db: DatabaseConnection,
        profile: DatabaseProfile | None = None,
        summary: DatabaseSummary | None = None,
        prompts: PromptManager | None = None,
    ) -> None:
        self.db = db
        self.profile = profile
        self.summary = summary
        self.prompts = prompts or PromptManager()
        self._schema = db.get_schema()

    def get_table_names(self) -> list[str]:
        """List every table in the live schema.

        Returns:
            Table names in schema order.
        """
        return list(self._schema["tables"].keys())

    def format_table_list(self, tables: list[str]) -> str:
        """Render the table names plus the join map between them.

        Args:
            tables: Tables to list.

        Returns:
            A ``Tables:`` line, followed by the declared foreign keys when any links two
            listed tables, labelled partial so an undeclared join is not read as impossible.
        """
        listed = set(tables)
        joins = [f"{t}.{fk['column']} -> {fk['referred_table']}.{fk['referred_column']}"
                 for t in tables
                 for fk in self._schema["tables"].get(t, {}).get("foreign_keys", [])
                 if fk["referred_table"] in listed]
        return f"Tables: {', '.join(tables)}" + (
            f"\nDeclared joins (partial; tables also relate through columns not listed here): "
            f"{'; '.join(joins)}" if joins else "")

    def format_schema(
        self,
        tables: list[str] | None = None,
        detail: str = "full",
        fields: dict[str, list[str]] | None = None,
        explorable: bool = False,
    ) -> str:
        """Format the schema for prompt inclusion.

        Args:
            tables: Specific tables to include (None = all).
            detail: Which description each column line carries - ``name`` (none), ``short``,
                ``long``, or ``full`` (both).
            fields: Column-level subset ``{table: [columns]}``, the "focused schema". Narrows
                both tables and columns, and takes precedence over ``tables``.
            explorable: Also name the tables left out, so an agent can recover one the subset
                dropped. Off for single-shot generation, which can only invent columns for them.

        Returns:
            The rendered ``schema`` template.
        """
        target = list(fields) if fields else (
            tables or (list(self.profile.tables) if self.profile else self.get_table_names()))
        rows = [self._table(t, info, detail, set(fields[t]) if fields else None)
                for t in target if (info := self._schema["tables"].get(t))]
        return self.prompts.render(
            "schema", tables=rows,
            unlinked=[t for t in self._schema["tables"] if t not in target] if explorable else [])

    def column_description(self, table: str, col_name: str, detail: str) -> str:
        """The column's LLM summary at the requested detail level.

        Args:
            table: Table name.
            col_name: Column name.
            detail: Description level; ``name`` is the empty description and ``full`` is the
                short and long summaries together.

        Returns:
            The summary text, or ``""`` when unavailable.
        """
        if not self.summary or detail == "name":
            return ""
        if detail == "short":
            return self.summary.get_short(table, col_name)
        if detail == "long":
            return self.summary.get_long(table, col_name)
        return " ".join(dict.fromkeys(d for d in (self.summary.get_short(table, col_name),
                                                  self.summary.get_long(table, col_name)) if d))

    def _table(self, table: str, info: dict[str, Any], detail: str,
               only: set[str] | None) -> dict[str, Any]:
        """One table's template row: name, row count, rendered column lines and foreign keys,
        narrowed to ``only`` when given."""
        # When profiled, the profile's columns are the authoritative subset; a caller's
        # `only` narrows further, and both must admit a column.
        tp = self.profile.tables.get(table) if self.profile else None
        allowed = set(tp.columns) if tp else {str(c["name"]) for c in info["columns"]}
        # FKs stay visible even when `only` excludes them: a focused schema that hides the
        # join map cannot be joined.
        fk_allowed = set(allowed) if tp else None
        if only is not None:
            allowed &= only

        # Profiled JSON paths nest under the column they were read from.
        children: dict[str, list[str]] = {}
        for c in tp.columns if tp else ():
            if "." in c:
                children.setdefault(c.split(".", 1)[0], []).append(c)

        lines = []
        for col in info["columns"]:
            name = str(col["name"])
            # A JSON parent brings every leaf: the blob itself answers nothing. A linked leaf
            # brings its parent back, or it would render under nothing.
            wanted = name in allowed
            kids = [f for f in children.get(name, ()) if wanted or f in allowed]
            if not wanted and not kids:
                continue
            lines.append(self._column(table, name, col, tp, detail, ""))
            lines += [self._column(table, f, {"type": "JSON field"}, tp, detail, "  ")
                      for f in kids]
        return {
            "name": table,
            "row_count": tp.row_count if tp else None,
            "columns": lines,
            "fks": [fk for fk in info.get("foreign_keys", [])
                    if fk_allowed is None or fk["column"] in fk_allowed],
        }

    def _column(self, table: str, name: str, col: dict[str, Any], tp: Any, detail: str,
                indent: str) -> str:
        """Render ``table.column (type flags): description; stats``, each part dropped when
        empty. ``name`` is dotted and ``indent`` set for a profiled JSON path."""
        cp = tp.columns.get(name) if tp else None
        flags = (" [PK]" if col.get("primary_key") else "") + (
            "" if col.get("nullable", True) else " NOT NULL")
        tail = "; ".join(p for p in (self.column_description(table, name, detail),
                                     *_stats(cp)) if p)
        return (f"{indent}  {table}.{name} ({type_text(cp, str(col['type']))}{flags})"
                + (f": {tail}" if tail else ""))
