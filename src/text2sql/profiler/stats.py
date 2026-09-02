"""Per-column statistics profiler: counts, min/max, top-k, and value-shape analysis."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from typing import Any

from text2sql.db import DatabaseConnection

logger = logging.getLogger(__name__)

SEP = "|"
# Max dotted paths harvested from one JSON column.
_MAX_JSON_FIELDS = 50


def flat(prefix: str, table: str, column: str) -> str:
    """Build a ``db|table|column`` cache key.

    Args:
        prefix: Database prefix.
        table: Table name.
        column: Column name.

    Returns:
        The flat key.
    """
    return f"{prefix}{SEP}{table}{SEP}{column}"


def dotted(obj: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested dicts to dotted paths.

    Args:
        obj: The mapping to flatten.
        prefix: Path prefix for the recursion.

    Returns:
        ``{dotted path: leaf}``, leaves kept as-is.
    """
    out: dict[str, Any] = {}
    for k, v in obj.items():
        path = f"{prefix}{k}"
        out.update(dotted(v, f"{path}.") if isinstance(v, dict) else {path: v})
    return out


def entries(doc: dict[str, Any]) -> dict[str, Any]:
    """Unwrap a cache envelope.

    Args:
        doc: A cache envelope or a bare flat map.

    Returns:
        The flat ``db|table|column -> value`` map.
    """
    flat_map: dict[str, Any] = doc.get("columns", doc)
    return flat_map


def group(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Group any flat-keyed document into ``{table: {column: value}}``.

    The single reader of the key format, so every source interoperates: a cache envelope or
    a bare map, any ``db`` prefix, and the shipped ``{"column_meaning", "fields_meaning"}``
    form, which expands into dotted sibling columns.

    Args:
        doc: The document to group.

    Returns:
        ``{table: {column: value}}``.
    """
    out: dict[str, dict[str, Any]] = {}
    for key, value in entries(doc).items():
        parts = key.split(SEP)
        if len(parts) < 2:
            continue
        cols = out.setdefault(parts[-2], {})
        if isinstance(value, dict) and "fields_meaning" in value:
            cols[parts[-1]] = value.get("column_meaning", "")
            cols.update({f"{parts[-1]}.{p}": v for p, v in dotted(value["fields_meaning"]).items()})
        else:
            cols[parts[-1]] = value
    return out


class DictSerde:
    """Dict (de)serialization mixin; containers with nested dataclasses override
    ``from_dict`` to rebuild their children."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize the dataclass.

        Returns:
            A nested dict of its fields.
        """
        return asdict(self)  # type: ignore[call-overload, no-any-return]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Any:
        """Rebuild the dataclass from a dict.

        Args:
            data: Field values; unknown keys are ignored.

        Returns:
            The instance.
        """
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})  # type: ignore[attr-defined]


@dataclass
class ColumnProfile(DictSerde):
    """Profile statistics for a single column."""

    table_name: str
    column_name: str
    column_type: str
    row_count: int = 0
    null_count: int = 0
    non_null_count: int = 0
    distinct_count: int = 0
    min_value: str | None = None
    max_value: str | None = None
    top_k_values: list[dict[str, Any]] = field(default_factory=list)
    # Value shape analysis (from the sample)
    min_length: int | None = None
    max_length: int | None = None
    is_always_numeric: bool | None = None
    is_constant_length: bool | None = None
    common_patterns: list[str] = field(default_factory=list)

    @property
    def padded(self) -> bool:
        """Both stored bounds carry surrounding whitespace, so equality against an unpadded
        literal finds nothing and no declared type shows it."""
        ends = [str(v) for v in (self.min_value, self.max_value) if v is not None]
        return bool(ends) and all(v != v.strip() for v in ends)

    def to_english(self) -> str:
        """Render the profile as the English blob fed to the LLM summarizer.

        Returns:
            One sentence per statistic that is populated.
        """
        parts = [
            f"Column {self.column_name} (type: {self.column_type}) "
            f"has {self.null_count} NULL values out of {self.row_count} records.",
        ]

        if self.distinct_count > 0:
            parts.append(f"There are {self.distinct_count} distinct values.")

        if self.min_value is not None:
            parts.append(f"The minimum value is '{self.min_value}' and the maximum value is '{self.max_value}'.")

        if self.top_k_values:
            vals = ", ".join(f"'{v['value']}' ({v['count']})" for v in self.top_k_values[:10])
            parts.append(f"Most common non-NULL column values are: {vals}.")

        if self.is_constant_length and self.min_length is not None:
            parts.append(f"The values are always {self.min_length} characters long.")
        elif self.min_length is not None and self.max_length is not None:
            parts.append(f"Value lengths range from {self.min_length} to {self.max_length} characters.")

        if self.is_always_numeric:
            parts.append("Every column value looks like a number.")

        if self.common_patterns:
            patterns_str = ", ".join(self.common_patterns[:5])
            parts.append(f"Common value patterns: {patterns_str}.")

        return " ".join(parts)


@dataclass
class TableProfile(DictSerde):
    """Profile for an entire table."""

    table_name: str
    row_count: int = 0
    columns: dict[str, ColumnProfile] = field(default_factory=dict)


@dataclass
class DatabaseProfile(DictSerde):
    """Profile for an entire database."""

    dialect: str = ""
    tables: dict[str, TableProfile] = field(default_factory=dict)

    def to_flat(self, prefix: str) -> dict[str, Any]:
        """Flatten every column profile for caching.

        Args:
            prefix: Database prefix for the cache keys.

        Returns:
            ``{db|table|column: column profile}``.
        """
        return {flat(prefix, t, c): cp.to_dict()
                for t, tp in self.tables.items() for c, cp in tp.columns.items()}

    def table_meta(self) -> dict[str, dict[str, Any]]:
        """Collect the per-table extras that have no column key.

        Returns:
            ``{table: {"row_count": n}}``, cached under the document's ``meta``.
        """
        return {t: {"row_count": tp.row_count} for t, tp in self.tables.items()}

    @classmethod
    def from_flat(cls, doc: dict[str, Any], dialect: str = "") -> DatabaseProfile:
        """Rebuild a profile from a cached flat document.

        Args:
            doc: The cache envelope.
            dialect: Dialect name to stamp on the profile.

        Returns:
            The profile; a table may appear with no columns.
        """
        meta, by_table = doc.get("meta", {}), group(doc)
        dp = cls(dialect=dialect)
        for t in meta.keys() | by_table.keys():  # a table may be profiled with no columns
            dp.tables[t] = TableProfile(
                table_name=t,
                row_count=int(meta.get(t, {}).get("row_count", 0)),
                columns={c: ColumnProfile.from_dict(v) for c, v in by_table.get(t, {}).items()},
            )
        return dp


class StatsProfiler:
    """Collects per-column statistics with two queries per table (aggregate + sample)."""

    # Max columns aggregated in a single SELECT (bounds SQL width on wide tables).
    _AGG_CHUNK = 50

    def __init__(
        self,
        db: DatabaseConnection,
        top_k: int = 10,
        sample_size: int = 10_000,
        *,
        exact_top_k: bool = False,
        approx_distinct: bool = True,
        sample_aggregates: bool = False,
    ) -> None:
        self.db = db
        self.top_k = top_k
        self.sample_size = sample_size
        self.exact_top_k = exact_top_k
        self.approx_distinct = approx_distinct
        self.sample_aggregates = sample_aggregates

    def iter_profile(
        self, selection: dict[str, list[str]] | None = None
    ) -> Iterator[tuple[DatabaseProfile, str, int, int]]:
        """Profile one table per step.

        Args:
            selection: ``{table: [columns]}`` to restrict what is profiled, or None for
                everything.

        Yields:
            ``(profile, table, done, total)``. The same ``DatabaseProfile`` is mutated and
            re-yielded each step, so callers can show live per-table progress.
        """
        profile = DatabaseProfile(dialect=self.db.dialect_name)
        table_names = list(selection) if selection else self.db.get_table_names()
        total = len(table_names)

        logger.info("Profiling %d tables in %s database", total, self.db.dialect_name)

        for i, table_name in enumerate(table_names, 1):
            logger.info("Profiling table: %s", table_name)
            try:
                profile.tables[table_name] = self._profile_table(
                    table_name, selection.get(table_name) if selection else None)
            except Exception as e:
                logger.error("Failed to profile table %s: %s", table_name, e)
            yield profile, table_name, i, total

    def profile_database(self, selection: dict[str, list[str]] | None = None) -> DatabaseProfile:
        """Profile all tables, or a chosen subset.

        Args:
            selection: ``{table: [columns]}`` to restrict what is profiled, or None for
                everything.

        Returns:
            The finished profile.
        """
        profile = DatabaseProfile(dialect=self.db.dialect_name)
        for profile, _, _, _ in self.iter_profile(selection):
            pass
        return profile

    def _profile_table(self, table_name: str, columns: list[str] | None = None) -> TableProfile:
        """Profile one table with one aggregate query and one sample query.

        Args:
            table_name: Table to profile.
            columns: Columns to restrict to; an empty list means the whole table, as in
                ``Text2SQL._filter``.

        Returns:
            The table profile, JSON paths included as dotted columns.
        """
        # Column (name, type) pairs, filtered to the requested subset.
        schema = self.db.get_schema()
        col_infos = schema["tables"].get(table_name, {}).get("columns", [])
        if columns:  # an empty list means "the whole table", as in Text2SQL._filter
            wanted = set(columns)
            col_infos = [c for c in col_infos if c["name"] in wanted]
        cols = [(c["name"], c["type"]) for c in col_infos]

        if not cols:
            rows, _ = self.db.execute_safe(f"SELECT COUNT(*) AS n FROM {self.db.quote_identifier(table_name)}")
            n = self._row_get(rows[0], "n") if rows else 0
            return TableProfile(table_name=table_name, row_count=int(n or 0))

        row_count, agg = self._aggregate_stats(table_name, cols)
        samples = self._sample_rows(table_name, cols)

        table_profile = TableProfile(table_name=table_name, row_count=row_count)
        for name, col_type in cols:
            a = agg.get(name, {})
            profile = ColumnProfile(
                table_name=table_name, column_name=name, column_type=col_type, row_count=row_count,
                non_null_count=a.get("nn", 0),
                null_count=max(row_count - a.get("nn", 0), 0),
                distinct_count=a.get("dc", 0),
                min_value=a.get("min"),
                max_value=a.get("max"),
            )
            values = samples.get(name, [])
            if values:
                self._analyze_value_shapes(profile, values)
            profile.top_k_values = (
                self._exact_top_k(table_name, name) if self.exact_top_k
                else self._top_k_from_sample(values, self.top_k)
            )
            table_profile.columns[name] = profile
            for path, vals in self._json_fields(values).items():
                table_profile.columns[f"{name}.{path}"] = self._field_profile(
                    table_name, f"{name}.{path}", row_count, vals)

        return table_profile

    def _json_fields(self, values: list[str]) -> dict[str, list[str]]:
        """Harvest the JSON paths inside a sampled column.

        Args:
            values: The column's sampled values.

        Returns:
            ``{dotted path: values}`` when the sample is all JSON objects, else ``{}``.
        """
        try:
            if not values[:3] or not all(isinstance(json.loads(v), dict) for v in values[:3]):
                return {}
        except (ValueError, TypeError):
            return {}
        fields: dict[str, list[str]] = {}
        for v in values:
            try:
                paths = dotted(json.loads(v))
            except ValueError:
                continue
            for path, leaf in paths.items():
                if leaf is not None and (path in fields or len(fields) < _MAX_JSON_FIELDS):
                    fields.setdefault(path, []).append(str(leaf))
        return fields

    def _field_profile(self, table: str, name: str, row_count: int, values: list[str]) -> ColumnProfile:
        """Profile one JSON path as an ordinary dotted column, from the sample alone.

        Args:
            table: Table name.
            name: Dotted column name.
            row_count: The table's row count.
            values: The path's sampled values.

        Returns:
            The column profile.
        """
        key = ((lambda v: float(v.replace(",", "")))
               if values and all(self._is_numeric(v) for v in values) else None)
        profile = ColumnProfile(
            table_name=table, column_name=name, column_type="JSON field", row_count=row_count,
            non_null_count=len(values), null_count=max(row_count - len(values), 0),
            distinct_count=len(set(values)),
            min_value=min(values, key=key), max_value=max(values, key=key),
        )
        profile.top_k_values = self._top_k_from_sample(values, self.top_k)
        self._analyze_value_shapes(profile, values)
        return profile

    def _aggregate_stats(
        self, table_name: str, cols: list[tuple[str, str]]
    ) -> tuple[int, dict[str, dict[str, Any]]]:
        """Aggregate counts, distincts and min/max for every column.

        Args:
            table_name: Table to query.
            cols: ``(name, type)`` pairs to aggregate.

        Returns:
            ``(row_count, {col: {nn, dc, min, max}})``. Each chunk of columns is one SELECT,
            and a failing chunk retries per-column so one bad column cannot fail the table.
        """
        row_count = 0
        stats: dict[str, dict[str, Any]] = {}
        for start in range(0, len(cols), self._AGG_CHUNK):
            chunk = cols[start : start + self._AGG_CHUNK]
            rc, part = self._run_aggregate(table_name, chunk)
            if part is None:  # chunk failed: degrade to one column per query
                for col in chunk:
                    rc1, part1 = self._run_aggregate(table_name, [col])
                    if part1 is not None:
                        row_count = rc1 or row_count
                        stats.update(part1)
                    else:
                        logger.warning("Failed to profile column %s.%s", table_name, col[0])
            else:
                row_count = rc or row_count
                stats.update(part)
        return row_count, stats

    def _run_aggregate(
        self, table_name: str, cols: list[tuple[str, str]]
    ) -> tuple[int | None, dict[str, dict[str, Any]] | None]:
        """Run one aggregate SELECT over a chunk of columns.

        Args:
            table_name: Table to query.
            cols: ``(name, type)`` pairs to aggregate.

        Returns:
            ``(row_count, stats)``, or ``(None, None)`` when the query fails.
        """
        q_table = self.db.quote_identifier(table_name)
        selects = ["COUNT(*) AS n_rows"]
        for i, (name, _type) in enumerate(cols):
            q = self.db.quote_identifier(name)
            txt = self.db.cast_to_text(q)
            distinct = self.db.approx_count_distinct(q) if self.approx_distinct else f"COUNT(DISTINCT {q})"
            selects += [
                f"COUNT({q}) AS c{i}_nn",
                f"{distinct} AS c{i}_dc",
                f"MIN({txt}) AS c{i}_min",
                f"MAX({txt}) AS c{i}_max",
            ]
        if self.sample_aggregates:
            projection = ", ".join(self.db.quote_identifier(name) for name, _ in cols)
            source = f"(SELECT {projection} FROM {q_table} LIMIT {self.sample_size}) AS __s"
        else:
            source = q_table
        rows, err = self.db.execute_safe(f"SELECT {', '.join(selects)} FROM {source}")
        if err or not rows:
            return None, None

        row = rows[0]
        stats: dict[str, dict[str, Any]] = {}
        for i, (name, _type) in enumerate(cols):
            mn = self._row_get(row, f"c{i}_min")
            mx = self._row_get(row, f"c{i}_max")
            stats[name] = {
                "nn": int(self._row_get(row, f"c{i}_nn") or 0),
                "dc": int(self._row_get(row, f"c{i}_dc") or 0),
                "min": str(mn) if mn is not None else None,
                "max": str(mx) if mx is not None else None,
            }
        return int(self._row_get(row, "n_rows") or 0), stats

    def _sample_rows(self, table_name: str, cols: list[tuple[str, str]]) -> dict[str, list[str]]:
        """Sample rows for client-side value-shape analysis.

        Args:
            table_name: Table to query.
            cols: ``(name, type)`` pairs to sample.

        Returns:
            ``{column: [non-null string values]}`` from one ``LIMIT``-n scan.
        """
        q_table = self.db.quote_identifier(table_name)
        selects = [f"{self.db.quote_identifier(name)} AS c{i}" for i, (name, _) in enumerate(cols)]
        rows, err = self.db.execute_safe(f"SELECT {', '.join(selects)} FROM {q_table} LIMIT {self.sample_size}")
        result: dict[str, list[str]] = {name: [] for name, _ in cols}
        if err or not rows:
            return result
        for row in rows:
            for i, (name, _) in enumerate(cols):
                v = self._row_get(row, f"c{i}")
                if v is not None:
                    result[name].append(str(v))
        return result

    def _exact_top_k(self, table_name: str, col_name: str) -> list[dict[str, Any]]:
        """Count the exact top-k values of one column, via its own GROUP BY.

        Args:
            table_name: Table to query.
            col_name: Column to count.

        Returns:
            ``[{"value", "count"}]``, most frequent first.
        """
        q_col = self.db.quote_identifier(col_name)
        q_table = self.db.quote_identifier(table_name)
        sql = (
            f"SELECT {self.db.cast_to_text(q_col)} AS val, COUNT(*) AS cnt "
            f"FROM {q_table} WHERE {q_col} IS NOT NULL "
            f"GROUP BY {q_col} ORDER BY cnt DESC LIMIT {self.top_k}"
        )
        rows, err = self.db.execute_safe(sql)
        if err or not rows:
            return []
        return [{"value": str(self._row_get(r, "val")), "count": self._row_get(r, "cnt")} for r in rows]

    def _top_k_from_sample(self, values: list[str], k: int) -> list[dict[str, Any]]:
        """Approximate the top-k values by counting the shared sample, with no extra query.

        Args:
            values: The column's sampled values.
            k: How many to keep.

        Returns:
            ``[{"value", "count"}]``, most frequent first.
        """
        counts: dict[str, int] = {}
        for v in values:
            counts[v] = counts.get(v, 0) + 1
        ordered = sorted(counts.items(), key=lambda x: -x[1])[:k]
        return [{"value": v, "count": c} for v, c in ordered]

    @staticmethod
    def _row_get(row: dict[str, Any], alias: str) -> Any:
        """Read a value by alias, case-insensitively: Snowflake upper-cases aliases and
        Postgres lower-cases them.

        Args:
            row: The result row.
            alias: The alias to read.

        Returns:
            The value, or None when the alias is absent.
        """
        if alias in row:
            return row[alias]
        return next((v for k, v in row.items() if k.lower() == alias.lower()), None)

    def _analyze_value_shapes(self, profile: ColumnProfile, values: list[str]) -> None:
        """Derive lengths, numeric-ness and format patterns from sampled values.

        Args:
            profile: The column profile, mutated in place.
            values: The column's sampled values.
        """
        if not values:
            return
        lengths = [len(v) for v in values]
        profile.min_length = min(lengths)
        profile.max_length = max(lengths)
        profile.is_constant_length = profile.min_length == profile.max_length
        profile.is_always_numeric = all(self._is_numeric(v) for v in values)
        profile.common_patterns = self._detect_patterns(values)

    @staticmethod
    def _is_numeric(value: str) -> bool:
        """Check whether a string value looks like a number.

        Args:
            value: The value to test.

        Returns:
            True when it parses as a float.
        """
        try:
            float(value.replace(",", ""))
            return True
        except (ValueError, AttributeError):
            return False

    @staticmethod
    def _detect_patterns(values: list[str], max_patterns: int = 5) -> list[str]:
        """Bucket values by shape, mapping letters to ``A`` and digits to ``D``.

        Args:
            values: The column's sampled values.
            max_patterns: How many patterns to keep.

        Returns:
            The most common patterns, e.g. ``DDDD-DD-DD`` for ``2024-01-15``.
        """
        counts: dict[str, int] = {}
        for value in values[:1000]:
            pattern = re.sub(r"[0-9]", "D", re.sub(r"[a-zA-Z]", "A", value))
            counts[pattern] = counts.get(pattern, 0) + 1
        return [p for p, _ in sorted(counts.items(), key=lambda x: -x[1])[:max_patterns]]

