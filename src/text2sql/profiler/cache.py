"""Profile caching: local JSON files or S3."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from text2sql.profiler.stats import SEP, group

logger = logging.getLogger(__name__)

CACHE_VERSION = 1  # Bump to invalidate all cached artifacts on format changes.

# Filename suffixes, one artifact each. ``meaning_base_*`` matches the shipped
# ``data/<db>/<db>_column_meaning_base.json`` format so the two are interchangeable.
KINDS = ("profile", "meaning_base_short", "meaning_base_long", "kb")


class ProfileCache:
    """Cache a database's profiling artifacts to local disk or S3.

    One versioned JSON file per ``(database, kind)``, each a flat ``db|table|column`` map.
    Saves *upsert* by key, so partial/selective profiling runs accumulate into a single
    growing document rather than fragmenting into per-selection files.
    """

    def __init__(self, cache_dir: str) -> None:
        self._is_s3 = cache_dir.startswith("s3://")
        if self._is_s3:
            path = cache_dir.removeprefix("s3://")
            parts = path.split("/", 1)
            self._s3_bucket = parts[0]
            self._s3_prefix = parts[1].rstrip("/") if len(parts) > 1 else ""
            self._local_dir = None
        else:
            self._s3_bucket = self._s3_prefix = ""
            self._local_dir = Path(cache_dir)
            self._local_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def cache_key(db_uri: str) -> str:
        """Derive a cache key from a database URI.

        Args:
            db_uri: The database URI.

        Returns:
            The filename stem, with scheme, path and extension stripped.
        """
        name = db_uri.split("://")[-1].split("?")[0].rsplit("/", 1)[-1]
        return name.rsplit(".", 1)[0].replace(" ", "_") or "default"

    def load(self, key: str, kind: str) -> dict[str, Any]:
        """Read one artifact.

        Args:
            key: Cache key.
            kind: One of :data:`KINDS`.

        Returns:
            The document; a missing file or stale version reads as empty.
        """
        doc = self._read(f"{key}_{kind}.json")
        return doc if doc and doc.get("version") == CACHE_VERSION else {}

    def save(self, key: str, kind: str, entries: dict[str, Any],
             extra: dict[str, dict[str, Any]] | None = None, *, replace: bool = False) -> None:
        """Upsert entries by key, stamping ``version`` and per-table ``meta``.

        Args:
            key: Cache key.
            kind: One of :data:`KINDS`.
            entries: The flat ``db|table|column -> value`` map to write.
            extra: Per-table values that have no column key, such as ``row_count``.
            replace: Write ``entries`` as the whole document, for the KB, whose keys are
                LLM-chosen names and would otherwise accumulate duplicates.
        """
        doc, extra = self.load(key, kind), extra or {}
        now = datetime.now(UTC).isoformat()
        columns = entries if replace else {**doc.get("columns", {}), **entries}
        touched = set(group(entries)) | set(extra)  # a table may be profiled with no columns
        meta = {**doc.get("meta", {}),
                **{t: {"profiled_at": now, **extra.get(t, {})} for t in touched}}
        self._write(f"{key}_{kind}.json",
                    {"version": CACHE_VERSION, "profiled_at": now, "columns": columns, "meta": meta})

    def cached_tables(self, key: str) -> dict[str, str]:
        """List the tables already profiled.

        Args:
            key: Cache key.

        Returns:
            ``{table: ISO timestamp}``, which drives the UI badges and ask picker.
        """
        return {t: m.get("profiled_at", "") for t, m in self.load(key, "profile").get("meta", {}).items()}

    def cached_columns(self, key: str) -> dict[str, list[str]]:
        """List the columns already profiled per table.

        Args:
            key: Cache key.

        Returns:
            ``{table: [columns]}``, which drives the Ask-tab picker.
        """
        return {t: list(cols) for t, cols in group(self.load(key, "profile")).items()}

    def delete(self, key: str, table: str, columns: list[str] | None = None) -> None:
        """Remove a table, or some of its columns, from every artifact.

        Args:
            key: Cache key.
            table: Table to remove from.
            columns: Columns to drop, or None to drop the whole table. A table leaves
                ``meta`` once its last column goes.
        """
        targets = tuple(columns or ())

        def keep(cache_key: str) -> bool:
            """Whether one cache key survives the deletion.

            Args:
                cache_key: The flat ``db|table|column`` key; a knowledge entry naming more
                    than one table comma-joins them in the table segment.

            Returns:
                True to keep the entry.
            """
            parts = cache_key.split(SEP)
            if len(parts) < 2 or table not in parts[-2].split(","):
                return True
            # A column takes its dotted JSON fields with it.
            return bool(targets) and not any(
                parts[-1] == c or parts[-1].startswith(f"{c}.") for c in targets)

        for kind in KINDS:
            doc = self.load(key, kind)
            if not doc:
                continue
            doc["columns"] = {k: v for k, v in doc.get("columns", {}).items() if keep(k)}
            if table not in group(doc):
                doc.get("meta", {}).pop(table, None)
            self._write(f"{key}_{kind}.json", doc)

    def _write(self, filename: str, data: Any) -> None:
        """Write one artifact to disk or S3.

        Args:
            filename: Artifact filename.
            data: JSON-serializable document.
        """
        body = json.dumps(data, indent=2, default=str)
        if self._is_s3:
            key = self._s3_key(filename)
            self._s3().put_object(Bucket=self._s3_bucket, Key=key,
                                  Body=body.encode(), ContentType="application/json")
            logger.info("Cached to S3: s3://%s/%s", self._s3_bucket, key)
        else:
            assert self._local_dir is not None
            (self._local_dir / filename).write_text(body, encoding="utf-8")
            logger.info("Cached: %s", self._local_dir / filename)

    def _read(self, filename: str) -> Any | None:
        """Read one artifact from disk or S3.

        Args:
            filename: Artifact filename.

        Returns:
            The parsed document, or None when it is missing or unreadable.
        """
        if self._is_s3:
            s3, key = self._s3(), self._s3_key(filename)
            try:
                return json.loads(s3.get_object(Bucket=self._s3_bucket, Key=key)["Body"].read())
            except s3.exceptions.NoSuchKey:
                return None
            except Exception as e:
                logger.warning("S3 load failed for %s: %s", key, e)
                return None
        assert self._local_dir is not None
        path = self._local_dir / filename
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def _s3_key(self, filename: str) -> str:
        """Qualify an artifact filename with the configured S3 prefix.

        Args:
            filename: Artifact filename.

        Returns:
            The object key.
        """
        return f"{self._s3_prefix}/{filename}" if self._s3_prefix else filename

    def _s3(self) -> Any:
        """Create an S3 client; imported lazily so boto3 stays an optional dependency.

        Returns:
            The boto3 S3 client.
        """
        import boto3
        return boto3.client("s3")
