"""MinHash LSH over shingled field values: "which fields contain this literal?"."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from datasketch import MinHash, MinHashLSH

from text2sql.profiler.stats import DatabaseProfile

logger = logging.getLogger(__name__)

NUM_PERM = 64
SHINGLE = 3
# Minimum MinHash resemblance for a question literal to match a stored value.
THRESHOLD = 0.5

# A literal matching more than this share of indexed fields is a stopword, not a value:
# "10" from "top 10 customers" matched 17 of credit's 120 fields, every one a count column.
MAX_FIELD_RATIO = 0.10


def shingles(text: str, size: int = SHINGLE) -> list[bytes]:
    """Shingle a value into character n-grams.

    Args:
        text: The value to shingle.
        size: N-gram length.

    Returns:
        Lowercased n-grams, or the whole string when shorter than ``size``.
    """
    t = " ".join(text.lower().split())
    if not t:
        return []
    if len(t) <= size:
        return [t.encode()]
    return [g.encode() for g in {t[i:i + size] for i in range(len(t) - size + 1)}]


def sketch(text: str, size: int = SHINGLE, num_perm: int = NUM_PERM) -> MinHash | None:
    """Build the MinHash of a value's shingle set.

    Args:
        text: The value to hash.
        size: N-gram length.
        num_perm: MinHash permutations.

    Returns:
        The sketch, or None when there is nothing to hash.
    """
    grams = shingles(text, size)
    if not grams:
        return None
    mh = MinHash(num_perm=num_perm)
    mh.update_batch(grams)
    return mh


@dataclass(frozen=True)
class ValueMatch:
    """A field whose sampled values approximately contain a queried literal.

    Attributes:
        table: Table name.
        column: Column name.
        value: The stored value that matched.
        resemblance: MinHash Jaccard resemblance to the literal.
    """
    table: str
    column: str
    value: str
    resemblance: float

    def __str__(self) -> str:
        """Render the match as one human-readable line."""
        return f"{self.table}.{self.column} ~ {self.value!r} ({self.resemblance:.3f})"


class ValueIndex:
    """Maps a literal to the fields holding it, over the profile's top-k values.

    Usage::

        index = ValueIndex.from_profile(profile)
        index.fields_containing("Acme Corp")   # -> [ValueMatch("customers", "name", ...)]
    """

    def __init__(self, values: dict[str, list[str]], *, threshold: float = THRESHOLD,
                 shingle_size: int = SHINGLE, num_perm: int = NUM_PERM,
                 max_field_ratio: float = MAX_FIELD_RATIO) -> None:
        """Initialize the index.

        Args:
            values: ``{"table.column": [values]}`` to index.
            threshold: Minimum resemblance for a match.
            shingle_size: N-gram length.
            num_perm: MinHash permutations.
            max_field_ratio: Share of all fields above which a literal is treated as a
                stopword and ignored.
        """
        self.values = values
        self.threshold = threshold
        self.shingle_size = shingle_size
        self.num_perm = num_perm
        self.max_field_ratio = max_field_ratio
        self._lsh: MinHashLSH | None = None

    @classmethod
    def from_profile(cls, profile: DatabaseProfile, **kw) -> ValueIndex:
        """Build an index from the values profiling already collected, with no DB access.

        Args:
            profile: The database profile.
            **kw: Passed through to the constructor.

        Returns:
            The index.
        """
        values = {
            f"{table}.{column}": vals
            for table, tp in profile.tables.items()
            for column, cp in tp.columns.items()
            if (vals := [str(v["value"]) for v in cp.top_k_values if v.get("value") is not None])
        }
        logger.info("Value index: %d values across %d field(s)",
                    sum(len(v) for v in values.values()), len(values))
        return cls(values, **kw)

    def fields_containing(self, literal: str) -> list[ValueMatch]:
        """Find the fields holding a value approximately matching a literal.

        Args:
            literal: The literal to look up.

        Returns:
            Matches by resemblance, at most one per field, so a field of near-duplicate
            values cannot crowd out the rest.
        """
        query = self._sketch(literal)
        if query is None:
            return []
        best: dict[str, ValueMatch] = {}
        for hit in self._index().query(query):
            key, _, i = hit.rpartition("|")
            table, _, column = key.partition(".")
            value = self.values[key][int(i)]
            if (mh := self._sketch(value)) is None:
                continue
            score = query.jaccard(mh)
            if score >= self.threshold and score > getattr(best.get(key), "resemblance", 0.0):
                best[key] = ValueMatch(table, column, value, score)
        return sorted(best.values(), key=lambda m: -m.resemblance)

    def fields_for(self, literals: list[str]) -> dict[str, set[str]]:
        """Collect every field matching any of a question's literals.

        Args:
            literals: The literals to look up.

        Returns:
            ``{table: {columns}}``. A literal matching more than ``max_field_ratio`` of all
            fields is skipped: it describes the database rather than the question.
        """
        cap = max(1, math.floor(len(self.values) * self.max_field_ratio))
        fields: dict[str, set[str]] = {}
        for literal in literals:
            matches = self.fields_containing(literal)
            if len(matches) > cap:
                logger.debug("Value index: ignoring %r, matches %d/%d fields",
                             literal, len(matches), len(self.values))
                continue
            for m in matches:
                fields.setdefault(m.table, set()).add(m.column)
        return fields

    def _sketch(self, text: str) -> MinHash | None:
        """Sketch one value with this index's settings.

        Args:
            text: The value to hash.

        Returns:
            The sketch, or None when there is nothing to hash.
        """
        return sketch(text, self.shingle_size, self.num_perm)

    def _index(self) -> MinHashLSH:
        """Build the LSH on first use, one entry per ``(field, value)``.

        Returns:
            The LSH index.
        """
        if self._lsh is None:
            self._lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
            for key, vals in self.values.items():
                for i, val in enumerate(vals):
                    if (mh := self._sketch(val)) is not None:
                        try:
                            self._lsh.insert(f"{key}|{i}", mh)
                        except ValueError:
                            pass  # duplicate key
        return self._lsh
