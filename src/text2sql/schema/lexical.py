"""Lexical ranking primitives: BM25 with subword matching, and reciprocal-rank fusion.

Shared by schema linking (columns, knowledge entries) and the agent's retrieval tools.
"""

from __future__ import annotations

import math
import re
from collections import Counter

# BM25 term saturation and length normalisation.
BM25_K1 = 1.5
BM25_B = 0.75

# Reciprocal-rank fusion constant.
RRF_K = 60

# A question word this long may match *inside* a document token, at reduced weight.
SUBWORD_MIN = 4
SUBWORD_WEIGHT = 0.5

# Shortest compound run kept whole beside its split parts.
COMPOUND_MIN = 4

# Description levels ranked and fused; each misses what the others catch.
DETAIL_LEVELS = ("name", "short", "long")

type Documents[K] = tuple[dict[K, Counter[str]], dict[str, set[K]]]


def token_list(text: str) -> list[str]:
    """Lowercased word tokens of length >= 3, split on case and ``_``, plus each identifier
    rejoined wherever splitting broke a compound name apart.

    Args:
        text: Arbitrary text to tokenise.

    Returns:
        Tokens in order of appearance, with duplicates kept.
    """
    parts: list[str] = []
    for run in re.split(r"[^a-zA-Z0-9_]+", text):
        split = [p.lower() for p in re.split(r"_+|(?<=[a-z0-9])(?=[A-Z])", run) if len(p) >= 3]
        parts += split
        # Rejoin spans an identifier only, so `chatmsg/sesscount` stays two tokens.
        joined = run.replace("_", "").lower()
        if len(joined) >= COMPOUND_MIN and split != [joined]:
            parts.append(joined)
    return parts


def tokens(text: str) -> set[str]:
    """The distinct tokens of ``text``.

    Args:
        text: Arbitrary text to tokenise.

    Returns:
        Set of distinct tokens.
    """
    return set(token_list(text))


def documents[K](items: dict[K, str]) -> Documents[K]:
    """Build the term counts and inverted token index that :func:`bm25` scores over.

    Args:
        items: Mapping of document key to document text.

    Returns:
        ``({key: term counts}, {token: keys})``.
    """
    docs = {key: Counter(token_list(text)) for key, text in items.items()}
    index: dict[str, set[K]] = {}
    for key, terms in docs.items():
        for token in terms:
            index.setdefault(token, set()).add(key)
    return docs, index


def bm25[K](words: set[str], docs: dict[K, Counter[str]], index: dict[str, set[K]],
            *, subword: bool) -> list[K]:
    """Rank tokenised documents against a set of query words.

    Args:
        words: Query tokens.
        docs: Per-document term counts, from :func:`documents`.
        index: Inverted token index, from :func:`documents`.
        subword: Also score a long query word found inside a document token, at half weight.

    Returns:
        Document keys, best first.
    """
    if not docs or not words:
        return []
    total = len(docs)
    lengths = {key: sum(terms.values()) for key, terms in docs.items()}
    avg_length = sum(lengths.values()) / total
    idf = {token: math.log(1 + (total - len(keys) + 0.5) / (len(keys) + 0.5))
           for token, keys in index.items()}
    scores: dict[K, float] = {}
    for word in words:
        exact: set[K] = index.get(word, set())
        for key in exact:
            freq = docs[key][word]
            scores[key] = scores.get(key, 0.0) + idf[word] * freq * (BM25_K1 + 1) / (
                freq + BM25_K1 * (1 - BM25_B + BM25_B * lengths[key] / avg_length))
        if not subword or len(word) < SUBWORD_MIN:
            continue
        # Scored once per document and only where the word is absent outright, so verbosity
        # is not rewarded.
        bonus = SUBWORD_WEIGHT * idf.get(word, math.log(total))
        near = {key for token, keys in index.items()
                if len(token) >= SUBWORD_MIN and (word in token or token in word)
                for key in keys}
        for key in near - exact:
            scores[key] = scores.get(key, 0.0) + bonus

    # Ties break on the key so the ranking is reproducible across processes.
    return sorted(scores, key=lambda k: (-scores[k], k))


def fuse[K](rankings: list[list[K]]) -> list[K]:
    """Combine rankings by reciprocal-rank fusion.

    Args:
        rankings: One ranking per description level.

    Returns:
        Fused ranking, best first.
    """
    if len(rankings) < 2:
        return rankings[0] if rankings else []
    scores: dict[K, float] = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking):
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
    # Ties break on the key so the ranking is reproducible across processes.
    return sorted(scores, key=lambda k: (-scores[k], k))
