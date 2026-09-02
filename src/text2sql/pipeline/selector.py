"""Candidate selection by executing each query and clustering the results.

Modes: ``single`` (first candidate), ``majority`` (largest cluster, random on a tie),
``confidence`` (same, but an LLM adjudicates when every candidate disagrees).

Reference: DeepEye-SQL (Li et al.): N-version generation + confidence-aware selection.
"""

from __future__ import annotations

import hashlib
import logging
import random
from collections import Counter
from collections.abc import Sequence
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from text2sql.db import DatabaseConnection
from text2sql.llm import LLMClient, parse_llm_json
from text2sql.prompts.manager import PromptManager

logger = logging.getLogger(__name__)

# Sentinel cluster key for a result set with no rows.
_EMPTY_RESULTS = "empty"


def _normalize_value(value: Any) -> str:
    """Reduce a cell to a canonical string, so cosmetic differences do not split clusters.

    Args:
        value: The cell value.

    Returns:
        Dates as ``YYYY-MM-DD`` and numerics rounded to 4 places, since float formatting
        and Decimal scale differ harmlessly between equivalent queries.
    """
    if isinstance(value, (date, datetime)):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, (float, Decimal)):
        return f"{float(value):.4f}"
    if value is None:
        return "None"
    return str(value).strip()


class CandidateSelector:
    """Selects the best SQL candidate by executing and clustering the candidates."""

    def __init__(
        self,
        db: DatabaseConnection,
        llm: LLMClient | None = None,
        prompt_manager: PromptManager | None = None,
        mode: str = "majority",
    ) -> None:
        self.db = db
        self.llm = llm
        self.prompt_manager = prompt_manager
        self.mode = mode

    async def select(
        self,
        candidates: list[str],
        question: str = "",
        schema_text: str = "",
    ) -> tuple[str, list[dict[str, Any]] | None, dict[str, Any]]:
        """Execute candidates and select the best one.

        Args:
            candidates: List of candidate SQL queries.
            question: Original question (needed for LLM adjudication).
            schema_text: Schema text (needed for LLM adjudication).

        Returns:
            ``(selected SQL, results or None, selection metadata)``.
        """
        if not candidates:
            return "", None, {"error": "No candidates provided"}

        if self.mode == "single" or len(candidates) == 1:
            results, error = self.db.execute_safe(candidates[0])
            return candidates[0], results, {"method": "single_candidate", "error": error}

        # Execute all candidates
        executions = [(sql, *self.db.execute_safe(sql)) for sql in candidates]
        successful = [(sql, res) for sql, res, err in executions if res is not None]

        if not successful:
            return candidates[0], None, {"method": "all_failed", "errors": [e for _, _, e in executions]}
        if len(successful) == 1:
            return successful[0][0], successful[0][1], {"method": "single_success"}

        # Cluster by result-set hash. On a tie prefer a cluster with rows: every empty
        # result lands in one bucket, so an over-filtering cluster would win ties by accident.
        result_hashes = [self._hash_results(res) for _, res in successful]
        hash_counts = Counter(result_hashes)
        most_common_hash, count = max(
            hash_counts.items(), key=lambda kv: (kv[1], kv[0] != _EMPTY_RESULTS),
        )
        confidence = count / len(successful)

        # Counts, not thresholds: at the 3-candidate ceiling the only distinctions available
        # are unanimous, a plurality, and no agreement at all.
        if count >= 2:
            return self._pick(
                most_common_hash, result_hashes, successful,
                method="unanimous" if count == len(successful) else "majority_vote",
                agreement=count, total=len(successful), confidence=round(confidence, 3),
                **({} if count == len(successful) else {"uncertain": self.mode == "confidence"}),
            )
        if self.mode == "confidence" and self.llm and self.prompt_manager and question:
            return await self._llm_adjudicate(successful, question, schema_text)
        sql, results = random.choice(successful)
        return sql, results, {"method": "random_selection", "reason": "no_agreement",
                              "total": len(successful), "confidence": round(confidence, 3)}

    @staticmethod
    def _pick(
        most_common_hash: str,
        result_hashes: list[str],
        successful: Sequence[tuple[str, list[dict[str, Any]] | None]],
        **meta: Any,
    ) -> tuple[str, list[dict[str, Any]] | None, dict[str, Any]]:
        """Return the first successful candidate whose result hash matches.

        Args:
            most_common_hash: The winning cluster's hash, drawn from ``result_hashes`` by
                the caller, so a match is guaranteed by construction.
            result_hashes: One hash per successful candidate, in order.
            successful: The ``(sql, results)`` pairs that executed.
            **meta: Selection metadata returned alongside the pick.

        Returns:
            ``(selected SQL, results, metadata)``.

        Raises:
            AssertionError: The hash was not drawn from ``result_hashes``.
        """
        for i, h in enumerate(result_hashes):
            if h == most_common_hash:
                return successful[i][0], successful[i][1], meta
        raise AssertionError("most_common_hash must exist in result_hashes")

    @staticmethod
    def _hash_results(results: list[dict[str, Any]] | None) -> str:
        """Hash a result set for comparison, ignoring column names and row order.

        Mirrors the benchmark's BIRD-EX comparison: values in SELECT order, as an unordered
        set of rows. Aliases are excluded so two candidates differing only in ``AS total``
        against ``AS cnt`` still form a majority.

        Args:
            results: The rows the candidate returned.

        Returns:
            A stable digest, so recorded traces are comparable across processes.
        """
        if not results:
            return _EMPTY_RESULTS
        rows = frozenset(
            tuple(_normalize_value(v) for v in row.values()) for row in results
        )
        return hashlib.sha1(  # noqa: S324 - comparison key, not security
            "\x1e".join(sorted("\x1f".join(r) for r in rows)).encode()
        ).hexdigest()

    async def _llm_adjudicate(
        self,
        candidates: Sequence[tuple[str, list[dict[str, Any]] | None]],
        question: str,
        schema_text: str,
    ) -> tuple[str, list[dict[str, Any]] | None, dict[str, Any]]:
        """Ask the LLM to adjudicate between candidates that all disagree.

        Args:
            candidates: The ``(sql, results)`` pairs that executed.
            question: The user's question.
            schema_text: Schema for the prompt.

        Returns:
            ``(selected SQL, results, metadata)``; falls back to the first candidate when
            the call or its reply fails.
        """
        # Untruncated: the template shows the first five and reports the true total.
        candidate_data = [{"sql": sql, "results": res, "error": None} for sql, res in candidates]
        prompt = self.prompt_manager.render(
            "select_candidate", question=question, schema=schema_text, candidates=candidate_data,
        )
        try:
            response = await self.llm.chat(prompt)
            parsed = parse_llm_json(response)
            idx = parsed.get("selected", 1) - 1
            if 0 <= idx < len(candidates):
                return candidates[idx][0], candidates[idx][1], {
                    "method": "llm_adjudication", "selected_index": idx + 1,
                    "reasoning": parsed.get("reasoning", ""), "total": len(candidates),
                }
        except Exception as e:
            logger.warning("LLM adjudication failed: %s", e)

        return candidates[0][0], candidates[0][1], {"method": "adjudication_fallback", "total": len(candidates)}
