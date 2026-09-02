"""Benchmark outcomes: one result per example, the report over them, and their JSON shape.

`run.json` is written from here and read back here, so the layout lives in one place:
`meta` / `summary` / `breakdown` / `instances`, each instance clustered the way the pipeline
runs. Loading, running and scoring are the package's ``__init__``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from text2sql.db import split_statements

logger = logging.getLogger(__name__)


STAGES = {"profiling": "profiling", "linking": "schema_linking", "generation": "sql_generation",
          "repair": "sql_repair", "selection": "selection"}


def write_predictions(results: list[BenchmarkResult], path: Path,
                      sql_of: Any = None) -> Path:
    """Write the official submission file: each source record verbatim, plus ``pred_sqls``.

    Args:
        results: The results to write.
        path: The output file.
        sql_of: Picks the SQL for one result; defaults to its own ``pred_sqls``.

    Returns:
        The path written. Rows are sorted by instance_id, since the official wrapper zips
        its sorted results against the input positionally.
    """
    with open(path, "w", encoding="utf-8") as f:
        for r in sorted(results, key=lambda r: r.id):
            pred_sqls = split_statements(sql_of(r)) if sql_of else r.pred_sqls
            f.write(json.dumps({**r.record, "pred_sqls": pred_sqls},
                               ensure_ascii=False, default=str) + "\n")
    return path


def _prf(gold: set[str], pred: set[str], total: int = 0) -> dict[str, Any] | None:
    """Score one predicted name set against gold.

    Args:
        gold: The names the gold query references.
        pred: The names linking produced.
        total: How many names the schema holds, so ``kept`` reports the share of it that
            survived. 0 leaves ``kept`` out.

    Returns:
        Precision, recall, F1, kept share, coverage, exact match and both difference lists;
        None when the gold set is empty, since there is nothing to score and set equality
        would call that a perfect match.
    """
    if not gold:
        return None
    hit = len(gold & pred)
    precision = hit / len(pred) if pred else 0.0
    recall = hit / len(gold) if gold else 0.0
    return {
        "gold": len(gold), "linked": len(pred), "hit": hit,
        **({"total": total, "kept": round(len(pred) / total, 4)} if total else {}),
        "precision": round(precision, 4), "recall": round(recall, 4),
        "f1": round(2 * precision * recall / (precision + recall), 4)
        if precision + recall else 0.0,
        # Generation cannot recover a name it never saw, so coverage - every gold name
        # linked - is the recall that decides an instance; the micro average is not.
        "covered": not gold - pred,
        "exact_match": gold == pred,
        "missing": sorted(gold - pred),  # gold names linking dropped: fatal downstream
        "extra": sorted(pred - gold),    # names the gold query does not need: prompt bloat
    }


def linking_outputs(trace: dict[str, Any]) -> dict[str, Any]:
    """The schema-linking step's trace outputs, empty when the stage never ran.

    Args:
        trace: The pipeline trace.

    Returns:
        The step's outputs: ``linked``, ``linked_tables`` and the ``schema`` sizes.
    """
    return next((step.get("outputs") or {} for step in trace.get("steps", [])
                 if step.get("step") == "schema_linking"), {})


def linked_fields(trace: dict[str, Any]) -> dict[str, list[str]] | None:
    """Read the linked fields out of a pipeline trace.

    Args:
        trace: The pipeline trace.

    Returns:
        The ``{table: [columns]}`` linking produced, or None when it never ran.
    """
    outputs = linking_outputs(trace)
    if (linked := outputs.get("linked")) is not None:
        return dict(linked)
    # Linking-disabled runs (and older traces) carry only the table list.
    tables = outputs.get("linked_tables")
    return None if tables is None else {t: [] for t in tables}


def linking_metrics(gold_sql: str, trace: dict[str, Any]) -> dict[str, Any] | None:
    """Score schema linking against what the gold SQL actually references.

    Gold SQL is used for scoring only, exactly like execution accuracy, never as pipeline
    input.

    Args:
        gold_sql: The instance's gold SQL.
        trace: The pipeline trace.

    Returns:
        A ``{"table", "column"}`` pair of :func:`_prf` blocks, columns compared as
        ``table.column`` so a right column on the wrong table counts as a miss. None when
        linking did not run or the gold SQL parses to no tables, so "not measured" is never
        confused with a score of 0.
    """
    from text2sql.schema.linker import SchemaLinker

    linked = linked_fields(trace)
    gold = SchemaLinker.extract_fields(gold_sql)
    if linked is None or not gold:
        return None

    def names(fields: dict[str, list[str]]) -> tuple[set[str], set[str]]:
        """Reduce fields to comparable name sets.

        Args:
            fields: ``{table: [columns]}``.

        Returns:
            ``(lowercased tables, lowercased table.column pairs)``.
        """
        return ({t.lower() for t in fields},
                {f"{t.lower()}.{c.lower()}" for t, cols in fields.items() for c in cols})

    gold_tables, gold_columns = names(gold)
    linked_tables, linked_columns = names(linked)
    size = linking_outputs(trace).get("schema") or {}
    return {"table": _prf(gold_tables, linked_tables, size.get("table", 0)),
            "column": _prf(gold_columns, linked_columns, size.get("column", 0))}


def apply_verdicts(results: list[BenchmarkResult], verdicts: dict[str, dict[str, Any]]) -> None:
    """Fold official verdicts into results, in place.

    Shared by a live run and ``text2sql eval`` so both derive correctness the same way: a
    pass is the evaluator's ``status``, and an instance is executable unless the evaluator
    hit an execution or timeout error. An instance that produced no SQL is neither, since a
    bespoke test case can pass vacuously against an empty statement.

    Args:
        results: The results to annotate.
        verdicts: The evaluator's verdicts, keyed by instance id.
    """
    for r in results:
        verdict = verdicts.get(r.id)
        if verdict is None:
            logger.warning("No official verdict for %s", r.id)
            continue
        r.verdict = verdict
        produced = bool((r.predicted_sql or "").strip())
        r.execution_match = produced and verdict["status"] == "success"
        r.executable = produced and not (
            verdict["execution_error"] or verdict["timeout_error"])


@dataclass
class BenchmarkResult:
    """Result of running a single benchmark example."""

    id: str
    question: str
    predicted_sql: str
    gold_sql: str
    execution_match: bool | None = None
    pipeline_error: str | None = None
    latency_seconds: float = 0.0
    trace: dict[str, Any] = field(default_factory=dict)
    predicted_results: list[dict[str, Any]] | None = None
    gold_results: list[dict[str, Any]] | None = None
    linking: dict[str, Any] | None = None
    pred_sqls: list[str] = field(default_factory=list)
    candidates: list[str] = field(default_factory=list)
    candidate_correct: list[bool] = field(default_factory=list)
    executable: bool = False
    # This instance's own LLM usage; the trace carries the engine's *running* total, which
    # summed over instances over-counts by roughly n/2.
    usage: dict[str, Any] = field(default_factory=dict)
    empty_result: bool = False
    category: str = ""
    difficulty: str = ""
    db_name: str = ""
    verdict: dict[str, Any] = field(default_factory=dict)
    record: dict[str, Any] = field(default_factory=dict)

    @property
    def oracle(self) -> bool:
        """Whether any candidate was correct: the ceiling selection could have reached."""
        return any(self.candidate_correct)

    @property
    def error(self) -> str | None:
        """The pipeline's own message and the evaluator's, derived so re-scoring cannot
        stack a stale verdict message onto a fresh one."""
        return " | ".join(dict.fromkeys(
            m for m in (self.pipeline_error, self.verdict.get("error_message")) if m)) or None

    @property
    def row_graded(self) -> bool:
        """Whether the grader compares returned rows: no bespoke test case, so a read."""
        return not self.verdict.get("total_test_cases")

    @property
    def linking_recall(self) -> float | None:
        """Table-level recall, the headline linking number; None when unmeasured."""
        return self.linking["table"]["recall"] if self.linking else None

    @property
    def linking_extra(self) -> int | None:
        """How many linked tables the gold query never needed; None when unmeasured."""
        return len(self.linking["table"]["extra"]) if self.linking else None

    def to_dict(self, max_rows: int = 50) -> dict[str, Any]:
        """Serialize one instance, clustered the way the pipeline runs.

        Args:
            max_rows: Result rows kept from each of the predicted and gold sets.

        Returns:
            The instance as a JSON-compatible dict.
        """
        steps = {s["step"]: s for s in self.trace.get("steps", [])}

        def stage(name: str, **extra: Any) -> dict[str, Any]:
            """One pipeline step: its duration, its recorded outputs, and any extra."""
            step = steps.get(name) or {}
            return {"seconds": step.get("duration"),
                    **(step.get("outputs") or {}), **extra}

        generation = stage("sql_generation")
        # The conversations belong to the instance, not to one stage: repair and selection
        # add their own, and a run is debugged by reading them in order.
        conversations = [{"stage": "generation", "messages": m}
                         for m in generation.pop("conversations", None) or []]
        return {
            "id": self.id,
            "category": self.category,
            "difficulty": self.difficulty,
            "db": self.db_name,
            "question": self.question,
            "verdict": {
                "execution_match": self.execution_match,
                "executable": self.executable,
                "empty_result": self.empty_result,
                "oracle": self.oracle,
                "row_graded": self.row_graded,
                "error": self.error,
                "pipeline_error": self.pipeline_error,
                **{k: self.verdict.get(k) for k in
                   ("execution_error", "timeout_error", "assertion_error")},
                # Meaningless where the grader compares rows: the report writes (1/0) there.
                **{k: None if self.row_graded else self.verdict.get(k)
                   for k in ("passed_test_cases", "total_test_cases")},
            },
            # Pipeline order, so the file reads the way the run happened.
            "stages": {
                "profiling": stage(STAGES["profiling"]),
                "linking": stage(STAGES["linking"], metrics=self.linking),
                "generation": generation,
                "repair": stage(STAGES["repair"]),
                "selection": stage(STAGES["selection"],
                                   candidate_correct=self.candidate_correct),
            },
            "sql": {"predicted": self.predicted_sql, "gold": self.gold_sql},
            "results": {"predicted": (self.predicted_results or [])[:max_rows],
                        "gold": (self.gold_results or [])[:max_rows]},
            "usage": {**(self.usage or {}), "latency_seconds": round(self.latency_seconds, 2)},
            "conversations": conversations,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], record: dict[str, Any]) -> BenchmarkResult:
        """Rebuild a result from ``run.json``, the inverse of :meth:`to_dict`.

        Args:
            data: One instance block from ``run.json``.
            record: Its source dataset row.

        Returns:
            The result, with the verdict fields left at their defaults for re-scoring to
            supply, so ``text2sql eval`` runs through the same report code as a live run.

        Raises:
            KeyError: The file predates this layout; its numbers are final, not re-scorable.
        """
        if "stages" not in data:
            raise KeyError(
                f"{data.get('id', '?')}: run.json predates the stage layout and cannot be "
                "re-scored; re-run the benchmark instead")
        sql, stages, usage = data["sql"], data["stages"], data.get("usage") or {}
        # Rebuild the trace so re-scoring rewrites the file with everything it read: the
        # stage durations the summary sums, and the conversations lifted out of generation.
        generation = stages["generation"] | {
            "conversations": [c["messages"] for c in data.get("conversations") or []]}
        outputs = {**stages, "generation": generation}
        trace = {"steps": [{"step": step, "duration": outputs[name].get("seconds"),
                            "outputs": {k: v for k, v in outputs[name].items() if k != "seconds"}}
                           for name, step in STAGES.items()]}
        return cls(
            id=data["id"], question=data.get("question", ""),
            predicted_sql=sql.get("predicted", ""), gold_sql=sql.get("gold", ""),
            pred_sqls=split_statements(sql.get("predicted") or ""),
            candidates=stages["generation"].get("candidates") or [],
            candidate_correct=stages["selection"].get("candidate_correct") or [],
            pipeline_error=data["verdict"].get("pipeline_error"),
            usage=usage, latency_seconds=usage.get("latency_seconds") or 0.0,
            trace=trace, linking=stages["linking"].get("metrics"),
            empty_result=bool(data["verdict"].get("empty_result")),
            predicted_results=(data.get("results") or {}).get("predicted"),
            gold_results=(data.get("results") or {}).get("gold"),
            category=data.get("category", ""), difficulty=data.get("difficulty", ""),
            db_name=data.get("db", ""), record=record)


@dataclass
class BenchmarkReport:
    """Aggregate metrics from a benchmark run."""

    total: int = 0
    correct: int = 0
    incorrect: int = 0
    errors: int = 0
    execution_accuracy: float = 0.0
    avg_latency: float = 0.0
    avg_linking_recall: float | None = None
    avg_linking_extra: float | None = None
    results: list[BenchmarkResult] = field(default_factory=list)

    def compute(self) -> None:
        """Compute the official accuracy, where errors count against you, plus the extras."""
        self.total = len(self.results)
        self.correct = sum(1 for r in self.results if r.execution_match is True)
        self.errors = sum(1 for r in self.results if not r.executable)
        self.incorrect = self.total - self.correct - self.errors
        self.execution_accuracy = self.correct / self.total if self.total else 0.0

        latencies = [r.latency_seconds for r in self.results if r.latency_seconds > 0]
        self.avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        # Averaged over measured examples only; None when linking never ran.
        recalls = [r.linking_recall for r in self.results if r.linking_recall is not None]
        self.avg_linking_recall = sum(recalls) / len(recalls) if recalls else None
        extras = [r.linking_extra for r in self.results if r.linking_extra is not None]
        self.avg_linking_extra = sum(extras) / len(extras) if extras else None

    def _usage(self, key: str) -> float:
        """Total one usage counter across the run.

        Args:
            key: The counter name.

        Returns:
            The sum over every result.
        """
        return sum((r.usage or {}).get(key) or 0 for r in self.results)

    def linking(self) -> dict[str, dict[str, float]] | None:
        """Build the schema-linking scorecard.

        Returns:
            Each level averaged over the examples where it was measured, since a star-select
            scores tables but not columns; None when linking never ran.
        """
        out = {}
        for level in ("table", "column"):
            if m := [r.linking[level] for r in self.results if r.linking and r.linking[level]]:
                out[level] = {
                    metric: round(sum(float(x[metric]) for x in scored) / len(scored), 4)
                    for metric in ("kept", "precision", "recall", "covered", "f1",
                                   "exact_match")
                    if (scored := [x for x in m if metric in x])} | {"n": len(m)}
        return out or None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the run totals, grouped the way they are read.

        Returns:
            The scorecard as a JSON-compatible dict. A run halted before SQL exists was
            never scored, so its accuracy fields are None rather than 0.
        """
        oracle = sum(1 for r in self.results if r.oracle)
        cost = self._usage("total_cost_usd")
        scored = any(r.execution_match is not None for r in self.results)
        seconds: dict[str, float] = {}
        for r in self.results:
            for step in r.trace.get("steps", []):
                seconds[step["step"]] = round(
                    seconds.get(step["step"], 0.0) + (step.get("duration") or 0.0), 2)
        return {
            "accuracy": {
                "total": self.total,
                "correct": self.correct if scored else None,
                "incorrect": self.incorrect if scored else None,
                "errors": self.errors if scored else None,
                "execution_accuracy": round(self.execution_accuracy, 4) if scored else None,
                # Oracle@N is the ceiling selection could reach; the gap is what it threw away.
                "oracle_accuracy": round(oracle / self.total, 4) if scored and self.total else None,
                "selection_loss": round((oracle - self.correct) / self.total, 4)
                if scored and self.total else None,
            },
            "execution": {
                "executable_rate": self._rate(lambda r: r.executable) if scored else None,
                # Reads only: a write returns nothing by nature, so over everything this
                # would measure the read/write mix instead of query quality.
                "empty_result_rate": self._rate(
                    lambda r: r.empty_result, lambda r: r.row_graded) if scored else None,
                "row_graded": sum(1 for r in self.results if r.row_graded),
                # The vendored wrapper contends over its ephemeral copies: one gold run
                # scored 269/270 and the next 270/270. Such a failure is not a wrong answer.
                "harness_errors": sum(1 for r in self.results
                                      if "database connection" in (r.error or "")),
            },
            "cost": {
                "total_tokens": self._usage("total_tokens"),
                "total_cost_usd": round(cost, 4),
                "cost_per_correct_usd": round(cost / self.correct, 4) if self.correct else None,
                "avg_latency_seconds": round(self.avg_latency, 2),
            },
            "linking": self.linking(),
            "stage_seconds": seconds,
        }

    def scores(self) -> list[dict[str, Any]]:
        """Build the per-instance verdict patch, which exists only after scoring.

        A streamed run emits every result while it is still unscored, since scoring is one
        batch at the end, so a live consumer holds rows whose verdicts are missing.

        Returns:
            That delta, one row per instance.
        """
        return [{"id": r.id, "verdict": {
            "execution_match": r.execution_match, "executable": r.executable,
            "empty_result": r.empty_result, "oracle": r.oracle, "error": r.error}}
            for r in self.results]

    def _rate(self, predicate: Any, over: Any = None) -> float:
        """Share of the results satisfying a predicate.

        Args:
            predicate: Counted when true.
            over: Restricts the denominator; defaults to every result.

        Returns:
            The rate, or 0.0 when nothing qualifies.
        """
        rows = [r for r in self.results if over is None or over(r)]
        return round(sum(1 for r in rows if predicate(r)) / len(rows), 4) if rows else 0.0

    def breakdowns(self) -> dict[str, dict[str, dict[str, Any]]]:
        """Split accuracy by category, difficulty and database, which the official report
        does not report.

        Returns:
            ``{dimension: {group: {n, correct, accuracy}}}``.
        """
        out: dict[str, dict[str, dict[str, Any]]] = {}
        for dimension, key in (("category", lambda r: r.category),
                               ("difficulty", lambda r: r.difficulty),
                               ("database", lambda r: r.db_name)):
            groups: dict[str, list[BenchmarkResult]] = {}
            for r in self.results:
                groups.setdefault(key(r) or "unknown", []).append(r)
            out[dimension] = {
                name: {"n": len(rs),
                       "correct": sum(1 for r in rs if r.execution_match is True),
                       "accuracy": round(sum(1 for r in rs if r.execution_match is True) / len(rs), 4)}
                for name, rs in sorted(groups.items())}
        return out
