"""Pipeline tracing: captures full execution details for debugging and analysis."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StepTrace:
    """Trace for a single pipeline step."""
    step_name: str
    started_at: float = 0.0
    completed_at: float = 0.0
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def duration_seconds(self) -> float:
        """Wall-clock seconds the step took, or 0.0 while it is still running."""
        return self.completed_at - self.started_at if self.completed_at else 0.0


@dataclass
class PipelineTrace:
    """Full trace for a pipeline execution."""
    question: str = ""
    db_uri: str = ""
    model: str = ""
    started_at: float = 0.0
    completed_at: float = 0.0
    steps: list[StepTrace] = field(default_factory=list)
    final_sql: str = ""
    final_results: list[dict[str, Any]] | None = None
    llm_usage: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def duration_seconds(self) -> float:
        """Wall-clock seconds the pipeline took, or 0.0 while it is still running."""
        return self.completed_at - self.started_at if self.completed_at else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Flatten the trace for JSON output.

        Returns:
            The run's metadata, usage and per-step durations and outputs.
        """
        return {
            "question": self.question,
            "db_uri": self.db_uri,
            "model": self.model,
            "duration_seconds": round(self.duration_seconds, 3),
            "final_sql": self.final_sql,
            "num_results": len(self.final_results) if self.final_results else 0,
            "llm_usage": self.llm_usage,
            "error": self.error,
            "steps": [
                {
                    "step": s.step_name,
                    "duration": round(s.duration_seconds, 3),
                    "error": s.error,
                    "outputs": s.outputs,
                }
                for s in self.steps
            ],
        }


class PipelineTracer:
    """Context manager for tracing pipeline execution."""

    def __init__(self) -> None:
        self.trace = PipelineTrace()
        self._current_step: StepTrace | None = None

    def start_pipeline(self, question: str, db_uri: str, model: str) -> None:
        """Begin a run.

        Args:
            question: The user's question.
            db_uri: Database the run targets.
            model: Model the run uses.
        """
        self.trace.question = question
        self.trace.db_uri = db_uri
        self.trace.model = model
        self.trace.started_at = time.time()

    def end_pipeline(self, sql: str = "", results: list[dict[str, Any]] | None = None,
                     llm_usage: dict[str, Any] | None = None, error: str | None = None) -> None:
        """Close the run and record its outcome.

        Args:
            sql: The final SQL.
            results: The rows it returned.
            llm_usage: Token and cost totals.
            error: The failure, when the run did not complete.
        """
        self.trace.completed_at = time.time()
        self.trace.final_sql = sql
        self.trace.final_results = results
        self.trace.llm_usage = llm_usage or {}
        self.trace.error = error

    def start_step(self, name: str, **inputs: Any) -> StepTrace:
        """Begin a pipeline step.

        Args:
            name: Step name.
            **inputs: Inputs recorded on the step.

        Returns:
            The started :class:`StepTrace`, to pass back to :meth:`end_step`.
        """
        step = StepTrace(step_name=name, started_at=time.time(), inputs=inputs)
        self._current_step = step
        self.trace.steps.append(step)
        return step

    def end_step(self, step: StepTrace, error: str | None = None, **outputs: Any) -> None:
        """Close a pipeline step.

        Args:
            step: The step returned by :meth:`start_step`.
            error: The failure, when the step did not complete.
            **outputs: Outputs recorded on the step.
        """
        step.completed_at = time.time()
        step.outputs = outputs
        step.error = error
        self._current_step = None
