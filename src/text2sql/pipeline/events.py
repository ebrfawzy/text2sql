"""Pipeline event model and emitter for streaming progress.

Structured events describing each stage of the pipeline, from profiling through to final
selection. Events are yielded by ``Text2SQL.ask()``.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class Stage(StrEnum):
    """Pipeline stage identifiers."""

    PROFILING = "profiling"
    SCHEMA_LINKING = "schema_linking"
    SQL_GENERATION = "sql_generation"
    SQL_REPAIR = "sql_repair"
    SELECTION = "selection"
    PIPELINE = "pipeline"


class Status(StrEnum):
    """Event status values."""

    STARTED = "started"
    PROGRESS = "progress"
    COMPLETED = "completed"
    ERROR = "error"


# Stage-specific emoji for descriptive messages.
_STAGE_ICONS: dict[str, str] = {
    Stage.PROFILING: "🔬",
    Stage.SCHEMA_LINKING: "🔗",
    Stage.SQL_GENERATION: "🤖",
    Stage.SQL_REPAIR: "🛠️",
    Stage.SELECTION: "🗳️",
    Stage.PIPELINE: "🏁",
}

_STATUS_ICONS: dict[str, str] = {
    Status.STARTED: "⏳",
    Status.PROGRESS: "📝",
    Status.COMPLETED: "✅",
    Status.ERROR: "❌",
}


@dataclass
class PipelineEvent:
    """A single event emitted during pipeline execution.

    Attributes:
        stage: Pipeline stage identifier (e.g. ``"profiling"``).
        status: Event status: ``"started"``, ``"progress"``, ``"completed"`` or ``"error"``.
        message: Human-readable progress description.
        data: Stage-specific structured payload.
        elapsed_seconds: Seconds since the pipeline started.
        timestamp: ISO-8601 UTC timestamp.
    """

    stage: str
    status: str
    message: str
    data: dict[str, Any] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize the event.

        Returns:
            A JSON-compatible dict.
        """
        return {
            "stage": self.stage,
            "status": self.status,
            "message": self.message,
            "data": self.data,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "timestamp": self.timestamp,
        }

    def __str__(self) -> str:
        """Render the event as one human-readable line."""
        icon = _STATUS_ICONS.get(self.status, "")
        return f"[{self.stage}] {icon} {self.status} - {self.message}"


@dataclass
class TokenDelta:
    """A single streamed token chunk from the LLM, yielded by ``ask()``.

    Attributes:
        text: The token text fragment.
        is_thinking: ``True`` when the token is part of the model's internal reasoning.
    """

    text: str
    is_thinking: bool = False


class EventEmitter:
    """Lightweight helper that creates timestamped :class:`PipelineEvent` instances.

    Usage::

        emitter = EventEmitter()
        event = emitter.emit(Stage.PROFILING, Status.STARTED, "Loading profile...")
        # ... do work ...
        event = emitter.emit(Stage.PROFILING, Status.COMPLETED, "Done", tables_profiled=8)
    """

    def __init__(self) -> None:
        self._start_time = time.monotonic()

    def emit(
        self,
        stage: str,
        status: str,
        message: str,
        **data: Any,
    ) -> PipelineEvent:
        """Create a :class:`PipelineEvent` with elapsed time and timestamp.

        Args:
            stage: Pipeline stage identifier.
            status: Event status.
            message: Human-readable description.
            **data: Arbitrary key-value pairs for the event payload.

        Returns:
            A fully populated ``PipelineEvent``.
        """
        icon = _STAGE_ICONS.get(stage, "")
        decorated_message = f"{icon} {message}" if icon else message

        return PipelineEvent(
            stage=stage,
            status=status,
            message=decorated_message,
            data=data,
            elapsed_seconds=time.monotonic() - self._start_time,
            timestamp=datetime.now(UTC).isoformat(),
        )


async def collect_result(stream: AsyncIterator[Any]) -> Any:
    """Drain a ``Text2SQL.ask()`` stream and return its final result.

    Args:
        stream: The stream to consume.

    Returns:
        The terminal ``Text2SQLResult``, or None when the stream held only progress items.
    """
    result: Any = None
    async for item in stream:
        if not isinstance(item, (PipelineEvent, TokenDelta)):
            result = item
    return result
