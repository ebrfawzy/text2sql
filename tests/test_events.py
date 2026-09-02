"""Tests for text2sql.pipeline.events — PipelineEvent, EventEmitter.

Covers event construction, serialization, string representation with
emoji icons, emitter timestamps, and elapsed time monotonicity.
"""

from __future__ import annotations

import time

import pytest

from text2sql.pipeline.events import (
    EventEmitter,
    PipelineEvent,
    Stage,
    Status,
    TokenDelta,
)

# ── Stage & Status enums ─────────────────────────────────────────


class TestEnumValues:
    """The wire format is the string value — SSE consumers match on it."""

    def test_stage_values(self):
        assert [s.value for s in Stage] == [
            "profiling", "schema_linking", "sql_generation", "sql_repair",
            "selection", "pipeline"]

    def test_status_values(self):
        assert [s.value for s in Status] == ["started", "progress", "completed", "error"]


# ── PipelineEvent ────────────────────────────────────────────────


class TestPipelineEvent:
    def test_create_event(self):
        event = PipelineEvent(
            stage="profiling", status="started",
            message="Loading...", data={"tables": 5},
            elapsed_seconds=1.23, timestamp="2026-01-01T00:00:00+00:00",
        )
        assert event.stage == "profiling"
        assert event.status == "started"
        assert event.data == {"tables": 5}

    def test_to_dict(self):
        event = PipelineEvent(
            stage="schema_linking", status="completed",
            message="Linked 3 tables",
            data={"linked_tables": ["a", "b", "c"]},
            elapsed_seconds=2.5678,
            timestamp="2026-01-01T00:00:02+00:00",
        )
        d = event.to_dict()
        assert d["stage"] == "schema_linking"
        assert d["status"] == "completed"
        assert d["elapsed_seconds"] == 2.568  # rounded to 3 decimals
        assert d["data"]["linked_tables"] == ["a", "b", "c"]

    def test_defaults_and_dict_keys(self):
        event = PipelineEvent(stage="x", status="y", message="z")
        assert event.data == {} and event.timestamp == ""
        assert set(event.to_dict()) == {"stage", "status", "message", "data",
                                        "elapsed_seconds", "timestamp"}

    @pytest.mark.parametrize("status, icon", [
        ("started", "⏳"),
        ("progress", "📝"),
        ("completed", "✅"),
        ("error", "❌"),
    ])
    def test_str_status_icons(self, status, icon):
        event = PipelineEvent(stage="pipeline", status=status, message="msg")
        assert icon in str(event)

    def test_str_contains_stage(self):
        event = PipelineEvent(stage="profiling", status="started", message="Loading")
        assert "[profiling]" in str(event)
        assert "Loading" in str(event)


# ── EventEmitter ─────────────────────────────────────────────────


class TestEventEmitter:
    def test_emit_creates_event(self):
        emitter = EventEmitter()
        event = emitter.emit(Stage.PROFILING, Status.STARTED, "Loading...")
        assert event.stage == "profiling"
        assert event.status == "started"
        assert "🔬" in event.message
        assert event.timestamp != ""
        assert event.elapsed_seconds >= 0

    def test_elapsed_time_increases(self):
        emitter = EventEmitter()
        e1 = emitter.emit(Stage.PROFILING, Status.STARTED, "Start")
        time.sleep(0.05)
        e2 = emitter.emit(Stage.PROFILING, Status.COMPLETED, "Done")
        assert e2.elapsed_seconds > e1.elapsed_seconds

    def test_data_kwargs(self):
        emitter = EventEmitter()
        event = emitter.emit(
            Stage.SCHEMA_LINKING, Status.COMPLETED, "Done",
            linked_tables=["a", "b"], count=2,
        )
        assert event.data["linked_tables"] == ["a", "b"]
        assert event.data["count"] == 2

    def test_unknown_stage_no_icon(self):
        emitter = EventEmitter()
        event = emitter.emit("unknown_stage", Status.STARTED, "Msg")
        # Message should not have a leading icon (no empty space + icon)
        assert event.message == "Msg"

    def test_multiple_emits_use_same_clock(self):
        emitter = EventEmitter()
        events = [emitter.emit(Stage.PROFILING, Status.STARTED, f"E{i}") for i in range(5)]
        elapsed = [e.elapsed_seconds for e in events]
        assert elapsed == sorted(elapsed)  # monotonically non-decreasing


# ── TokenDelta ───────────────────────────────────────────────────


class TestTokenDelta:
    def test_thinking_flag_defaults_off(self):
        assert TokenDelta(text="hello").is_thinking is False
        assert TokenDelta(text="reasoning", is_thinking=True).is_thinking is True
