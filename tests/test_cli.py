"""Tests for text2sql.cli — Typer CLI commands.

Uses Typer's CliRunner to invoke commands in-process.
All LLM/DB interactions are mocked to avoid external dependencies.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from text2sql.benchmark import BenchmarkReport
from text2sql.cli import app
from text2sql.core import Text2SQLResult

runner = CliRunner()

#: Every ask test invokes the same command with a different tail flag.
ASK = ["ask", "Q", "--db", "sqlite:///test.db", "--no-stream"]


def answer(sql="SELECT 1", error=None, rows=None):
    return Text2SQLResult(
        sql=sql,
        results=rows if rows is not None else (None if error else [{"cnt": 5}]),
        error=error,
        trace={"llm_usage": {"num_calls": 1, "total_tokens": 100, "total_cost_usd": 0.001}},
    )


@pytest.fixture
def engine():
    """A mock engine standing in for `Text2SQL.build()`: `engine.answers(...)` sets what
    `ask()` streams, and `engine.build` is the patched class the command constructed it
    through.
    """
    eng = MagicMock()
    eng.__enter__ = MagicMock(return_value=eng)
    eng.__exit__ = MagicMock(return_value=False)
    eng.settings.model = "gpt-4o-mini"
    eng.db._safe_uri.return_value = "sqlite:///test.db"
    eng.profile_database = AsyncMock()

    def answers(*items):
        async def ask(question):
            for item in items:
                yield item
        eng.ask = ask

    eng.answers = answers
    answers(answer())

    with patch("text2sql.cli.Text2SQL") as cls:
        cls.build.return_value = eng
        eng.cls = cls
        yield eng


@pytest.fixture
def benchmark_run(tmp_path, engine):
    """Stub out dataset loading and the runner; the CLI keeps the final report."""
    engine.settings.benchmark_use_knowledge = False
    engine.settings.benchmark_dataset_folder = str(tmp_path)
    engine.settings.benchmark_data_jsonl = None
    engine.settings.benchmark_output_dir = str(tmp_path / "results")
    report = BenchmarkReport()
    report.compute()

    async def fake_run(*args, **kwargs):
        # run() is a streaming async generator; the CLI keeps the last report.
        yield report

    with patch("text2sql.benchmark.load_examples", return_value=[]), \
            patch("text2sql.benchmark.BenchmarkRunner") as mock_runner:
        mock_runner.return_value.run = fake_run
        yield engine


# ── version command ──────────────────────────────────────────────


class TestVersionCommand:
    def test_version_output(self):
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "text2sql-toolkit" in result.output
        assert any(c.isdigit() for c in result.output)


# ── ask command ──────────────────────────────────────────────────


class TestAskCommand:
    def test_ask_no_stream(self, engine):
        result = runner.invoke(app, ASK)
        assert result.exit_code == 0
        assert "SELECT 1" in result.output

    def test_ask_error_result(self, engine):
        engine.answers(answer(error="LLM failed"))
        result = runner.invoke(app, ASK)
        assert result.exit_code == 1
        assert "LLM failed" in result.output

    def test_ask_json_output(self, engine):
        result = runner.invoke(app, [*ASK, "--json"])
        assert result.exit_code == 0
        assert "sql" in result.output

    def test_ask_with_trace(self, engine):
        result = runner.invoke(app, [*ASK, "--trace"])
        assert result.exit_code == 0
        assert "Trace" in result.output

    def test_ask_with_config(self, engine):
        """The --config path is routed through Text2SQL.build(config, ...)."""
        result = runner.invoke(app, ["ask", "Q", "--config", "config.yaml", "--no-stream"])
        assert result.exit_code == 0
        engine.cls.build.assert_called_once()
        assert engine.cls.build.call_args.args[0] == "config.yaml"

    def test_ask_none_result(self, engine):
        engine.answers()  # the generator finishes without producing a result
        assert runner.invoke(app, ASK).exit_code == 1

    def test_ask_many_rows_truncated(self, engine):
        engine.answers(answer(rows=[{"x": i} for i in range(30)]))
        result = runner.invoke(app, ASK)
        assert result.exit_code == 0
        assert "30 rows total" in result.output


# ── profile command ──────────────────────────────────────────────


class TestProfileCommand:
    def test_profile_basic(self, engine):
        result = runner.invoke(app, ["profile", "sqlite:///test.db"])
        assert result.exit_code == 0
        assert "cached" in result.output.lower()

    def test_profile_with_output(self, engine, tmp_path):
        engine.cache.cache_key.return_value = "test_key"
        engine.cache.load_profile.return_value = MagicMock(to_dict=lambda: {"tables": {}})
        engine.cache.load_summary.return_value = MagicMock(to_dict=lambda: {"columns": {}})

        out_file = tmp_path / "out.json"
        result = runner.invoke(app, ["profile", "sqlite:///test.db", "--output", str(out_file)])
        assert result.exit_code == 0
        assert out_file.exists()


# ── serve command ────────────────────────────────────────────────


class TestServeCommand:
    def test_serve_is_registered(self):
        """If uvicorn is missing, serve should fail gracefully rather than at import."""
        with patch.dict("sys.modules", {"uvicorn": None}):
            from text2sql.cli import serve
            assert serve is not None


# ── benchmark & eval commands ────────────────────────────────────


class TestBenchmarkCommand:
    def test_benchmark_basic(self, benchmark_run, tmp_path):
        result = runner.invoke(app, [
            "benchmark", "--dataset-folder", str(tmp_path),
            "--output", str(tmp_path / "results"),
        ])
        assert result.exit_code == 0

    def test_stop_after_reaches_settings(self, benchmark_run):
        """--stop-after is a plain Settings override, so it works for any stage."""
        benchmark_run.settings.stop_after = "schema_linking"
        result = runner.invoke(app, ["benchmark", "--stop-after", "schema_linking"])

        assert result.exit_code == 0
        assert benchmark_run.cls.build.call_args.kwargs["stop_after"] == "schema_linking"
        assert "Execution Accuracy" not in result.stdout  # nothing was executed

    def test_eval_rescoring(self, tmp_path):
        """`text2sql eval` re-runs the official evaluator over a finished run."""
        report = BenchmarkReport(results=[])
        report.compute()
        with patch("text2sql.benchmark.rescore", return_value=report) as mock:
            result = runner.invoke(app, ["eval", str(tmp_path), "--mode", "gold"])

        assert result.exit_code == 0
        assert mock.call_args.args[0] == str(tmp_path)
        assert mock.call_args.kwargs["mode"] == "gold"
