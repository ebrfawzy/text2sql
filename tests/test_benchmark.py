"""Tests for text2sql.benchmark — runner, report, linking metrics, official driver.

Correctness itself is never computed here: the runner records a run and the
vendored LiveSQLBench evaluator judges it. These tests cover the recording, the
artifacts, the folding-back of verdicts, and the reporting on top.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from text2sql.benchmark import (
    BenchmarkExample,
    BenchmarkReport,
    BenchmarkResult,
    BenchmarkRunner,
    apply_verdicts,
    linking_metrics,
    official,
    write_predictions,
)
from text2sql.config import Settings
from text2sql.core import Text2SQL, Text2SQLResult


def result(id="a", **kw):
    """A BenchmarkResult with the four always-required fields filled in."""
    kw.setdefault("question", "Q")
    kw.setdefault("predicted_sql", "S")
    kw.setdefault("gold_sql", "G")
    return BenchmarkResult(id=id, **kw)


# ── Report ───────────────────────────────────────────────────────


class TestBenchmarkReport:
    def test_compute_mixed_results(self):
        report = BenchmarkReport(results=[
            result("1", execution_match=True, latency_seconds=1.5),
            result("2", execution_match=False, latency_seconds=2.5),
        ])
        report.compute()
        assert (report.total, report.correct) == (2, 1)
        assert report.execution_accuracy == 0.5
        assert report.avg_latency == 2.0

    def test_compute_empty(self):
        report = BenchmarkReport()
        report.compute()
        assert report.total == 0 and report.execution_accuracy == 0.0
        assert {"accuracy", "execution", "cost", "linking"} <= set(report.to_dict())

    def test_oracle_and_selection_loss(self):
        report = BenchmarkReport(results=[
            # selected wrong, but one candidate was right — pure selection loss
            result("1", execution_match=False, executable=True, candidate_correct=[False, True]),
            result("2", execution_match=True, executable=True, candidate_correct=[True]),
        ])
        report.compute()
        acc = report.to_dict()["accuracy"]
        assert acc["execution_accuracy"] == 0.5
        assert acc["oracle_accuracy"] == 1.0
        assert acc["selection_loss"] == 0.5

    def test_breakdowns_split_by_category_and_difficulty(self):
        report = BenchmarkReport(results=[
            result("1", execution_match=True, category="Query", difficulty="Simple"),
            result("2", execution_match=False, category="Management", difficulty="Simple"),
        ])
        report.compute()
        by = report.breakdowns()
        assert by["category"]["Query"]["accuracy"] == 1.0
        assert by["category"]["Management"]["accuracy"] == 0.0
        assert by["difficulty"]["Simple"]["n"] == 2

    def test_agent_telemetry_reaches_the_row(self):
        """The counters ride on the generation stage, beside the conversation they summarize."""
        row = result(trace={"steps": [{"step": "sql_generation", "duration": 1.5, "outputs": {
            "agent": {"turns": 4, "tool_calls": 6, "termination": "submit", "max_turns": 20},
            "conversations": [[{"role": "user", "content": "q"}]]}}]}).to_dict()
        gen = row["stages"]["generation"]
        assert (gen["agent"]["turns"], gen["agent"]["tool_calls"]) == (4, 6)
        assert gen["agent"]["termination"] == "submit" and gen["seconds"] == 1.5
        # Lifted to the instance: repair and selection add their own exchanges.
        assert row["conversations"] == [{"stage": "generation",
                                         "messages": [{"role": "user", "content": "q"}]}]
        assert "conversations" not in gen


# ── Linking metrics ──────────────────────────────────────────────


class TestLinkingMetrics:
    """Schema linking scored against the gold SQL (scoring only)."""

    GOLD = "SELECT u.name FROM users u JOIN orders o ON o.user_id = u.id"

    @staticmethod
    def _trace(linked, key="linked"):
        return {"steps": [{"step": "schema_linking", "outputs": {key: linked}}]}

    def test_perfect_link(self):
        m = linking_metrics(self.GOLD, self._trace({"users": ["name", "id"],
                                                    "orders": ["user_id"]}))
        assert (m["table"]["precision"], m["table"]["recall"], m["table"]["f1"]) == (1.0, 1.0, 1.0)
        assert m["table"]["exact_match"] and m["column"]["exact_match"]
        assert m["table"]["missing"] == m["table"]["extra"] == []

    def test_partial_and_extra(self):
        m = linking_metrics(self.GOLD, self._trace({"users": ["name"], "products": ["sku"]}))
        assert m["table"] == {"gold": 2, "linked": 2, "hit": 1, "precision": 0.5, "recall": 0.5,
                              "covered": False, "f1": 0.5, "exact_match": False,
                              "missing": ["orders"], "extra": ["products"]}
        # Columns: users.name hit; users.id + orders.user_id missed; products.sku extra.
        assert m["column"]["hit"] == 1
        assert m["column"]["missing"] == ["orders.user_id", "users.id"]
        assert m["column"]["extra"] == ["products.sku"]

    def test_case_insensitive(self):
        m = linking_metrics("SELECT Name FROM Users", self._trace({"users": ["name"]}))
        assert m["table"]["recall"] == m["column"]["recall"] == 1.0

    def test_falls_back_to_table_list(self):
        """Traces carrying only `linked_tables` still score at the table level."""
        m = linking_metrics(self.GOLD, self._trace(["users", "orders"], key="linked_tables"))
        assert m["table"]["recall"] == 1.0
        assert m["column"]["linked"] == 0

    def test_kept_reports_the_share_of_the_schema_that_survived(self):
        """Precision is bounded by the gold query's own size, so a run that linked every
        table still scored 0.27 and read as a reduction it never made."""
        trace = self._trace({"users": ["name", "id"]})
        trace["steps"][0]["outputs"]["schema"] = {"table": 4, "column": 20}
        m = linking_metrics(self.GOLD, trace)
        assert (m["table"]["kept"], m["column"]["kept"]) == (0.25, 0.1)
        # The second instance carries no recorded schema size: `kept` averages over the
        # instances that have it rather than counting the others as zero.
        report = BenchmarkReport(results=[
            result("1", linking=m),
            result("2", linking=linking_metrics(self.GOLD, self._trace({"users": ["name"]})))])
        report.compute()
        assert report.linking()["table"]["kept"] == 0.25

    def test_empty_gold_column_set_is_not_an_exact_match(self):
        """`SELECT *` yields no gold columns, and `set() == set()` scored as a perfect
        column match — 21 of them fabricated a 0.0802 exact-match rate on one run."""
        m = linking_metrics("SELECT * FROM users", self._trace({"users": []}))
        assert m["table"]["exact_match"] is True
        assert m["column"] is None

    def test_unmeasured_returns_none(self):
        """Linking skipped, step absent, or no gold tables → not measured, not 0.0."""
        assert linking_metrics("SELECT * FROM users", {"steps": []}) is None
        assert linking_metrics("SELECT * FROM users", self._trace(None, "linked_tables")) is None
        assert linking_metrics("SELECT 1", self._trace({"users": []})) is None

    def test_report_averages_only_measured(self):
        report = BenchmarkReport(results=[
            result("1", predicted_sql="", gold_sql="",
                   linking=linking_metrics(self.GOLD, self._trace({"users": [], "orders": []}))),
            result("2", predicted_sql="", gold_sql="",
                   linking=linking_metrics(self.GOLD, self._trace(
                       {"users": [], "products": [], "reviews": []}))),
            result("3", predicted_sql="", gold_sql=""),  # unmeasured
        ])
        report.compute()
        assert report.avg_linking_recall == 0.75  # (1.0 + 0.5) / 2
        assert report.avg_linking_extra == 1.0  # (0 + 2) / 2
        assert report.linking()["table"] == {"precision": 0.6666, "recall": 0.75,
                                             "covered": 0.5, "f1": 0.7, "exact_match": 0.5,
                                             "n": 2}

    def test_each_level_averages_over_its_own_measured_instances(self):
        """A star-select measures tables but not columns, so the two levels have different
        denominators — sharing one reported the unmeasurable half as zero."""
        report = BenchmarkReport(results=[
            result("1", linking=linking_metrics(self.GOLD, self._trace({"users": ["name"]}))),
            result("2", linking=linking_metrics("SELECT * FROM users",
                                                self._trace({"users": []}))),
        ])
        assert report.linking()["table"]["n"] == 2
        assert report.linking()["column"]["n"] == 1

    def test_report_none_when_never_measured(self):
        report = BenchmarkReport(results=[result("1")])
        report.compute()
        assert report.avg_linking_recall is None


# ── Runner: recording a run ──────────────────────────────────────


class TestRunRecording:
    @pytest.fixture
    def runner(self, sample_db):
        """A runner with a live engine but no output dir — `_evaluate` only needs the DB."""
        runner = BenchmarkRunner.__new__(BenchmarkRunner)
        with Text2SQL(db_uri=sample_db) as engine:
            runner.engine = engine
            yield runner

    def test_evaluate_records_the_run_without_judging_it(self, sample_db):
        """Correctness is the official evaluator's call, applied later for the whole run."""
        runner = BenchmarkRunner.__new__(BenchmarkRunner)
        runner.engine = SimpleNamespace(db=SimpleNamespace(execute_safe=lambda sql: ([], None)))
        example = BenchmarkExample(
            id="1", question="Q", db_uri=sample_db,
            gold_sql="SELECT COUNT(*) AS cnt FROM users",
            record={"sol_sql": ["SELECT COUNT(*) AS cnt FROM users"], "test_cases": []})
        res = runner._evaluate(
            example, Text2SQLResult(sql="SELECT COUNT(*) FROM users; ", results=[], trace={}), 1.0)
        assert res.pred_sqls == ["SELECT COUNT(*) FROM users;"]
        assert res.execution_match is None and res.executable is False
        assert res.empty_result is True

    def test_gold_rows_are_fetched_for_side_by_side_comparison(self, runner, sample_db):
        """The UI compares predicted vs gold rows, so both must reach the row dict."""
        res = runner._evaluate(
            BenchmarkExample(id="1", question="Q", db_uri=sample_db,
                             gold_sql="SELECT name FROM users ORDER BY id"),
            Text2SQLResult(sql="SELECT 1", results=[{"x": 1}], trace={}), 1.0)
        assert res.gold_results and "name" in res.gold_results[0]
        assert res.to_dict()["results"]["gold"] == res.gold_results

    def test_gold_writes_never_mutate_the_database(self, runner, sample_db):
        """Management-task gold is DML; db.execute rolls back, so fetching it is safe."""
        from pathlib import Path

        path = Path(sample_db.removeprefix("sqlite:///"))
        before = path.read_bytes()
        runner._evaluate(
            BenchmarkExample(id="1", question="Q", db_uri=sample_db, gold_sql="DELETE FROM users"),
            Text2SQLResult(sql="SELECT 1", results=[], trace={}), 1.0)
        assert path.read_bytes() == before

    async def test_run_one_records_sql_and_errors(self, sample_db, tmp_path):
        async def ok(question):
            yield Text2SQLResult(sql="SELECT 1", results=[{"x": 1}], trace={}, error=None)

        async def boom(question):
            raise RuntimeError("boom")
            yield  # makes this an async generator

        example = BenchmarkExample(id="1", question="Q", db_uri=sample_db, gold_sql="SELECT 1")
        for ask, check in ((ok, lambda r: r.predicted_sql == "SELECT 1"),
                           (boom, lambda r: r.error == "boom")):
            engine = MagicMock()
            engine.ask = ask
            runner = BenchmarkRunner(engine, output_dir=str(tmp_path))
            assert check(await runner._run_one(example))

    def test_each_run_gets_its_own_directory(self, tmp_path):
        """Ablation arms and reruns must never overwrite each other."""
        runner = BenchmarkRunner(SimpleNamespace(), output_dir=str(tmp_path))
        assert runner.output_dir.parent == tmp_path
        assert runner.output_dir.is_dir()


# ── Runner: scoring and artifacts ────────────────────────────────


class TestRunArtifacts:
    @staticmethod
    def _runner(tmp_path, **settings):
        runner = BenchmarkRunner.__new__(BenchmarkRunner)
        runner.output_dir = tmp_path
        runner.engine = SimpleNamespace(
            settings=Settings(bedrock_aws_secret_access_key="shhh", **settings))
        return runner

    @staticmethod
    def _verdict(ok, error="", execution_error=False):
        return {"status": "success" if ok else "failed", "error_message": error,
                "execution_error": execution_error, "timeout_error": False,
                "assertion_error": not ok and not execution_error}

    def test_apply_verdicts_sets_correctness(self):
        """The official status is the single source of truth for a pass, while `executable`
        tracks the execution-error flag: a query that ran but asserted wrong is executable.
        """
        results = [result("a"), result("b")]
        apply_verdicts(results, {"a": self._verdict(True),
                                 "b": self._verdict(False, "syntax error",
                                                    execution_error=True)})
        assert (results[0].execution_match, results[0].executable) == (True, True)
        assert (results[1].execution_match, results[1].executable) == (False, False)
        assert results[1].error == "syntax error"

    def test_no_sql_can_never_be_a_pass(self):
        """A bespoke test case can pass vacuously: museum_M_5 counts rows its own
        `preprocess_sql` was meant to insert, so when that setup cannot run an *empty*
        prediction satisfies it — the run reported Correct: 1, Errors: 1, Incorrect: -1."""
        r = result("museum_M_5")
        r.predicted_sql = ""
        r.pipeline_error = "NOT NULL constraint failed"
        apply_verdicts([r], {"museum_M_5": self._verdict(True)})
        assert (r.execution_match, r.executable) == (False, False)
        report = BenchmarkReport(results=[r])
        report.compute()
        assert (report.correct, report.errors, report.incorrect) == (0, 1, 0)

    def test_rescoring_keeps_cost_and_does_not_stack_error_messages(self):
        """`from_dict` dropped `usage`, so re-scoring rewrote run.json with 0 tokens and
        $0.00; and it restored `error`, onto which the new verdict message was appended."""
        r = result("a")
        r.usage = {"total_tokens": 14, "total_cost_usd": 0.02, "num_calls": 2}
        r.pipeline_error = "no SQL"
        apply_verdicts([r], {"a": self._verdict(False, "first")})

        back = BenchmarkResult.from_dict(r.to_dict(), r.record)
        assert {k: back.usage[k] for k in r.usage} == r.usage
        assert back.pipeline_error == "no SQL"

        apply_verdicts([back], {"a": self._verdict(False, "second")})
        assert back.error == "no SQL | second"   # not "no SQL | first | second"

    def test_a_pre_stage_run_json_is_refused_rather_than_half_read(self):
        """The layout changed with no shim: a silent partial read would rewrite a finished
        run with zeroed cost, which is exactly the failure the shim was meant to prevent."""
        with pytest.raises(KeyError, match="predates the stage layout"):
            BenchmarkResult.from_dict(
                {"id": "a", "total_tokens": 14, "cost_usd": 0.02, "llm_calls": 2}, {})

    def test_rescoring_preserves_everything_it_did_not_score(self):
        """`from_dict` must be the exact inverse: it once read a `trace` key that `to_dict`
        no longer writes, so re-scoring would have rewritten a finished run with its stage
        timings and conversations blanked - the same class of loss as the zeroed cost."""
        r = result("a", trace={"steps": [
            {"step": "sql_generation", "duration": 2.0,
             "outputs": {"candidates": ["SELECT 1"],
                         "conversations": [[{"role": "user", "content": "q"}]]}},
            {"step": "sql_repair", "duration": 0.5, "outputs": {"issues": ["[null] x"]}}]})
        first = r.to_dict()
        again = BenchmarkResult.from_dict(first, r.record).to_dict()
        assert again["stages"] == first["stages"]
        assert again["conversations"] == first["conversations"]
        assert list(again["stages"]) == [
            "profiling", "linking", "generation", "repair", "selection"]

    def test_scores_carries_the_post_evaluator_delta(self):
        """Rows are streamed before scoring, so the UI needs the verdicts separately.

        Regression: the UI's Match and Error columns stayed empty for a whole run, since
        its only per-instance payload was emitted before the evaluator had run.
        """
        results = [result("a"), result("b")]
        apply_verdicts(results, {"a": self._verdict(True),
                                 "b": self._verdict(False, "assertion failed")})
        report = BenchmarkReport(results=results)
        assert report.scores() == [
            {"id": "a", "verdict": {"execution_match": True, "executable": True,
                                    "empty_result": False, "oracle": False, "error": None}},
            {"id": "b", "verdict": {"execution_match": False, "executable": True,
                                    "empty_result": False, "oracle": False,
                                    "error": "assertion failed"}},
        ]

    def test_result_dict_reports_its_own_token_usage(self):
        """Regression: per-example tokens rendered as 0 — only flat scalars were exposed."""
        r = result("a")
        r.usage = {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14,
                   "total_cost_usd": 0.02, "num_calls": 2}
        d = r.to_dict()
        assert (d["usage"]["total_tokens"], d["usage"]["num_calls"],
                d["usage"]["total_cost_usd"]) == (14, 2, 0.02)

    def test_row_graded_gates_the_two_metrics_that_only_apply_to_reads(self):
        """The vendored report writes `(1/0) test cases passed` for every read, so summing
        those counts gave 65 passes out of 2; and empty_result_rate over all instances
        measured the read/write mix, not query quality."""
        read = result("r", empty_result=False)
        write = result("w", empty_result=True)
        apply_verdicts([read, write], {
            "r": self._verdict(True) | {"passed_test_cases": 1, "total_test_cases": 0},
            "w": self._verdict(True) | {"passed_test_cases": 2, "total_test_cases": 2}})

        assert (read.row_graded, write.row_graded) == (True, False)
        assert read.to_dict()["verdict"]["total_test_cases"] is None
        assert read.to_dict()["verdict"]["passed_test_cases"] is None
        assert write.to_dict()["verdict"]["total_test_cases"] == 2

        report = BenchmarkReport(results=[read, write])
        report.compute()
        # The write is empty by nature and excluded, so the rate is the read's alone.
        assert report.to_dict()["execution"]["empty_result_rate"] == 0.0

    def test_predictions_file_is_what_the_official_evaluator_eats(self, tmp_path):
        """Source record verbatim + pred_sqls, sorted by instance_id."""
        results = [
            result("b", predicted_sql="SELECT 2", pred_sqls=["SELECT 2;"],
                   record={"instance_id": "b", "query": "Q"}),
            result("a", predicted_sql="SELECT 1", pred_sqls=["SELECT 1;"],
                   record={"instance_id": "a", "query": "Q"}),
        ]
        path = write_predictions(results, tmp_path / "predictions.jsonl")
        lines = [json.loads(x) for x in path.read_text().splitlines()]
        # Sorted — the wrapper zips its sorted results onto this file positionally.
        assert [x["instance_id"] for x in lines] == ["a", "b"]
        assert lines[1] == {"instance_id": "b", "query": "Q", "pred_sqls": ["SELECT 2;"]}
        # A candidate pass writes the same shape from a different SQL source.
        cand = write_predictions(results, tmp_path / "cand.jsonl", lambda r: "SELECT 9")
        assert json.loads(cand.read_text().splitlines()[0])["pred_sqls"] == ["SELECT 9;"]

    def test_score_runs_the_evaluator_once_per_candidate_slot(self, tmp_path):
        """Oracle@N: each slot is scored by the same evaluator, so selection loss is real."""
        runner = self._runner(tmp_path, benchmark_dataset_folder="data")
        results = [result("a", predicted_sql="BAD", pred_sqls=["BAD;"],
                          candidates=["BAD", "GOOD"], record={"instance_id": "a"})]
        # The selected SQL fails; the second candidate would have passed.
        passes = {"predictions.jsonl": False, "predictions_cand0.jsonl": False,
                  "predictions_cand1.jsonl": True}

        def fake_evaluate(predictions, db_path, **kwargs):
            assert db_path == "data"
            return {"a": self._verdict(passes[predictions.name])}

        with patch("text2sql.benchmark.official.evaluate", side_effect=fake_evaluate):
            runner.score(results)

        assert results[0].execution_match is False
        assert results[0].candidate_correct == [False, True]
        assert results[0].oracle is True

    def test_score_skips_candidate_passes_when_voting_is_off(self, tmp_path):
        runner = self._runner(tmp_path, benchmark_dataset_folder="data")
        results = [result("a", pred_sqls=["S;"], candidates=["S"], record={"instance_id": "a"})]
        with patch("text2sql.benchmark.official.evaluate",
                   return_value={"a": self._verdict(True)}) as mock:
            runner.score(results)

        assert mock.call_count == 1  # no extra database work for a single candidate
        assert results[0].candidate_correct == [True]

    def test_save_report_writes_run_json(self, tmp_path):
        runner = self._runner(tmp_path)
        report = BenchmarkReport(results=[
            result("a", predicted_sql="SELECT 1", execution_match=True,
                   record={"instance_id": "a"})])
        report.compute()
        runner._save_report(report)

        assert {p.name for p in tmp_path.iterdir()} == {"run.json"}
        run = json.loads((tmp_path / "run.json").read_text())
        assert set(run) == {"meta", "summary", "breakdown", "instances"}
        assert run["meta"]["config"]["generation_mode"] == "direct"
        assert not [k for k in run["meta"]["config"] if k.startswith(("bedrock_", "athena_"))]

    def test_linking_run_writes_a_scorecard(self, tmp_path):
        """A stop_after=schema_linking arm reports linking, not execution accuracy."""
        runner = self._runner(tmp_path, stop_after="schema_linking")
        trace = {"steps": [{"step": "schema_linking", "outputs": {"linked": {"users": ["id"]}}}]}
        report = BenchmarkReport(results=[
            result("a", predicted_sql="", gold_sql="SELECT id FROM users", trace=trace,
                   record={"instance_id": "a"},
                   linking=linking_metrics("SELECT id FROM users", trace))])
        report.compute()
        runner._save_report(report)

        assert {p.name for p in tmp_path.iterdir()} == {"run.json", "linking.jsonl"}
        row = json.loads((tmp_path / "linking.jsonl").read_text().splitlines()[0])
        assert row["linked"] == {"users": ["id"]}
        assert row["table"]["exact_match"] and row["column"]["exact_match"]
        run = json.loads((tmp_path / "run.json").read_text())
        assert run["meta"]["stop_after"] == "schema_linking"
        assert run["summary"]["linking"]["column"]["f1"] == 1.0
        # Never scored — reporting 0% would read as "got everything wrong".
        assert run["summary"]["accuracy"]["execution_accuracy"] is None
        assert run["summary"]["accuracy"]["correct"] is None


# ── Official evaluator driver ────────────────────────────────────

REPORT = """--------------------------------------------------
BIRD CRITIC Stack Overflow Result Statistics (Postgres, Multi-Thread):
Number of Instances: 3
Overall Accuracy: 33.33%

Question_alien_1: (2/2) test cases passed, failed test cases: None
Question_credit_4: (0/1) test cases passed, failed test cases: test_case | Eval Phase: Assertion Error
Question_gaming_2: (0/1) test cases passed, failed test cases: None | Eval Phase: Timeout Error
"""


class TestLocalDataset:
    """Loading LiveSQLBench rows, including the knowledge the instance mandates."""

    @pytest.fixture
    def dataset(self, tmp_path, sample_db):
        """A one-instance master file beside a ``credit/`` folder with its own KB."""
        import shutil

        folder = tmp_path / "credit"
        folder.mkdir()
        shutil.copy(sample_db.removeprefix("sqlite:///"), folder / "credit.sqlite")
        (folder / "credit_kb.jsonl").write_text("\n".join(json.dumps(e) for e in [
            {"id": 7, "knowledge": "Repeat Buyer", "description": "buys twice",
             "definition": "orders >= 2 within Window", "type": "domain_knowledge",
             "children_knowledge": [9]},
            {"id": 8, "knowledge": "Unused", "description": "no", "definition": "no"},
            {"id": 9, "knowledge": "Window", "description": "the span counted",
             "definition": "90 days"},
        ]), encoding="utf-8")
        master = tmp_path / "master.jsonl"
        master.write_text(json.dumps({
            "instance_id": "credit_4", "selected_database": "credit",
            "query": "How many repeat buyers?", "external_knowledge": [7, 99],
            "sol_sql": ["SELECT 1"], "difficulty_tier": "moderate",
        }), encoding="utf-8")
        return tmp_path, master

    def test_mandated_knowledge_is_appended_to_the_question(self, dataset):
        """The rule text is the model's only definition of a domain term, so a term named
        inside one comes too: 75 of the 270 instances mandate a term whose definition names
        another, and the agent spent whole turn budgets hunting the missing one. An id the KB
        does not carry must be skipped rather than emitted as a blank rule."""
        from text2sql.benchmark import load_local_dataset

        folder, master = dataset
        example, = load_local_dataset(str(folder), str(master), use_knowledge=True)
        assert example.question == (
            "How many repeat buyers?\n<definitions> MUST USE\n"
            "- Repeat Buyer: buys twice\n  Definition: orders >= 2 within Window\n"
            "- Window: the span counted\n  Definition: 90 days\n"
            "</definitions>")

    def test_knowledge_is_opt_in(self, dataset):
        from text2sql.benchmark import load_local_dataset

        folder, master = dataset
        example, = load_local_dataset(str(folder), str(master))
        assert example.question == "How many repeat buyers?"
        assert example.id == "credit_4" and example.difficulty == "moderate"


class TestOfficialDriver:
    """The evaluator itself is upstream code run unmodified; what is tested here is
    that we invoke it correctly and read its artifacts back faithfully."""

    def test_parse_report_counts_and_phase_flags(self, tmp_path):
        """Per-instance counts and phase flags survive only in the text report."""
        report = tmp_path / "r.txt"
        report.write_text(REPORT, encoding="utf-8")
        parsed = official._parse_report(report)
        assert parsed["alien_1"]["passed_test_cases"] == 2
        assert parsed["alien_1"]["assertion_error"] is False
        assert parsed["credit_4"] == {"passed_test_cases": 0, "total_test_cases": 1,
                                      "execution_error": False, "timeout_error": False,
                                      "assertion_error": True}
        assert parsed["gaming_2"]["timeout_error"] is True

    def test_missing_report_is_not_fatal(self, tmp_path):
        assert official._parse_report(tmp_path / "nope.txt") == {}

    def test_verdicts_merge_both_artifacts(self, tmp_path):
        predictions = tmp_path / "predictions.jsonl"
        predictions.write_text("\n".join(json.dumps(r) for r in [
            {"instance_id": "alien_1", "status": "success", "error_message": None},
            {"instance_id": "credit_4", "status": "failed", "error_message": "test_case failed"},
        ]), encoding="utf-8")
        # The wrapper names its artifacts off the predictions file's stem.
        (tmp_path / "predictions_simple_output_with_status.jsonl").write_text(
            predictions.read_text(), encoding="utf-8")
        (tmp_path / "predictions_simple_report.txt").write_text(REPORT, encoding="utf-8")

        verdicts = official._verdicts(predictions)
        assert verdicts["alien_1"]["status"] == "success"
        assert verdicts["alien_1"]["total_test_cases"] == 2
        assert verdicts["credit_4"]["assertion_error"] is True
        assert verdicts["credit_4"]["error_message"] == "test_case failed"

    def test_summarize(self):
        assert official.summarize([
            {"status": "success"},
            {"status": "failed", "execution_error": True},
            {"status": "failed", "assertion_error": True},
        ]) == {"total": 3, "passed": 1, "execution_errors": 1, "timeout_errors": 0,
               "assertion_errors": 1, "accuracy": 0.3333}
        assert official.summarize([])["accuracy"] == 0.0

    def test_vendored_entrypoints_exist(self):
        assert (official.EVAL_DIR / "wrapper_evaluation_sqlite.py").is_file()
        assert (official.EVAL_DIR / "single_instance_eval_sqlite.py").is_file()

    def test_wrapper_imports_resolve_from_its_own_directory(self):
        """Flat imports (`from utils import …`) — hence the cwd=EVAL_DIR invocation."""
        import subprocess
        import sys

        proc = subprocess.run(
            [sys.executable, "-c", "import wrapper_evaluation_sqlite"],
            cwd=official.EVAL_DIR, capture_output=True, text=True, check=False)
        assert proc.returncode == 0, proc.stderr

class TestKnowledgeSource:
    """Question knowledge is a property of the question, not of the metadata."""

    def test_question_knowledge_reaches_the_loader(self, tmp_path, monkeypatch):
        """These ids index the dataset's own `<db>_kb.jsonl` and belong to the question, not
        the metadata. Forcing them off for some runs changed the question set between arms."""
        from text2sql.config import Settings
        seen = {}

        def fake_loader(**kw):
            seen.update(kw)
            return []

        monkeypatch.setattr("text2sql.benchmark.load_local_dataset", fake_loader)
        from text2sql.benchmark import load_examples

        load_examples(Settings(benchmark_data_jsonl=str(tmp_path / "d.jsonl"),
                               benchmark_dataset_folder=str(tmp_path),
                               benchmark_use_knowledge=True))
        assert seen["use_knowledge"] is True


class TestInstanceFilter:
    """`benchmark_instance_id` is how one run scores a named handful of questions."""

    def test_a_comma_separated_list_selects_the_union_of_prefixes(self, tmp_path):
        from text2sql.benchmark import load_instances

        data = tmp_path / "d.jsonl"
        data.write_text("\n".join(json.dumps({"instance_id": i}) for i in
                                  ("credit_1", "credit_10", "credit_2", "fake_9", "news_7")))
        def picked(spec):
            return [r["instance_id"] for r in load_instances(data, None, spec)]

        assert picked("credit_1,fake_9") == ["credit_1", "credit_10", "fake_9"]
        assert picked(" news_7 , credit_2 ") == ["credit_2", "news_7"]  # order follows the file
        assert len(picked(None)) == len(picked("")) == 5


class TestGoldCheck:
    """`gold_check` is the fairness control on the scorer, so it must feed the evaluator
    exactly what a real run feeds it — only with `sol_sql` in the prediction slot."""

    def test_submission_is_the_source_record_with_sol_sql_as_the_prediction(self, tmp_path):
        """Regression: the record must go through verbatim, else the evaluator scores a
        different question, and the rows must be sorted, since the official wrapper zips
        its results onto the input positionally."""
        master = tmp_path / "master.jsonl"
        master.write_text("\n".join(json.dumps(r) for r in [
            {"instance_id": "b_1", "selected_database": "b", "sol_sql": ["SELECT 2;"],
             "preprocess_sql": ["PRAGMA x;"], "test_cases": ["def test_case(): pass"]},
            {"instance_id": "a_1", "selected_database": "a", "sol_sql": ["SELECT 1;"]},
        ]), encoding="utf-8")

        seen = {}

        def fake_evaluate(predictions, db_path, mode="pred"):
            seen["rows"] = [json.loads(line) for line in
                            Path(predictions).read_text(encoding="utf-8").splitlines()]
            seen["mode"] = mode
            return {"a_1": {"status": "success"},
                    "b_1": {"status": "failed", "error_message": " stale date \n"}}

        with patch("text2sql.benchmark.official.evaluate", side_effect=fake_evaluate):
            from text2sql.benchmark import gold_check
            summary = gold_check(master, tmp_path, tmp_path / "out")

        assert seen["mode"] == "gold"
        assert [r["instance_id"] for r in seen["rows"]] == ["a_1", "b_1"]
        assert seen["rows"][1]["pred_sqls"] == ["SELECT 2;"]
        assert seen["rows"][1]["test_cases"] == ["def test_case(): pass"]
        assert seen["rows"][1]["preprocess_sql"] == ["PRAGMA x;"]
        assert summary["passed"] == 1 and summary["total"] == 2
        assert summary["failures"] == [{"id": "b_1", "error": "stale date"}]
        assert json.loads((tmp_path / "out" / "gold_check.json").read_text())["accuracy"] == 0.5

    def test_an_unmatched_instance_filter_raises_instead_of_scoring_nothing(self, tmp_path):
        """An empty submission would come back 0/0 = 0.0 accuracy, which reads as a broken
        harness rather than a typo in --instance-id."""
        master = tmp_path / "master.jsonl"
        master.write_text(json.dumps({"instance_id": "a_1", "sol_sql": ["SELECT 1;"]}),
                          encoding="utf-8")
        from text2sql.benchmark import gold_check

        with pytest.raises(ValueError, match="nope"):
            gold_check(master, tmp_path, tmp_path / "out", instance_id="nope")

class TestPreparedDatabase:
    """The evaluator applies `preprocess_sql` before scoring predictions; without the same
    setup the pipeline explores and repairs against a schema its answer is not graded on."""

    def _example(self, tmp_path, preprocess):
        import sqlite3

        from text2sql.benchmark import BenchmarkExample
        path = tmp_path / "db_template.sqlite"
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE t (a INTEGER)")
        conn.commit()
        conn.close()
        return path, BenchmarkExample(
            id="i_1", question="q", db_uri=f"sqlite:///{path}", gold_sql="",
            record={"preprocess_sql": preprocess})

    def _runner(self, tmp_path):
        from text2sql.benchmark import BenchmarkRunner
        return BenchmarkRunner(object(), output_dir=str(tmp_path / "out"))

    def test_preprocess_is_applied_to_a_copy(self, tmp_path):
        import sqlite3
        path, example = self._example(tmp_path, ["ALTER TABLE t ADD COLUMN b INTEGER"])
        runner = self._runner(tmp_path)  # holds the temp dir; dropping it deletes the copy
        uri = runner._prepare(example)
        assert uri != example.db_uri
        copy = uri.removeprefix("sqlite:///")
        assert [r[1] for r in sqlite3.connect(copy).execute("PRAGMA table_info(t)")] == ["a", "b"]
        # The shipped database must not be touched.
        assert [r[1] for r in sqlite3.connect(path).execute("PRAGMA table_info(t)")] == ["a"]

    def test_the_filename_survives_so_the_profile_cache_still_hits(self, tmp_path):
        from text2sql.profiler.cache import ProfileCache
        _, example = self._example(tmp_path, ["ALTER TABLE t ADD COLUMN b INTEGER"])
        runner = self._runner(tmp_path)
        uri = runner._prepare(example)
        assert ProfileCache.cache_key(uri) == ProfileCache.cache_key(example.db_uri)

    def test_an_example_without_preprocess_is_left_alone(self, tmp_path):
        _, example = self._example(tmp_path, [])
        assert self._runner(tmp_path)._prepare(example) == example.db_uri


class TestGoldAudit:
    """`gold_check` cannot see these: a degenerate gold still matches itself, scores 100%,
    and leaves the instance unwinnable for any correct pipeline."""

    def _dataset(self, tmp_path, sql, category="Query", preprocess=()):
        import sqlite3
        (tmp_path / "db").mkdir()
        conn = sqlite3.connect(tmp_path / "db" / "db_template.sqlite")
        conn.execute("CREATE TABLE t (a INTEGER)")
        conn.execute("INSERT INTO t VALUES (1)")
        conn.commit()
        conn.close()
        data = tmp_path / "d.jsonl"
        data.write_text(json.dumps(
            {"instance_id": "i_1", "selected_database": "db", "sol_sql": [sql],
             "category": category, "preprocess_sql": list(preprocess)}) + "\n")
        return data

    @pytest.mark.parametrize("sql, defect", [
        ("SELECT strftime('%Y', 'now', 'start of quarter') AS q FROM t", "all_null_column"),
        ("SELECT a FROM t WHERE a > 999", "empty"),
        ("SELECT nope FROM t", "errors"),
    ])
    def test_degenerate_golds_are_reported(self, tmp_path, sql, defect):
        from text2sql.benchmark import gold_audit
        found = gold_audit(self._dataset(tmp_path, sql), tmp_path)
        assert [f["defect"] for f in found] == [defect]

    def test_a_write_gold_is_skipped(self, tmp_path):
        """A CTE-wrapped UPDATE reads as a query unless the whole statement is classified,
        and its correct empty result as a defect."""
        from text2sql.benchmark import gold_audit
        sql = "WITH x AS (SELECT a FROM t) UPDATE t SET a = 2 WHERE a IN (SELECT a FROM x)"
        assert gold_audit(self._dataset(tmp_path, sql, "Management"), tmp_path) == []

    def test_preprocess_sql_runs_before_the_gold(self, tmp_path):
        """`fake_M_4` adds the very column its gold reads; without it the gold looks broken."""
        from text2sql.benchmark import gold_audit
        data = self._dataset(tmp_path, "SELECT b FROM t WHERE b IS NOT NULL",
                             preprocess=["ALTER TABLE t ADD COLUMN b INTEGER DEFAULT 7"])
        assert gold_audit(data, tmp_path) == []

    def test_a_sound_gold_reports_nothing(self, tmp_path):
        from text2sql.benchmark import gold_audit
        assert gold_audit(self._dataset(tmp_path, "SELECT a FROM t"), tmp_path) == []


class TestUsageAccounting:
    """Per-instance cost must be that instance's own, not the run's running total."""

    def test_usage_is_a_delta_not_a_cumulative_snapshot(self):
        from text2sql.benchmark import _usage_delta
        d = _usage_delta({"num_calls": 5, "total_cost_usd": 0.10},
                         {"num_calls": 7, "total_cost_usd": 0.17})
        assert d["num_calls"] == 2
        assert abs(d["total_cost_usd"] - 0.07) < 1e-9

    async def test_the_streaming_path_records_usage_too(self, tmp_path):
        """Regression: the delta lived only in `_run_one`, so the CLI — which streams —
        reported every instance as null and the whole run as $0."""
        from types import SimpleNamespace

        from text2sql.benchmark import BenchmarkExample, BenchmarkRunner
        from text2sql.llm import LLMUsage
        from text2sql.pipeline.events import PipelineEvent, Stage, Status

        usage = LLMUsage()

        async def ask(_question):
            usage.num_calls += 1
            usage.prompt_tokens += 100
            usage.total_cost_usd += 0.02
            yield PipelineEvent(stage=Stage.COMPLETE, status=Status.COMPLETED)
            yield SimpleNamespace(sql="SELECT 1", results=None, metadata={})

        runner = BenchmarkRunner.__new__(BenchmarkRunner)
        runner.output_dir = tmp_path
        runner.engine = SimpleNamespace(
            llm=SimpleNamespace(usage=usage), ask=ask,
            settings=Settings(stop_after="sql_generation"))
        runner._switch_db = lambda example: None
        runner._save_report = lambda report: None

        example = BenchmarkExample(id="a", question="q", db_uri="sqlite://", gold_sql="")
        rows = [item async for item in runner.run([example])
                if isinstance(item, BenchmarkResult)]
        assert rows[0].usage["num_calls"] == 1
        assert rows[0].to_dict()["usage"]["total_cost_usd"] == 0.02

    def test_run_total_sums_per_instance_usage(self):
        """Regression: instances recorded the engine's cumulative counter (1, 2, 3 ... n), so
        summing them overstated the run by (n+1)/2 — measured at 7.2x on a 15-question run,
        turning $0.59 of Bedrock spend into a reported $4.21."""
        rows = [BenchmarkResult(id=str(i), question="q", predicted_sql="SELECT 1",
                                gold_sql="SELECT 1",
                                usage={"total_cost_usd": 0.04, "num_calls": 2})
                for i in range(15)]
        report = BenchmarkReport(results=rows)
        assert abs(report._usage("total_cost_usd") - 0.60) < 1e-9
        assert report._usage("num_calls") == 30

