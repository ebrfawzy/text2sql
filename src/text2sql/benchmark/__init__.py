"""Benchmark harness for Text-to-SQL evaluation.

Loads datasets from HuggingFace (BIRD MiniDev) or a local LiveSQLBench release, runs them
through the pipeline, and scores them with the official evaluator.
"""

from __future__ import annotations

import json
import logging
import shutil
import sqlite3
import tempfile
import time
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import sqlparse
from sqlalchemy.engine import make_url

from text2sql.benchmark import official
from text2sql.benchmark.results import (
    BenchmarkReport,
    BenchmarkResult,
    apply_verdicts,
    linked_fields,
    linking_metrics,
    write_predictions,
)
from text2sql.db import DatabaseConnection, split_statements
from text2sql.pipeline.events import PipelineEvent, TokenDelta, collect_result
from text2sql.profiler.knowledge import DatabaseKnowledge
from text2sql.prompts.manager import PromptManager

logger = logging.getLogger(__name__)

# Reported stage name -> tracer step name. One map, so `to_dict` and `from_dict` cannot
# disagree and re-scoring a run cannot quietly drop what it did not understand.










def _usage_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Difference two usage snapshots.

    Args:
        before: The counter before the block.
        after: The counter after it.

    Returns:
        What the block itself cost.
    """
    return {k: (v - before.get(k, 0)) if isinstance(v, (int, float)) else v
            for k, v in after.items()}


@contextmanager
def _measured(llm: Any, take: Callable[[dict[str, Any]], None]) -> Generator[None, None, None]:
    """Measure one block's own LLM usage.

    Both run paths must go through it, else a run reports zero cost or the engine's
    running total.

    Args:
        llm: The client whose cumulative counter is sampled.
        take: Receives the block's usage on exit.

    Yields:
        None.
    """
    before = llm.usage.summary()
    try:
        yield
    finally:
        take(_usage_delta(before, llm.usage.summary()))




@dataclass
class BenchmarkExample:
    """A single benchmark example, as loaded from a dataset."""

    id: str
    question: str
    db_uri: str
    gold_sql: str = ""
    db_name: str = ""
    difficulty: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
    record: dict[str, Any] = field(default_factory=dict)  # source row, passed through to file 1




class BenchmarkRunner:
    """Runs benchmark examples through the Text2SQL pipeline.

    ``run()`` is a streaming async generator over ``ask()``: it yields per-example markers
    and events, then a final :class:`BenchmarkReport`.

    Usage::

        runner = BenchmarkRunner(engine)
        async for item in runner.run(examples):
            if isinstance(item, BenchmarkReport):
                print(item.to_dict())
    """

    def __init__(
        self,
        engine: Any,  # Text2SQL instance (import avoided for circular deps)
        output_dir: str = "results",
    ) -> None:
        self.engine = engine
        self._prepared: tempfile.TemporaryDirectory[str] | None = None
        # One directory per run, so ablation arms and reruns never overwrite each other.
        self.output_dir = Path(output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def run(
        self,
        examples: list[BenchmarkExample] | Iterator[BenchmarkExample],
        *,
        max_examples: int | None = None,
    ):
        """Run benchmark examples as a stream.

        Args:
            examples: The examples to run.
            max_examples: Stop after this many.

        Yields:
            Per example, the :class:`BenchmarkExample` as a start marker, every ``ask()``
            event passed straight through, then its :class:`BenchmarkResult`; finally the
            aggregate :class:`BenchmarkReport`.
        """
        report = BenchmarkReport()

        for i, example in enumerate(examples):
            if max_examples and i >= max_examples:
                break

            logger.info("Benchmark %d: %s", i + 1, example.question)
            yield example  # start marker
            start = time.time()
            final = None
            usage: dict[str, Any] = {}
            with _measured(self.engine.llm, usage.update):
                try:
                    self._switch_db(example)
                    async for item in self.engine.ask(example.question):
                        if isinstance(item, (PipelineEvent, TokenDelta)):
                            yield item  # pass-through (same shape as /ask)
                        else:
                            final = item
                    result = self._evaluate(example, final, time.time() - start)
                except Exception as e:
                    result = BenchmarkResult(
                        id=example.id,
                        question=example.question,
                        predicted_sql="",
                        gold_sql=example.gold_sql,
                        pipeline_error=str(e),
                        latency_seconds=time.time() - start,
                        record=example.record,
                    )
            result.usage = usage
            report.results.append(result)
            yield result

        # The official evaluator scores a whole predictions file in one batch, so it runs
        # once, here. A run halted before SQL exists has nothing to score.
        if not self.engine.settings.stop_after:
            self.score(report.results)
        report.compute()
        self._save_report(report)
        yield report

    def score(self, results: list[BenchmarkResult]) -> None:
        """Score results in place with the official evaluator, main pass then Oracle@N.

        Args:
            results: The results to score.
        """
        db_path = self.engine.settings.benchmark_dataset_folder
        apply_verdicts(results, official.evaluate(
            write_predictions(results, self.output_dir / "predictions.jsonl"), db_path))

        # Oracle@N: the same evaluator per candidate slot, so selection loss is measurable.
        # Database-only, no LLM cost, but pointless when voting is off.
        slots = max((len(r.candidates) for r in results), default=0)
        if slots < 2:  # single candidate: the selected SQL *is* the only candidate
            for r in results:
                r.candidate_correct = [r.execution_match is True]
            return
        for r in results:
            r.candidate_correct = []
        for i in range(slots):
            eligible = [r for r in results if len(r.candidates) > i]
            verdicts = official.evaluate(
                write_predictions(eligible, self.output_dir / f"predictions_cand{i}.jsonl",
                                  lambda r: r.candidates[i]), db_path)
            for r in eligible:
                r.candidate_correct.append(verdicts.get(r.id, {}).get("status") == "success")

    def _prepare(self, example: BenchmarkExample) -> str:
        """Apply the example's ``preprocess_sql``, as the evaluator does.

        Without it the pipeline explores, repairs and selects against a different schema
        than the one grading its answer.

        Args:
            example: The example to prepare.

        Returns:
            A URI for the prepared copy, whose filename is kept so the profile cache still
            hits; the original URI when there is nothing to apply.
        """
        if self._prepared:
            self._prepared.cleanup()
            self._prepared = None
        setup = (example.record or {}).get("preprocess_sql") or []
        source = make_url(example.db_uri).database if example.db_uri.startswith("sqlite") else None
        if not setup:
            return example.db_uri
        if not source:
            logger.warning("%s: cannot apply preprocess_sql to %s", example.id, example.db_uri)
            return example.db_uri
        self._prepared = tempfile.TemporaryDirectory()
        copy = Path(self._prepared.name) / Path(source).name
        shutil.copy(source, copy)
        conn = sqlite3.connect(copy)
        with conn:
            for statement in setup:
                try:
                    conn.execute(statement)
                except sqlite3.Error as e:
                    # What the official `run_preprocessing` does: the error is caught and
                    # the flag discarded, so the evaluator scores the instance anyway.
                    logger.warning("%s: preprocess_sql failed, continuing as the official "
                                   "evaluator does: %s", example.id, e)
                    break
        conn.close()
        return f"sqlite:///{copy}"

    def _switch_db(self, example: BenchmarkExample) -> None:
        """Point the engine at this example's database, discarding the previous profile.

        Args:
            example: The example to switch to.
        """
        uri = self._prepare(example)
        if self.engine.settings.db_uri != uri:
            self.engine.db.close()
            self.engine.settings.db_uri = uri
            self.engine.db = DatabaseConnection(
                uri, connect_args=self.engine.settings.athena_connect_args(uri),
            )
            self.engine._profile = None
            self.engine._summary = None
            self.engine._knowledge = None  # every database has its own domain terms

    async def _run_one(self, example: BenchmarkExample) -> BenchmarkResult:
        """Run one example without streaming.

        Args:
            example: The example to run.

        Returns:
            Its result.
        """
        start = time.time()
        self._switch_db(example)
        usage: dict[str, Any] = {}
        with _measured(self.engine.llm, usage.update):
            try:
                result = await collect_result(self.engine.ask(example.question))
                row = self._evaluate(example, result, time.time() - start)
            except Exception as e:
                row = BenchmarkResult(
                    id=example.id, question=example.question, predicted_sql="",
                    gold_sql=example.gold_sql, pipeline_error=str(e),
                    latency_seconds=time.time() - start, record=example.record)
        row.usage = usage
        return row

    def _evaluate(self, example: BenchmarkExample, result: Any, latency: float) -> BenchmarkResult:
        """Turn one pipeline result into a scoreable row.

        Execution correctness is not decided here; :meth:`score` settles that for the whole
        run at once.

        Args:
            example: The example that was run.
            result: The pipeline's result, or None.
            latency: Wall-clock seconds it took.

        Returns:
            What the pipeline produced, plus the schema-linking comparison against gold.
        """
        if result is None:
            return BenchmarkResult(
                id=example.id, question=example.question, predicted_sql="",
                gold_sql=example.gold_sql, pipeline_error="Pipeline returned no result",
                latency_seconds=latency, record=example.record)

        rows = result.results or []
        # Gold rows are for eyeballing a mismatch side by side, never for scoring.
        # `db.execute()` always rolls back, so a Management task's gold DML is safe here.
        gold_rows, _ = (self.engine.db.execute_safe(example.gold_sql) if example.gold_sql
                        else (None, None))
        candidates: list[str] = next(
            (s["outputs"].get("candidates", []) for s in result.trace.get("steps", [])
             if s.get("step") == "sql_generation"), [])

        return BenchmarkResult(
            id=example.id, question=example.question, predicted_sql=result.sql,
            gold_sql=example.gold_sql, pred_sqls=split_statements(result.sql),
            pipeline_error=result.error, latency_seconds=latency, trace=result.trace,
            predicted_results=rows[:20], gold_results=(gold_rows or [])[:20],
            candidates=candidates,
            linking=linking_metrics(example.gold_sql, result.trace) if example.gold_sql else None,
            empty_result=not rows,
            category=example.record.get("category", ""), difficulty=example.difficulty,
            db_name=example.db_name, record=example.record)

    def _config(self) -> dict[str, Any]:
        """Collect the run's effective settings.

        Returns:
            Every setting except the credentials.
        """
        settings = getattr(getattr(self, "engine", None), "settings", None)
        return {k: v for k, v in (settings.model_dump() if settings else {}).items()
                if not k.startswith(("bedrock_", "athena_"))}

    def _save_report(self, report: BenchmarkReport) -> None:
        """Write the artifacts the official harness does not report.

        ``predictions.jsonl``, the official submission file, is written by :meth:`score`.

        Args:
            report: The finished report.
        """
        run = self.output_dir / "run.json"
        with open(run, "w", encoding="utf-8") as f:
            json.dump({
                "meta": {"timestamp": datetime.now().isoformat(timespec="seconds"),
                         "scorer": official.SCORER,
                         "model": self.engine.settings.model,
                         "generation_mode": self.engine.settings.generation_mode,
                         "stop_after": self.engine.settings.stop_after,
                         "config": self._config()},
                "summary": report.to_dict(),
                "breakdown": report.breakdowns(),
                "instances": [r.to_dict() for r in report.results],
            }, f, indent=2, default=str)
        logger.info("Report saved: %s", run)

        # One flat row per instance: what a linking ablation gets analysed from.
        if measured := [r for r in report.results if r.linking]:
            linking = self.output_dir / "linking.jsonl"
            with open(linking, "w", encoding="utf-8") as f:
                for r in measured:
                    f.write(json.dumps({
                        "id": r.id, "db": r.db_name, "category": r.category,
                        "difficulty": r.difficulty, "question": r.question,
                        "gold_sql": r.gold_sql, "linked": linked_fields(r.trace), **(r.linking or {}),
                    }, ensure_ascii=False, default=str) + "\n")
            logger.info("Linking scorecard saved: %s", linking)


def load_instances(data_jsonl: str | Path, testcases_jsonl: str | Path | None = None,
                  instance_id: str | None = None) -> list[dict[str, Any]]:
    """Load the dataset's rows, merged from LiveSQLBench's two release files.

    The public release ships questions, databases and conditions; the ground truth arrives
    separately and is applied over it by instance id.

    Args:
        data_jsonl: The public release file.
        testcases_jsonl: The ground-truth file.
        instance_id: Comma-separated id prefixes, so ``credit`` selects a whole database,
            ``credit_1`` selects both ``credit_1`` and ``credit_10``, and ``a,b`` selects
            the union.

    Returns:
        The merged rows.
    """
    def rows(path: str | Path) -> list[dict[str, Any]]:
        """Read one JSONL file.

        Args:
            path: The file to read.

        Returns:
            Its rows.
        """
        return [json.loads(line) for line
                in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]

    truth = {r["instance_id"]: r for r in rows(testcases_jsonl)} if testcases_jsonl else {}
    merged = [{**r, **truth.get(r.get("instance_id", ""), {})} for r in rows(data_jsonl)]
    wanted = tuple(s for p in (instance_id or "").split(",") if (s := p.strip()))
    return [r for r in merged
            if not wanted or str(r.get("instance_id", "")).startswith(wanted)]


def gold_check(
    data_jsonl: str | Path,
    dataset_folder: str | Path,
    output_dir: str | Path = "results/gold_check",
    instance_id: str | None = None,
    testcases_jsonl: str | Path | None = None,
) -> dict[str, Any]:
    """Score the dataset's own ``sol_sql`` as if the pipeline had predicted it.

    The fairness control on the scorer: the submission's ``pred_sqls`` is ``sol_sql``, and
    it goes through the same :func:`official.evaluate` path a real run uses, so anything
    short of 100% is a property of the dataset or the environment rather than of a model.

    Args:
        data_jsonl: The public release file.
        dataset_folder: Root directory holding the database folders.
        output_dir: Where the submission and summary are written.
        instance_id: Filters by prefix.
        testcases_jsonl: The ground-truth file.

    Returns:
        The official summary, plus a ``failures`` list of ``{id, error}``.

    Raises:
        ValueError: No instance matched.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_instances(data_jsonl, testcases_jsonl, instance_id)
    if not rows:
        raise ValueError(f"No instances in {data_jsonl} matching {instance_id!r}")

    # Sorted by instance_id: the official wrapper zips its results on positionally.
    predictions = output_dir / "predictions.jsonl"
    with open(predictions, "w", encoding="utf-8") as f:
        for row in sorted(rows, key=lambda r: str(r.get("instance_id", ""))):
            f.write(json.dumps({**row, "pred_sqls": row.get("sol_sql") or []},
                               ensure_ascii=False, default=str) + "\n")

    verdicts = official.evaluate(predictions, dataset_folder, mode="gold")
    summary = official.summarize(verdicts.values())
    summary["failures"] = [
        {"id": iid, "error": (v.get("error_message") or "").strip()}
        for iid, v in sorted(verdicts.items()) if v.get("status") != "success"]
    (output_dir / "gold_check.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Gold check: %s/%s passed", summary["passed"], summary["total"])
    return summary


def gold_audit(data_jsonl: str | Path, dataset_folder: str | Path,
               instance_id: str | None = None,
               testcases_jsonl: str | Path | None = None) -> list[dict[str, Any]]:
    """Find the instances whose own gold SQL answers a degenerate question.

    :func:`gold_check` cannot catch these: a gold that returns garbage still matches itself,
    so the run scores 100% and the instance is unwinnable. Write golds are checked for
    errors only, since they return nothing by nature, but a setup that cannot run is a
    defect for any gold.

    Args:
        data_jsonl: The public release file.
        dataset_folder: Root directory holding the database folders.
        instance_id: Filters by prefix.
        testcases_jsonl: The ground-truth file.

    Returns:
        One ``{id, defect, detail}`` per finding.
    """
    findings: list[dict[str, Any]] = []
    for row in load_instances(data_jsonl, testcases_jsonl, instance_id):
        iid = str(row.get("instance_id", ""))
        sql = "\n".join(row.get("sol_sql") or []).strip().rstrip(";")
        if not sql:
            continue
        db = row.get("selected_database", "")
        path = Path(dataset_folder) / db / f"{db}_template.sqlite"
        # One call, so the rollback covers the setup too.
        script = [s.strip().rstrip(";") for s in row.get("preprocess_sql") or []] + [sql]
        with DatabaseConnection(f"sqlite:///{path}") as conn:
            results, error = conn.execute_safe("; ".join(script))
        if error:
            findings.append({"id": iid, "defect": "errors", "detail": error[:200]})
        elif row.get("category") == "Management":
            continue  # nothing to return is the correct outcome for a write
        elif not results:
            findings.append({"id": iid, "defect": "empty", "detail": "gold returns no rows"})
        else:
            findings += [{"id": iid, "defect": "all_null_column", "detail": column}
                         for column in results[0]
                         if all(r.get(column) is None for r in results)]
    logger.info("Gold audit: %d degenerate instance(s)", len(findings))
    return findings


def rescore(run_dir: str | Path, db_path: str | None = None, mode: str = "pred") -> BenchmarkReport:
    """Re-run the official evaluator over a finished run and rewrite its ``run.json``.

    The run's own artifacts are enough to rebuild every result, so this costs only database
    time. ``candidate_correct`` is carried over untouched: Oracle@N is not recomputed.

    Args:
        run_dir: A finished results directory.
        db_path: Dataset folder; defaults to the one recorded in ``run.json``.
        mode: ``pred`` scores the predictions, ``gold`` scores the gold SQL.

    Returns:
        The fresh report.

    Raises:
        ValueError: No dataset folder was recorded or passed.
    """
    run_dir = Path(run_dir)
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    predictions = run_dir / "predictions.jsonl"
    records = {str(r.get("instance_id", "")): r for r in
               (json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines())}

    results = [BenchmarkResult.from_dict(d, records.get(d["id"], {})) for d in run["instances"]]
    db_path = db_path or run["meta"]["config"].get("benchmark_dataset_folder")
    if not db_path:
        raise ValueError("No dataset folder in run.json meta; pass one explicitly")
    apply_verdicts(results, official.evaluate(
        write_predictions(results, predictions), db_path, mode=mode))

    report = BenchmarkReport(results=results)
    report.compute()
    run["meta"] |= {"scorer": official.SCORER, "eval_mode": mode,
                    "rescored_at": datetime.now().isoformat(timespec="seconds")}
    run |= {"summary": report.to_dict(), "breakdown": report.breakdowns(),
            "instances": [r.to_dict() for r in results]}
    (run_dir / "run.json").write_text(json.dumps(run, indent=2, default=str), encoding="utf-8")
    logger.info("Rescored %s (%s): %s", run_dir, mode, report.execution_accuracy)
    return report


def load_bird_minidev(db_root: str, split: str = "minidev") -> list[BenchmarkExample]:
    """Load examples from BIRD MiniDev dataset.

    Args:
        db_root: Root directory containing database folders.
        split: Dataset split (default "minidev").

    Returns:
        The examples.

    Raises:
        ImportError: The optional ``datasets`` package is not installed.
        FileNotFoundError: A database named by the dataset is missing.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets package required for benchmarks. Install with: uv sync --extra benchmark")

    ds = load_dataset("birdsql/bird-minidev", split=split)
    examples = []

    for row in ds:
        db_name = row.get("db_id", "")
        db_path = Path(db_root) / db_name / f"{db_name}.sqlite"
        if not db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")
        db_uri = f"sqlite:///{db_path}"

        examples.append(
            BenchmarkExample(
                id=str(row.get("question_id", len(examples))),
                question=row.get("question", ""),
                db_uri=db_uri,
                gold_sql=row.get("SQL", row.get("sql", "")),
                db_name=db_name,
                difficulty=row.get("difficulty", ""),
                record=row,
            )
        )

    logger.info("Loaded %d examples from BIRD MiniDev", len(examples))
    return examples


def _table_names(db_path: Path) -> set[str]:
    """The database's lowercased table names, which tell a KB row count from a magnitude."""
    conn = sqlite3.connect(db_path)
    try:
        return {t.lower() for (t,) in
                conn.execute("select name from sqlite_master where type='table'")}
    finally:
        conn.close()


def load_local_dataset(
    dataset_folder: str,
    data_jsonl: str,
    instance_id: str | None = None,
    use_knowledge: bool = False,
    prompts: PromptManager | None = None,
    testcases_jsonl: str | None = None,
) -> list[BenchmarkExample]:
    """Load examples from a local LiveSQLBench JSONL file.

    Args:
        dataset_folder: Root directory containing database folders.
        data_jsonl: Path to the LiveSQLBench release JSONL.
        instance_id: Optional ID to filter by.
        use_knowledge: Whether to append the instance's external knowledge to the
            question, rendered by ``question_knowledge.j2``.
        prompts: Template manager to render with; defaults to the bundled templates.
        testcases_jsonl: The ground-truth file.

    Returns:
        The examples.

    Raises:
        FileNotFoundError: A database named by the dataset is missing.
    """
    examples = []
    dataset_path = Path(dataset_folder)
    prompts = prompts or PromptManager()
    kb_cache: dict[str, DatabaseKnowledge] = {}

    for row in load_instances(data_jsonl, testcases_jsonl, instance_id):
        db_name = row.get("selected_database", "")

        for suffix in ("_template_copy.sqlite", "_template.sqlite", ".sqlite"):
            db_path = dataset_path / db_name / f"{db_name}{suffix}"
            if db_path.exists():
                break
        else:
            raise FileNotFoundError(
                f"Database not found for '{db_name}' in {dataset_path / db_name}")
        db_uri = f"sqlite:///{db_path}"

        question = row.get("query", "")

        if use_knowledge and row.get("external_knowledge"):
            if db_name not in kb_cache:
                kb_path = dataset_path / db_name / f"{db_name}_kb.jsonl"
                kb_cache[db_name] = DatabaseKnowledge.from_jsonl(
                    kb_path.read_text(encoding="utf-8"),
                    _table_names(db_path)) if kb_path.exists() else DatabaseKnowledge()
            base = kb_cache[db_name]
            # A term the instance names is useless without the terms its definition names.
            if entries := base.with_children(
                    [e for i in row["external_knowledge"] if (e := base.entries.get(i))]):
                question = prompts.render("question_knowledge",
                                          question=question, entries=entries)

        sol_sqls = row.get("sol_sql", [])
        gold_sql = sqlparse.format(
            "\n".join(sol_sqls), reindent=True, keyword_case="upper") if sol_sqls else ""

        examples.append(
            BenchmarkExample(
                id=str(row.get("instance_id", len(examples))),
                question=question,
                db_uri=db_uri,
                gold_sql=gold_sql,
                db_name=db_name,
                difficulty=row.get("difficulty_tier", ""),
                extra={"category": row.get("category", "")},
                record=row,  # passed through verbatim to the submission file
            )
        )

    logger.info("Loaded %d examples from local dataset (instance_id=%s)", len(
        examples), instance_id or "all")
    return examples


def load_examples(settings: Any) -> list[BenchmarkExample]:
    """Load benchmark examples from the dataset configured in settings.

    Shared by the CLI, ``/benchmark`` and ``/benchmark/preview``, so example loading lives
    in one place.

    Args:
        settings: The effective settings.

    Returns:
        The examples, from the local LiveSQLBench loader when both
        ``benchmark_data_jsonl`` and ``benchmark_dataset_folder`` are set, else BIRD
        MiniDev.
    """
    # `external_knowledge` is part of the *question*, so a run with generated schema
    # metadata still asks the question the benchmark defines.
    use_knowledge = settings.benchmark_use_knowledge

    if settings.benchmark_data_jsonl and settings.benchmark_dataset_folder:
        return load_local_dataset(
            dataset_folder=settings.benchmark_dataset_folder,
            data_jsonl=settings.benchmark_data_jsonl,
            testcases_jsonl=settings.benchmark_testcases_jsonl,
            instance_id=settings.benchmark_instance_id,
            use_knowledge=use_knowledge,
            prompts=PromptManager(template_dir=settings.prompt_template_dir,
                                  version=settings.prompt_version),
        )
    return load_bird_minidev(settings.benchmark_dataset_folder or ".")
