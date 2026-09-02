"""The official LiveSQLBench SQLite evaluator, driven as shipped.

The scripts under ``live_sql_bench_sqlite/evaluation`` are vendored **unmodified** and
run as a subprocess, exactly the way the benchmark authors run them: the wrapper forks one
worker per instance against ephemeral copies of ``<db>_template.sqlite`` and writes a status
JSONL plus a text report. This module is the only place that knows about them: it shells
out, then reads their artifacts back into ``{instance_id: verdict}``.

Nothing here re-implements the metric: a disagreement between this and the published
leaderboard would be a bug in how we *invoke* the evaluator, not in how we score.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).parent / "live_sql_bench_sqlite" / "evaluation"
SCORER = "livesqlbench-base-lite sqlite (official evaluation scripts)"

_REPORT_LINE = re.compile(
    r"^Question_(?P<id>.+?): \((?P<passed>\d+)/(?P<total>\d+)\) test cases passed, "
    r"failed test cases: (?P<failed>.*)$")


def evaluate(
    predictions: Path | str,
    db_path: Path | str,
    *,
    mode: str = "pred",
    num_threads: int = 4,
) -> dict[str, dict[str, Any]]:
    """Score a predictions JSONL with the official wrapper.

    Args:
        predictions: JSONL of dataset records plus a ``pred_sqls`` list, sorted by
            ``instance_id``: the wrapper zips its sorted results onto the input
            positionally, so an unsorted file mis-assigns statuses.
        db_path: Dataset folder holding ``<db>/<db>_template.sqlite``.
        mode: ``pred`` scores ``pred_sqls``; ``gold`` scores ``sol_sql`` instead, a sanity
            run that must come out at 100%.
        num_threads: Wrapper worker threads, and the ephemeral copies per database.

    Returns:
        ``{instance_id: verdict}``; see :func:`_verdicts` for the verdict keys.

    Raises:
        RuntimeError: The evaluator exited non-zero.
    """
    predictions = Path(predictions).resolve()
    cmd = [sys.executable, "wrapper_evaluation_sqlite.py",
           "--jsonl_file", str(predictions),
           "--db_path", str(Path(db_path).resolve()),
           "--mode", mode,
           "--num_threads", str(num_threads)]
    # The scripts use flat imports and a relative path to the worker script, and open the
    # submission without an encoding, which is cp1252 on Windows. Their workers are spawned as
    # a bare `python3`, which Windows has none of: the Store alias answers and every worker
    # dies, so a copy of this interpreter goes beside it, where the venv still resolves.
    home = Path(sys.executable).parent
    if os.name == "nt" and not (shim := home / "python3.exe").exists():
        shutil.copy2(sys.executable, shim)
    env = {**os.environ, "PYTHONUTF8": "1",
           "PATH": f"{home}{os.pathsep}{os.environ['PATH']}"}
    logger.info("Running official evaluator (%s) on %s", mode, predictions)
    proc = subprocess.run(cmd, cwd=EVAL_DIR, env=env, capture_output=True, text=True, check=False)
    logger.debug("Official evaluator output:\n%s", proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Official evaluator failed (exit {proc.returncode}):\n"
            f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    return _verdicts(predictions)


def _verdicts(predictions: Path) -> dict[str, dict[str, Any]]:
    """Read the wrapper's two artifacts back into one verdict per instance.

    Args:
        predictions: The submission file whose artifacts sit beside it.

    Returns:
        ``{instance_id: verdict}``, carrying status and message from the status JSONL and
        the test-case counts and phase flags from the text report, which is the only place
        they survive.
    """
    base = predictions.with_suffix("")
    detail = _parse_report(base.with_name(f"{base.name}_simple_report.txt"))

    verdicts: dict[str, dict[str, Any]] = {}
    status_file = base.with_name(f"{base.name}_simple_output_with_status.jsonl")
    with open(status_file, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            instance_id = str(row.get("instance_id", ""))
            verdicts[instance_id] = {
                "status": row.get("status", "failed"),
                "error_message": row.get("error_message") or "",
                "passed_test_cases": 0, "total_test_cases": 0,
                "execution_error": False, "timeout_error": False, "assertion_error": False,
                **detail.get(instance_id, {}),
            }
    return verdicts


def _parse_report(report: Path) -> dict[str, dict[str, Any]]:
    """Parse the wrapper's text report.

    Args:
        report: The report file.

    Returns:
        ``{instance_id: {counts and phase flags}}``; empty when the report is missing.
    """
    if not report.exists():
        logger.warning("Official report missing: %s", report)
        return {}

    out: dict[str, dict[str, Any]] = {}
    for line in report.read_text(encoding="utf-8").splitlines():
        if not (m := _REPORT_LINE.match(line)):
            continue
        out[m["id"]] = {
            "passed_test_cases": int(m["passed"]),
            "total_test_cases": int(m["total"]),
            "execution_error": "Execution Error" in line,
            "timeout_error": "Timeout Error" in line,
            "assertion_error": "Assertion Error" in line,
        }
    return out


def summarize(verdicts: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate verdicts into the official summary counts, as the wrapper prints them.

    Args:
        verdicts: The per-instance verdicts.

    Returns:
        Totals, pass count, per-phase error counts and accuracy.
    """
    rows = list(verdicts)
    passed = sum(1 for v in rows if v.get("status") == "success")
    return {
        "total": len(rows),
        "passed": passed,
        "execution_errors": sum(1 for v in rows if v.get("execution_error")),
        "timeout_errors": sum(1 for v in rows if v.get("timeout_error")),
        "assertion_errors": sum(1 for v in rows if v.get("assertion_error")),
        "accuracy": round(passed / len(rows), 4) if rows else 0.0,
    }
