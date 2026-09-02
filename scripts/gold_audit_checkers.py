#!/usr/bin/env python3
"""Report which repair checkers fire on the LiveSQLBench golds, or on a run's predictions.

A checker that asks for a rewrite on a gold query is wrong by construction, so the bar for a
new one is zero firings here. Run from the repository root:

    python3 scripts/gold_audit_checkers.py
    python3 scripts/gold_audit_checkers.py --predictions results/<run>/run.json
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from text2sql.benchmark import load_local_dataset  # noqa: E402
from text2sql.db import DatabaseConnection  # noqa: E402
from text2sql.pipeline.repair import run_checkers  # noqa: E402

DATA = pathlib.Path("data")
CASES = DATA / "livesqlbench_sqlite_gt_kg_testcases_20260601.jsonl"
QUESTIONS = DATA / "livesqlbench_data_sqlite.jsonl"


def _load(path: pathlib.Path) -> dict[str, dict]:
    return {r["instance_id"]: r for r in (json.loads(line) for line in path.open(encoding="utf-8"))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", type=pathlib.Path,
                    help="A run.json; audits its predicted SQL instead of the golds.")
    ap.add_argument("--instance", help="Audit one instance only.")
    args = ap.parse_args()

    cases, questions = _load(CASES), _load(QUESTIONS)
    # The question the pipeline passes, definitions block included - the checkers that read it
    # are written against that string, not against the bare `query` field.
    asked = {e.id: e.question for e in load_local_dataset(
        str(DATA), str(QUESTIONS), testcases_jsonl=str(CASES), use_knowledge=True)}
    verdicts: dict[str, bool] = {}
    if args.predictions:
        run = json.loads(args.predictions.read_text(encoding="utf-8"))
        sqls = {i["id"]: i["sql"]["predicted"] for i in run["instances"]}
        verdicts = {i["id"]: i["verdict"]["execution_match"] for i in run["instances"]}
    else:
        sqls = {k: "\n".join(v["sol_sql"]) for k, v in cases.items()}

    fired: dict[str, list[str]] = collections.defaultdict(list)
    connections: dict[str, DatabaseConnection] = {}
    for iid, sql in sorted(sqls.items()):
        if args.instance and iid != args.instance:
            continue
        db_name = questions[iid]["selected_database"]
        if db_name not in connections:
            uri = f"sqlite:///{DATA / db_name / f'{db_name}_template.sqlite'}"
            connections[db_name] = DatabaseConnection(uri)
        db = connections[db_name]
        question = asked.get(iid, questions[iid].get("query", ""))
        try:
            found = run_checkers(sql or "", db, question)
        except Exception as exc:  # a checker must never take the audit down
            found = [("AUDIT-ERROR", exc)]  # type: ignore[list-item]
        for name, issue in found:
            mark = "" if not verdicts else (" pass" if verdicts[iid] else " FAIL")
            fired[f"{name} ({getattr(issue, 'severity', 'error')})"].append(f"{iid}{mark}")

    label = "predictions" if args.predictions else "golds"
    print(f"{len(sqls)} {label}, {len(connections)} databases\n")
    for name, hits in sorted(fired.items(), key=lambda kv: -len(kv[1])):
        print(f"{len(hits):4}  {name}")
        print(f"      {' '.join(hits)}\n")
    if not fired:
        print("no checker fired")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
