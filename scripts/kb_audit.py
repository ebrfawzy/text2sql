#!/usr/bin/env python3
"""Score a generated knowledge base against the shape the corpus questions actually use.

The generated base is a substitute for a curated one, so what matters is whether it names the
terms questions ask for and relates the columns a query has to join — not how many entries it
has. Run from the repository root:

    python3 scripts/kb_audit.py .cache/profiles-deepseek4flash
    python3 scripts/kb_audit.py --shipped          # the corpus's own base, as the target
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import sqlite3
import sys

import sqlglot
from sqlglot import exp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from typing import Any  # noqa: E402

from text2sql.profiler.knowledge import DatabaseKnowledge  # noqa: E402

DATA = pathlib.Path("data")
WORD = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")


def columns_by_table(db: str) -> dict[str, set[str]]:
    """Map every table of a database to its lowercased column names."""
    conn = sqlite3.connect(DATA / db / f"{db}_template.sqlite")
    return {t: {r[1].lower() for r in conn.execute(f"PRAGMA table_info({t})")}
            for (t,) in conn.execute("select name from sqlite_master where type='table'")}


def question_terms() -> dict[str, set[str]]:
    """The terms each database's questions rely on: the ids they cite, plus every term those
    definitions depend on, since a cited definition may name another term the question omits."""
    dbs = {json.loads(line)["instance_id"]: json.loads(line)["selected_database"]
           for line in (DATA / "livesqlbench_data_sqlite.jsonl").open(encoding="utf-8")}
    bases = {db: {e.id: e for e in DatabaseKnowledge.from_jsonl(
        (DATA / db / f"{db}_kb.jsonl").read_text(encoding="utf-8")).entries.values()} for db in set(dbs.values())}
    out: dict[str, set[str]] = collections.defaultdict(set)
    for line in (DATA / "livesqlbench_sqlite_gt_kg_testcases_20260601.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        db = dbs[row["instance_id"]]
        need, seen = list(row.get("external_knowledge") or []), set()
        while need:
            if (e := bases[db].get(need.pop())) is None:
                continue
            out[db].add(bare(e.knowledge))
            need += [c for c in e.children if c not in seen and not seen.add(c)]
    return out


def unsound(definition: str, profile: dict[str, Any]) -> list[str]:
    """Every comparison in a definition that the profiled values contradict.

    Args:
        definition: The definition to check.
        profile: ``{column: (values, is_numeric)}`` from the cached statistics.

    Returns:
        One message per equality literal that column never stores, or number compared to a
        text column. Only equality is checked: a range bound or a GLOB pattern is sound
        without ever being a stored value.
    """
    try:
        parsed = sqlglot.maybe_parse(definition)
    except Exception:
        return []
    out = []
    for cmp_ in parsed.find_all(exp.Predicate):
        if isinstance(cmp_, exp.Like):
            continue
        for col, other in ((cmp_.this, cmp_.args.get("expression")), (cmp_.args.get("expression"), cmp_.this)):
            if not isinstance(col, exp.Column) or col.name.lower() not in profile:
                continue
            values, numeric = profile[col.name.lower()]
            lits = [x for x in cmp_.args.get("expressions", [])] if isinstance(cmp_, exp.In) else [other]
            equality = isinstance(cmp_, exp.EQ | exp.NEQ | exp.In)
            for node in (x for x in lits if isinstance(x, exp.Literal)):
                if node.is_string and equality and node.this not in values:
                    out.append(f"'{node.this}' is not a value of {col.name}")
                elif not node.is_string and not numeric:
                    out.append(f"{node.this} compared to the text column {col.name}")
    return out


def profiled(path: pathlib.Path) -> dict[str, Any]:
    """Map each profiled column to its frequent values and whether they are all numeric."""
    out: dict[str, Any] = {}
    for key, stats in json.loads(path.read_text(encoding="utf-8"))["columns"].items():
        got = [str(v["value"]) for v in stats.get("top_k_values", [])]
        column = key.split("|")[-1].lower()
        values, numeric = out.get(column, (set(), True))
        out[column] = (values | set(got),
                       numeric and all(re.fullmatch(r"-?\d+(\.\d+)?", g) for g in got if g))
    return out


def bare(term: str) -> str:
    """A term's name with its acronym and punctuation stripped, for comparison."""
    return re.sub(r"[^a-z0-9]", "", term.split("(")[0].lower())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cache_dir", nargs="?", help="A profile cache directory to audit.")
    ap.add_argument("--shipped", action="store_true",
                    help="Audit the corpus's own <db>_kb.jsonl instead, as the target shape.")
    args = ap.parse_args()
    if not args.cache_dir and not args.shipped:
        ap.error("give a cache directory or --shipped")

    asked = question_terms()
    tally: collections.Counter[str] = collections.Counter()
    named: set[str] = set()
    for db_dir in sorted(p.name for p in DATA.iterdir() if (p / f"{p.name}_kb.jsonl").exists()):
        stats: dict[str, Any] = {}
        if args.shipped:
            kb = DatabaseKnowledge.from_jsonl((DATA / db_dir / f"{db_dir}_kb.jsonl").read_text(encoding="utf-8"))
        else:
            cached = pathlib.Path(args.cache_dir) / f"{db_dir}_template_kb.json"
            if not cached.exists():
                continue
            kb = DatabaseKnowledge.from_flat(json.loads(cached.read_text(encoding="utf-8")))
            stats = profiled(cached.with_name(f"{db_dir}_template_profile.json"))
        cols = columns_by_table(db_dir)
        for e in kb.entries.values():
            words = {w.lower() for w in WORD.findall(e.definition)}
            touched = [t for t, c in cols.items() if words & c]
            lhs = e.definition.split("=")[0].strip().lower() if "=" in e.definition else ""
            tally["entries"] += 1
            tally["cross_table"] += len(touched) > 1
            tally["single_table"] += len(touched) == 1
            tally["no_columns"] += not touched
            tally["identity"] += bool(WORD.fullmatch(lhs)) and any(lhs in c for c in cols.values())
            tally["refuted"] += e.refuted
            tally["composite"] += bool(e.children) and not touched
            tally["unsound"] += bool(stats) and bool(unsound(e.definition, stats))
            if bare(e.knowledge) in asked[db_dir]:
                named.add(f"{db_dir}:{bare(e.knowledge)}")

    total = tally["entries"] or 1
    want = sum(len(v) for v in asked.values())
    print(f"{'shipped' if args.shipped else args.cache_dir}: {tally['entries']} entries\n")
    for label, k in (("cross-table", "cross_table"), ("single-table", "single_table"),
                     ("no columns (term-only)", "no_columns"),
                     ("stored-result identity", "identity"),
                     ("term composite", "composite"), ("refuted by the data", "refuted"),
                     ("unsound against the data", "unsound")):
        print(f"  {label:24} {tally[k]:5}  {tally[k]/total:6.1%}")
    print(f"\n  question terms named      {len(named):5}  of {want} ({len(named)/want:.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
