#!/usr/bin/env python3
"""Profile every LiveSQLBench database in ``data/`` with one config, one at a time.

Each database is a separate ``text2sql profile`` process, so one failure does not lose the
rest. Run from the repository root:

    python scripts/profile_all.py --config configs/deepseek4flash_agent.yaml
    python scripts/profile_all.py --config configs/deepseek4flash_agent.yaml --kb-only
    python scripts/profile_all.py --config configs/... --only credit mental

    uv self update
    uv sync
    python scripts/profile_all.py --config configs/deepseek4flash_agent.yaml
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess

DATA = pathlib.Path("data")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="YAML config passed to every run.")
    ap.add_argument("--kb-only", action="store_true",
                    help="Re-derive only the knowledge base, reusing the cached statistics.")
    ap.add_argument("--only", nargs="+", metavar="DB",
                    help="Database names to restrict the run to. Default: all of them.")
    args = ap.parse_args()

    databases = sorted(p for p in DATA.glob("*/*_template.sqlite"))
    if args.only:
        wanted = set(args.only)
        databases = [p for p in databases if p.parent.name in wanted]
        if missing := wanted - {p.parent.name for p in databases}:
            ap.error(f"no such database: {', '.join(sorted(missing))}")
    if not databases:
        ap.error(f"no *_template.sqlite under {DATA}/ — run from the repository root")

    failed = []
    for i, path in enumerate(databases, 1):
        name = path.parent.name
        print(f"\n[{i}/{len(databases)}] {name}", flush=True)
        command = ["uv", "run", "text2sql", "profile", f"sqlite:///{path.as_posix()}",
                   "--config", args.config] + (["--kb-only"] if args.kb_only else [])
        if subprocess.run(command).returncode:
            failed.append(name)

    print(f"\nProfiled {len(databases) - len(failed)}/{len(databases)}")
    if failed:
        print(f"Failed: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
