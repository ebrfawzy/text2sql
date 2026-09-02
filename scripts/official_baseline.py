#!/usr/bin/env python3
"""Run the vendored LiveSQLBench-Lite baseline end to end: prompt, generate, post-process, score.

The three stages the release's README describes are its own scripts, run unmodified as
subprocesses; only the generation step in the middle is ours, and it is one OpenAI-compatible
chat call per instance with no pipeline around it. Everything lands in one output directory,
and a rerun asks only for the answers that are missing, so an interrupted run resumes.

    uv run python scripts/official_baseline.py --model deepseek-v4-flash
    uv run python scripts/official_baseline.py --limit 5 --out results/smoke
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "src" / "text2sql" / "benchmark" / "live_sql_bench_sqlite"


def run_vendored(script: str, **args: str) -> None:
    """Run one of the release's scripts from its own directory, as its README does.

    Args:
        script: Path of the script relative to the vendored root.
        **args: Command-line flags, passed as ``--name value``.

    Raises:
        RuntimeError: The script exited non-zero.
    """
    command = [sys.executable, Path(script).name]
    for name, value in args.items():
        command += [f"--{name}", value]
    # PYTHONPATH reaches the sibling `prompt` package, PATH gives the evaluator's bare
    # `python3` workers this interpreter, and PYTHONUTF8 covers the reads and writes the
    # scripts make without an encoding, which are cp1252 on Windows.
    env = {**os.environ, "PYTHONPATH": str(BENCH), "PYTHONUTF8": "1",
           "PATH": f"{interpreter_dir()}{os.pathsep}{os.environ['PATH']}"}
    if subprocess.run(command, cwd=(BENCH / script).parent, env=env).returncode:
        raise RuntimeError(f"{script} failed")


def interpreter_dir() -> Path:
    """Directory to put first on PATH so the evaluator's hardcoded ``python3`` resolves here.

    Windows ships no ``python3.exe`` and the bare name reaches the Microsoft Store alias, which
    kills every worker, so a copy is made beside the interpreter, where the venv still resolves.

    Returns:
        The interpreter's own directory.
    """
    home = Path(sys.executable).parent
    if os.name == "nt" and not (shim := home / "python3.exe").exists():
        shutil.copy2(sys.executable, shim)
    return home


def read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file, or nothing when it does not exist.

    Args:
        path: The file to read.

    Returns:
        Its rows.
    """
    if not path.exists():
        return []
    return [json.loads(line) for line
            in path.read_text(encoding="utf-8").splitlines() if line.strip()]


async def generate(prompts: list[dict], responses: Path, client: AsyncOpenAI,
                   model: str, temperature: float, concurrency: int) -> None:
    """Answer every prompt still unanswered, appending each reply as it arrives.

    Args:
        prompts: Rows carrying ``instance_id`` and ``prompt``.
        responses: JSONL appended to, and read first for what is already answered.
        client: The OpenAI-compatible endpoint.
        model: Model id, as that endpoint names it.
        temperature: Sampling temperature.
        concurrency: In-flight requests.
    """
    done = {row["instance_id"] for row in read_jsonl(responses) if row.get("response")}
    todo = [row for row in prompts if row["instance_id"] not in done]
    print(f"Generating {len(todo)} responses ({len(done)} already answered) with {model}")

    limit, write = asyncio.Semaphore(concurrency), asyncio.Lock()
    with open(responses, "a", encoding="utf-8") as out, tqdm(total=len(todo)) as bar:
        async def answer(row: dict) -> None:
            """Ask for one instance and record the reply, empty when the call failed.

            Args:
                row: The prompt row.
            """
            record = {"instance_id": row["instance_id"], "response": "", "usage": {}}
            async with limit:
                try:
                    reply = await client.chat.completions.create(
                        model=model, temperature=temperature,
                        messages=[{"role": "user", "content": row["prompt"]}])
                    record["response"] = reply.choices[0].message.content or ""
                    record["usage"] = reply.usage.model_dump() if reply.usage else {}
                except Exception as exc:  # a dead instance must not lose the run
                    bar.write(f"{row['instance_id']}: {exc}")
            async with write:
                out.write(json.dumps(record) + "\n")
                out.flush()
                bar.update()

        await asyncio.gather(*(answer(row) for row in todo))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="deepseek-v4-flash",
                    help="Model id as the endpoint names it, with no LiteLLM provider prefix.")
    ap.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com"),
                    help="OpenAI-compatible endpoint.")
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "livesqlbench_data_sqlite.jsonl",
                    help="The public release file.")
    ap.add_argument("--testcases", type=Path,
                    default=ROOT / "data" / "livesqlbench_sqlite_gt_kg_testcases_20260601.jsonl",
                    help="The ground-truth file, merged in before scoring.")
    ap.add_argument("--db-path", type=Path, default=ROOT / "data",
                    help="Folder holding <db>/<db>_template.sqlite and the per-database prompt files.")
    ap.add_argument("--out", type=Path,
                    help="Output directory. Default: results/baseline_<model>_<stamp>.")
    ap.add_argument("--limit", type=int, help="Answer only the first N instances.")
    ap.add_argument("--concurrency", type=int, default=8, help="In-flight requests.")
    ap.add_argument("--threads", type=int, default=4, help="Evaluator worker threads.")
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        ap.error("set DEEPSEEK_API_KEY (or OPENAI_API_KEY) in .env or the environment")

    # Resolved: the vendored scripts run from their own directories.
    out = (args.out or ROOT / "results" / f"baseline_{args.model}_{datetime.now():%Y%m%d_%H%M%S}").resolve()
    out.mkdir(parents=True, exist_ok=True)

    prompts = out / "prompts.jsonl"
    if not prompts.exists():
        run_vendored("utils/prompt_generator.py", data_path=str(args.data.resolve()),
                     prompt_path=str(prompts), prompt_type="assistant",
                     data_path_base=str(args.db_path.resolve()))

    # Retries cover a blip; a longer outage leaves the instance empty for the next run.
    client = AsyncOpenAI(api_key=key, base_url=args.base_url, max_retries=5, timeout=600)
    asyncio.run(generate(read_jsonl(prompts)[: args.limit], out / "responses.jsonl",
                         client, args.model, args.temperature, args.concurrency))

    answers = {row["instance_id"]: row for row in read_jsonl(out / "responses.jsonl")}
    tokens = sum(row.get("usage", {}).get("total_tokens", 0) for row in answers.values())
    if unanswered := [i for i, row in answers.items() if not row["response"]]:
        print(f"\n{len(unanswered)} instances have no response and will score as failures. "
              f"Rerun with the same --out to retry just those.\n  {', '.join(unanswered[:10])}")
    print(f"{tokens} tokens used")

    processed = out / "post_processed.jsonl"
    run_vendored("utils/post_process.py", input_path=str(out / "responses.jsonl"),
                 output_path=str(processed))

    # The wrapper zips its sorted results back on positionally, so the submission must be sorted.
    predicted = {row["instance_id"]: row.get("pred_sqls") or [] for row in read_jsonl(processed)}
    truth = {row["instance_id"]: row for row in read_jsonl(args.testcases)}
    submission = out / "predictions.jsonl"
    with open(submission, "w", encoding="utf-8") as f:
        for row in sorted(read_jsonl(args.data), key=lambda r: r["instance_id"]):
            if (instance_id := row["instance_id"]) in predicted:
                f.write(json.dumps({**row, **truth.get(instance_id, {}),
                                    "pred_sqls": predicted[instance_id]}) + "\n")

    run_vendored("evaluation/wrapper_evaluation_sqlite.py", jsonl_file=str(submission),
                 db_path=str(args.db_path.resolve()), mode="pred", num_threads=str(args.threads))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
