"""CLI entry point using Typer: ``ask``, ``profile``, ``tables``, ``benchmark``, ``eval``,
``gold``, ``serve`` and ``version``."""

from __future__ import annotations

import asyncio
import json

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from text2sql import __version__
from text2sql.core import Text2SQL, Text2SQLResult
from text2sql.pipeline.events import PipelineEvent, TokenDelta, collect_result
from text2sql.profiler.cache import KINDS

app = typer.Typer(
    name="text2sql",
    help="Text-to-SQL with automatic database profiling. Powered by LiteLLM.",
    no_args_is_help=True,
)
console = Console()


async def _run_streaming(engine: Text2SQL, question: str) -> Text2SQLResult | None:
    """Run ``ask()``, printing streamed progress and tokens to the console.

    Args:
        engine: The engine to run.
        question: The user's question.

    Returns:
        The final result, or None when the stream produced none.
    """
    result: Text2SQLResult | None = None
    in_thinking = False

    async for item in engine.ask(question):
        if isinstance(item, TokenDelta):
            if item.is_thinking and not in_thinking:
                console.print("\n[dim italic]<thinking>[/]", end="")
                in_thinking = True
            elif not item.is_thinking and in_thinking:
                console.print("[dim italic]</thinking>[/]\n")
                in_thinking = False
            style = "dim" if item.is_thinking else ""
            console.print(
                f"[{style}]{item.text}[/]" if style else item.text, end="", highlight=False)
        elif isinstance(item, PipelineEvent):
            if in_thinking:
                console.print("[dim italic]</thinking>[/]\n")
                in_thinking = False
            console.print(str(item))
        else:
            if in_thinking:
                console.print("[dim italic]</thinking>[/]\n")
                in_thinking = False
            console.print()  # newline after streamed tokens
            result = item

    return result


async def _run_collect(engine: Text2SQL, question: str) -> Text2SQLResult | None:
    """Run ``ask()`` quietly, collecting only the final result.

    Args:
        engine: The engine to run.
        question: The user's question.

    Returns:
        The final result, or None when the stream produced none.
    """
    return await collect_result(engine.ask(question))


@app.command()
def ask(
    question: str = typer.Argument(...,
                                   help="Natural language question to convert to SQL."),
    db: str = typer.Option(
        None, "--db", "-d", help="SQLAlchemy database URI.", envvar="TEXT2SQL_DB_URI"),
    model: str = typer.Option(
        None, "--model", "-m", help="LiteLLM model string.", envvar="TEXT2SQL_MODEL"),
    config: str = typer.Option(
        None, "--config", "-c", help="Path to YAML config file."),
    output_json: bool = typer.Option(
        False, "--json", "-j", help="Output full result as JSON."),
    show_trace: bool = typer.Option(
        False, "--trace", "-t", help="Show execution trace."),
    stream: bool = typer.Option(
        True, "--stream/--no-stream", help="Enable/disable live streaming progress."),
) -> None:
    """Ask a natural language question and get SQL + results."""
    overrides = {k: v for k, v in {"db_uri": db,
                                   "model": model}.items() if v is not None}
    engine = Text2SQL.build(config, **overrides)

    with engine:
        console.print(f"\n[bold blue]Model:[/] {engine.settings.model}")
        console.print(f"[bold blue]Database:[/] {engine.db._safe_uri()}")
        console.print(f"[bold blue]Question:[/] {question}\n")

        if stream:
            result = asyncio.run(_run_streaming(engine, question))
        else:
            with console.status("[bold green]Running pipeline..."):
                result = asyncio.run(_run_collect(engine, question))

        if result is None:
            console.print(
                Panel("[red]Pipeline returned no result[/red]", title="❌ Error"))
            raise typer.Exit(1)

        if result.error:
            console.print(Panel(f"[red]{result.error}[/red]", title="❌ Error"))
            raise typer.Exit(1)

        console.print(
            Panel(result.sql, title="✅ Generated SQL", border_style="green"))

        if result.results:
            table = Table(title="Results", show_lines=True)
            if result.results:
                for col in result.results[0].keys():
                    table.add_column(col)
                for row in result.results[:20]:
                    table.add_row(*[str(v) for v in row.values()])
                if len(result.results) > 20:
                    table.add_row(*["..." for _ in result.results[0].keys()])
            console.print(table)
            console.print(f"\n[dim]({len(result.results)} rows total)[/dim]")

        if show_trace or output_json:
            trace = result.trace
            if output_json:
                console.print_json(json.dumps(
                    {"sql": result.sql, "trace": trace}, default=str))
            else:
                console.print(Panel(json.dumps(trace, indent=2,
                              default=str), title="Trace", border_style="dim"))

        usage = result.trace.get("llm_usage", {})
        if usage:
            console.print(
                f"\n[dim]LLM: {usage.get('num_calls', 0)} calls, "
                f"{usage.get('total_tokens', 0)} tokens, "
                f"${usage.get('total_cost_usd', 0):.4f}[/dim]"
            )


def _parse_include(specs: list[str], db) -> dict[str, list[str]]:
    """Parse ``table`` / ``table:col,col`` specs.

    Args:
        specs: The ``--include`` values.
        db: Connection supplying the real column names.

    Returns:
        ``{table: [columns]}``; a bare table name expands to all its columns.
    """
    full = db.list_columns()
    sel: dict[str, list[str]] = {}
    for spec in specs:
        table, _, cols = spec.partition(":")
        table = table.strip()
        sel[table] = [c.strip() for c in cols.split(",") if c.strip()] if cols else full.get(table, [])
    return sel


def _interactive_selection(db) -> dict[str, list[str]]:
    """Walk the tables and columns, prompting which to include in profiling.

    Args:
        db: Connection supplying the tables and columns.

    Returns:
        ``{table: [columns]}`` of everything confirmed.
    """
    sel: dict[str, list[str]] = {}
    for table, cols in db.list_columns().items():
        if not typer.confirm(f"Include table '{table}' ({len(cols)} cols)?", default=True):
            continue
        if typer.confirm(f"  All {len(cols)} columns?", default=True):
            sel[table] = cols
        else:
            sel[table] = [c for c in cols if typer.confirm(f"    include {table}.{c}?", default=True)]
    return sel


@app.command()
def tables(
    db: str = typer.Argument(..., help="SQLAlchemy database URI to introspect."),
) -> None:
    """List a database's tables and columns (helper for `profile --include`)."""
    engine = Text2SQL.build(db_uri=db)
    with engine:
        for table, cols in engine.db.list_columns().items():
            console.print(f"[bold]{table}[/] [dim]({len(cols)})[/]: {', '.join(cols)}")


@app.command()
def profile(
    db: str = typer.Argument(..., help="SQLAlchemy database URI to profile."),
    model: str = typer.Option(
        None, "--model", "-m", help="LiteLLM model for summarization.", envvar="TEXT2SQL_MODEL"),
    config: str = typer.Option(
        None, "--config", "-c", help="Path to YAML config file."),
    output: str = typer.Option(
        None, "--output", "-o", help="Output JSON file path."),
    include: list[str] = typer.Option(
        None, "--include", "-t",
        help="Restrict to a table or table:col,col (repeatable). Default: all tables."),
    interactive: bool = typer.Option(
        False, "--interactive", "-i", help="Interactively pick tables and columns."),
    kb_only: bool = typer.Option(
        False, "--kb-only", help="Derive only the knowledge base, reusing the cached "
                                 "statistics and column descriptions."),
) -> None:
    """Profile a database and generate metadata summaries."""
    engine = Text2SQL.build(config, db_uri=db, **({"model": model} if model else {}))

    with engine:
        if interactive:
            engine.settings.profile_selection = _interactive_selection(engine.db)
        elif include:
            engine.settings.profile_selection = _parse_include(include, engine.db)

        what = "knowledge base" if kb_only else "database"
        console.print(f"\n[bold blue]Profiling:[/] {engine.db._safe_uri()}")
        with console.status(f"[bold green]Deriving {what}..."):
            asyncio.run(engine.profile_database(build=not kb_only))
        console.print(f"[green]✅ Cached the {what}.[/green]")

        if output:
            export = {kind: engine.cache.load(engine.cache_key, kind) for kind in KINDS}
            with open(output, "w") as f:
                json.dump(export, f, indent=2, default=str)
            console.print(f"[green]Exported to {output}[/green]")


@app.command()
def version() -> None:
    """Show version."""
    console.print(f"text2sql-toolkit v{__version__}")


@app.command()
def benchmark(
    model: str = typer.Option(
        None, "--model", "-m", help="LiteLLM model string.", envvar="TEXT2SQL_MODEL"),
    config: str = typer.Option(
        None, "--config", "-c", help="Path to YAML config file."),
    max_examples: int = typer.Option(
        None, "--max", "-n", help="Max examples to run."),
    output_dir: str | None = typer.Option(
        None, "--output", "-o", help="Output directory for results.", envvar="TEXT2SQL_BENCHMARK_OUTPUT_DIR"
    ),
    use_knowledge: bool | None = typer.Option(
        None,
        "--use-knowledge",
        help="Append external knowledge to question.",
        envvar="TEXT2SQL_BENCHMARK_USE_KNOWLEDGE",
    ),
    dataset_folder: str | None = typer.Option(
        None, "--dataset-folder", help="Path to the local dataset folder.", envvar="TEXT2SQL_BENCHMARK_DATASET_FOLDER"
    ),
    data_jsonl: str | None = typer.Option(
        None, "--data-jsonl", help="Path to the LiveSQLBench release JSONL.",
        envvar="TEXT2SQL_BENCHMARK_DATA_JSONL"
    ),
    testcases_jsonl: str | None = typer.Option(
        None, "--testcases-jsonl", help="Path to the ground-truth JSONL (sol_sql, test_cases).",
        envvar="TEXT2SQL_BENCHMARK_TESTCASES_JSONL"
    ),
    instance_id: str = typer.Option(
        None, "--instance-id",
        help="Comma-separated instance id prefixes to benchmark ('credit_1,fake_9')."),
    stop_after: str | None = typer.Option(
        None,
        "--stop-after",
        help="Stop each run after this stage (profiling | schema_linking | sql_generation) "
        "— scores that stage alone, no SQL execution.",
        envvar="TEXT2SQL_STOP_AFTER",
    ),
) -> None:
    """Run benchmarks against datasets."""
    from text2sql.benchmark import BenchmarkReport, BenchmarkResult, BenchmarkRunner, load_examples

    overrides = {
        k: v
        for k, v in {
            "model": model,
            "benchmark_output_dir": output_dir,
            "benchmark_use_knowledge": use_knowledge,
            "benchmark_dataset_folder": dataset_folder,
            "benchmark_data_jsonl": data_jsonl,
            "benchmark_testcases_jsonl": testcases_jsonl,
            "benchmark_instance_id": instance_id,
            "stop_after": stop_after,
        }.items()
        if v is not None
    }
    engine = Text2SQL.build(config, **overrides)

    with engine:
        console.print(f"\n[bold blue]Model:[/] {engine.settings.model}")
        examples = load_examples(engine.settings)
        console.print(f"[bold blue]Examples:[/] {len(examples)}")

        runner = BenchmarkRunner(
            engine, output_dir=engine.settings.benchmark_output_dir)

        async def drain() -> BenchmarkReport | None:
            """Consume the streaming run, printing one line per question.

            Returns:
                The final report, or None when the run produced none.
            """
            report: BenchmarkReport | None = None
            async for item in runner.run(examples, max_examples=max_examples):
                if isinstance(item, BenchmarkResult):
                    mark = "[red]✗[/]" if item.error else (
                        "[green]✓[/]" if item.execution_match else "[yellow]•[/]")
                    console.print(
                        f"  {mark} {item.id} ({item.latency_seconds:.1f}s)")
                elif isinstance(item, BenchmarkReport):
                    report = item
            return report

        report = asyncio.run(drain())
        if report is None:
            return

        lines = [f"Total: {report.total}"]
        if not engine.settings.stop_after:
            lines += [f"Correct: {report.correct}",
                      f"Incorrect: {report.incorrect}",
                      f"Errors: {report.errors}",
                      f"Execution Accuracy: {report.execution_accuracy:.1%}"]
        for level, m in (report.linking() or {}).items():
            extra = (f" · +{report.avg_linking_extra:.1f} extra/query"
                     if level == "table" and report.avg_linking_extra is not None else "")
            kept = f"kept {m['kept']:.1%} of schema · " if "kept" in m else ""
            lines.append(f"Linking ({level}): {kept}covered {m['covered']:.1%} · "
                         f"P {m['precision']:.1%} · R {m['recall']:.1%} · "
                         f"F1 {m['f1']:.1%}{extra}")
        lines += [f"Avg Latency: {report.avg_latency:.1f}s",
                  f"Saved to: {runner.output_dir}"]
        console.print(
            Panel("\n".join(lines), title="Benchmark Results", border_style="green"))


@app.command(name="eval")
def eval_run(
    run_dir: str = typer.Argument(..., help="A benchmark results directory (holds run.json)."),
    dataset_folder: str | None = typer.Option(
        None, "--dataset-folder", help="Dataset folder; defaults to the one recorded in run.json.",
        envvar="TEXT2SQL_BENCHMARK_DATASET_FOLDER"),
    mode: str = typer.Option(
        "pred", "--mode", help="'pred' scores the predictions, 'gold' scores the gold SQL "
        "(a harness sanity check — it must come out at 100%)."),
) -> None:
    """Re-score a finished benchmark run with the official evaluator."""
    from text2sql.benchmark import rescore

    report = rescore(run_dir, dataset_folder, mode=mode)
    console.print(
        Panel(f"Mode: {mode}\n"
              f"Total: {report.total}\n"
              f"Correct: {report.correct}\n"
              f"Errors: {report.errors}\n"
              f"Execution Accuracy: {report.execution_accuracy:.1%}\n"
              f"Updated: {run_dir}/run.json",
              title="Official Evaluation", border_style="green"))


@app.command(name="gold-check")
def gold_check_cmd(
    data_jsonl: str = typer.Option(
        None, "--data-jsonl", help="LiveSQLBench release JSONL.",
        envvar="TEXT2SQL_BENCHMARK_DATA_JSONL"),
    testcases_jsonl: str = typer.Option(
        None, "--testcases-jsonl", help="Ground-truth JSONL (holds sol_sql).",
        envvar="TEXT2SQL_BENCHMARK_TESTCASES_JSONL"),
    dataset_folder: str = typer.Option(
        None, "--dataset-folder", help="Dataset folder holding <db>/<db>_template.sqlite.",
        envvar="TEXT2SQL_BENCHMARK_DATASET_FOLDER"),
    instance_id: str = typer.Option(
        None, "--instance-id", "-i",
        help="Comma-separated id prefixes; only instances starting with one of them."),
    output_dir: str = typer.Option(
        "results/gold_check", "--output-dir", "-o", help="Where to write the artifacts."),
    audit: bool = typer.Option(
        False, "--audit", help="Instead of scoring, report golds whose own output is "
        "degenerate (all-NULL column, no rows, error) — those instances are unwinnable."),
    config: str = typer.Option(None, "--config", "-c", help="Path to YAML config file."),
) -> None:
    """Score the dataset's own gold SQL; the scorer is fair only if this comes out at 100%."""
    from text2sql.benchmark import gold_audit, gold_check
    from text2sql.config import Settings

    settings = Settings.from_yaml(config) if config else Settings()
    jsonl = data_jsonl or settings.benchmark_data_jsonl
    truth = testcases_jsonl or settings.benchmark_testcases_jsonl
    folder = dataset_folder or settings.benchmark_dataset_folder
    if not jsonl or not folder:
        console.print("[red]--data-jsonl and --dataset-folder are required[/red]")
        raise typer.Exit(1)

    if audit:
        found = gold_audit(jsonl, folder, instance_id, truth)
        console.print(Panel(
            "\n".join(f"{f['id']}: {f['defect']} — {f['detail']}" for f in found)
            or "No degenerate gold queries.",
            title=f"Gold Audit ({len(found)} finding(s))",
            border_style="yellow" if found else "green"))
        return

    summary = gold_check(jsonl, folder, output_dir, instance_id, truth)
    failures = summary["failures"]
    lines = [f"Total: {summary['total']}",
             f"Passed: {summary['passed']}",
             f"Execution Accuracy: {summary['accuracy']:.2%}",
             f"Saved to: {output_dir}/gold_check.json"]
    if failures:
        lines.append("")
        lines.append(f"[yellow]{len(failures)} gold queries did not pass:[/yellow]")
        lines += [f"  {f['id']}: {' '.join(f['error'].split())[:120]}" for f in failures]
    console.print(Panel("\n".join(lines), title="Gold SQL Sanity Check",
                        border_style="green" if not failures else "yellow"))


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", help="Host to bind to."),
    port: int = typer.Option(8000, "--port", "-p", help="Port to listen on."),
    config: str = typer.Option(
        None, "--config", "-c", help="Path to YAML config file."),
) -> None:
    """Start the FastAPI server for HTTP access."""
    try:
        import uvicorn
    except ImportError:
        console.print(
            "[red]uvicorn required. Install with: uv sync --extra api[/red]")
        raise typer.Exit(1)

    from text2sql.api import create_app

    console.print(
        f"\n[bold green]Starting server at http://{host}:{port}[/bold green]")
    uvicorn.run(create_app(config), host=host, port=port)


def main() -> None:
    """Entry point for the CLI, called by ``pyproject.toml``'s ``[project.scripts]``."""
    app()


if __name__ == "__main__":
    main()
