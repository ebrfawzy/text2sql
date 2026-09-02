"""FastAPI server and Lambda handler for the Text2SQL API.

SSE streaming REST endpoints matching the CLI commands 1:1 (``/ask``, ``/profile``,
``/benchmark``, ``/version``, ``/health``), plus an AWS Lambda handler via mangum.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from text2sql.api.model import (
    AskRequest,
    BenchmarkRequest,
    CacheDeleteRequest,
    ProfileRequest,
    SchemaRequest,
)
from text2sql.api.schema import build_config_schema

__all__ = [
    "AskRequest",
    "BenchmarkRequest",
    "CacheDeleteRequest",
    "ProfileRequest",
    "SchemaRequest",
    "create_app",
    "lambda_handler",
]

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


def render_ui() -> str:
    """Render the UI page.

    ``index.html`` is a Jinja template only so repeated blocks can live in
    ``_macros.html`` once and expand at each call site with full Alpine reactivity.
    Rendering happens per request, which keeps edits live without a restart.

    Returns:
        The rendered page.
    """
    from jinja2 import Environment, FileSystemLoader

    # No autoescaping: the only values interpolated are Alpine expressions, and escaping
    # them would mangle the JS.
    env = Environment(loader=FileSystemLoader(STATIC_DIR), autoescape=False)  # noqa: S701
    return env.get_template("index.html").render()


_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _sse(event: str, data: Any) -> str:
    """Format a single Server-Sent Event.

    Args:
        event: Event name.
        data: JSON-serializable payload.

    Returns:
        The wire-format event.
    """
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def create_app(config: str | None = None):
    """Create and configure the FastAPI application.

    Args:
        config: Optional path to a YAML config file. When given, the server's
            default engine is built from it so every request uses those
            settings (per-request ``config`` in the body still overrides).

    Returns:
        FastAPI app with ``/ask``, ``/profile``, ``/benchmark``,
        ``/version``, and ``/health`` endpoints.
    """
    try:
        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse, StreamingResponse
    except ImportError:
        raise ImportError("fastapi is required for the API server. Install with: uv sync --extra api")

    from text2sql import __version__
    from text2sql.core import Text2SQL
    from text2sql.pipeline.events import (
        EventEmitter,
        PipelineEvent,
        Stage,
        Status,
        TokenDelta,
    )

    app = FastAPI(
        title="Text2SQL API",
        description="Natural language to SQL query generation API.",
        version=__version__,
    )

    _engine: Text2SQL | None = None

    def _build_engine(req_config: str | None = None, **overrides: Any) -> Text2SQL:
        """Build or reuse a Text2SQL engine.

        Args:
            req_config: Per-request YAML config path.
            **overrides: Per-request Settings overrides.

        Returns:
            A fresh engine when the request carries a config or overrides, else the cached
            default built from the server's startup config.
        """
        nonlocal _engine
        if req_config or overrides:
            return Text2SQL.build(req_config or config, **overrides)
        if _engine is None:
            # Only load a config file when serve was given one; else env + defaults.
            _engine = Text2SQL.build(config) if config else Text2SQL()
        return _engine

    def _engine_from(req: ProfileRequest) -> Text2SQL:
        """Build the engine a ProfileRequest asks for.

        Args:
            req: The request body.

        Returns:
            An engine from its YAML path, with every set field applied as an override.
        """
        overrides = {k: v for k, v in req.model_dump(exclude={"db_uri", "config"}).items() if v is not None}
        return _build_engine(req.config, db_uri=req.db_uri, **overrides)

    def _stream(generator):
        """Wrap a generator as an SSE response.

        Args:
            generator: Yields wire-format events.

        Returns:
            The streaming response.
        """
        return StreamingResponse(generator, media_type="text/event-stream", headers=_SSE_HEADERS)

    @app.post("/ask")
    async def ask(req: AskRequest):
        """Generate SQL from a natural language question.

        Args:
            req: The request body.

        Returns:
            An SSE stream of ``progress``, ``token``, ``result`` and ``error`` events.
        """
        overrides = {k: v for k, v in req.model_dump(exclude={"question", "config"}).items() if v is not None}
        engine = _build_engine(req.config, **overrides)

        async def gen():
            """Yield the endpoint's SSE events.

            Yields:
                Wire-format events, ending in ``result`` or ``error``.
            """
            try:
                async for item in engine.ask(req.question):
                    if isinstance(item, TokenDelta):
                        yield _sse(
                            "token",
                            {"text": item.text, "is_thinking": item.is_thinking},
                        )
                    elif isinstance(item, PipelineEvent):
                        yield _sse("progress", item.to_dict())
                    else:
                        yield _sse(
                            "result",
                            {
                                "sql": item.sql,
                                "results": item.results,
                                "error": item.error,
                                "trace": item.trace,
                            },
                        )
            except Exception as e:
                logger.error("Ask error: %s", e, exc_info=True)
                yield _sse("error", {"error": str(e)})

        return _stream(gen())

    @app.post("/schema")
    async def schema(req: SchemaRequest):
        """List a database's tables and columns for the selection UI.

        Args:
            req: The request body.

        Returns:
            ``tables``, plus ``cached`` (``table -> ISO timestamp``, the freshness badges)
            and ``cached_columns`` (``table -> [profiled columns]``), so the Ask-tab picker
            can restrict questions to what the cache holds.
        """
        from text2sql.db import DatabaseConnection
        from text2sql.profiler.cache import ProfileCache

        settings = _build_engine().settings
        cache = ProfileCache(settings.profile_cache_dir)
        key = cache.cache_key(req.db_uri)
        with DatabaseConnection(req.db_uri, connect_args=settings.athena_connect_args(req.db_uri)) as db:
            return {
                "tables": db.list_columns(),
                "cached": cache.cached_tables(key),
                "cached_columns": cache.cached_columns(key),
            }

    @app.post("/cache")
    async def cache_get(req: ProfileRequest):
        """Read the effective artifacts without profiling.

        Args:
            req: The request body.

        Returns:
            The profile, meaning bases and knowledge base.
        """
        with _engine_from(req) as engine:
            return engine.artifacts()

    @app.post("/cache/delete")
    async def cache_delete(req: CacheDeleteRequest):
        """Delete a table, or some of its columns, from every cached artifact.

        Args:
            req: The request body.

        Returns:
            ``{"ok": True}``.
        """
        from text2sql.profiler.cache import ProfileCache

        cache = ProfileCache(_build_engine().settings.profile_cache_dir)
        cache.delete(cache.cache_key(req.db_uri), req.table, req.columns)
        return {"ok": True}

    @app.post("/profile")
    async def profile(req: ProfileRequest):
        """Profile a database and generate its metadata summaries.

        Args:
            req: The request body.

        Returns:
            An SSE stream of ``progress``, ``result`` and ``error`` events.
        """
        engine = _engine_from(req)

        async def gen():
            emitter = EventEmitter()
            try:
                yield _sse("progress", emitter.emit(Stage.PROFILING, Status.STARTED, "Profiling database...").to_dict())
                with engine:
                    async for item in engine.profile_stream(emitter):
                        if isinstance(item, PipelineEvent):
                            yield _sse("progress", item.to_dict())
                yield _sse("progress", emitter.emit(Stage.PROFILING, Status.COMPLETED, "Profile ready").to_dict())
                yield _sse("result", engine.artifacts())
            except Exception as e:
                logger.error("Profile error: %s", e, exc_info=True)
                yield _sse("error", {"error": str(e)})

        return _stream(gen())

    def _benchmark_engine(req: BenchmarkRequest) -> Text2SQL:
        """Build the engine a BenchmarkRequest asks for.

        Args:
            req: The request body; every key but the request-only ones is a Settings field.

        Returns:
            The engine.
        """
        overrides = {
            k: v
            for k, v in req.model_dump(exclude={"config", "max_examples"}).items()
            if v is not None
        }
        return _build_engine(req.config, **overrides)

    @app.post("/benchmark")
    async def benchmark(req: BenchmarkRequest):
        """Run a benchmark, forwarding the runner's stream 1:1.

        Args:
            req: The request body.

        Returns:
            An SSE stream of ``progress``, ``example_start``, ``token``, ``example``,
            ``scores`` (per-instance verdicts, once scoring is done), ``result`` (the final
            report) and ``error`` events.
        """
        from text2sql.benchmark import (
            BenchmarkExample,
            BenchmarkReport,
            BenchmarkResult,
            BenchmarkRunner,
            load_examples,
        )

        engine = _benchmark_engine(req)

        async def gen():
            """Yield the endpoint's SSE events.

            Yields:
                Wire-format events, ending in ``result`` or ``error``.
            """
            try:
                with engine:
                    examples = load_examples(engine.settings)
                    total = min(len(examples), req.max_examples or len(examples))
                    yield _sse(
                        "progress",
                        {
                            "stage": "benchmark",
                            "status": "started",
                            "total_examples": total,
                            "model": engine.settings.model,
                        },
                    )
                    runner = BenchmarkRunner(engine, output_dir=engine.settings.benchmark_output_dir)
                    index = 0
                    async for item in runner.run(examples, max_examples=req.max_examples):
                        if isinstance(item, BenchmarkExample):
                            yield _sse(
                                "example_start",
                                {
                                    "index": index,
                                    "id": item.id,
                                    "question": item.question,
                                    "db_name": item.db_name,
                                    "total": total,
                                },
                            )
                            index += 1
                        elif isinstance(item, TokenDelta):
                            yield _sse("token", {"text": item.text, "is_thinking": item.is_thinking})
                        elif isinstance(item, PipelineEvent):
                            yield _sse("progress", item.to_dict())
                        elif isinstance(item, BenchmarkResult):
                            yield _sse("example", item.to_dict())
                        elif isinstance(item, BenchmarkReport):
                            # Rows were streamed before the official evaluator ran; this
                            # carries the verdicts/errors back to them.
                            yield _sse("scores", item.scores())
                            yield _sse("result", item.to_dict())
            except Exception as e:
                logger.error("Benchmark error: %s", e, exc_info=True)
                yield _sse("error", {"error": str(e)})

        return _stream(gen())

    @app.post("/benchmark/preview")
    async def benchmark_preview(req: BenchmarkRequest):
        """List the records a run would execute, without running them.

        Args:
            req: The request body.

        Returns:
            The total and the first 200 matching examples.
        """
        from text2sql.benchmark import load_examples

        engine = _benchmark_engine(req)
        with engine:
            examples = load_examples(engine.settings)
        limit = req.max_examples or len(examples)
        return {
            "total": len(examples),
            "examples": [
                {
                    "id": ex.id,
                    "db_name": ex.db_name,
                    "question": ex.question,
                    "gold_sql": ex.gold_sql,
                    "difficulty": ex.difficulty,
                    "category": ex.extra.get("category", ""),
                }
                for ex in examples[: min(limit, 200)]
            ],
        }

    @app.get("/version")
    async def version():
        """Report the package version.

        Returns:
            ``{"version": ...}``.
        """
        return {"version": __version__}

    @app.get("/health")
    async def health():
        """Report service health.

        Returns:
            ``{"status": "ok"}``.
        """
        return {"status": "ok"}

    @app.get("/config/schema")
    async def config_schema(config: str | None = None):
        """Describe every configurable pipeline field, grouped and typed.

        Drives the web UI's config form: each field lists its control type, allowed
        options and bounds, default, and the endpoints it applies to.

        Args:
            config: YAML config path whose values become the defaults, so the UI can
                preview a config before running. Defaults to the server's startup config.

        Returns:
            ``{"groups": [...]}``.

        Raises:
            HTTPException: The config file is missing (404) or unloadable (400).
        """
        from fastapi import HTTPException

        try:
            settings = _build_engine(config).settings if config else _build_engine().settings
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to load config: {e}")
        return {"groups": build_config_schema(settings)}

    @app.get("/config/files")
    async def config_files():
        """List the YAML config files under ``configs/``, for the settings dropdown.

        Returns:
            ``{"files": [...]}``.
        """
        from pathlib import Path

        configs = Path("configs")
        files = sorted(str(p) for p in configs.glob("*.yaml")) + sorted(str(p) for p in configs.glob("*.yml"))
        return {"files": files}

    from fastapi.staticfiles import StaticFiles

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        """Render the single-page UI; ``app.css`` and ``app.js`` are served from /static.

        Returns:
            The rendered page.
        """
        return render_ui()

    return app


def lambda_handler(event: dict, context: Any) -> dict:
    """AWS Lambda entry point, via mangum; configure with ``TEXT2SQL_*`` env vars.

    Args:
        event: The Lambda event.
        context: The Lambda context.

    Returns:
        The API Gateway response.
    """
    try:
        from mangum import Mangum
    except ImportError:
        raise ImportError("mangum is required for Lambda deployment. Install with: uv sync --extra lambda")
    return Mangum(create_app())(event, context)
