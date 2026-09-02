"""Tests for text2sql.api — FastAPI endpoints using httpx.AsyncClient.

Tests the /health, /ask, and /ask/stream endpoints with mocked
Text2SQL engine to avoid real LLM calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from text2sql.api import AskRequest

# ── Settings-derived request models & config form ────────────────


class TestGeneratedRequestModels:
    """The request bodies are generated from Settings + SECTION_SCOPE."""

    @pytest.mark.parametrize("endpoint", ["ask", "profile", "benchmark"])
    def test_covers_every_scoped_setting_and_no_credentials(self, endpoint):
        import text2sql.api.model as api_model
        from text2sql.config import section_fields

        fields = set(getattr(api_model, f"{endpoint.capitalize()}Request").model_fields)
        assert set(section_fields(endpoint)) <= fields
        assert not [f for f in fields if f.startswith(("bedrock_", "athena_"))]

    @pytest.mark.parametrize("endpoint", ["ask", "profile", "benchmark"])
    def test_every_endpoint_accepts_a_yaml_config_path(self, endpoint):
        """Without this the UI's config picker is silently ignored for that endpoint."""
        import text2sql.api.model as api_model

        assert "config" in getattr(api_model, f"{endpoint.capitalize()}Request").model_fields

    def test_optional_fields_and_constraints(self):
        from pydantic import ValidationError

        req = AskRequest(question="Q", db_uri="sqlite:///x.db")
        assert (req.question, req.db_uri) == ("Q", "sqlite:///x.db")
        assert req.model is None and req.temperature is None  # unset stays None, never a default
        for bad in ({"temperature": 9.0}, {"selection_mode": "nope"}):
            with pytest.raises(ValidationError):
                AskRequest(question="Q", **bad)


class TestConfigSchema:
    @pytest.fixture
    def fields(self):
        from text2sql.api.schema import build_config_schema
        from text2sql.config import Settings

        groups = build_config_schema(Settings(
            num_candidates=3, generation_strategy="diverse", generation_mode="direct"))
        return groups, {f["name"]: f for g in groups for f in g["fields"]}

    def test_every_field_is_renderable_and_documented(self, fields):
        _, by_name = fields
        assert by_name
        for f in by_name.values():
            assert f["control"] in {"toggle", "select", "multi", "number", "text"}
            assert f["label"] and f["help"] and f["endpoints"], f["name"]
            assert f["control"] not in ("select", "multi") or f["options"], f["name"]

    def test_a_list_of_literals_renders_as_a_multi_select(self, fields):
        """Regression: an unhandled annotation returned no control, which dropped the
        field from the form entirely rather than failing loudly."""
        _, by_name = fields
        modes = by_name["schema_linking_modes"]
        assert modes["control"] == "multi"
        assert modes["options"] == ["direct", "reversed", "value"]
        assert modes["default"] == ["value"]

    def test_derives_each_field_from_its_annotation_and_settings(self, fields):
        from text2sql.config import Settings

        _, by_name = fields
        assert by_name["generation_mode"] == {
            "name": "generation_mode", "label": "Mode", "control": "select",
            "options": ["direct", "agent"], "default": "direct",
            "help": Settings.model_fields["generation_mode"].description,  # tooltip = description
            "endpoints": ["ask", "benchmark"]}
        assert by_name["selection_mode"]["options"] == ["majority", "confidence", "single"]
        assert by_name["num_candidates"]["options"] == [1, 2, 3]  # bounded int → dropdown
        assert by_name["num_candidates"]["default"] == 3          # from the live Settings
        assert by_name["log_level"]["default"] == "INFO"                   # StrEnum → plain string
        assert (by_name["temperature"]["min"], by_name["temperature"]["max"]) == (0.0, 2.0)
        assert "profile_selection" not in by_name  # nested dict → no control → skipped

    def test_groups_match_the_yaml_sections(self, fields):
        from text2sql.config import SECTION_SCOPE

        groups, _ = fields
        assert [g["key"] for g in groups] == list(SECTION_SCOPE)
        for g in groups:
            assert g["fields"][0]["endpoints"] == list(SECTION_SCOPE[g["key"]][1])

    def test_nested_subsections_are_flattened_into_their_parent_group(self, fields):
        groups, _ = fields
        gen = next(g for g in groups if g["key"] == "sql_generation")
        names = [f["name"] for f in gen["fields"]]
        assert names[0] == "generation_mode"                    # the fork leads the group
        assert {"agent_max_turns", "prompt_version"} <= set(names)

    def test_dependent_fields_carry_their_controller(self, fields):
        _, by_name = fields
        assert by_name["agent_mode"]["depends_on"] == {
            "field": "generation_mode", "values": ["agent"]}
        assert by_name["selection_mode"]["depends_on"] == {
            "field": "num_candidates", "values": [2, 3]}
        assert "depends_on" not in by_name["generation_mode"]  # nothing gates the fork itself

    def test_a_chained_field_advertises_only_its_immediate_controller(self, fields):
        """The client walks the chain itself, so the wire format stays one hop deep —
        scenarios_file names agent_mode, not the generation_mode two hops up."""
        _, by_name = fields
        assert by_name["scenarios_file"]["depends_on"] == {
            "field": "agent_mode", "values": ["retrieval"]}
        assert by_name["reversed_knowledge"]["depends_on"] == {
            "field": "schema_linking_modes", "values": ["reversed"]}


# ── Single-page UI (index.html + _macros.html + app.js) ──────────


class TestUI:
    """The page is split across files, so the seams need checking."""

    @pytest.fixture(scope="class")
    def page(self):
        from text2sql.api import render_ui
        return render_ui()

    def test_renders_and_links_its_assets(self, page):
        assert '<body' in page and 'x-data="app()"' in page
        assert '/static/app.js' in page and '/static/app.css' in page
        assert "{{" not in page and "{%" not in page  # every macro expanded

    def test_every_config_field_can_reach_the_form(self, page):
        """The sidebar renders whole groups, so no field can be stranded like stop_after was."""
        assert 'x-for="group in groups()"' in page
        assert 'x-html="fieldHTML(f)"' in page
        assert "RELOCATED" not in page  # the old hand-picked allowlist is gone

    def test_markup_only_calls_functions_the_script_defines(self, page):
        """Catches a macro left pointing at a renamed/removed component method."""
        import re
        from pathlib import Path

        from text2sql.api import STATIC_DIR

        js = Path(STATIC_DIR / "app.js").read_text(encoding="utf-8")
        defined = set(re.findall(r"^\s{4}(?:async )?(\w+)\(", js, re.M))  # component methods
        defined |= set(re.findall(r"^(?:const|function) (\w+)", js, re.M))  # module-level
        builtins = {"if", "for", "Object", "JSON", "String", "Number", "Boolean", "Array", "Date",
                    "hljs", "app", "includes", "keys", "entries", "trim", "toFixed", "replace",
                    "writeText", "stringify", "length", "filter", "map", "join", "slice"}
        called = set()
        for attr in re.findall(r'(?:x-(?:text|html|show|if|for|model|effect|init)|@\w+(?:\.\w+)*)="([^"]*)"', page):
            called |= set(re.findall(r"\b(\w+)\s*\(", attr))
        missing = called - defined - builtins
        assert not missing, f"markup calls undefined helpers: {sorted(missing)}"


# ── FastAPI endpoint tests ───────────────────────────────────────


@pytest.fixture
def mock_engine():
    """Create a mock Text2SQL engine for API tests."""
    engine = MagicMock()
    engine.settings.model = "gpt-4o-mini"
    engine.db._safe_uri.return_value = "sqlite:///test.db"
    return engine


@pytest.fixture
def mock_result():
    """Create a mock Text2SQLResult."""
    from text2sql.core import Text2SQLResult
    return Text2SQLResult(
        sql="SELECT COUNT(*) AS cnt FROM users",
        results=[{"cnt": 5}],
        error=None,
        trace={"llm_usage": {"num_calls": 1}},
    )


class TestHealthEndpoint:
    async def test_health(self):
        from httpx import ASGITransport, AsyncClient

        from text2sql.api import create_app
        app = create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
            assert resp.status_code == 200
            assert resp.json() == {"status": "ok"}


class TestCacheEndpoint:
    async def test_cache_reads_artifacts_with_a_config_path(self, sample_db, tmp_path):
        """``config`` is a build argument, not a Settings override — passing it on twice is a 500."""
        from httpx import ASGITransport, AsyncClient

        from text2sql.api import create_app

        yaml_path = tmp_path / "c.yaml"
        yaml_path.write_text("profiling:\n  cache_dir: " + str(tmp_path / "cache") + "\n")
        transport = ASGITransport(app=create_app())
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/cache", json={"db_uri": sample_db, "config": str(yaml_path)})
        assert resp.status_code == 200
        assert set(resp.json()) >= {"profile", "kb"}


class TestAskEndpoint:
    async def test_ask_success(self, mock_result):
        from httpx import ASGITransport, AsyncClient

        from text2sql.api import create_app

        with patch("text2sql.api.create_app"):
            # Create a real app but patch the engine inside
            app = create_app()

            # Patch get_settings and Text2SQL inside the module
            with patch("text2sql.config.get_settings") as mock_settings, \
                 patch("text2sql.core.Text2SQL") as mock_t2s:
                mock_settings.return_value = MagicMock()
                engine = MagicMock()
                async def mock_ask(question):
                    yield mock_result
                engine.ask = mock_ask
                mock_t2s.return_value = engine

                # Re-create app with patched imports
                app = create_app()
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    resp = await client.post("/ask", json={"question": "How many users?"})
                    # /ask is an SSE stream: the final SQL/results arrive as a `result` event.
                    assert resp.status_code == 200
                    assert "text/event-stream" in resp.headers["content-type"]
                    body = resp.text
                    assert "event: result" in body
                    assert "SELECT COUNT(*) AS cnt FROM users" in body
                    assert "cnt" in body

    async def test_ask_with_overrides(self, mock_result):
        from httpx import ASGITransport, AsyncClient

        from text2sql.api import create_app

        with patch("text2sql.config.get_settings") as mock_settings, \
             patch("text2sql.core.Text2SQL") as mock_t2s:
            mock_settings.return_value = MagicMock()
            engine = MagicMock()
            async def mock_ask(question):
                yield mock_result
            engine.ask = mock_ask
            mock_t2s.return_value = engine

            app = create_app()
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post("/ask", json={
                    "question": "Q",
                    "db_uri": "sqlite:///other.db",
                    "model": "gpt-4o",
                })
                assert resp.status_code == 200

    async def test_ask_server_error(self):
        from httpx import ASGITransport, AsyncClient

        from text2sql.api import create_app

        with patch("text2sql.config.get_settings") as mock_settings, \
             patch("text2sql.core.Text2SQL") as mock_t2s:
            mock_settings.return_value = MagicMock()
            engine = MagicMock()
            async def mock_ask(question):
                raise RuntimeError("DB down")
                yield  # make it an async generator
            engine.ask = mock_ask
            mock_t2s.return_value = engine

            app = create_app()
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post("/ask", json={"question": "Q"})
                # SSE streams start with 200; a mid-stream failure surfaces as an `error` event.
                assert resp.status_code == 200
                body = resp.text
                assert "event: error" in body
                assert "DB down" in body


class TestAskStreamEndpoint:
    async def test_stream_returns_sse(self, mock_result):
        from httpx import ASGITransport, AsyncClient

        from text2sql.api import create_app
        from text2sql.pipeline.events import EventEmitter, Stage, Status, TokenDelta

        emitter = EventEmitter()

        with patch("text2sql.config.get_settings") as mock_settings, \
             patch("text2sql.core.Text2SQL") as mock_t2s:
            mock_settings.return_value = MagicMock()
            engine = MagicMock()

            # Make ask return a proper async generator with token deltas
            async def mock_ask(question):
                yield emitter.emit(Stage.PROFILING, Status.STARTED, "Profiling...")
                yield emitter.emit(Stage.PROFILING, Status.COMPLETED, "Done")
                yield TokenDelta(text="SELECT", is_thinking=False)
                yield TokenDelta(text=" 1", is_thinking=False)
                yield mock_result

            engine.ask = mock_ask
            mock_t2s.return_value = engine

            app = create_app()
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post("/ask", json={"question": "Q"})
                assert resp.status_code == 200
                assert "text/event-stream" in resp.headers["content-type"]
                body = resp.text
                assert "event: progress" in body
                assert "event: token" in body
                assert "event: result" in body


# ── Lambda handler ───────────────────────────────────────────────


class TestLambdaHandler:
    def test_lambda_handler_import_error(self):
        """Verify lambda_handler raises ImportError without mangum."""
        from text2sql.api import lambda_handler
        with patch.dict("sys.modules", {"mangum": None}):
            with pytest.raises(ImportError, match="mangum"):
                lambda_handler({}, None)
