"""Tests for text2sql.core — the Text2SQL orchestrator.

Covers the streaming `ask()` contract, candidate diversity, `stop_after`,
retrieval context mode, repair hand-off, and the profiling/caching paths.
Stage-local behaviour lives in the per-module test files.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from conftest import mock_sql_llm, stub_agent, stub_repair

from text2sql.core import Text2SQL, Text2SQLResult
from text2sql.pipeline.events import EventEmitter, PipelineEvent, TokenDelta
from text2sql.profiler.knowledge import DatabaseKnowledge, KnowledgeEntry
from text2sql.profiler.stats import group
from text2sql.profiler.summarizer import ColumnSummary, DatabaseSummary
from text2sql.schema.loader import SchemaLoader


async def drain(engine, question="How many users?"):
    """Run `ask()` to completion, returning (events, result)."""
    events, result = [], None
    async for item in engine.ask(question):
        if isinstance(item, PipelineEvent):
            events.append(item)
        elif not isinstance(item, TokenDelta):
            result = item
    return events, result


# ── Construction ─────────────────────────────────────────────────


class TestGenerationKnowledge:
    """The question can already define a term; repeating it as domain knowledge is noise."""

    def test_a_term_the_question_defines_is_not_repeated(self, make_engine):
        from text2sql.profiler.knowledge import DatabaseKnowledge

        kb = DatabaseKnowledge({
            0: KnowledgeEntry(0, "Net Worth", "assets minus liabilities", "totassets - totliabs"),
            1: KnowledgeEntry(1, "Leverage Ratio", "liabilities over assets", "totliabs / totassets"),
        })
        engine = make_engine()
        scope = {"t": ["totassets", "totliabs"]}
        assert [e.knowledge for e in engine._generation_knowledge(kb, scope, "show net worth")] \
            == ["Leverage Ratio"]

        # Regression: the refuted entry is the one that matters, and the question naming it is
        # exactly why — `use_knowledge` put `Net Worth` in HINTS with the false
        # equality, and the correction was filtered out of all six credit_10 arms.
        refuted = DatabaseKnowledge({0: KnowledgeEntry(
            0, "Net Worth", "", "networth != totassets - totliabs", refuted=True)})
        kept = engine._generation_knowledge(refuted, scope, "show net worth")
        assert [e.knowledge for e in kept] == ["Net Worth"]

        # A refuted entry the question never mentions corrects nothing: the generator invents
        # these, and inlining every one of them put ~466 chars of arithmetic about columns
        # nobody asked for into 119 of 270 prompts.
        assert engine._generation_knowledge(refuted, scope, "show the leverage ratio") == []

        # The term is matched by acronym too, but only one long enough to mean something:
        # "MI" matches "minimum", which would keep every entry again.
        acronym = DatabaseKnowledge({0: KnowledgeEntry(
            0, "Net Worth (NTW)", "", "networth != totassets - totliabs", refuted=True)})
        assert engine._generation_knowledge(acronym, scope, "report NTW per client")
        short = DatabaseKnowledge({0: KnowledgeEntry(
            0, "Memory Intensity (MI)", "", "memint != used / total", refuted=True)})
        assert engine._generation_knowledge(short, scope, "show the minimum memory") == []
        assert len(engine._generation_knowledge(kb, scope, "show everything")) == 2
        engine.settings.generation_knowledge = "off"
        assert engine._generation_knowledge(kb, scope, "show everything") == []


class TestConstruction:
    def test_context_manager_and_close(self):
        with Text2SQL(db_uri="sqlite:///:memory:") as engine:
            assert engine.settings.db_uri == "sqlite:///:memory:"
        engine.close()  # idempotent — closing twice must not raise

    def test_from_config(self, sample_yaml_config):
        engine = Text2SQL.from_config(sample_yaml_config)
        assert engine.settings.model == "gpt-4o"
        engine.close()

    def test_package_exports_the_streaming_types(self):
        """`from text2sql import ...` is the documented public surface."""
        import text2sql
        assert {"Text2SQL", "Text2SQLResult", "PipelineEvent", "TokenDelta"} <= set(dir(text2sql))


# ── Profile context ──────────────────────────────────────────────


class TestAsk:
    async def test_yields_events_then_result(self, make_engine):
        engine = mock_sql_llm(
            make_engine(tables=("users", "orders"), model="gpt-4o-mini",
                        generation_mode="direct", use_schema_linking=False,
                        use_repair=False, num_candidates=1),
            "SELECT COUNT(*) AS cnt\nFROM users")

        events, result = await drain(engine)
        engine.close()

        assert isinstance(result, Text2SQLResult)
        assert result.sql == "SELECT COUNT(*) AS cnt\nFROM users"
        assert result.results[0]["cnt"] == 5  # really executed against the sample DB
        assert result.error is None and "question" in result.trace

        assert len(events) >= 4 and events[0].stage == "profiling"
        assert {"started", "completed"} <= {e.status for e in events if e.stage == "profiling"}
        assert all(e.timestamp != "" and e.elapsed_seconds >= 0 for e in events)

    async def test_error_yields_error_event_and_result(self, make_engine):
        engine = make_engine(generation_mode="direct")
        engine._get_or_build_profile = AsyncMock(side_effect=RuntimeError("DB connection failed"))

        events, result = await drain(engine)
        engine.close()

        assert [e for e in events if e.status == "error"]
        assert result.sql == "" and "DB connection failed" in result.error


# ── Candidate diversity ──────────────────────────────────────────


class TestAgentCandidateDiversity:
    """Agent-mode candidates must not be byte-identical.

    At temperature 0 the agent is deterministic, so N identical runs would make
    majority voting cluster N copies of one answer and report false agreement.
    """

    SCHEMA = (
        "Table: users\n"
        "  id INTEGER\n  name TEXT\n  email TEXT\n  age INTEGER\n"
        "  city TEXT\n  state TEXT\n  zip TEXT\n  country TEXT"
    )

    async def test_every_candidate_gets_its_own_tool_set(self, make_engine, monkeypatch):
        """One shared tool set meant candidate 1 spent the submit gate's refusal budget and
        2-3 submitted unchecked."""
        built = []
        engine = make_engine(generation_mode="agent", num_candidates=3,
                             generation_strategy="diverse")
        monkeypatch.setattr(engine, "_build_agent",
                            lambda _q="": built.append(_q) or stub_agent())
        async for _ in engine._generate_candidates("How many users?", self.SCHEMA, "",
                                                   EventEmitter()):
            pass
        assert len(built) == 3

    async def test_schema_order_varies_across_candidates(self, make_engine, monkeypatch):
        import random

        seen: list[str] = []
        engine = make_engine(generation_mode="agent", num_candidates=3,
                             generation_strategy="diverse")
        monkeypatch.setattr(engine, "_build_agent", lambda _q="": stub_agent(seen=seen))

        random.seed(1234)  # deterministic shuffle for a stable assertion
        async for _ in engine._generate_candidates(
            "How many users?", self.SCHEMA, "", EventEmitter(),
        ):
            pass

        assert len(seen) == 3
        assert seen[0] == self.SCHEMA  # first candidate uses the canonical order
        assert len(set(seen)) == 3  # each later candidate is a distinct variant
        # Shuffling must preserve content — only the field order changes.
        for variant in seen[1:]:
            assert sorted(variant.splitlines()) == sorted(self.SCHEMA.splitlines())
            assert variant.startswith("Table: users")

    async def test_no_candidates_surfaces_as_error(self, make_engine):
        """Generation returning nothing must fail loudly, not yield empty SQL.

        A silent empty result is indistinguishable from a wrong answer in benchmark
        reports — this is how a model that can't do tool calls looked like "Errors: 0".
        """
        engine = make_engine(generation_mode="agent", use_schema_linking=False,
                             use_repair=False, num_candidates=1)
        engine._build_agent = lambda _question="": stub_agent(dead=True)

        _, result = await drain(engine)
        assert result.sql == ""
        assert result.error and "no candidates" in result.error.lower()


# ── stop_after ───────────────────────────────────────────────────


class TestStopAfter:
    """`stop_after` halts the pipeline early without breaking the streaming contract."""

    @pytest.mark.parametrize("stage", ["profiling", "schema_linking"])
    async def test_stops_with_a_result_and_no_generation(self, make_engine, stage):
        engine = make_engine(stop_after=stage, use_schema_linking=False, num_candidates=1)
        _, result = await drain(engine)
        assert isinstance(result, Text2SQLResult)  # same contract: result comes last
        assert result.sql == "" and result.error is None
        steps = [s["step"] for s in result.trace["steps"]]
        assert "sql_generation" not in steps  # no LLM spent past the stage under test
        assert steps[-1] == stage

    async def test_linked_columns_reach_the_trace(self, make_engine):
        """The benchmark scores columns, so linking must record them, not just tables."""
        engine = make_engine(stop_after="schema_linking", use_schema_linking=True,
                             num_candidates=1)
        engine._build_agent = lambda _question="": None  # never reached

        async def fake_link(self, question, emitter=None):
            yield {"users": ["id", "name"]}

        with patch("text2sql.schema.linker.SchemaLinker.link_stream", new=fake_link):
            _, result = await drain(engine)

        step = next(s for s in result.trace["steps"] if s["step"] == "schema_linking")
        assert step["outputs"]["linked"] == {"users": ["id", "name"]}
        assert step["outputs"]["linked_tables"] == ["users"]

    async def test_linking_off_records_none_not_an_empty_set(self, make_engine):
        """`{}` means linking ran and linked nothing, which scores 0 recall. Linking *off*
        must be `None`, or the benchmark reports a fabricated zero as a measurement."""
        engine = make_engine(stop_after="schema_linking", use_schema_linking=False,
                             num_candidates=1)
        _, result = await drain(engine)

        step = next(s for s in result.trace["steps"] if s["step"] == "schema_linking")
        assert step["outputs"]["linked"] is None
        assert step["outputs"]["linked_tables"] is None


    async def test_the_value_index_follows_the_profile(self, make_engine, db_profile):
        """Regression: the index was memoized on the engine, so a benchmark run served the
        *first* database's values to all 270 questions — every later database then logged
        its matches as 'non-existent table' and dropped them."""
        from copy import deepcopy

        engine = make_engine(use_schema_linking=True, schema_linking_modes=["value"])
        first = engine._value_index(db_profile)
        assert engine._value_index(db_profile) is first  # same profile, one build
        assert engine._value_index(deepcopy(db_profile)) is not first  # new profile, new index


    async def test_a_deleted_knowledge_base_rebuilds_without_resummarizing(
            self, sample_db, tmp_path, monkeypatch):
        """`--kb-only`: knowledge regenerated only for tables summarized in the same run, so
        deleting the KB rebuilt nothing while the descriptions stayed cached."""
        async def fake_summarize(self, profile, **kw):
            ds = DatabaseSummary()
            for t, tp in profile.tables.items():
                ds.columns[t] = {c: ColumnSummary(t, c, "s", "l") for c in tp.columns}
            return ds

        asked = []

        async def fake_knowledge(self, profile, describe, only=None, joins=""):
            asked.append(only)
            return DatabaseKnowledge()

        monkeypatch.setattr("text2sql.core.ProfileSummarizer.summarize_database", fake_summarize)
        monkeypatch.setattr("text2sql.core.KnowledgeGenerator.generate", fake_knowledge)
        engine = Text2SQL(db_uri=sample_db, profile_cache_dir=str(tmp_path / "cache"))
        with engine:
            await engine.profile_database()                      # warm stats + descriptions
            engine.cache.save(engine.cache_key, "kb", {}, replace=True)   # delete just the KB
            engine._profile = engine._summary = None
            asked.clear()
            await engine.profile_database(build=False)            # --kb-only
            assert asked and set(asked[0]) == {"users", "orders"}

    async def test_a_failed_summarization_does_not_fail_the_request(self, make_engine, tmp_path):
        """Regression: one column lacking a cached description made the engine summarize on
        the ask path, and an expired token there took the whole database offline — 15
        benchmark instances died mid-sweep on a stage that needs no LLM at all. A missing
        description degrades that column to name-only; it must never fail the request."""
        engine = make_engine(profile_cache_dir=str(tmp_path / "cache"))
        engine._profile = engine._summary = None  # cold cache: this path must summarize
        with patch("text2sql.profiler.summarizer.ProfileSummarizer.summarize_database",
                   side_effect=RuntimeError("token expired")):
            items = [x async for x in engine._profile_stream(EventEmitter(), build=False)]

        events = [x for x in items if isinstance(x, PipelineEvent)]
        assert any(e.data.get("summarization_failed") for e in events)
        profile, summary, _ = items[-1]
        assert profile.tables and not summary.columns  # profiled, just undescribed


# ── Agent context / repair hand-off ──────────────────────────────


class TestRetrievalContextMode:
    """`retrieval` withholds the schema from the agent — and only from the agent."""

    @staticmethod
    def _engine(make_engine, **kw):
        return make_engine(generation_mode="agent", agent_mode="retrieval",
                           use_schema_linking=False,
                           num_candidates=1, **kw)

    async def test_the_prompt_and_the_tools_never_show_a_term_twice(self, make_engine):
        """`search_columns` appends definitions of its own, so the entries the system prompt
        already inlined must not come back through a tool result."""
        engine = self._engine(make_engine)
        engine._shown_terms = {7}
        tools = {t.name: t for t in engine._build_agent("q").tools}
        assert "search_columns" in tools

    async def test_applies_without_schema_linking(self, make_engine):
        """The override used to live inside `if linked:`, so linking-off silently
        handed the agent the full DDL — the opposite of what retrieval asks for."""
        seen: list[str] = []
        engine = self._engine(make_engine, use_repair=False)
        engine._build_agent = lambda _question="": stub_agent(seen=seen)

        await drain(engine)
        assert seen and seen[0].startswith("Tables: ")
        # Unlinked renders every column somewhere; the agent's own copy must stay the list.
        # The descriptions used to arrive in a second block that reached 33k chars.
        assert "CREATE TABLE" not in seen[0] and "Field " not in seen[0]

    async def test_repair_still_receives_the_full_schema(self, make_engine):
        """Only the agent's prompt is stripped; repair must not lose the schema."""
        repaired: list[tuple[str, str]] = []
        engine = self._engine(make_engine, use_repair=True)
        engine._build_agent = lambda _question="": stub_agent()

        with patch("text2sql.core.SQLRepair", stub_repair(repaired)):
            await drain(engine)
        assert repaired and not repaired[0][1].startswith("Tables: ")


# ── Profiling: summary modes, sources, accumulation ──────────────


class TestSummaryModes:
    """profile_summary picks which meaning bases get written."""

    @staticmethod
    async def _run(sample_db, tmp_path, monkeypatch, mode):
        seen = {}

        async def fake_summarize(self, profile, *, generate_short=True, generate_long=True, **kw):
            seen.update(short=generate_short, long=generate_long)
            ds = DatabaseSummary()
            for t, tp in profile.tables.items():
                ds.columns[t] = {c: ColumnSummary(t, c, "s" if generate_short else "",
                                                  "l" if generate_long else "") for c in tp.columns}
            return ds

        monkeypatch.setattr("text2sql.core.ProfileSummarizer.summarize_database", fake_summarize)
        engine = Text2SQL(db_uri=sample_db, profile_cache_dir=str(tmp_path / "cache"),
                          profile_kb=False, profile_summary=mode)
        with engine:
            await engine.profile_database()
            return engine, seen

    @pytest.mark.parametrize("mode, short, long", [
        ("short_and_long", True, True), ("short", True, False), ("long", False, True),
    ])
    async def test_mode_drives_generation_flags(self, sample_db, tmp_path, monkeypatch,
                                                mode, short, long):
        engine, seen = await self._run(sample_db, tmp_path, monkeypatch, mode)
        assert seen == {"short": short, "long": long}
        has_text = {
            kind: any(v for cols in group(engine.cache.load(engine.cache_key, kind)).values()
                      for v in cols.values())
            for kind in ("meaning_base_short", "meaning_base_long")
        }
        assert has_text == {"meaning_base_short": short, "meaning_base_long": long}

    async def test_short_only_still_feeds_generation(self, sample_db, tmp_path, monkeypatch):
        """The unlinked schema renders `short`, so a short-only profile still carries meanings."""
        engine, _ = await self._run(sample_db, tmp_path, monkeypatch, "short")
        with engine:
            engine._profile = engine._summary = None
            _, summary, _ = await engine._get_or_build_profile()
            loader = SchemaLoader(engine.db, engine._profile, summary,
                                  prompts=engine.prompt_manager)
        assert "name" in loader.format_schema(detail="short")


class TestProfileAccumulation:
    """The cache is one accumulating file per DB; profiling upserts, ask() reads."""

    @staticmethod
    def _engine(sample_db, tmp_path, monkeypatch):
        async def fake_summarize(self, profile, **kw):
            ds = DatabaseSummary()
            for t, tp in profile.tables.items():
                ds.columns[t] = {c: ColumnSummary(table_name=t, column_name=c) for c in tp.columns}
            return ds

        monkeypatch.setattr("text2sql.core.ProfileSummarizer.summarize_database", fake_summarize)
        return Text2SQL(db_uri=sample_db, profile_cache_dir=str(tmp_path / "cache"),
                        profile_kb=False)

    async def test_partial_profiling_accumulates(self, sample_db, tmp_path, monkeypatch):
        engine = self._engine(sample_db, tmp_path, monkeypatch)
        with engine:
            engine.settings.profile_selection = {"users": ["id", "name"]}
            await engine.profile_database()
            engine._profile = engine._summary = None  # drop the in-memory memo
            engine.settings.profile_selection = {"orders": ["id"]}
            await engine.profile_database()
        # Both partial runs upserted into one file rather than overwriting.
        assert set(engine.cache.cached_tables(engine.cache_key)) == {"users", "orders"}

    async def test_ask_path_skips_uncached_and_filters(self, sample_db, tmp_path, monkeypatch):
        engine = self._engine(sample_db, tmp_path, monkeypatch)
        with engine:
            engine.settings.profile_selection = {"users": ["id", "name"]}
            await engine.profile_database()  # only `users` is cached
            engine._profile = engine._summary = None
            # ask()-path selecting an uncached table: it is skipped, not built.
            engine.settings.profile_selection = {"orders": ["id"]}
            profile, _, _ = await engine._get_or_build_profile()
        assert "orders" not in profile.tables  # never profiled on the read path
        assert set(engine.cache.cached_tables(engine.cache_key)) == {"users"}
