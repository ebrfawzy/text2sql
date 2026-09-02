"""Core Text2SQL orchestrator: the package's main entry point.

Wires profiling, schema linking, SQL generation, repair and selection into a single async
streaming pipeline.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from text2sql.config import Settings, get_settings
from text2sql.db import DatabaseConnection
from text2sql.llm import LLMClient
from text2sql.pipeline.agent import SQLAgent
from text2sql.pipeline.events import EventEmitter, PipelineEvent, Stage, Status, TokenDelta
from text2sql.pipeline.examples import ExampleStore
from text2sql.pipeline.generator import SQLGenerator, randomize_schema_order, strategy_for
from text2sql.pipeline.repair import SQLRepair
from text2sql.pipeline.selector import CandidateSelector
from text2sql.pipeline.tools import build_tools
from text2sql.pipeline.tracer import PipelineTracer
from text2sql.profiler import (
    DatabaseKnowledge,
    KnowledgeGenerator,
    ProfileCache,
    ProfileSummarizer,
    StatsProfiler,
    ValueIndex,
)
from text2sql.profiler.knowledge import KnowledgeEntry
from text2sql.profiler.stats import DatabaseProfile, TableProfile, entries
from text2sql.profiler.summarizer import DatabaseSummary
from text2sql.prompts.manager import PromptManager
from text2sql.schema.linker import SchemaLinker, VariantSpec
from text2sql.schema.loader import SchemaLoader

logger = logging.getLogger(__name__)


def _agent_telemetry(traces: list[dict[str, Any]], max_turns: int) -> dict[str, Any]:
    """Summarize the agent runs of one request for the trace.

    Args:
        traces: One trace per candidate.
        max_turns: The configured turn budget, which a review finding can raise.

    Returns:
        The counters under ``agent``, empty when the request ran no agent. The calls
        themselves are not repeated here - the conversation carries them in full.
    """
    return {"agent": {
        "turns": sum(t["turns"] for t in traces),
        "tool_calls": sum(len(t["tool_calls"]) for t in traces),
        "termination": ",".join(t["termination"] for t in traces),
        "recovered": sum("recovered_from" in t for t in traces),
        "granted": sum(t.get("granted", 0) for t in traces),
        "max_turns": max(t.get("limit", max_turns) for t in traces),
        # Without this an `error` termination has no cause anywhere in the artefacts.
        **({"error": "; ".join(e for t in traces if (e := t.get("error")))}
           if any(t.get("error") for t in traces) else {}),
    }} if traces else {}


@dataclass
class Text2SQLResult:
    """Result of a Text2SQL pipeline execution.

    Attributes:
        sql: The selected SQL.
        results: The rows it returned, or None when it did not run.
        trace: The full pipeline trace.
        error: The failure, when the run did not complete.
    """

    sql: str
    results: list[dict[str, Any]] | None
    trace: dict[str, Any]
    error: str | None = None


class Text2SQL:
    """Main orchestrator for the Text-to-SQL pipeline.

    Usage::

        engine = Text2SQL(db_uri="sqlite:///my.db", model="gpt-4o-mini")
        async for event in engine.ask("How many users?"):
            if isinstance(event, PipelineEvent):
                print(event)          # progress message
            elif isinstance(event, TokenDelta):
                print(event.text, end='')  # streamed tokens
            else:
                result = event        # final Text2SQLResult

    Or with a config file::

        engine = Text2SQL.from_config("configs/config.yaml")
    """

    def __init__(
        self,
        db_uri: str | None = None,
        model: str | None = None,
        settings: Settings | None = None,
        **overrides: Any,
    ) -> None:
        """Initialize the pipeline.

        Args:
            db_uri: SQLAlchemy connection URI (overrides settings).
            model: LiteLLM model string (overrides settings).
            settings: Pre-built Settings object (optional).
            **overrides: Any other settings overrides.
        """
        if db_uri:
            overrides["db_uri"] = db_uri
        if model:
            overrides["model"] = model

        self.settings = settings or get_settings(**overrides)
        self.db = DatabaseConnection(
            self.settings.db_uri,
            connect_args=self.settings.athena_connect_args(),
        )
        self.llm = LLMClient(
            model=self.settings.model,
            temperature=self.settings.temperature,
            max_retries=self.settings.llm_retries,
            idle_ms=self.settings.llm_idle_ms,
            reasoning_effort=self.settings.reasoning_effort,
            **self.settings.bedrock_llm_kwargs(),
        )
        self.cache = ProfileCache(self.settings.profile_cache_dir)
        self.prompt_manager = PromptManager(
            template_dir=self.settings.prompt_template_dir,
            version=self.settings.prompt_version,
        )

        # Lazy-loaded components (in-memory memo of the last profiling artifacts)
        self._profile: DatabaseProfile | None = None
        self._summary: DatabaseSummary | None = None
        self._knowledge: DatabaseKnowledge | None = None
        self._linked: dict[str, list[str]] = {}
        self._shown_terms: set[int] = set()
        self._cached_value_index: tuple[DatabaseProfile, ValueIndex] | None = None
        self._traces: list[dict[str, Any]] = []  # agent traces of the current request
        self._conversations: list[list[dict[str, Any]]] = []  # every LLM exchange of it

    @classmethod
    def from_config(cls, config_path: str, **overrides: Any) -> Text2SQL:
        """Create an engine from a YAML config file.

        Args:
            config_path: The YAML file.
            **overrides: Values that win over the file, env and defaults.

        Returns:
            The engine.
        """
        settings = Settings.from_yaml(config_path, **overrides)
        return cls(settings=settings)

    @classmethod
    def build(cls, config: str | None = None, **overrides: Any) -> Text2SQL:
        """Construct an engine from an optional config file plus overrides.

        The shared entry point for the CLI and API.

        Args:
            config: YAML config path, or None for env and defaults.
            **overrides: Settings overrides; the caller filters out the None values.

        Returns:
            The engine.
        """
        if config:
            return cls.from_config(config, **overrides)
        return cls(**overrides)

    async def ask(self, question: str) -> AsyncIterator[PipelineEvent | TokenDelta | Text2SQLResult]:
        """Run the full pipeline.

        Args:
            question: Natural language question.

        Yields:
            A :class:`PipelineEvent` per stage milestone, at the detail
            ``settings.event_verbosity`` asks for, a :class:`TokenDelta` per streamed token
            during generation, then one terminal :class:`Text2SQLResult`.
        """
        emitter = EventEmitter()
        tracer = PipelineTracer()
        tracer.start_pipeline(question, self.settings.db_uri, self.settings.model)
        verbosity = self.settings.event_verbosity
        agent = self.settings.generation_mode == "agent"
        self.db.reset_cache()  # fresh per-request execution memo
        self._traces = []

        try:
            # 1. Profiling (uses the accumulated cache; see _profile_stream)
            yield emitter.emit(Stage.PROFILING, Status.STARTED, "Loading database profile...")
            step = tracer.start_step("profiling")
            profile, summary, knowledge = await self._get_or_build_profile()
            tracer.end_step(step, tables_profiled=len(profile.tables),
                            knowledge_entries=len(knowledge.entries))
            yield emitter.emit(
                Stage.PROFILING,
                Status.COMPLETED,
                f"Profiled {len(profile.tables)} tables, "
                f"{sum(len(t.columns) for t in profile.tables.values())} columns",
                tables_profiled=len(profile.tables),
            )
            if self.settings.stop_after == Stage.PROFILING:
                for stop_item in self._stopped(Stage.PROFILING, emitter, tracer):
                    yield stop_item
                return

            # 2. Schema Linking
            yield emitter.emit(
                Stage.SCHEMA_LINKING,
                Status.STARTED,
                f'Identifying relevant tables for: "{question[:80]}"',
            )
            step = tracer.start_step("schema_linking", question=question)
            schema_loader = SchemaLoader(self.db, profile=profile, summary=summary,
                                         prompts=self.prompt_manager)
            linked: dict[str, list[str]] = {}
            linked_tables: list[str] | None = None

            if self.settings.use_schema_linking:
                linker = SchemaLinker(
                    schema_loader,
                    self.llm,
                    self.prompt_manager,
                    modes=self.settings.schema_linking_modes,
                    direct=VariantSpec(
                        tuple(self.settings.direct_schema_scope),
                        tuple(self.settings.direct_descriptions),
                        self.settings.direct_knowledge),
                    reversed_=VariantSpec(
                        tuple(self.settings.reversed_schema_scope),
                        tuple(self.settings.reversed_descriptions),
                        self.settings.reversed_knowledge),
                    value_index=self._value_index(profile),
                    # Names + descriptions only (the template omits definitions): enough to
                    # map domain wording onto columns without spending their tokens yet.
                    knowledge=list(knowledge.entries.values()),
                    top_k=self.settings.value_top_k,
                    event_verbosity=verbosity,
                )
                async for linking in linker.link_stream(question, emitter):
                    if isinstance(linking, PipelineEvent):
                        yield linking
                    else:
                        linked = linking
                linked_tables = list(linked.keys()) if linked else None

            tables = self.db.get_schema()["tables"]
            tracer.end_step(step, linked_tables=linked_tables,
                            linked=linked if self.settings.use_schema_linking else None,
                            schema={"table": len(tables),
                                    "column": sum(len(i.get("columns", []))
                                                  for i in tables.values())})
            self._linked = linked  # describe_table types these and names the rest
            tables_desc = ", ".join(linked_tables) if linked_tables else "all tables"
            yield emitter.emit(
                Stage.SCHEMA_LINKING,
                Status.COMPLETED,
                f"Linked {len(linked)} tables: {tables_desc}",
                linked_tables=linked_tables,
            )
            if self.settings.stop_after == Stage.SCHEMA_LINKING:
                for stop_item in self._stopped(Stage.SCHEMA_LINKING, emitter, tracer):
                    yield stop_item
                return

            # One rendering whatever linking did, so linking on and off differ only in which
            # columns are present.
            schema_text = schema_loader.format_schema(
                fields=linked or None, explorable=agent)

            # Retrieval withholds the detail so the agent pulls it per table; only the agent's
            # own prompt is stripped, repair and selection keep the full schema.
            retrieval = agent and self.settings.agent_mode == "retrieval"
            agent_schema_text = schema_loader.format_table_list(
                list(linked or schema_loader.get_table_names())) if retrieval else schema_text

            scope = linked or {t: list(tp.columns) for t, tp in profile.tables.items()}
            terms = self._generation_knowledge(knowledge, scope, question)
            # Retrieval pulls definitions through `search_knowledge`; a *refuted* one is
            # inlined because it contradicts the question's own HINT, which is exactly when
            # the agent will not think to ask.
            inlined = [e for e in terms if e.refuted] if retrieval else terms
            # `search_columns` appends definitions too; without this it repeats them.
            self._shown_terms = {e.id for e in inlined}
            context = {
                "knowledge": inlined,
                "knowledge_full": self.settings.generation_knowledge == "full",
            }

            # 3. SQL Generation
            mode_label = "ReAct agent" if agent else "direct prompting"
            yield emitter.emit(
                Stage.SQL_GENERATION,
                Status.STARTED,
                f"Generating {self.settings.num_candidates} SQL candidates using {mode_label}...",
                num_candidates=self.settings.num_candidates,
                generation_mode=self.settings.generation_mode,
            )
            step = tracer.start_step(
                "sql_generation",
                num_candidates=self.settings.num_candidates,
                generation_mode=self.settings.generation_mode,
            )

            candidates: list[str] = []
            async for item in self._generate_candidates(
                    question, agent_schema_text, context, emitter):
                if isinstance(item, list):
                    candidates = item  # terminal item: the candidate list
                else:
                    yield item  # PipelineEvent or TokenDelta

            # Candidates are recorded so the benchmark can score each one (oracle@N).
            tracer.end_step(step, num_candidates=len(candidates), candidates=candidates,
                            mode=self.settings.generation_mode,
                            conversations=self._conversations,
                            **_agent_telemetry(self._traces, self.settings.agent_max_turns))
            # Raise so the failure travels the error path: silently continuing yields empty
            # SQL with no error, indistinguishable from a merely wrong answer.
            if not candidates:
                raise RuntimeError(
                    f"SQL generation produced no candidates using {mode_label}; "
                    f"see earlier SQL_GENERATION events for the underlying error"
                )
            yield emitter.emit(
                Stage.SQL_GENERATION,
                Status.COMPLETED,
                f"Generated {len(candidates)} candidates",
                num_candidates=len(candidates),
            )
            if self.settings.stop_after == Stage.SQL_GENERATION:
                for stop_item in self._stopped(Stage.SQL_GENERATION, emitter, tracer, sql=candidates[0]):
                    yield stop_item
                return

            # 4. SQL Repair
            if self.settings.use_repair and candidates:
                yield emitter.emit(
                    Stage.SQL_REPAIR,
                    Status.STARTED,
                    "Running the validation cascade "
                    f"(up to {self.settings.max_repair_retries} LLM fixes per candidate)...",
                )
                step = tracer.start_step("sql_repair")
                repair = SQLRepair(
                    self.llm,
                    self.db,
                    self.prompt_manager,
                    max_retries=self.settings.max_repair_retries,
                    # So a repair prompt renders only the tables the failing query touches
                    # instead of everything linking selected.
                    schema_loader=schema_loader,
                    knowledge=knowledge,
                )
                repaired: list[str] = []
                all_issues: list[str] = []
                for idx, sql in enumerate(candidates):
                    fixed, issues = await repair.repair(sql, question, schema_text)
                    repaired.append(fixed)
                    all_issues.extend(issues)

                    if verbosity in ("detailed", "verbose") and issues:
                        yield emitter.emit(
                            Stage.SQL_REPAIR,
                            Status.PROGRESS,
                            f"Candidate {idx + 1}: {len(issues)} issue(s): " + "; ".join(issues[:3]),
                            candidate_index=idx + 1,
                            issues=issues,
                        )

                candidates = repaired
                tracer.end_step(step, issues=all_issues)
                yield emitter.emit(
                    Stage.SQL_REPAIR,
                    Status.COMPLETED,
                    f"Repair complete: {len(all_issues)} total issues processed",
                    issues_count=len(all_issues),
                )

            # 5. Selection
            yield emitter.emit(
                Stage.SELECTION,
                Status.STARTED,
                "Executing candidates and selecting best...",
            )
            step = tracer.start_step("selection")
            selector = CandidateSelector(
                db=self.db,
                llm=self.llm,
                prompt_manager=self.prompt_manager,
                mode=self.settings.selection_mode,
            )
            selected_sql, results, sel_meta = await selector.select(
                candidates,
                question=question,
                schema_text=schema_text,
            )
            tracer.end_step(step, **sel_meta)
            method = sel_meta.get("method", "unknown")
            confidence = sel_meta.get("confidence", "")
            conf_str = f" ({confidence:.0%} agreement)" if isinstance(confidence, float) else ""
            yield emitter.emit(
                Stage.SELECTION,
                Status.COMPLETED,
                f"Selected via {method}{conf_str}",
                **sel_meta,
            )

            # 6. Pipeline complete
            tracer.end_pipeline(sql=selected_sql, results=results, llm_usage=self.llm.usage.summary())
            num_rows = len(results) if results else 0
            yield emitter.emit(
                Stage.PIPELINE,
                Status.COMPLETED,
                f"Final SQL ready ({len(selected_sql)} chars, {num_rows} result rows)",
                sql=selected_sql,
                num_results=num_rows,
            )
            yield Text2SQLResult(
                sql=selected_sql,
                results=results,
                trace=tracer.trace.to_dict(),
            )

        except Exception as e:
            logger.error("Pipeline failed: %s", e, exc_info=True)
            tracer.end_pipeline(error=str(e), llm_usage=self.llm.usage.summary())
            yield emitter.emit(
                Stage.PIPELINE,
                Status.ERROR,
                f"Pipeline failed: {e}",
                error=str(e),
            )
            yield Text2SQLResult(
                sql="",
                results=None,
                trace=tracer.trace.to_dict(),
                error=str(e),
            )

    def _stopped(
        self,
        stage: Stage,
        emitter: EventEmitter,
        tracer: PipelineTracer,
        sql: str = "",
    ) -> list[PipelineEvent | Text2SQLResult]:
        """Build the terminal items for a run halted by ``stop_after``.

        Args:
            stage: The stage the run stopped after.
            emitter: Event emitter.
            tracer: The run's tracer, whose trace already holds the stage's own outputs.
            sql: SQL the stopping stage produced, if any.

        Returns:
            A completed ``PIPELINE`` event and one :class:`Text2SQLResult`, the same
            contract as a full run, so every caller works unchanged on a partial one.
        """
        tracer.end_pipeline(sql=sql, llm_usage=self.llm.usage.summary())
        return [
            emitter.emit(Stage.PIPELINE, Status.COMPLETED, f"Stopped after {stage}", sql=sql),
            Text2SQLResult(sql=sql, results=None, trace=tracer.trace.to_dict()),
        ]

    async def _generate_candidates(
        self,
        question: str,
        schema_text: str,
        context: dict[str, Any],
        emitter: EventEmitter,
    ) -> AsyncIterator[PipelineEvent | TokenDelta | list[str]]:
        """Generate SQL candidates through the configured generation mode.

        Args:
            question: The user's question.
            schema_text: Schema for the generation prompt.
            context: Extra template arguments (knowledge).
            emitter: Event emitter.

        Yields:
            :class:`PipelineEvent` and :class:`TokenDelta` progress items, then the
            ``list[str]`` of candidates as the terminal item.
        """
        candidates: list[str] = []

        if self.settings.generation_mode == "agent":
            self._traces = []
            for i in range(self.settings.num_candidates):
                # Per candidate, so no tool state (the repeat-call guard, the review log)
                # carries from one attempt into the next.
                agent = self._build_agent(question)
                yield emitter.emit(
                    Stage.SQL_GENERATION,
                    Status.PROGRESS,
                    f"Generating candidate {i + 1}/{self.settings.num_candidates}...",
                    candidate_index=i + 1,
                )

                # Diversity comes from the strategy; the reshuffled schema and seed are
                # secondary, and `seed` is dropped by some providers without notice.
                async for item in agent.generate(
                    question,
                    randomize_schema_order(schema_text) if i > 0 else schema_text,
                    context=context,
                    dialect=self.db.dialect_name,
                    strategy=strategy_for(self.settings.generation_strategy, i),
                    seed=i,
                    emitter=emitter,
                ):
                    if isinstance(item, tuple):
                        sql, trace = item
                        self._traces.append(trace)
                        if sql:
                            candidates.append(sql)
                    else:
                        yield item
            self._conversations = [t["conversation"] for t in self._traces if t.get("conversation")]
        else:
            generator = SQLGenerator(self.llm, self.db, self.prompt_manager)
            async for gen_item in generator.generate(
                question,
                schema_text,
                num_candidates=self.settings.num_candidates,
                context=context,
                strategy_mode=self.settings.generation_strategy,
            ):
                if isinstance(gen_item, TokenDelta):
                    yield gen_item
                elif isinstance(gen_item, list):
                    candidates = gen_item  # final list[str]
            self._conversations = generator.conversations

        yield candidates

    def _build_agent(self, question: str = "") -> SQLAgent:
        """Build an agent with the configured tool set.

        Args:
            question: The user's question, which gates the agent's submission.

        Returns:
            The agent.
        """
        return SQLAgent(
            self.llm,
            self.prompt_manager,
            build_tools(
                self.db,
                SchemaLoader(self.db, self._profile, self._summary,
                             prompts=self.prompt_manager),
                self._knowledge or DatabaseKnowledge(),
                ExampleStore(self.settings.scenarios_file),
                tools=self.settings.agent_mode,
                question=question,
                linked=self._linked,
                shown=self._shown_terms,
            ),
            max_turns=self.settings.agent_max_turns,
            mode=self.settings.agent_mode,
            event_verbosity=self.settings.event_verbosity,
        )

    async def profile_database(self, build: bool = True) -> None:
        """Explicitly (re)profile the selected tables and upsert them into the cache.

        Args:
            build: False reuses the cached statistics and descriptions and derives only what
                the cache lacks, such as a knowledge base deleted to be regenerated.
        """
        async for _ in self._profile_stream(EventEmitter(), build=build):
            pass

    @property
    def cache_key(self) -> str:
        """Profile-cache key for this engine's DB URI; one set of files per database."""
        return self.cache.cache_key(self.settings.db_uri)

    async def _get_or_build_profile(self) -> tuple[DatabaseProfile, DatabaseSummary, DatabaseKnowledge]:
        """Load the artifacts for an ``ask()``, building only on a cold cache.

        Returns:
            ``(profile, summary, knowledge)``.
        """
        result: tuple[DatabaseProfile, DatabaseSummary, DatabaseKnowledge] | None = None
        async for item in self._profile_stream(EventEmitter(), build=False):
            if not isinstance(item, PipelineEvent):
                result = item
        assert result is not None
        return result

    def _value_index(self, profile: DatabaseProfile) -> ValueIndex | None:
        """Build the value index over the profile's own values.

        Args:
            profile: The database profile.

        Returns:
            The index, memoized against the profile it was built from so repeated ``ask()``
            calls on one database pay for it once and a new profile always gets its own.
            None when no selected mode has a use for it.
        """
        if not self.settings.value_index_enabled:
            return None
        if self._cached_value_index is None or self._cached_value_index[0] is not profile:
            index = ValueIndex.from_profile(profile)
            self._cached_value_index = (profile, index)
        return self._cached_value_index[1]

    def _generation_knowledge(
        self, knowledge: DatabaseKnowledge, scope: dict[str, list[str]], question: str
    ) -> list[KnowledgeEntry]:
        """Pick the domain terms the generation prompt should carry.

        Args:
            knowledge: The database's knowledge base.
            scope: The linked ``{table: [columns]}``, or everything profiled.
            question: The user's question, which may define terms itself.

        Returns:
            The terms the linked columns mention, minus any the question already defines. A
            refuted term is the exception both ways: it is carried only when the question does
            name it, since correcting the question's own copy is the whole reason to send it.
        """
        if self.settings.generation_knowledge == "off":
            return []
        selected = knowledge.select({n.lower() for t, c in scope.items() for n in (t, *c)},
                                    question)
        return [e for e in selected if e.refuted == e.named_in(question)]

    def _joins(self) -> str:
        """The declared foreign keys, one ``a.b -> c.d`` per line, for knowledge generation."""
        return "\n".join(
            f"{t}.{fk['column']} -> {fk['referred_table']}.{fk['referred_column']}"
            for t, info in self.db.get_schema()["tables"].items()
            for fk in info.get("foreign_keys", []))

    def _mismatches(self, table: str, predicate: str) -> int | None:
        """Count the rows a predicate selects.

        Args:
            table: Table to query.
            predicate: The WHERE clause.

        Returns:
            The row count, or None when the expression cannot run, which the knowledge
            verifiers must tell apart from a predicate no row satisfies.
        """
        rows, error = self.db.execute_safe(f"SELECT COUNT(*) AS n FROM {table} WHERE {predicate}")
        return None if error or not rows else int(rows[0]["n"])

    def _load_metadata(self, key: str) -> tuple[DatabaseSummary, DatabaseKnowledge]:
        """Load descriptions and knowledge from the configured source, never merging the two.

        The shipped and generated knowledge bases number their entries independently, so a
        merged view can only renumber one of them and nothing downstream could then resolve
        an id against it.

        Args:
            key: Cache key.

        Returns:
            ``(summary, knowledge)``.
        """
        short = entries(self.cache.load(key, "meaning_base_short"))
        long = entries(self.cache.load(key, "meaning_base_long"))
        return (DatabaseSummary.from_flat(short, long),
                DatabaseKnowledge.from_flat(self.cache.load(key, "kb")))

    def artifacts(self) -> dict[str, dict[str, Any]]:
        """Read the four artifacts in cache-document form.

        Returns:
            The profile, both meaning bases and the KB.
        """
        key = self.cache_key
        summary, knowledge = self._load_metadata(key)
        return {
            "profile": self.cache.load(key, "profile"),
            "meaning_base_short": {"columns": summary.to_flat(key, long=False)},
            "meaning_base_long": {"columns": summary.to_flat(key, long=True)},
            "kb": {"columns": knowledge.to_flat(key)},
        }

    def profile_stream(self, emitter: EventEmitter) -> AsyncIterator[Any]:
        """Stream a full (re)profile of the selection, for the API's ``/profile``.

        Args:
            emitter: Event emitter.

        Returns:
            The stream of progress events, ending in ``(profile, summary, knowledge)``.
        """
        return self._profile_stream(emitter, build=True)

    async def _profile_stream(self, emitter: EventEmitter, *, build: bool) -> AsyncIterator[Any]:
        """Load or build the profiling artifacts.

        The cache is a single accumulating document per database, so a partial run upserts
        into it rather than replacing it. Summaries are (re)generated for the tables built
        this run plus any lacking a cached summary.

        Args:
            emitter: Event emitter.
            build: True (re)profiles the selected tables, or the whole database when there
                is no selection. False reads the cache as-is and builds only when it is
                completely cold, so the inference path stays light and a
                selected-but-uncached table is skipped with a warning.

        Yields:
            Progress events, then the terminal ``(profile, summary, knowledge)``, filtered
            to ``settings.profile_selection``.
        """
        if self._profile and self._summary:
            yield self._profile, self._summary, self._knowledge or DatabaseKnowledge()
            return

        key = self.cache_key
        selection = self.settings.profile_selection
        profile = DatabaseProfile.from_flat(self.cache.load(key, "profile"), self.db.dialect_name)
        summary, knowledge = self._load_metadata(key)

        built_cols: dict[str, set[str]] = {}  # columns actually (re)profiled this run
        if build or not profile.tables:
            profiler = StatsProfiler(
                self.db,
                top_k=self.settings.profile_top_k,
                sample_size=self.settings.profile_sample_size,
                exact_top_k=self.settings.profile_exact_top_k,
                approx_distinct=self.settings.profile_approx_distinct,
                sample_aggregates=self.settings.profile_sample_aggregates,
            )
            fresh = DatabaseProfile(dialect=self.db.dialect_name)
            for fresh, table, idx, total in profiler.iter_profile(selection):
                tp = fresh.tables.get(table)
                ncols = len(tp.columns) if tp else 0
                yield emitter.emit(
                    Stage.PROFILING,
                    Status.PROGRESS,
                    f"Profiled {table}: {ncols} column(s) ({idx}/{total})",
                    table=table,
                    columns_profiled=ncols,
                    tables_done=idx,
                    tables_total=total,
                )
                await asyncio.sleep(0)  # hand control back so the event flushes
            built_cols = {t: set(tp.columns) for t, tp in fresh.tables.items()}
            # Upsert, then reload so the result is the merged document, not just this run.
            self.cache.save(key, "profile", fresh.to_flat(key), fresh.table_meta())
            profile = DatabaseProfile.from_flat(self.cache.load(key, "profile"), self.db.dialect_name)
        else:
            cached = set(profile.tables)
            missing = sorted((set(selection) if selection else cached) - cached)
            if missing:
                yield emitter.emit(
                    Stage.PROFILING,
                    Status.PROGRESS,
                    f"Skipping {len(missing)} unprofiled table(s): {', '.join(missing)}",
                    skipped_tables=missing,
                )
            yield emitter.emit(Stage.PROFILING, Status.PROGRESS, f"Loaded cached profile ({len(cached)} tables)")

        # Summarize only freshly-profiled columns plus any still missing a cached summary;
        # never re-summarize columns that are already cached and unchanged.
        pending: dict[str, list[str]] = {}
        for t, tp in profile.tables.items():
            have = {c for c, s in summary.columns.get(t, {}).items() if s.short_summary or s.long_summary}
            want = [c for c in tp.columns if c in built_cols.get(t, set()) or c not in have]
            if want:
                pending[t] = want

        # Tables with no entries at all, so a deleted KB rebuilds without re-summarizing.
        covered = {t for e in knowledge.entries.values() for t in e.tables}
        missing_kb = [t for t in profile.tables if t not in covered]

        if pending:
            total_cols = sum(len(v) for v in pending.values())
            # Batches (tables or columns) are summarized concurrently inside the summarizer,
            # so progress here is coarse: one event before, one after.
            yield emitter.emit(
                Stage.PROFILING,
                Status.PROGRESS,
                f"Summarizing {total_cols} column(s) across {len(pending)} table(s)",
                columns_done=0,
                columns_total=total_cols,
            )
            await asyncio.sleep(0)
            summarizer = ProfileSummarizer(
                self.llm,
                self.prompt_manager,
                one_call_per_table=self.settings.profile_one_call_per_table,
            )
            mode = self.settings.profile_summary
            # A description is an enrichment, not a precondition: a column without one still
            # links by name, and failing here would take a whole database offline.
            try:
                merged = await summarizer.summarize_database(
                    profile, only=pending,
                    generate_short=mode != "long", generate_long=mode != "short",
                )
                for long in (False, True):  # column-level merge into each cached meaning base
                    self.cache.save(key, f"meaning_base_{'long' if long else 'short'}",
                                    merged.to_flat(key, long=long))
                summary, knowledge = self._load_metadata(key)
                yield emitter.emit(
                    Stage.PROFILING,
                    Status.PROGRESS,
                    f"Summarized {total_cols} column(s)",
                    columns_done=total_cols,
                    columns_total=total_cols,
                )
            except Exception as e:
                logger.warning("Summarizing %d column(s) failed, continuing without them: %s",
                               total_cols, e)
                pending, missing_kb = {}, []  # nothing described, so nothing to derive either
                yield emitter.emit(
                    Stage.PROFILING,
                    Status.PROGRESS,
                    f"Could not summarize {total_cols} column(s); using names only",
                    columns_total=total_cols,
                    summarization_failed=True,
                )

        # Knowledge covers whole tables: any summarized this run, else any lacking entries.
        stale = list(pending) or missing_kb
        if self.settings.profile_kb and stale:
            yield emitter.emit(Stage.PROFILING, Status.PROGRESS,
                               f"Deriving knowledge for {len(stale)} table(s)")
            await asyncio.sleep(0)
            generated = await KnowledgeGenerator(self.llm, self.prompt_manager).generate(
                profile, summary.get_short, stale, joins=self._joins())
            generated = generated.verified(
                {t: set(tp.columns) for t, tp in profile.tables.items()}, self._mismatches)
            # A re-profiled table replaces its own entries; merge() keeps the ids unique.
            kept = DatabaseKnowledge.from_flat(self.cache.load(key, "kb")).without(set(stale))
            self.cache.save(key, "kb", kept.merge(generated).to_flat(key), replace=True)
            _, knowledge = self._load_metadata(key)
            yield emitter.emit(Stage.PROFILING, Status.PROGRESS,
                               f"Derived {len(generated.entries)} knowledge entries")

        profile, summary, knowledge = self._filter(profile, summary, knowledge, selection)
        self._profile, self._summary, self._knowledge = profile, summary, knowledge
        yield profile, summary, knowledge

    @staticmethod
    def _filter(
        profile: DatabaseProfile, summary: DatabaseSummary, knowledge: DatabaseKnowledge,
        selection: dict[str, list[str]] | None,
    ) -> tuple[DatabaseProfile, DatabaseSummary, DatabaseKnowledge]:
        """Trim the artifacts to a selected subset.

        Args:
            profile: The database profile.
            summary: The column summaries.
            knowledge: The knowledge base.
            selection: ``{table: [columns]}``; an empty column list means the whole table.

        Returns:
            The trimmed ``(profile, summary, knowledge)``.
        """
        if not selection:
            return profile, summary, knowledge
        fp, fs = DatabaseProfile(dialect=profile.dialect), DatabaseSummary()
        for table, cols in selection.items():
            tp = profile.tables.get(table)
            if not tp:
                continue
            want = set(cols) if cols else set(tp.columns)
            ftp = TableProfile(table_name=tp.table_name, row_count=tp.row_count)
            ftp.columns = {c: v for c, v in tp.columns.items() if c in want}
            fp.tables[table] = ftp
            if table in summary.columns:
                fs.columns[table] = {c: v for c, v in summary.columns[table].items() if c in want}
        fk = DatabaseKnowledge(
            {i: e for i, e in knowledge.entries.items()
             if set(e.tables) <= selection.keys() or not e.tables})
        return fp, fs, fk

    def close(self) -> None:
        """Close the database connection."""
        self.db.close()

    def __enter__(self) -> Text2SQL:
        """Enter the context manager.

        Returns:
            This engine.
        """
        return self

    def __exit__(self, *args: Any) -> None:
        """Close the engine on exit."""
        self.close()
