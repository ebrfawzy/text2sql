"""Centralized configuration via Pydantic Settings."""

from __future__ import annotations

import logging
import os
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, get_args

import yaml
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# PyYAML is YAML 1.1: a bare off/no resolves to False and a valueless key to None, so
# `knowledge: off` and `stop_after:` reach their str Literal as the wrong type entirely.
_YAML_TOKENS: dict[Any, tuple[str, ...]] = {True: ("on", "yes"), False: ("off", "no"), None: ("",)}

def _token(field: str, value: Any) -> Any:
    """Map a YAML-resolved bare token back onto the literal its field declares.

    Args:
        field: The Settings field name.
        value: The value PyYAML resolved.

    Returns:
        The matching literal, or the value unchanged.
    """
    if not isinstance(value, bool) and value is not None:
        return value
    options = get_args(Settings.model_fields[field].annotation)
    return next((o for o in options if o in _YAML_TOKENS[value]), value)

class LogLevel(StrEnum):
    """Supported log levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

# Nested YAML ``section -> {yaml_key: field}``; a dict value adds one nesting level.
YAML_SECTION_MAP: dict[str, dict[str, Any]] = {
    "general": {"log_level": "log_level", "log_transcript": "log_transcript",
                "stop_after": "stop_after"},
    "streaming": {"event_verbosity": "event_verbosity"},
    # Key order is what the UI renders, so a controller is listed before what it gates:
    # reasoning_effort forces temperature=1, num_candidates decides selection_mode.
    "litellm": {
        "model": "model",
        "reasoning_effort": "reasoning_effort",
        "temperature": "temperature",
        "retries": "llm_retries",
        "idle_ms": "llm_idle_ms",
    },
    "bedrock": {
        "aws_access_key_id": "bedrock_aws_access_key_id",
        "aws_secret_access_key": "bedrock_aws_secret_access_key",
        "aws_session_token": "bedrock_aws_session_token",
        "aws_region": "bedrock_aws_region",
    },
    "sqlalchemy": {"db_uri": "db_uri"},
    "athena": {
        "aws_access_key_id": "athena_aws_access_key_id",
        "aws_secret_access_key": "athena_aws_secret_access_key",
        "aws_session_token": "athena_aws_session_token",
        "aws_region": "athena_aws_region",
        "s3_staging_dir": "athena_s3_staging_dir",
    },
    "profiling": {
        "cache_dir": "profile_cache_dir",
        "top_k": "profile_top_k",
        "sample_size": "profile_sample_size",
        "selection": "profile_selection",
        "one_call_per_table": "profile_one_call_per_table",
        "exact_top_k": "profile_exact_top_k",
        "approx_distinct": "profile_approx_distinct",
        "sample_aggregates": "profile_sample_aggregates",
        "summary": "profile_summary",
        "kb": "profile_kb",
    },
    "schema_linking": {
        "enabled": "use_schema_linking",
        "modes": "schema_linking_modes",
        "direct": {
            "schema_scope": "direct_schema_scope",
            "descriptions": "direct_descriptions",
            "knowledge": "direct_knowledge",
        },
        "reversed": {
            "schema_scope": "reversed_schema_scope",
            "descriptions": "reversed_descriptions",
            "knowledge": "reversed_knowledge",
        },
        "value": {
            "top_k": "value_top_k",
        },
    },
    "sql_generation": {
        "mode": "generation_mode",
        "strategy": "generation_strategy",
        "num_candidates": "num_candidates",
        "knowledge": "generation_knowledge",
        "prompts": {
            "template_dir": "prompt_template_dir",
            "version": "prompt_version",
        },
        "agent": {
            "max_turns": "agent_max_turns",
            "mode": "agent_mode",
            "scenarios_file": "scenarios_file",
        },
    },
    "verification": {
        "repair": "use_repair",
        "repair_max_retries": "max_repair_retries",
        "selection_mode": "selection_mode",
    },
    "benchmark": {
        "instance_id": "benchmark_instance_id",
        "output_dir": "benchmark_output_dir",
        "data_jsonl": "benchmark_data_jsonl",
        "testcases_jsonl": "benchmark_testcases_jsonl",
        "dataset_folder": "benchmark_dataset_folder",
        "use_knowledge": "benchmark_use_knowledge",
    },
}

_PIPELINE = ("ask", "benchmark")
_EVERY = ("ask", "profile", "benchmark")

# Section -> (UI label, endpoints). Omitted sections (bedrock, athena) stay off the wire.
SECTION_SCOPE: dict[str, tuple[str, tuple[str, ...]]] = {
    "general": ("General", _EVERY),
    "streaming": ("Streaming", _PIPELINE),
    "litellm": ("LLM / LiteLLM", _EVERY),
    "sqlalchemy": ("Database", ("ask", "profile")),
    "profiling": ("Profiling", _EVERY),
    "schema_linking": ("Schema Linking", _PIPELINE),
    "sql_generation": ("SQL Generation", _PIPELINE),
    "verification": ("Verification", _PIPELINE),
    "benchmark": ("Benchmark", ("benchmark",)),
}

# Field -> (controlling field, values that make it live); anything else is inert.
# A scalar controller must be *one of* the values; a list controller must contain *all*
# of them, which is how a field gated on two selected modes at once is expressed.
FIELD_DEPENDS: dict[str, tuple[str, tuple[Any, ...]]] = {
    # Extended thinking forces temperature=1, so the configured value is never read.
    "temperature": ("reasoning_effort", ("none",)),
    "agent_max_turns": ("generation_mode", ("agent",)),
    "agent_mode": ("generation_mode", ("agent",)),
    "scenarios_file": ("agent_mode", ("retrieval",)),
    "schema_linking_modes": ("use_schema_linking", (True,)),
    "direct_schema_scope": ("schema_linking_modes", ("direct",)),
    "direct_descriptions": ("schema_linking_modes", ("direct",)),
    "direct_knowledge": ("schema_linking_modes", ("direct",)),
    "reversed_schema_scope": ("schema_linking_modes", ("reversed",)),
    "reversed_descriptions": ("schema_linking_modes", ("reversed",)),
    "reversed_knowledge": ("schema_linking_modes", ("reversed",)),
    # Candidate ranking serves `value` mode *and* any `focused` scope, so these stay live for
    # the whole stage — gating them on the mode would grey them out for a focused direct run.
    "value_top_k": ("use_schema_linking", (True,)),
    "max_repair_retries": ("use_repair", (True,)),
    "selection_mode": ("num_candidates", (2, 3)),
}

def host_path(value: str) -> str:
    """Rewrite a Windows-style path for a POSIX host, such as a Docker container.

    Args:
        value: A path or URI as written.

    Returns:
        The path with separators normalized, or unchanged when it already resolves.
    """
    if os.sep == "\\" or "\\" not in value:
        return value
    scheme, sep, rest = value.partition("://")
    if sep:  # a URI: only the part after the scheme is a path
        return scheme + sep + rest.replace("\\", "/")
    return value if Path(value).exists() else value.replace("\\", "/")

def section_items(section: str) -> list[tuple[str, str]]:
    """List a section's fields, nested subsections flattened.

    Args:
        section: The YAML section name.

    Returns:
        ``(display key, field name)`` pairs. A subsection keeps its name in the display key,
        so siblings can each carry a ``threshold`` without rendering as two "Threshold"s.
    """
    out: list[tuple[str, str]] = []
    for key, value in YAML_SECTION_MAP[section].items():
        out += ([(f"{key} {k}", f) for k, f in value.items()]
                if isinstance(value, dict) else [(key, value)])
    return out

def section_fields(endpoint: str) -> list[str]:
    """List the fields an endpoint accepts.

    Args:
        endpoint: ``ask``, ``profile`` or ``benchmark``.

    Returns:
        Every ``Settings`` field it accepts, in section order.
    """
    return [f for section, (_, endpoints) in SECTION_SCOPE.items() if endpoint in endpoints
            for _, f in section_items(section)]

class Settings(BaseSettings):
    """Application settings, grouped by pipeline stage.

    Precedence, highest first: constructor kwargs, env vars (``TEXT2SQL_``), ``.env``, YAML,
    defaults.
    """

    model_config = SettingsConfigDict(
        env_prefix="TEXT2SQL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── General ──────────────────────────────────────────────────────
    log_level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Root logging level; third-party libraries stay at WARNING regardless.",
    )
    log_transcript: bool = Field(
        default=False,
        description="Log every message sent to the model and every reply, tool calls "
        "included. Independent of log_level.",
    )
    stop_after: Literal["", "profiling", "schema_linking", "sql_generation"] = Field(
        default="",
        description="Halt the pipeline after this stage and return the partial result — for "
        "stage-level ablations (e.g. benchmarking schema linking alone). Empty runs every stage.",
    )

    # ── Streaming ─────────────────────────────────────────────────────
    event_verbosity: Literal["minimal", "detailed", "verbose"] = Field(
        default="verbose",
        description="Streaming event detail level.",
    )

    # ── LiteLLM ──────────────────────────────────────────────────────
    model: str = Field(
        default="gpt-4o-mini",
        description="LiteLLM-compatible model identifier "
        "(e.g. gpt-4o, anthropic/claude-sonnet-4-20250514, ollama/llama3).",
    )
    reasoning_effort: Literal["none", "low", "medium", "high"] = Field(
        default="none",
        description="Extended-thinking depth. 'none' keeps the configured temperature; any "
        "other value forces temperature=1, which Anthropic requires while thinking is on.",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for every LLM call. 0 keeps generation deterministic.",
    )
    llm_retries: int = Field(
        default=3,
        ge=0,
        description="Number of retries on transient LLM errors (rate limits, timeouts).",
    )
    llm_idle_ms: int = Field(
        default=1000,
        ge=0,
        description="Milliseconds to wait between LLM retries (backoff base).",
    )

    # ── AWS Bedrock (LLM account) ────────────────────────────────────
    # Routed to LiteLLM; unset falls back to the ambient AWS_* / boto3 chain.
    bedrock_aws_access_key_id: str | None = Field(default=None)
    bedrock_aws_secret_access_key: str | None = Field(default=None)
    bedrock_aws_session_token: str | None = Field(
        default=None,
        description="Session token for temporary STS credentials.",
    )
    bedrock_aws_region: str | None = Field(default=None)

    # ── SQLAlchemy ───────────────────────────────────────────────────
    db_uri: str = Field(
        default="sqlite:///example.db",
        description="SQLAlchemy connection URI.",
    )

    # ── AWS Athena (database account) ────────────────────────────────
    # Routed to pyathena connect_args for ``awsathena://`` URIs only; unset falls back to boto3.
    athena_aws_access_key_id: str | None = Field(default=None)
    athena_aws_secret_access_key: str | None = Field(default=None)
    athena_aws_session_token: str | None = Field(
        default=None,
        description="Session token for temporary STS credentials.",
    )
    athena_aws_region: str | None = Field(default=None)
    athena_s3_staging_dir: str | None = Field(
        default=None,
        description="S3 location for Athena query results (pyathena s3_staging_dir).",
    )

    # ── Profiling ────────────────────────────────────────────────────
    profile_cache_dir: str = Field(
        default=".cache/profiles",
        description="Directory (or s3://bucket/prefix) for cached profiles.",
    )
    profile_top_k: int = Field(
        default=10,
        ge=1,
        description="Number of top-k frequent values to collect per column.",
    )
    profile_sample_size: int = Field(
        default=10_000,
        ge=100,
        description="Max rows to sample for value shape analysis and minhash.",
    )
    profile_selection: dict[str, list[str]] | None = Field(
        default=None,
        description="{table: [columns]} subset to profile / consume; None uses the whole DB. "
        "At ask() time this filters which already-cached tables reach the pipeline.",
    )
    profile_one_call_per_table: bool = Field(
        default=True,
        description="Summarize a whole table per LLM call (True) vs one call per column (False). "
        "Table mode issues far fewer inferences; both produce short + long summaries.",
    )
    profile_exact_top_k: bool = Field(
        default=True,
        description="Exact top-k via per-column GROUP BY (True, extra queries) vs from the shared "
        "sample (False, cheaper). Exact improves linking recall on large tables.",
    )
    profile_approx_distinct: bool = Field(
        default=False,
        description="Approximate distinct-count where the dialect supports it, exact elsewhere. "
        "Far cheaper on warehouses and only affects LLM-facing prose, never pipeline decisions.",
    )
    profile_sample_aggregates: bool = Field(
        default=False,
        description="Aggregate over a profile_sample_size sample (True, approximate but far fewer "
        "bytes scanned on huge tables) instead of an exact full scan (False).",
    )
    profile_summary: Literal["short", "long", "short_and_long"] = Field(
        default="short_and_long",
        description="Which column descriptions to generate; cached as meaning_base_short/_long, "
        "and consumers fall back to whichever exists.",
    )
    profile_kb: bool = Field(
        default=True,
        description="Derive a knowledge base of cross-column relations (one LLM call per table). "
        "Where it is injected is set per prompt, by the linking and generation knowledge levels.",
    )

    # ── Schema Linking ───────────────────────────────────────────────
    use_schema_linking: bool = Field(
        default=True,
        description="Enable schema linking pass.",
    )
    schema_linking_modes: list[Literal["direct", "reversed", "value"]] = Field(
        default=["value"],
        description="Which linkers run, unioned: schema-to-question (direct), question-to-SQL "
        "(reversed), and name/value matching against the profile (value). Any combination.",
    )
    direct_schema_scope: list[Literal["full", "focused"]] = Field(
        default=["full"],
        description="Which schema direct linking is shown. 'focused' is a non-LLM pre-filter — "
        "fields whose name or description the question matches, plus fields holding values it "
        "quotes — not a product of linking. Selecting it alone caps recall at whatever that "
        "filter found, so pair it with 'full'. Its value half needs the 'value' mode selected.",
    )
    direct_descriptions: list[Literal["short", "long"]] = Field(
        default=["short"],
        description="Which generated column descriptions accompany direct linking's schema. "
        "Each scope x description pair is one prompt; the answers are unioned.",
    )
    direct_knowledge: Literal["off", "terms", "full"] = Field(
        default="full",
        description="Domain knowledge in the direct linking prompt: none, term names with their "
        "descriptions (terms), or those plus each term's definition (full).",
    )
    reversed_schema_scope: list[Literal["full", "focused"]] = Field(
        default=["full"],
        description="Which schema reversed linking is shown. 'focused' is a non-LLM pre-filter — "
        "fields whose name or description the question matches, plus fields holding values it "
        "quotes — not a product of linking. Selecting it alone caps recall at whatever that "
        "filter found, so pair it with 'full'. Its value half needs the 'value' mode selected.",
    )
    reversed_descriptions: list[Literal["short", "long"]] = Field(
        default=["short"],
        description="Which generated column descriptions accompany reversed linking's schema. "
        "Each scope x description pair is one generated query; the fields are unioned.",
    )
    reversed_knowledge: Literal["off", "terms", "full"] = Field(
        default="full",
        description="Domain knowledge in the reversed linking prompt: none, term names with "
        "their descriptions (terms), or those plus each term's definition (full).",
    )

    value_top_k: int = Field(
        default=30,
        ge=1,
        le=500,
        description="Floor under the candidate budget, which is 45% of the schema. Over 270 "
        "questions on 18 LiveSQLBench databases that budget keeps 49% of the columns and "
        "carries every gold column of 85.5% of questions - 95% of those whose gold columns "
        "all exist in the schema, the rest naming a CTE or an output alias. Raising it keeps "
        "buying recall (61% of the schema reaches 88%) but the candidate set is also the "
        "generation prompt's schema, where a wider one measured worse end to end.",
    )

    @property
    def value_index_enabled(self) -> bool:
        """Whether the value index is needed; implied by the modes, never set separately."""
        return self.use_schema_linking and "value" in self.schema_linking_modes

    # ── SQL Generation ───────────────────────────────────────────────

    generation_mode: Literal["direct", "agent"] = Field(
        default="direct",
        description="Single-shot prompting (direct), or the ReAct agent that executes and "
        "refines its own query (agent).",
    )
    generation_strategy: Literal["direct", "decompose", "query_plan", "diverse"] = Field(
        default="direct",
        description="Reasoning style per candidate; 'diverse' cycles through all of them so "
        "candidates disagree structurally rather than by sampling noise.",
    )
    num_candidates: int = Field(
        default=1,
        ge=1,
        le=3,
        description="Candidate SQL queries to generate. Above 1 requires "
        "generation_strategy='diverse', whose three styles are also the ceiling.",
    )
    generation_knowledge: Literal["off", "terms", "full"] = Field(
        default="full",
        description="Domain knowledge in the SQL generation and agent prompts: none, term names "
        "with their descriptions (terms), or those plus each term's definition (full). Only terms "
        "the linked columns mention are sent, minus any the question already defines — unless the "
        "data refuted that definition. agent_mode='retrieval' withholds it entirely.",
    )
    prompt_template_dir: str | None = Field(
        default=None,
        description="Root directory for Jinja2 prompt templates. Defaults to bundled templates.",
    )
    prompt_version: str = Field(
        default="v1",
        description="Template version subdirectory to load (e.g. 'v1', 'v2').",
    )

    # ── Agent (generation_mode="agent") ──────────────────────────────
    agent_max_turns: int = Field(
        default=20,
        ge=1,
        description="Maximum agent turns before it must answer with what it has.",
    )
    agent_mode: Literal["schema_preloaded", "retrieval"] = Field(
        default="retrieval",
        description="What the agent is given and may call: the linked schema plus execute/review/"
        "submit (schema_preloaded); or table names only, so it pulls detail and definitions through the "
        "schema and knowledge tools (retrieval, which scales to wide schemas).",
    )
    scenarios_file: str | None = Field(
        default=None,
        description="Path to scenarios.md for the lookup_example tool (agent mode='retrieval').",
    )

    # ── Verification ─────────────────────────────────────────────────
    use_repair: bool = Field(
        default=True,
        description="Enable SQL repair/validation pass.",
    )
    max_repair_retries: int = Field(
        default=3,
        ge=0,
        description="Max LLM retries per candidate during repair.",
    )
    selection_mode: Literal["majority", "confidence", "single"] = Field(
        default="single",
        description="Most common result set (majority), the same plus LLM adjudication when "
        "every candidate disagrees (confidence), or the first candidate (single).",
    )

    # ── Benchmark ────────────────────────────────────────────────────
    benchmark_instance_id: str | None = Field(
        default=None,
        description="Run only these instances: a comma-separated list of id prefixes, so "
                    "'credit' is a whole database and 'credit_1,fake_9' is two questions "
                    "(a prefix also matches longer ids, so 'credit_1' takes 'credit_10').",
    )
    benchmark_output_dir: str = Field(
        default="results",
        description="Output directory for benchmark reports.",
    )
    benchmark_data_jsonl: str | None = Field(
        default=None,
        description="Path to the LiveSQLBench release JSONL (questions, databases, conditions).",
    )
    benchmark_testcases_jsonl: str | None = Field(
        default=None,
        description="Path to the ground-truth JSONL (sol_sql, external_knowledge, test_cases).",
    )
    benchmark_dataset_folder: str | None = Field(
        default=None,
        description="Path to the local dataset folder for benchmarking.",
    )
    benchmark_use_knowledge: bool = Field(
        default=True,
        description="Append each instance's external knowledge (the dataset's own KB entries, "
                    "LaTeX rewritten as SQL) to the question. "
                    "the ids index the dataset KB the loader reads directly.",
    )

    @field_validator(
        "db_uri", "profile_cache_dir", "prompt_template_dir", "scenarios_file",
        "benchmark_output_dir", "benchmark_data_jsonl", "benchmark_testcases_jsonl",
        "benchmark_dataset_folder",
    )
    @classmethod
    def normalize_path(cls, v: str | None) -> str | None:
        """Accept host-written paths on any platform.

        Args:
            v: The configured path.

        Returns:
            The path, normalized by :func:`host_path`.
        """
        return host_path(v) if isinstance(v, str) else v

    @field_validator("db_uri")
    @classmethod
    def validate_db_uri(cls, v: str) -> str:
        """Check that the DB URI carries a scheme.

        Args:
            v: The configured URI.

        Returns:
            The URI unchanged.

        Raises:
            ValueError: The URI has no ``://``.
        """
        if "://" not in v:
            raise ValueError(f"Invalid SQLAlchemy URI (missing ://): {v}")
        return v

    def inactive(self) -> list[str]:
        """Find the fields this config rules out.

        Returns:
            Every field whose :data:`FIELD_DEPENDS` chain is not satisfied. Dependencies
            chain, so a field is live only when its whole chain is: switching schema linking
            off takes the per-mode settings with it, even though the mode list still holds
            the values they name.
        """
        memo: dict[str, bool] = {}

        def live(field: str) -> bool:
            """Whether a field's whole dependency chain is satisfied.

            Args:
                field: The field name.

            Returns:
                True when the field is live.
            """
            if field not in FIELD_DEPENDS:
                return True
            if field not in memo:
                dep, ok = FIELD_DEPENDS[field]
                value = getattr(self, dep)
                held = set(ok) <= set(value) if isinstance(value, list) else value in ok
                memo[field] = held and live(dep)
            return memo[field]

        return [f for f in FIELD_DEPENDS if not live(f)]

    @model_validator(mode="after")
    def finalize(self) -> Settings:
        """Configure logging, then reject unusable combinations and flag inert settings.

        Returns:
            The validated settings.

        Raises:
            ValueError: The settings ask for candidates that could only differ by sampling.
        """
        from rich.logging import RichHandler

        logging.basicConfig(
            level=self.log_level.value, format="%(message)s", datefmt="%H:%M:%S", force=True,
            handlers=[RichHandler(rich_tracebacks=True, show_path=False, markup=False)],
        )
        for noisy in ("LiteLLM", "httpcore", "httpx", "botocore", "boto3", "urllib3", "s3transfer"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
        # sqlglot warns per statement on SQLite DDL it cannot model; `parse_sql` handles it.
        logging.getLogger("sqlglot").setLevel(logging.ERROR)
        logging.getLogger("text2sql.llm").setLevel(
            logging.DEBUG if self.log_transcript else logging.INFO)

        if self.num_candidates > 1 and self.generation_strategy != "diverse":
            raise ValueError(
                f"num_candidates={self.num_candidates} needs generation_strategy='diverse': other "
                "strategies vary only by sampling, and bedrock anthropic drops `seed` entirely"
            )
        if self.num_candidates > 1 and self.selection_mode == "single":
            logger.warning(
                "selection_mode='single' takes candidate 1, so the other %d are generated and "
                "discarded; one measured run threw away the only candidate that submitted",
                self.num_candidates - 1,
            )
        # Only a deliberately non-default value is worth warning about.
        inert = sorted(f for f in set(self.inactive()) & self.model_fields_set
                       if getattr(self, f) != type(self).model_fields[f].default)
        if inert:
            logger.warning(
                "These settings have no effect as configured: %s",
                "; ".join(f"{f} (needs {(d := FIELD_DEPENDS[f][0])}, is {getattr(self, d)!r})"
                          for f in inert),
            )
        return self

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides: Any) -> Settings:
        """Load a YAML config, merged with env and defaults.

        Args:
            path: The YAML file.
            **overrides: Values that win over everything else.

        Returns:
            The settings. YAML reaches pydantic as constructor kwargs, which outrank its env
            sources, so env-supplied fields are dropped to restore the documented order.

        Raises:
            FileNotFoundError: The config file does not exist.
        """
        path = Path(host_path(str(path)))
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        # Flatten nested sections to flat field names.
        flat: dict[str, Any] = {}
        for section, mapping in YAML_SECTION_MAP.items():
            if not isinstance(block := raw.get(section), dict):
                continue
            for key, value in mapping.items():
                scope = block.get(key) if isinstance(value, dict) else block
                for yaml_key, field_name in (value.items() if isinstance(value, dict) else [(key, value)]):
                    if isinstance(scope, dict) and yaml_key in scope:
                        flat[field_name] = scope[yaml_key]

        for key in cls.model_fields:  # also accept top-level keys
            if key in raw:
                flat[key] = raw[key]
        flat = {k: _token(k, v) for k, v in flat.items()}

        # On a plain construction, model_fields_set is exactly what env/.env supplied. A
        # cross-field rule can make that construction invalid (env alone setting
        # num_candidates>1); only the names matter here, so fall back to the env itself.
        try:
            env_supplied = cls().model_fields_set
        except ValueError:
            prefix = cls.model_config["env_prefix"]
            env_supplied = {f for f in cls.model_fields if f"{prefix}{f}".upper() in os.environ}
        if shadowed := sorted(env_supplied & flat.keys()):
            logger.debug("YAML keys overridden by env/.env: %s", ", ".join(shadowed))
        flat = {k: v for k, v in flat.items() if k not in env_supplied}
        flat.update(overrides)  # overrides win

        return cls(**flat)

    def athena_connect_args(self, db_uri: str | None = None) -> dict[str, Any]:
        """Build the pyathena ``connect_args``.

        Args:
            db_uri: The per-request URI, when it may differ from settings.

        Returns:
            The connect args, or ``{}`` unless the URI is ``awsathena://``. Unset
            credentials are omitted so they fall back to the boto3 chain.
        """
        if not (db_uri or self.db_uri).startswith("awsathena"):
            return {}
        pairs = {
            "aws_access_key_id": self.athena_aws_access_key_id,
            "aws_secret_access_key": self.athena_aws_secret_access_key,
            "aws_session_token": self.athena_aws_session_token,
            "region_name": self.athena_aws_region,
            "s3_staging_dir": self.athena_s3_staging_dir,
        }
        return {k: v for k, v in pairs.items() if v}

    def bedrock_llm_kwargs(self) -> dict[str, Any]:
        """Build the LiteLLM Bedrock credentials.

        Returns:
            The credentials under ``LLMClient``'s own ``aws_*`` names; unset ones are
            omitted so they fall back to the boto3 chain.
        """
        pairs = {
            "aws_access_key_id": self.bedrock_aws_access_key_id,
            "aws_secret_access_key": self.bedrock_aws_secret_access_key,
            "aws_session_token": self.bedrock_aws_session_token,
            "aws_region_name": self.bedrock_aws_region,
        }
        return {k: v for k, v in pairs.items() if v}

def get_settings(**overrides: Any) -> Settings:
    """Create settings from the environment.

    Args:
        **overrides: Values that win over env and defaults.

    Returns:
        The settings.
    """
    return Settings(**overrides)
