"""Tests for text2sql.config — Settings, YAML loading, and validation.

Consolidates all config-related tests that were previously scattered
across test_core.py, test_modules.py, and test_streaming.py.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from text2sql.config import FIELD_DEPENDS, LogLevel, Settings

# ── Default values ───────────────────────────────────────────────


class TestSettingsDefaults:
    """Every Settings default must match what config.py and configs/config.yaml declare.

    These are the ablation baseline, so they are pinned here rather than read off the class:
    a silent default change re-bases every benchmark arm.
    """

    DEFAULTS = {
        # LLM
        "model": "gpt-4o-mini", "temperature": 0.0,
        # generation
        "generation_mode": "direct", "num_candidates": 1, "generation_strategy": "direct",
        # agent
        "agent_max_turns": 20, "agent_mode": "retrieval", "scenarios_file": None,
        # verification
        "use_repair": True, "max_repair_retries": 3, "selection_mode": "single",
        # profiling
        "profile_cache_dir": ".cache/profiles", "profile_top_k": 10,
        "profile_sample_size": 10_000,
        # schema linking (the value index is implied by the mode, never set directly)
        "use_schema_linking": True, "schema_linking_modes": ["value"],
        "direct_schema_scope": ["full"], "direct_descriptions": ["short"],
        "direct_knowledge": "full",
        "reversed_schema_scope": ["full"], "reversed_descriptions": ["short"],
        "reversed_knowledge": "full",
        "value_top_k": 30,
        # prompts, streaming, logging
        "prompt_template_dir": None, "prompt_version": "v1",
        "event_verbosity": "verbose", "log_level": LogLevel.INFO,
    }

    def test_defaults(self):
        s = Settings()
        assert {f: getattr(s, f) for f in self.DEFAULTS} == self.DEFAULTS


# ── Overrides ────────────────────────────────────────────────────


class TestSettingsOverrides:
    """Constructor kwargs override defaults."""

    @pytest.mark.parametrize(
        "field, value",
        [
            ("model", "anthropic/claude-sonnet-4-20250514"),
            ("temperature", 0.7),
            ("agent_mode", "retrieval"),
            ("use_repair", False),
            ("generation_mode", "direct"),
            ("agent_max_turns", 25),
            ("selection_mode", "confidence"),
            ("schema_linking_modes", ["direct", "value"]),
            ("prompt_version", "v2"),
            ("event_verbosity", "minimal"),
            ("agent_mode", "retrieval"),
            ("generation_strategy", "diverse"),
        ],
    )
    def test_override(self, field, value):
        s = Settings(**{field: value})
        assert getattr(s, field) == value

    def test_multiple_overrides(self):
        s = Settings(model="gpt-4o", num_candidates=3, generation_strategy="diverse",
                     use_repair=False)
        assert s.model == "gpt-4o"
        assert s.num_candidates == 3
        assert s.use_repair is False

    def test_profile_cache_dir_empty_string(self):
        """Empty string is valid (disables caching)."""
        s = Settings(profile_cache_dir="")
        assert s.profile_cache_dir == ""

    def test_scenarios_file_override(self):
        s = Settings(scenarios_file="scenarios.md")
        assert s.scenarios_file == "scenarios.md"


# ── Validation ───────────────────────────────────────────────────


class TestSettingsValidation:
    """Field validators and boundary conditions."""

    def test_valid_db_uri(self):
        assert Settings(db_uri="sqlite:///test.db").db_uri == "sqlite:///test.db"

    def test_an_unknown_linking_mode_is_rejected(self):
        """The linker no longer guards this itself — the Literal is the only check."""
        with pytest.raises(ValidationError):
            Settings(schema_linking_modes=["reversed", "bogus"])
        assert Settings(schema_linking_modes=[]).schema_linking_modes == []

    @pytest.mark.parametrize("field, low, high, extra", [
        ("temperature", 0.0, 2.0, {}),
        ("num_candidates", 1, 3, {"generation_strategy": "diverse"}),
    ])
    def test_bounds_accept_the_edges_and_reject_beyond(self, field, low, high, extra):
        step = 0.1 if isinstance(low, float) else 1
        for value in (low, high):
            assert getattr(Settings(**{field: value}, **extra), field) == value
        for value in (low - step, high + step):
            with pytest.raises(ValidationError):
                Settings(**{field: value}, **extra)

    @pytest.mark.parametrize("kwargs", [
        {"db_uri": "no-scheme-here"},
        {"selection_mode": "nope"},
        {"generation_mode": "telepathy"},
    ])
    def test_invalid_values_raise(self, kwargs):
        with pytest.raises(ValidationError):
            Settings(**kwargs)


# ── Field dependencies ───────────────────────────────────────────


class TestFieldDependencies:
    """FIELD_DEPENDS declares which settings are inert under which configuration."""

    def test_map_only_references_real_fields(self):
        from text2sql.config import FIELD_DEPENDS

        for field, (controller, _) in FIELD_DEPENDS.items():
            assert field in Settings.model_fields, field
            assert controller in Settings.model_fields, controller

    def test_the_dependency_graph_is_acyclic(self):
        """`inactive()` walks the chain upward, so a cycle would hang the process."""
        from text2sql.config import FIELD_DEPENDS

        for field in FIELD_DEPENDS:
            seen, cur = set(), field
            while cur in FIELD_DEPENDS:
                assert cur not in seen, f"cycle through {cur}"
                seen.add(cur)
                cur = FIELD_DEPENDS[cur][0]

    def test_a_field_dies_with_its_whole_chain(self):
        """Regression: sub-settings stayed visible after their section was switched off,
        because each dependency was checked in isolation rather than along its whole chain.
        """
        off = set(Settings(use_schema_linking=False).inactive())
        assert {"schema_linking_modes", "reversed_schema_scope", "reversed_knowledge",
                "value_top_k"} <= off

        # Two hops: scenarios_file -> agent_mode -> generation_mode.
        assert "scenarios_file" in Settings(
            generation_mode="direct", agent_mode="retrieval").inactive()
        assert "scenarios_file" not in Settings(
            generation_mode="agent", agent_mode="retrieval").inactive()

    def test_retrieval_always_carries_its_tools(self):
        """One field replaced two, so "table names only, no way to expand them" — which the
        old validator had to reject — cannot be expressed at all."""
        assert Settings(generation_mode="agent", agent_mode="retrieval").agent_mode == "retrieval"
        assert "scenarios_file" not in Settings(
            generation_mode="agent", agent_mode="retrieval").inactive()

    def test_only_selecting_value_builds_the_index(self):
        """The whole point of a mode list: `reversed` alone must be measurable without
        value matching leaking into its focused schema."""
        assert Settings(schema_linking_modes=["value"]).value_index_enabled
        assert Settings(schema_linking_modes=["reversed", "value"]).value_index_enabled
        assert not Settings(schema_linking_modes=["reversed"]).value_index_enabled
        assert not Settings(schema_linking_modes=["direct"]).value_index_enabled
        assert not Settings(schema_linking_modes=["value"],
                            use_schema_linking=False).value_index_enabled

    def test_a_list_controller_must_hold_every_listed_value(self):
        """A list controller reads as "contains all of" — plain membership for the
        single-value dependencies left in `FIELD_DEPENDS`. Keep `<=`, not `&`."""
        assert "reversed_schema_scope" in Settings(schema_linking_modes=["value"]).inactive()
        assert "reversed_schema_scope" not in Settings(
            schema_linking_modes=["value", "reversed"]).inactive()
        assert "direct_knowledge" in Settings(schema_linking_modes=["reversed"]).inactive()

    def test_inactive_reports_fields_their_controller_rules_out(self):
        agentless = Settings(generation_mode="direct").inactive()
        assert {"agent_max_turns", "agent_mode"} <= set(agentless)

        s = Settings(generation_mode="agent", num_candidates=3, generation_strategy="diverse",
                     selection_mode="majority", use_schema_linking=True,
                     schema_linking_modes=["reversed"])
        inactive = set(s.inactive())
        assert "agent_max_turns" not in inactive   # live: the agent is on
        assert "reversed_schema_scope" not in inactive  # live: reversed is selected
        assert "selection_mode" not in inactive    # live: more than one candidate

    def test_single_candidate_makes_selection_inert(self):
        assert "selection_mode" in Settings(num_candidates=1).inactive()
        assert "selection_mode" not in Settings(
            num_candidates=2, generation_strategy="diverse").inactive()

    def test_extra_candidates_need_a_strategy_that_differs(self):
        """Without `diverse` the candidates differ only by sampling — and on bedrock anthropic,
        which drops `seed`, one measured N=3 run returned identical SQL on 12 of 15 questions."""
        with pytest.raises(Exception, match="diverse"):
            Settings(num_candidates=2)
        assert Settings(num_candidates=3, generation_strategy="diverse").num_candidates == 3

    # configure_logging() runs basicConfig(force=True), which drops caplog's handler —
    # so these assert on the Rich handler's stdout instead.
    def test_explicitly_setting_an_inert_field_warns(self, capsys):
        Settings(generation_mode="direct", agent_max_turns=25)
        out = " ".join(capsys.readouterr().out.split())
        assert "agent_max_turns (needs generation_mode, is 'direct')" in out

    def test_a_select_control_holds_a_string_so_its_values_must_compare_loosely(self):
        """`num_candidates` renders as a <select>, whose value is always a string, so the UI's
        mirror of this walk compared '3' against [2, 3] and never revealed selection_mode."""
        assert "selection_mode" in Settings(num_candidates=1).inactive()
        assert "selection_mode" not in Settings(
            num_candidates=3, generation_strategy="diverse").inactive()
        assert all(isinstance(v, int) for v in FIELD_DEPENDS["selection_mode"][1])

    def test_every_stage_setting_dies_with_its_stage(self):
        """A field left visible once its stage is off is a knob that silently does nothing:
        `value_top_k` outlived schema linking, `temperature` outlived thinking."""
        off = Settings(use_schema_linking=False).inactive()
        assert {"value_top_k", "schema_linking_modes"} <= set(off)
        assert "temperature" in Settings(reasoning_effort="high").inactive()
        assert "temperature" not in Settings(reasoning_effort="none").inactive()

    def test_paying_for_candidates_and_using_one_warns(self, capsys):
        """A measured run generated three candidates, took candidate 1's recovered diagnostic,
        and discarded the only one that had submitted."""
        Settings(num_candidates=3, generation_strategy="diverse", selection_mode="single")
        assert "takes candidate 1" in " ".join(capsys.readouterr().out.split())

    def test_default_valued_inert_fields_stay_quiet(self, capsys):
        """Only fields the user actually set are worth warning about."""
        Settings(generation_mode="direct")
        assert "no effect" not in capsys.readouterr().out


class TestShippedConfigYaml:
    #: Which machine and which data you point at is not a pipeline default — these are
    #: expected to differ per checkout and are exempt from the drift check below.
    ENVIRONMENT_SPECIFIC = {"log_level", "log_transcript", "model", "db_uri"}

    def test_pipeline_defaults_match_the_code(self):
        """configs/config.yaml is the annotated mirror of Settings — the tuning knobs must
        not drift, or the documented defaults stop describing what actually runs."""
        from pathlib import Path

        shipped = Settings.from_yaml(Path(__file__).parent.parent / "configs" / "config.yaml")
        defaults = Settings()
        exempt = self.ENVIRONMENT_SPECIFIC | {
            f for f in Settings.model_fields if f.startswith(("benchmark_", "bedrock_", "athena_"))
        }
        divergent = {f for f in Settings.model_fields
                     if f not in exempt and getattr(shipped, f) != getattr(defaults, f)}
        assert not divergent, f"configs/config.yaml diverges from Settings defaults: {divergent}"


class TestEnvExample:
    """.env.example is the env-var reference — unknown TEXT2SQL_ vars are silently
    ignored (``extra="ignore"``), so a stale name here fails quietly at runtime."""

    #: Read by PromptManager directly rather than being Settings fields.
    PROMPT_OVERRIDES = {"TEXT2SQL_PROMPT_GENERATE_SQL_PATH", "TEXT2SQL_PROMPT_REPAIR_SQL_PATH"}
    #: Nested dict — set via YAML or the UI, not an env var.
    NO_ENV_FORM = {"profile_selection"}

    @pytest.fixture
    def documented(self):
        import re
        from pathlib import Path

        text = (Path(__file__).parent.parent / ".env.example").read_text()
        return re.findall(r"^#?\s*(TEXT2SQL_[A-Z0-9_]+)=", text, re.M)

    def test_every_documented_var_is_a_real_field(self, documented):
        for var in documented:
            if var in self.PROMPT_OVERRIDES:
                continue
            assert var.removeprefix("TEXT2SQL_").lower() in Settings.model_fields, var

    def test_every_configurable_field_is_documented(self, documented):
        from text2sql.config import YAML_SECTION_MAP, section_items

        listed = {v.removeprefix("TEXT2SQL_").lower() for v in documented}
        for section in YAML_SECTION_MAP:
            for _, field in section_items(section):
                if field not in self.NO_ENV_FORM:
                    assert field in listed, f"TEXT2SQL_{field.upper()} missing from .env.example"


# ── YAML loading ─────────────────────────────────────────────────


class TestSettingsFromYaml:
    """Test loading from YAML config files."""

    def test_from_yaml_happy_path(self, sample_yaml_config):
        s = Settings.from_yaml(sample_yaml_config)
        assert s.db_uri == "sqlite:///from_yaml.db"
        assert s.model == "gpt-4o"
        assert s.temperature == 0.5
        assert s.num_candidates == 3
        assert s.use_repair is False          # verification.repair
        assert s.agent_max_turns == 10        # sql_generation.agent.max_turns
        assert s.agent_mode == "retrieval"

    def test_yaml_11_bare_tokens_reach_their_literal(self, tmp_path):
        """PyYAML resolves bare off/no to False and a blank value to None, so `knowledge: off`
        and `stop_after:` used to reach their str Literal as the wrong type and fail
        validation.
        """
        path = tmp_path / "c.yaml"
        path.write_text(
            "general:\n  stop_after:\n"
            "sql_generation:\n  knowledge: off\n"
            "schema_linking:\n  direct:\n    knowledge: off\n"
            "verification:\n  repair: no\n"
        )
        s = Settings.from_yaml(path)
        assert s.generation_knowledge == "off"
        assert s.direct_knowledge == "off"
        assert s.stop_after == ""
        assert s.use_repair is False  # a real bool field is untouched

    def test_from_yaml_file_not_found(self, tmp_path):
        missing = str(tmp_path / "nonexistent.yaml")
        with pytest.raises(FileNotFoundError):
            Settings.from_yaml(missing)

    def test_from_yaml_with_overrides(self, sample_yaml_config):
        """Explicit overrides win over YAML values."""
        s = Settings.from_yaml(sample_yaml_config, model="override-model")
        assert s.model == "override-model"

    def test_env_var_beats_yaml(self, sample_yaml_config, monkeypatch):
        """Env vars outrank YAML — YAML kwargs would otherwise win via pydantic."""
        monkeypatch.setenv("TEXT2SQL_MODEL", "env-model")
        monkeypatch.setenv("TEXT2SQL_NUM_CANDIDATES", "2")
        s = Settings.from_yaml(sample_yaml_config)
        assert s.model == "env-model"
        assert s.num_candidates == 2
        # Fields the env didn't set still come from the YAML.
        assert s.db_uri == "sqlite:///from_yaml.db"

    def test_overrides_beat_env_and_yaml(self, sample_yaml_config, monkeypatch):
        monkeypatch.setenv("TEXT2SQL_MODEL", "env-model")
        s = Settings.from_yaml(sample_yaml_config, model="override-model")
        assert s.model == "override-model"

    def test_from_yaml_top_level_keys(self, tmp_path):
        """YAML can also use top-level flat keys."""
        yaml_file = tmp_path / "flat.yaml"
        yaml_file.write_text("model: gpt-4o\nnum_candidates: 2\ngeneration_strategy: diverse\n")
        s = Settings.from_yaml(str(yaml_file))
        assert s.model == "gpt-4o"
        assert s.num_candidates == 2

    def test_from_yaml_empty_file(self, tmp_path):
        """Empty YAML file uses all defaults."""
        yaml_file = tmp_path / "empty.yaml"
        yaml_file.write_text("")
        s = Settings.from_yaml(str(yaml_file))
        assert s.model == "gpt-4o-mini"  # default


# ── LogLevel enum ────────────────────────────────────────────────


class TestLogLevel:
    """Test the LogLevel StrEnum."""

    @pytest.mark.parametrize(
        "member, value",
        [
            (LogLevel.DEBUG, "DEBUG"),
            (LogLevel.INFO, "INFO"),
            (LogLevel.WARNING, "WARNING"),
            (LogLevel.ERROR, "ERROR"),
            (LogLevel.CRITICAL, "CRITICAL"),
        ],
    )
    def test_member_values(self, member, value):
        assert member == value
        assert member.value == value
