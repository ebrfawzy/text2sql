"""Tests for text2sql.llm — extract_sql edge cases and LLMUsage tracking.

extract_sql is one of the most exercised functions in the pipeline
(called after every LLM response), so we parametrize across ~12 cases.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from text2sql.llm import LLMUsage, extract_sql, retry_with_backoff

# ── extract_sql ──────────────────────────────────────────────────


class TestPromptCaching:
    """The agent re-sends its system prompt every turn — 20.6k chars in a measured run."""

    def test_the_system_block_is_marked_for_caching(self):
        from text2sql.llm import cached_block
        block, = cached_block("SCHEMA")
        assert block == {"type": "text", "text": "SCHEMA",
                         "cache_control": {"type": "ephemeral"}}

    def test_usage_records_cache_tokens(self):
        """`drop_params=True` discards a marker a provider rejects, silently — as it already did
        with reasoning_effort. These counters are the only proof caching engaged."""
        from types import SimpleNamespace

        from text2sql.llm import LLMUsage
        usage = LLMUsage()
        usage.record(SimpleNamespace(usage=SimpleNamespace(
            prompt_tokens=100, completion_tokens=10,
            cache_read_input_tokens=4096, cache_creation_input_tokens=0)))
        assert usage.summary()["cache_read_tokens"] == 4096
        assert usage.summary()["cache_write_tokens"] == 0


class TestOutputCeiling:
    """There is no `max_tokens` setting: a configured cap nerfed every model above it (8,192
    against haiku's 64,000), so the model's own ceiling is always what is asked for."""

    def test_the_ceiling_comes_from_the_model(self):
        from text2sql.llm import _max_tokens

        assert _max_tokens("gpt-4o-mini") > 8192
        # No metadata means omit the parameter, not substitute a guess of ours — a 4,096
        # fallback constant was doing the nerfing for every model LiteLLM does not list.
        assert _max_tokens("nope/not-a-real-model") == 0


class TestExtractSQL:
    """Exhaustive edge-case coverage for SQL extraction from LLM responses."""

    @pytest.mark.parametrize(
        "raw, expected",
        [
            # Plain SQL — no fences
            ("SELECT * FROM users", "SELECT * FROM users"),
            # Markdown ```sql fence
            ("```sql\nSELECT * FROM users\n```", "SELECT * FROM users"),
            # Markdown ``` fence (no language)
            ("Here is the query:\n```\nSELECT COUNT(*) FROM orders\n```",
             "SELECT COUNT(*) FROM orders"),
            # postgresql language tag
            ("```postgresql\nSELECT 1\n```", "SELECT 1"),
            # mysql language tag
            ("```mysql\nSELECT 1\n```", "SELECT 1"),
            # sqlite language tag
            ("```sqlite\nSELECT 1\n```", "SELECT 1"),
            # Multi-line SQL inside fences
            (
                "```sql\nSELECT u.name, o.amount\nFROM users u\nJOIN orders o ON u.id = o.user_id\n```",
                "SELECT u.name, o.amount\nFROM users u\nJOIN orders o ON u.id = o.user_id",
            ),
            # Leading/trailing whitespace
            ("   SELECT 1   ", "SELECT 1"),
            # Fence with surrounding prose (first code block wins)
            (
                "Here is the answer:\n```sql\nSELECT 42\n```\nDone!",
                "SELECT 42",
            ),
            # Content on the tag's own line — the fence used to survive into the SQL
            ("```SQL SELECT 1 FROM t```", "SELECT 1 FROM t"),
            # Closing fence missing (truncated reply) — likewise
            ("```sql\nSELECT 1 FROM t", "SELECT 1 FROM t"),
        ],
        ids=[
            "plain_sql",
            "sql_fence",
            "no_lang_fence",
            "postgresql_tag",
            "mysql_tag",
            "sqlite_tag",
            "multiline",
            "whitespace",
            "prose_around_fence",
            "same_line_as_tag",
            "unterminated_fence",
        ],
    )
    def test_extraction(self, raw, expected):
        import sqlparse
        assert extract_sql(raw) == sqlparse.format(
            expected, reindent=True, keyword_case="upper")

    def test_empty_string(self):
        assert extract_sql("") == ""

    def test_whitespace_only(self):
        assert extract_sql("   \n\t  ") == ""

    def test_no_sql_text(self):
        """Non-SQL prose without fences is returned as-is (stripped)."""
        result = extract_sql("I don't know the answer")
        assert result == "I don't know the answer"

    def test_multiple_fences_first_wins(self):
        text = "```sql\nSELECT 1\n```\n\nAlternative:\n```sql\nSELECT 2\n```"
        assert extract_sql(text) == "SELECT 1"

    def test_fence_with_empty_content(self):
        result = extract_sql("```sql\n\n```")
        assert result == ""


# ── LLMUsage ─────────────────────────────────────────────────────


class TestLLMUsage:
    """Token usage tracking accumulation."""

    def test_initial_state(self):
        usage = LLMUsage()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_cost_usd == 0.0
        assert usage.num_calls == 0

    def test_summary_keys(self):
        usage = LLMUsage()
        summary = usage.summary()
        assert "prompt_tokens" in summary
        assert "completion_tokens" in summary
        assert "total_tokens" in summary
        assert "total_cost_usd" in summary
        assert "num_calls" in summary

    def test_summary_total_tokens(self):
        usage = LLMUsage(prompt_tokens=100, completion_tokens=50)
        assert usage.summary()["total_tokens"] == 150

    def test_cost_rounding(self):
        usage = LLMUsage(total_cost_usd=0.123456789)
        assert usage.summary()["total_cost_usd"] == 0.123457


# ── LLMClient init ───────────────────────────────────────────────


class TestLLMClientInit:
    """Verify LLMClient defaults without making actual API calls."""

    def test_default_fields(self):
        from text2sql.llm import LLMClient

        client = LLMClient()
        assert client.model == "gpt-4o-mini"
        assert client.temperature == 0.0

    def test_custom_fields(self):
        from text2sql.llm import LLMClient

        client = LLMClient(model="gpt-4o", temperature=0.5)
        assert client.model == "gpt-4o"
        assert client.temperature == 0.5

    def test_usage_starts_empty(self):
        from text2sql.llm import LLMClient

        client = LLMClient()
        assert client.usage.num_calls == 0


class TestReasoningEffortCoupling:
    """Extended thinking and sampling are coupled — Anthropic rejects temperature != 1
    while thinking is on, so the two settings cannot be sent independently."""

    def _capture(self, **kw):
        from unittest.mock import MagicMock, patch

        from text2sql.llm import LLMClient

        sent: dict = {}

        async def acompletion_mock(*_a, **kwargs):
            sent.update(kwargs)

            async def gen():
                chunk = MagicMock()
                chunk.choices = [MagicMock()]
                chunk.choices[0].delta.content = "ok"
                chunk.choices[0].delta.reasoning_content = None
                yield chunk

            return gen()

        usage = MagicMock()
        usage.usage.prompt_tokens = usage.usage.completion_tokens = 1
        client = LLMClient(model="test-model", **kw)
        with patch("text2sql.llm.acompletion", new=acompletion_mock), \
                patch("text2sql.llm.litellm.stream_chunk_builder", return_value=usage):
            import asyncio
            asyncio.run(client.chat("q"))
        return sent

    def test_none_omits_the_param_and_keeps_temperature(self):
        sent = self._capture(reasoning_effort="none", temperature=0.0)
        assert "reasoning_effort" not in sent
        assert sent["temperature"] == 0.0

    def test_any_effort_forces_temperature_one(self):
        sent = self._capture(reasoning_effort="high", temperature=0.0)
        assert sent["reasoning_effort"] == "high"
        assert sent["temperature"] == 1.0


# ── LLMUsage.record() ───────────────────────────────────────────


class TestLLMUsageRecord:
    """Test record() accumulates tokens from mock LiteLLM responses."""

    def test_record_accumulates(self):
        from unittest.mock import MagicMock
        usage = LLMUsage()
        response = MagicMock()
        response.usage.prompt_tokens = 100
        response.usage.completion_tokens = 50
        usage.record(response)
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.num_calls == 1

    def test_record_multiple(self):
        from unittest.mock import MagicMock
        usage = LLMUsage()
        for _ in range(3):
            response = MagicMock()
            response.usage.prompt_tokens = 10
            response.usage.completion_tokens = 5
            usage.record(response)
        assert usage.prompt_tokens == 30
        assert usage.completion_tokens == 15
        assert usage.num_calls == 3

    def test_record_no_usage_attr(self):
        from unittest.mock import MagicMock
        usage = LLMUsage()
        response = MagicMock(spec=[])  # no .usage attribute
        usage.record(response)
        assert usage.num_calls == 1
        assert usage.prompt_tokens == 0


# ── LLMClient with mocked litellm ────────────────────────────────


class TestLLMClientMocked:
    """Test LLMClient.chat/chat_messages/chat_for_sql with mocked litellm."""

    def _mock_acompletion(self, content="SELECT 1"):
        from unittest.mock import MagicMock

        async def mock_response_gen():
            mock_chunk = MagicMock()
            mock_chunk.choices = [MagicMock()]
            mock_chunk.choices[0].delta.content = content
            mock_chunk.choices[0].delta.reasoning_content = None
            yield mock_chunk

        async def acompletion_mock(*args, **kwargs):
            return mock_response_gen()

        return acompletion_mock

    async def test_chat_returns_text(self):
        from unittest.mock import MagicMock, patch

        from text2sql.llm import LLMClient

        mock_usage_resp = MagicMock()
        mock_usage_resp.usage.prompt_tokens = 10
        mock_usage_resp.usage.completion_tokens = 5

        client = LLMClient(model="test-model")
        with patch("text2sql.llm.acompletion", new=self._mock_acompletion("SELECT 1")), \
                patch("text2sql.llm.litellm.stream_chunk_builder", return_value=mock_usage_resp):
            result = await client.chat("test prompt")
        assert result == "SELECT 1"
        assert client.usage.num_calls == 1

    async def test_chat_with_system_prompt(self):
        from unittest.mock import MagicMock, patch

        from text2sql.llm import LLMClient

        mock_usage_resp = MagicMock()
        client = LLMClient()
        mock_comp = MagicMock(side_effect=self._mock_acompletion("OK"))

        with patch("text2sql.llm.acompletion", new=mock_comp), \
                patch("text2sql.llm.litellm.stream_chunk_builder", return_value=mock_usage_resp):
            await client.chat("user msg", system="sys msg")
            messages = mock_comp.call_args[1]["messages"]
            assert messages[0]["role"] == "system"
            assert messages[1]["role"] == "user"

    async def test_chat_messages(self):
        from unittest.mock import MagicMock, patch

        from text2sql.llm import LLMClient

        mock_usage_resp = MagicMock()
        client = LLMClient()
        with patch("text2sql.llm.acompletion", new=self._mock_acompletion("SELECT 42")), \
                patch("text2sql.llm.litellm.stream_chunk_builder", return_value=mock_usage_resp):
            result = await client.chat_messages([{"role": "user", "content": "hi"}])
        assert result == "SELECT 42"

    async def test_chat_for_sql_strips_fences(self):
        from unittest.mock import MagicMock, patch

        from text2sql.llm import LLMClient

        mock_usage_resp = MagicMock()
        client = LLMClient()
        with patch("text2sql.llm.acompletion", new=self._mock_acompletion("```sql\nSELECT 1\n```")), \
                patch("text2sql.llm.litellm.stream_chunk_builder", return_value=mock_usage_resp):
            result = await client.chat_for_sql("generate SQL")
        assert result == "SELECT 1"

    async def test_chat_with_seed(self):
        from unittest.mock import MagicMock, patch

        from text2sql.llm import LLMClient

        mock_usage_resp = MagicMock()
        client = LLMClient()
        mock_comp = MagicMock(side_effect=self._mock_acompletion("SELECT 1"))

        with patch("text2sql.llm.acompletion", new=mock_comp), \
                patch("text2sql.llm.litellm.stream_chunk_builder", return_value=mock_usage_resp):
            await client.chat("prompt", seed=42)
            assert mock_comp.call_args[1]["seed"] == 42


# ── retry_with_backoff ───────────────────────────────────────────


class TestRetryWithBackoff:
    """Every provider call goes through this wrapper, so its two exits matter."""

    async def test_returns_the_first_success_without_sleeping(self, monkeypatch):
        monkeypatch.setattr("asyncio.sleep", AsyncMock())
        calls = []

        async def fn():
            calls.append(1)
            return "ok"

        assert await retry_with_backoff(fn, max_retries=2, idle_ms=1,
                                        retry_on=RuntimeError) == "ok"
        assert len(calls) == 1

    async def test_retries_matching_errors_then_re_raises(self, monkeypatch):
        monkeypatch.setattr("asyncio.sleep", AsyncMock())
        calls = []

        async def fn():
            calls.append(1)
            raise RuntimeError("rate limited")

        with pytest.raises(RuntimeError, match="rate limited"):
            await retry_with_backoff(fn, max_retries=2, idle_ms=1, retry_on=RuntimeError)
        assert len(calls) == 3  # the initial attempt plus max_retries

    async def test_unmatched_errors_propagate_immediately(self, monkeypatch):
        monkeypatch.setattr("asyncio.sleep", AsyncMock())
        calls = []

        async def fn():
            calls.append(1)
            raise ValueError("bad request")

        with pytest.raises(ValueError):
            await retry_with_backoff(fn, max_retries=2, idle_ms=1, retry_on=RuntimeError)
        assert len(calls) == 1  # a non-retryable error is not worth a second call
