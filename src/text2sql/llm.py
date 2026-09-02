"""Unified LLM interface via LiteLLM.

A thin wrapper that reaches any LiteLLM provider through one API, tracks token usage and
cost per call, and retries transient failures.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import litellm
import sqlparse
from litellm import ModelResponse, TextCompletionResponse, acompletion
from litellm.exceptions import (
    APIConnectionError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)

# Transient errors that are safe to retry.
_RETRYABLE = (RateLimitError, ServiceUnavailableError,
              Timeout, APIConnectionError, InternalServerError)

logger = logging.getLogger(__name__)


async def retry_with_backoff[T](
    fn: Callable[[], Awaitable[T]],
    *,
    max_retries: int,
    idle_ms: int,
    retry_on: type[BaseException] | tuple[type[BaseException], ...],
    label: str = "operation",
) -> T:
    """Await a coroutine factory, retrying with exponential backoff.

    Args:
        fn: Called once per attempt.
        max_retries: Retries after the first attempt.
        idle_ms: Base delay; retry ``n`` waits ``(idle_ms / 1000) * 2**n`` seconds.
        retry_on: Exception types worth retrying; anything else propagates immediately.
        label: Name used in the log lines.

    Returns:
        Whatever ``fn()`` returns.

    Raises:
        BaseException: The last matching exception, once the retries run out.
    """
    last_exc: BaseException | None = None
    for attempt in range(max_retries + 1):
        try:
            return await fn()
        except retry_on as exc:
            last_exc = exc
            if attempt < max_retries:
                delay = (idle_ms / 1000) * (2 ** attempt)
                logger.warning(
                    "%s failed (attempt %d/%d): %s - retrying in %.1fs",
                    label, attempt + 1, max_retries + 1, exc, delay,
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    "%s failed after %d attempts: %s",
                    label, max_retries + 1, exc,
                )
    assert last_exc is not None
    raise last_exc


@dataclass
class LLMUsage:
    """Tracks cumulative token usage and cost across calls."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    total_cost_usd: float = 0.0
    num_calls: int = 0

    def record(self, response: ModelResponse | TextCompletionResponse) -> None:
        """Add one LiteLLM response's tokens and cost to the running totals.

        Args:
            response: The completed response.
        """
        if usage := getattr(response, "usage", None):
            self.prompt_tokens += getattr(usage, "prompt_tokens", 0)
            self.completion_tokens += getattr(usage, "completion_tokens", 0)
            # `drop_params=True` silently discards a cache marker a provider will not take,
            # so these counters are the only proof caching engaged.
            self.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0
            self.cache_write_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0

        try:
            self.total_cost_usd += litellm.completion_cost(
                completion_response=response)
        except Exception:
            pass  # Cost tracking is best-effort

        self.num_calls += 1

    def summary(self) -> dict[str, Any]:
        """Summarize the totals.

        Returns:
            Token counts, cost and call count, for tracing and logging.
        """
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "num_calls": self.num_calls,
        }


@dataclass
class LLMClient:
    """Async wrapper around LiteLLM for consistent LLM calls.

    Usage::

        client = LLMClient(model="gpt-4o-mini")
        text = await client.chat("Describe the table schema in one sentence.")
        print(client.usage.summary())

    Supports any LiteLLM model string:
        - ``gpt-4o``, ``gpt-4o-mini`` (OpenAI)
        - ``anthropic/claude-sonnet-4-20250514`` (Anthropic)
        - ``gemini/gemini-2.5-pro`` (Google)
        - ``ollama/llama3`` (local Ollama)
        - ``bedrock/...`` (AWS Bedrock)
        - etc.
    """

    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_retries: int = 3
    idle_ms: int = 1000
    # "none" omits the param entirely; any other value forces temperature=1
    # (Anthropic rejects other values while thinking is on). Overridable per call.
    reasoning_effort: str = "none"
    # AWS Bedrock credentials (optional). When set, forwarded to LiteLLM on
    # both the direct and LangChain paths; unset → boto3 default chain.
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_session_token: str | None = None
    aws_region_name: str | None = None
    usage: LLMUsage = field(default_factory=LLMUsage)
    # One warning per client when a provider silently ignores reasoning_effort.
    _warned_no_thinking: bool = False

    def _bedrock_kwargs(self) -> dict[str, Any]:
        """Collect the explicitly-set AWS Bedrock credentials.

        Returns:
            The non-None credentials under LiteLLM's own names; empty when none are set, so
            LiteLLM falls back to the ambient boto3 default chain.
        """
        pairs = {
            "aws_access_key_id": self.aws_access_key_id,
            "aws_secret_access_key": self.aws_secret_access_key,
            "aws_session_token": self.aws_session_token,
            "aws_region_name": self.aws_region_name,
        }
        return {k: v for k, v in pairs.items() if v}

    async def chat(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        **kwargs: Any,
    ) -> str:
        """Send a single user message (with optional system prompt) and return the text.

        Args:
            prompt: The user message.
            system: Optional system prompt.
            temperature: Override default temperature for this call.
            seed: Random seed for reproducibility / candidate diversity.
            **kwargs: Extra params passed to litellm.acompletion().

        Returns:
            The assistant's response text.
        """
        messages = ([{"role": "system", "content": cached_block(system)}]
                    if system else []) + [{"role": "user", "content": prompt}]
        return await self.chat_messages(messages, temperature=temperature, seed=seed, **kwargs)

    async def chat_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
        seed: int | None = None,
        **kwargs: Any,
    ) -> str:
        """Send a full message list and return the text response.

        Accumulates the stream from :meth:`stream_chat_messages`.

        Args:
            messages: List of {"role": ..., "content": ...} dicts.
            temperature: Override default temperature.
            seed: Random seed.
            **kwargs: Extra params passed to litellm.acompletion().

        Returns:
            The assistant's response text.
        """
        parts: list[str] = []
        async for item in self.stream_chat_messages(
            messages, temperature=temperature, seed=seed, **kwargs
        ):
            if isinstance(item, tuple) and not item[1]:
                parts.append(item[0])

        return "".join(parts)

    async def chat_for_sql(
        self,
        prompt: str,
        *,
        system: str | None = None,
        seed: int | None = None,
    ) -> str:
        """Send one message and extract the SQL from the reply.

        Args:
            prompt: The user message.
            system: Optional system prompt.
            seed: Random seed for reproducibility / candidate diversity.

        Returns:
            The SQL, with markdown fences and comments stripped.
        """
        return extract_sql(await self.chat(prompt, system=system, seed=seed))

    async def stream_chat_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
        seed: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[tuple[str, bool] | dict[str, Any]]:
        """Stream one completion.

        Args:
            messages: The conversation so far.
            temperature: Override the default temperature.
            seed: Random seed.
            **kwargs: Extra params passed to ``litellm.acompletion()``.

        Yields:
            ``(text, is_thinking)`` per chunk, covering both thinking conventions -
            ``delta.reasoning_content`` and ``<think>`` tags in ``delta.content``. When
            ``tools`` is passed, the final yield is instead the assembled assistant message
            for the caller to append to the conversation.
        """
        effort = kwargs.pop("reasoning_effort", self.reasoning_effort)
        temp = temperature if temperature is not None else self.temperature
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": 1.0 if effort != "none" else temp,
            "stream": True,
            **({} if effort == "none" else {"reasoning_effort": effort}),
            **({"max_tokens": mx} if (mx := _max_tokens(self.model)) else {}),
            "drop_params": True,
            **self._bedrock_kwargs(),
            **({"seed": seed} if seed is not None else {}),
            **kwargs,
        }

        response = await self._acompletion_with_retry(**params)
        chunks: list[Any] = []
        in_think_tag = False
        saw_thinking = False
        truncated = False

        async for chunk in response:
            chunks.append(chunk)
            truncated = truncated or (
                chunk.choices and chunk.choices[0].finish_reason == "length")
            if not (delta := chunk.choices[0].delta if chunk.choices else None):
                continue

            # Native reasoning_content (Anthropic, Deepseek, etc.)
            if reasoning := getattr(delta, "reasoning_content", None):
                saw_thinking = True
                yield reasoning, True
                continue

            # Regular content, with Ollama's <think> boundaries stripped out of it.
            if content := delta.content or "":
                if "<think>" in content:
                    in_think_tag = True
                    content = content.replace("<think>", "")
                if "</think>" in content:
                    in_think_tag = False
                    content = content.replace("</think>", "")
                if content:
                    yield content, in_think_tag

        if truncated:
            logger.warning("Reply from %s was cut off at its %s-token output limit.",
                           self.model, params.get("max_tokens", "provider-default"))

        if effort != "none" and not saw_thinking and not self._warned_no_thinking:
            self._warned_no_thinking = True
            logger.warning(
                "reasoning_effort=%s was requested but %s returned no thinking content: "
                "the parameter is being dropped. Note it still forces temperature=1, so a "
                "run configured this way differs from effort=none by sampling alone.",
                effort, self.model)

        # One reconstruction serves both usage tracking and the assembled tool-call turn.
        full = None
        try:
            if full := litellm.stream_chunk_builder(chunks, messages=messages):
                self.usage.record(full)
        except Exception:
            pass  # Usage tracking is best-effort

        msg = full.choices[0].message.model_dump() if full and full.choices else {}
        # Only role/content/tool_calls travel back: provider-specific extras are
        # rejected on resend.
        reply = {"role": "assistant",
                 **{k: msg[k] for k in ("content", "tool_calls") if msg.get(k)}}
        if kwargs.get("tools"):
            yield reply  # one turn of a loop: its caller logs the whole conversation at the end
        else:
            log_conversation(messages + [reply])

    async def _acompletion_with_retry(self, **params: Any) -> Any:
        """Call ``acompletion`` with retry and exponential backoff on transient errors.

        Args:
            **params: Passed straight to ``litellm.acompletion()``.

        Returns:
            The streaming response.
        """
        return await retry_with_backoff(
            lambda: acompletion(**params),
            max_retries=self.max_retries,
            idle_ms=self.idle_ms,
            retry_on=_RETRYABLE,
            label="LLM call",
        )


def log_conversation(messages: list[dict[str, Any]]) -> None:
    """Log a finished conversation in full, once.

    Args:
        messages: The whole exchange, replies and tool results included.
    """
    logger.debug("conversation (%d messages)\n%s", len(messages),
                 json.dumps(messages, indent=2, default=str))


def cached_block(text: str) -> list[dict[str, Any]]:
    """Wrap text in one content block marked as a cache breakpoint.

    Anthropic caches ``tools -> system -> messages`` in that order, so a marker on the
    system block also covers the tool definitions.

    Args:
        text: The block's text.

    Returns:
        The single-block content list.
    """
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


def with_cached_prefix(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Put a rolling cache breakpoint on the last user or tool message.

    The system marker covers only the static prefix, while an agent's cost is the
    conversation it re-sends every turn.

    Args:
        messages: The conversation so far.

    Returns:
        A copy with the last message marked, so the stored history stays plain strings.
    """
    if not messages or messages[-1].get("role") not in ("user", "tool") \
            or not isinstance(messages[-1].get("content"), str):
        return messages
    return [*messages[:-1], {**messages[-1],
                             "content": cached_block(messages[-1]["content"])}]


def _max_tokens(model: str) -> int:
    """Look up a model's own output ceiling.

    Args:
        model: The LiteLLM model string.

    Returns:
        The ceiling, or 0 when LiteLLM has no metadata for it, in which case the caller
        omits the parameter rather than guessing.
    """
    try:
        return int(litellm.get_max_tokens(model) or 0)
    except Exception:
        return 0


def extract_sql(text: str) -> str:
    """Extract and clean the SQL from an LLM response.

    Args:
        text: The raw reply.

    Returns:
        The SQL, unwrapped from any markdown fence, uncommented and reformatted.
    """
    # Unwrap a markdown fence: optional language tag, content possibly on the tag's own
    # line, closing fence possibly missing (truncated reply).
    if m := re.search(r"```[ \t]*(?:sqlite|postgresql|mysql|sql)?[ \t]*\n?([\s\S]*?)(?:```|\Z)",
                      text, re.IGNORECASE):
        text = m.group(1)
    text = text.strip()
    if len(text) >= 2 and text[0] in ('"', "'") and text[-1] == text[0]:
        text = text[1:-1]
    # Remove SQL comments (-- line and /* block */) while preserving string/identifier literals
    literal = r"'[^']*'|\"[^\"]*\"|`[^`]*`"
    text = re.sub(
        rf"--[^\n]*|/\*[\s\S]*?\*/|({literal})",
        lambda m: m.group(1) or "",
        text,
    )
    # Collapse consecutive blank lines while preserving string/identifier literals
    text = re.sub(
        rf"\n(?:[ \t]*\n)+|({literal})",
        lambda m: m.group(1) or "\n",
        text,
    )
    return sqlparse.format(text.strip(), reindent=True, keyword_case="upper")


def parse_llm_json(text: str) -> Any:
    """Parse JSON from an LLM response, tolerating common artefacts.

    Args:
        text: The raw reply.

    Returns:
        The parsed payload. Lossy cleanup - stripping comments and trailing commas - is
        applied only after a strict parse fails, so valid JSON containing ``//`` is never
        corrupted.

    Raises:
        json.JSONDecodeError: The payload is still unparseable after cleanup.
    """
    text = _extract_json_payload(text)
    try:
        return json.loads(text, strict=False)
    except json.JSONDecodeError:
        cleaned = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
        cleaned = re.sub(r"//[^\n]*", "", cleaned)
        cleaned = re.sub(r",\s*([\]}])", r"\1", cleaned)  # trailing commas
        return json.loads(cleaned, strict=False)


def _extract_json_payload(text: str) -> str:
    """Isolate the JSON slice of an LLM response.

    Args:
        text: The raw reply.

    Returns:
        The first fenced block, else the span between the outermost ``{}``/``[]``, which
        drops any surrounding prose.
    """
    if fence := re.search(r"```(?:json|jsonl)?\s*\n?(.*?)```", text, re.DOTALL | re.IGNORECASE):
        return fence.group(1).strip()
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    end = max(text.rfind("}"), text.rfind("]"))
    return text[min(starts):end + 1] if starts and end > min(starts) else text.strip()
