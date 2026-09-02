"""ReAct agent for SQL generation: an explicit tool loop over :class:`LLMClient`.

The model streams a turn, any tool calls are executed and fed back, and the loop ends when
it calls ``submit_sql``, answers without a tool, or hits the turn cap. Running on the client
directly keeps token/cost accounting, streaming and the trace exact.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncIterator
from typing import Any

from text2sql.db import sqlglot_name
from text2sql.llm import LLMClient, cached_block, extract_sql, log_conversation, with_cached_prefix
from text2sql.pipeline.events import EventEmitter, PipelineEvent, Stage, Status, TokenDelta
from text2sql.pipeline.repair import parse_sql
from text2sql.pipeline.tools import CLEAN, REVIEW, SUBMIT, Tool
from text2sql.prompts.manager import PromptManager

logger = logging.getLogger(__name__)

REVIEW_GRANT = 5     # turns a new review finding buys: fix, re-review, submit
GRANT_CEILING = 2    # grants stop at this multiple of the configured budget
PROBE = re.compile(r"\bsqlite_master\b|\bpragma[_\s]+\w+\s*\(", re.IGNORECASE)


class SQLAgent:
    """Generates SQL by iterating over tools until it submits an answer."""

    def __init__(
        self,
        llm: LLMClient,
        prompt_manager: PromptManager,
        tools: list[Tool],
        *,
        max_turns: int = 12,
        mode: str = "retrieval",
        event_verbosity: str = "verbose",
    ) -> None:
        self.llm = llm
        self.prompt_manager = prompt_manager
        self.tools = tools
        self.max_turns = max_turns
        self.mode = mode
        self.event_verbosity = event_verbosity

    async def generate(
        self,
        question: str,
        schema_text: str,
        *,
        context: dict[str, Any] | None = None,
        dialect: str = "sqlite",
        strategy: str = "",
        seed: int | None = None,
        emitter: EventEmitter | None = None,
    ) -> AsyncIterator[PipelineEvent | TokenDelta | tuple[str, dict[str, Any]]]:
        """Run the tool loop until the agent submits, answers, or runs out of turns.

        Args:
            question: The user's natural language question.
            schema_text: Schema rendered into the system prompt.
            context: Extra template arguments (knowledge, examples).
            dialect: Database dialect name.
            strategy: Reasoning style for candidate diversity.
            seed: Sampling seed.
            emitter: Event emitter for progress events.

        Yields:
            Progress events and streamed tokens, then ``(sql, trace)`` last.
        """
        emitter = emitter or EventEmitter()
        by_name = {t.name: t for t in self.tools}
        specs = [t.spec() for t in self.tools]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": cached_block(self.prompt_manager.render(
                "agent_system", schema=schema_text, **(context or {}),
                dialect=dialect, strategy=strategy, tools=self.tools,
                mode=self.mode, max_turns=self.max_turns))},
            {"role": "user", "content": question},
        ]
        trace: dict[str, Any] = {"turns": 0, "tool_calls": [], "termination": "max_turns"}
        sql = ""
        last_ok = ""    # most recent query that ran without error
        nominated = ""  # most recent query the agent put up for review
        limit, turn = self.max_turns, 0
        granted: set[str] = set()

        try:
            seen: set[str] = set()
            while turn < limit:
                turn += 1
                trace["turns"] = turn
                reviewed = any(c["name"] == REVIEW for c in trace["tool_calls"])
                if turn >= limit - 1:
                    # One turn out, not on the last: review then submit needs two. Anthropic
                    # carries tool results in a *user* block, so ride along on the last tool
                    # result rather than appending two consecutive user turns. The last
                    # turn offers only submitting, so it must not ask for a review.
                    nudge = (f"Turns are nearly out: call "
                             f"{SUBMIT if reviewed or turn == limit else REVIEW} now, "
                             "best query so far.")
                    if messages[-1].get("role") == "tool":
                        messages[-1]["content"] += f"\n{nudge}"
                    else:
                        messages.append({"role": "user", "content": nudge})
                # At the cap, submitting is the only tool left: the nudge alone gets ignored.
                # Before the first review it is the only one withheld, else the agent saves the
                # review until no turn is left to act on what it reports.
                submits = [s for s in specs if s["function"]["name"] == SUBMIT]
                if turn == limit and submits:
                    offered = submits
                elif reviewed or not any(s["function"]["name"] == REVIEW for s in specs):
                    offered = specs
                else:
                    offered = [s for s in specs if s not in submits]
                allowed = {s["function"]["name"] for s in offered}
                message: dict[str, Any] = {}
                async for item in self.llm.stream_chat_messages(
                        with_cached_prefix(messages), tools=offered, seed=seed):
                    if isinstance(item, tuple):
                        yield TokenDelta(item[0], item[1])
                    else:
                        message = item
                messages.append(message)

                if not (calls := message.get("tool_calls") or []):
                    # A turn with no tool call may be a query or prose. Parsing is not enough
                    # - prose can carry a query that parses over invented columns - so keep it
                    # only if it also runs.
                    sql = _answer(extract_sql(str(message.get("content", ""))))
                    if sql and (parse_sql(sql, sqlglot_name(dialect)) is None
                                or ((run := by_name.get("execute_sql"))
                                    and (await _run(run, {"sql": sql})).startswith("ERROR:"))):
                        sql = ""
                    if not (sql or nominated or last_ok) and turn < limit:
                        messages.append({"role": "user", "content":
                                         "No SQL yet. Make your best attempt with the columns "
                                         "you have and run it."})
                        continue
                    trace["termination"] = "no_tool_call"
                    break

                for index, call in enumerate(calls, 1):
                    name, raw = call["function"]["name"], call["function"]["arguments"]
                    try:  # arguments arrive as a JSON string; a malformed one is an empty call
                        args = json.loads(raw) if isinstance(raw, str) else raw
                    except ValueError:
                        args = {}
                    args = args if isinstance(args, dict) else {}
                    # A repeat cannot return anything new and costs a turn; compared
                    # whitespace- and case-insensitively, since a rerun is often reformatted.
                    key = f"{name}{sorted((k, ' '.join(str(v).split()).lower()) for k, v in args.items())}"
                    repeat = key in seen
                    seen.add(key)
                    # `by_name` holds every tool, so the dispatcher must enforce the offer too.
                    if repeat:
                        result = "ERROR: identical call already made; see its result above."
                    elif name in by_name and name not in allowed:
                        # A query the turn cannot take is still the agent's latest, and ranks
                        # by the tool it was offered to: review and submit nominate, run does not.
                        if cand := _answer(args.get("sql", "")):
                            if name == "execute_sql":
                                last_ok = cand
                            else:
                                nominated = cand
                        result = (f"ERROR: call {REVIEW} first." if name == SUBMIT
                                  else f"ERROR: {name} is not available on this turn.")
                    else:
                        result = await _run(by_name.get(name), args)
                    trace["tool_calls"].append(
                        {"turn": turn, "name": name, **args, "result": _clip(result, 300)})
                    # The budget only reaches the model through the results: the system prompt
                    # is cached and the final-turn nudge arrives too late to plan against.
                    messages.append({"role": "tool", "tool_call_id": call.get("id", name),
                                     "content": f"{result}\n[turn {turn}/{limit}]"})
                    # Also log it: the events below only reach a live stream consumer, so
                    # without this the loop is invisible to the benchmark and to --no-stream.
                    logger.debug("turn %d/%d call %d/%d %s(%s) → %s", turn, limit,
                                 index, len(calls), name,
                                 _clip(" ".join(map(str, args.values())), 400), _clip(result, 400))
                    detail = (f"Agent turn {turn}/{limit}"
                              if self.event_verbosity == "minimal" else
                              f"Agent called {name} ({index}/{len(calls)} this turn)"
                              if self.event_verbosity == "detailed" else
                              f"{name}({_clip(' '.join(map(str, args.values())))}) → {_clip(result)}")
                    yield emitter.emit(Stage.SQL_GENERATION, Status.PROGRESS, detail,
                                       turn=turn, tool=name)
                    # A finding buys turns to act on it; a repeat buys none, so a fix the
                    # agent cannot make still terminates.
                    if (name == REVIEW and result != CLEAN
                            and not result.startswith("ERROR:") and result not in granted):
                        granted.add(result)
                        limit = min(max(limit, turn + REVIEW_GRANT),
                                    self.max_turns * GRANT_CEILING)
                    if name == SUBMIT and not result.startswith("ERROR:"):
                        # `submit_sql` already normalized it; re-extracting would change the
                        # string review checked.
                        sql, trace["termination"] = result, "submit"
                        break  # nothing after a submission can change the answer
                    if name == "execute_sql" and not result.startswith("ERROR:"):
                        last_ok = _answer(args.get("sql", "")) or last_ok
                    elif (name == REVIEW and not result.startswith("ERROR:")
                            and "[result]" not in result):
                        # A query review says returns no answer must not outrank real rows.
                        nominated = _answer(args.get("sql", "")) or nominated
                if trace["termination"] == "submit":
                    break

        except Exception as e:
            trace["termination"], trace["error"] = "error", str(e)
            logger.error("Agent turn failed: %s", e, exc_info=True)
            yield emitter.emit(Stage.SQL_GENERATION, Status.ERROR, f"Agent failed: {e}", error=str(e))

        # Running out of turns, answering without SQL, or erroring mid-run all leave `sql`
        # empty, which fails the whole request. A query the agent nominated for review beats
        # the last one it happened to run, which may be a sanity-check probe.
        if not sql and (fallback := nominated or last_ok):
            sql, trace["recovered_from"] = fallback, trace["termination"]

        trace["limit"], trace["granted"] = limit, len(granted)
        trace["conversation"] = messages
        log_conversation(messages)
        logger.info("Agent finished: %s after %d turn(s), %d tool call(s)%s",
                    trace["termination"], trace["turns"], len(trace["tool_calls"]),
                    "; recovered an unsubmitted query" if trace.get("recovered_from") else "")
        yield sql, trace

def _answer(sql: str) -> str:
    """The SQL if it could be an answer; reading the schema catalogue never is.

    Args:
        sql: The SQL a tool was handed.

    Returns:
        The SQL, or ``""`` when it only inspects the catalogue.
    """
    return "" if not sql or PROBE.search(sql) else sql


def _clip(text: str, limit: int = 120) -> str:
    """Flatten and truncate text for a log line or event.

    Args:
        text: Text to clip.
        limit: Maximum characters kept.

    Returns:
        The whitespace-collapsed text, ellipsised when longer than ``limit``.
    """
    flat = " ".join(text.split())
    return flat[:limit] + "…" if len(flat) > limit else flat


async def _run(tool: Tool | None, args: dict[str, Any]) -> str:
    """Invoke one tool, turning a failure into a result the model can recover from.

    Args:
        tool: The tool to run, or None when the name is unknown.
        args: Tool arguments.

    Returns:
        The tool's output, or an ``ERROR:`` line.
    """
    if tool is None:
        return "ERROR: unknown tool"
    try:
        return await tool.run(**args)
    except Exception as e:
        return f"ERROR: {e}"
