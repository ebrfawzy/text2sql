"""Tests for text2sql.pipeline.agent — the ReAct tool loop.

The LLM is a scripted stub yielding the same shapes ``stream_chat_messages``
produces with ``tools=``: ``(text, is_thinking)`` chunks then an assistant message.
"""

from __future__ import annotations

import pytest
from conftest import message_text

from text2sql.pipeline.agent import SQLAgent
from text2sql.pipeline.events import PipelineEvent, TokenDelta
from text2sql.pipeline.tools import CLEAN, REVIEW, SUBMIT, Tool


def call(name, **args):
    """One tool call in the provider's wire shape."""
    import json
    return {"id": f"c_{name}", "function": {"name": name, "arguments": json.dumps(args)}}


class StubLLM:
    """Replays scripted turns; records the messages it was sent."""

    model = "stub"

    def __init__(self, turns):
        self.turns, self.seen, self.seeds, self.tools_seen = list(turns), [], [], []

    async def stream_chat_messages(self, messages, *, tools=None, seed=None, **kw):
        self.seen.append(list(messages))
        self.seeds.append(seed)
        self.tools_seen.append(list(tools or []))
        turn = self.turns.pop(0) if self.turns else {"role": "assistant", "content": ""}
        yield (str(turn.get("content") or ""), False)
        yield turn


async def _no_answer(sql: str) -> str:
    """A review reporting the one finding that means "this returns no answer"."""
    return "[result] Query returned only NULLs"


async def _echo(sql: str) -> str:
    return sql


async def _clean(sql: str) -> str:
    return CLEAN


def agent_with(pm, turns, tools=None, **kw):
    async def boom(**_):
        raise RuntimeError("tool exploded")

    tools = tools or [
        Tool(SUBMIT, "submit", {"sql": "sql"}, _echo),
        Tool("execute_sql", "run", {"sql": "sql"}, _echo),
        Tool("bad_tool", "fails", {"x": "x"}, boom),
    ]
    llm = StubLLM(turns)
    return SQLAgent(llm, pm, tools, **kw), llm


async def drain(agent, **kw):
    items = [i async for i in agent.generate("Q", "Schema", **kw)]
    return items[:-1], items[-1]


class TestTermination:
    async def test_submit_ends_the_loop(self, prompt_manager):
        agent, llm = agent_with(prompt_manager, [
            {"role": "assistant", "tool_calls": [call("execute_sql", sql="SELECT 1")]},
            {"role": "assistant", "tool_calls": [call(SUBMIT, sql="SELECT 2")]},
            {"role": "assistant", "content": "never reached"},
        ])
        _, (sql, trace) = await drain(agent)
        assert sql == "SELECT 2"
        assert trace["termination"] == "submit"
        assert trace["turns"] == 2
        assert llm.turns == [{"role": "assistant", "content": "never reached"}]

    async def test_a_prose_answer_parses_under_a_sqlalchemy_dialect_name(self, prompt_manager):
        """`postgresql` is not a sqlglot dialect — it raises, and the swallowed error dropped
        every prose answer on Postgres and Athena."""
        agent, _ = agent_with(prompt_manager, [
            {"role": "assistant", "content": "```sql\nSELECT 3\n```"}])
        _, (sql, trace) = await drain(agent, dialect="postgresql")
        assert "SELECT 3" in sql and trace["termination"] == "no_tool_call"

    async def test_answer_without_a_tool_call_is_accepted(self, prompt_manager):
        agent, _ = agent_with(prompt_manager, [{"role": "assistant", "content": "```sql\nSELECT 3\n```"}])
        _, (sql, trace) = await drain(agent)
        assert "SELECT 3" in sql
        assert trace["termination"] == "no_tool_call"

    async def test_the_nudge_leaves_a_turn_to_fix_a_refusal(self, prompt_manager):
        """It fired on the final turn, so a gate refusal there had nowhere to go."""
        agent, llm = agent_with(prompt_manager, [
            {"role": "assistant", "tool_calls": [call("execute_sql", sql="SELECT 1")]},
            {"role": "assistant", "tool_calls": [call("execute_sql", sql="SELECT 2")]},
            {"role": "assistant", "tool_calls": [call(SUBMIT, sql="SELECT 3")]},
        ], max_turns=3)
        await drain(agent)
        nudges = [m for m in llm.seen[-1]
                  if "content" in m and "Turns are nearly out" in message_text(m)]
        assert nudges, "the agent was never told its turns were running out"

    async def test_a_withheld_tool_is_refused_not_run(self, prompt_manager):
        """Offering fewer specs was not enough: Sonnet called execute_sql on its final turn and
        the dispatcher ran it, because `by_name` holds every tool regardless of what was sent."""
        agent, llm = agent_with(prompt_manager, [
            {"role": "assistant", "tool_calls": [call("execute_sql", sql="SELECT 1")]},
        ], max_turns=1)
        _, (_, trace) = await drain(agent)
        assert "not available on this turn" in trace["tool_calls"][0]["result"]

    async def test_the_final_turn_offers_only_submit(self, prompt_manager):
        """Sonnet took the nudge twice and ran execute_sql both times, so the run was cut off
        and the fallback shipped a `SELECT COUNT(*)` diagnostic as the answer."""
        agent, llm = agent_with(prompt_manager, [
            {"role": "assistant", "tool_calls": [call("execute_sql", sql="SELECT 1")]},
            {"role": "assistant", "tool_calls": [call(SUBMIT, sql="SELECT 2")]},
        ], max_turns=2)
        await drain(agent)
        assert [t["function"]["name"] for t in llm.tools_seen[0]] != [SUBMIT]
        assert [t["function"]["name"] for t in llm.tools_seen[-1]] == [SUBMIT]

    async def test_turn_cap_stops_an_agent_that_never_submits(self, prompt_manager):
        turns = [{"role": "assistant", "tool_calls": [call("execute_sql", sql="SELECT 1")]}] * 10
        agent, _ = agent_with(prompt_manager, turns, max_turns=3)
        _, (sql, trace) = await drain(agent)
        assert (trace["termination"], trace["turns"]) == ("max_turns", 3)
        # The cap is not a failure: the last query that ran cleanly is the answer.
        assert (sql, trace["recovered_from"]) == ("SELECT 1", "max_turns")


class TestRecovery:
    """An agent that stops without submitting still executed real queries — losing them
    fails the whole request downstream, so the last clean one is used instead."""

    async def test_a_query_reviewed_as_answerless_is_not_the_fallback(self, prompt_manager):
        """One measured run reviewed a query that returned only NULLs and then ran out of
        turns, and that same query was recovered as the answer. The repeated reviews spend
        the turns that finding buys."""
        agent, _ = agent_with(prompt_manager, [
            {"role": "assistant", "tool_calls": [call("execute_sql", sql="SELECT ran")]},
            *[{"role": "assistant", "tool_calls": [call(REVIEW, sql="SELECT empty")]}] * 3,
        ], tools=[Tool("execute_sql", "run", {"sql": "sql"}, _echo),
                  Tool(REVIEW, "review", {"sql": "sql"}, _no_answer)], max_turns=2)
        _, (sql, trace) = await drain(agent)
        assert sql == "SELECT ran" and trace["recovered_from"] == "max_turns"

    async def test_recovers_the_most_recent_successful_query(self, prompt_manager):
        turns = [
            {"role": "assistant", "tool_calls": [call("execute_sql", sql="SELECT 1")]},
            {"role": "assistant", "tool_calls": [call("execute_sql", sql="SELECT 2")]},
        ]
        # Three turns, so neither probe lands on the final one, where only submit is offered.
        agent, _ = agent_with(prompt_manager, turns, max_turns=3)
        _, (sql, _) = await drain(agent)
        assert sql == "SELECT 2"

    async def test_sql_refused_by_the_final_turn_is_still_recovered(self, prompt_manager):
        """`archeology_5`, `credit_8` and `virtual_M_4` each handed `execute_sql` their real
        answer on turn 20, where only submitting is offered; all three shipped an older probe."""
        agent, _ = agent_with(prompt_manager, [
            {"role": "assistant", "tool_calls": [call("execute_sql", sql="SELECT probe")]},
            {"role": "assistant", "tool_calls": [call("execute_sql", sql="SELECT answer")]},
        ], max_turns=2)
        _, (sql, trace) = await drain(agent)
        assert "not available on this turn" in trace["tool_calls"][-1]["result"]
        assert (sql, trace["recovered_from"]) == ("SELECT answer", "max_turns")

    async def test_a_refused_review_renominates(self, prompt_manager):
        """`museum_4` put a refined query up for review on turn 20, was refused, and shipped
        the draft it had reviewed twelve turns earlier."""
        agent, _ = agent_with(prompt_manager, [
            {"role": "assistant", "tool_calls": [call(REVIEW, sql="SELECT draft")]},
            {"role": "assistant", "tool_calls": [call(REVIEW, sql="SELECT refined")]},
        ], tools=[Tool(SUBMIT, "submit", {"sql": "sql"}, _echo),
                  Tool(REVIEW, "review", {"sql": "sql"}, _clean)], max_turns=2)
        _, (sql, _) = await drain(agent)
        assert sql == "SELECT refined"

    @pytest.mark.parametrize("probe", [
        "SELECT p.name FROM sqlite_master m, pragma_table_info(m.name) p",
        "PRAGMA table_info(marketstats); PRAGMA table_info(analyticsindicators);",
    ])
    async def test_a_schema_probe_is_never_the_answer(self, prompt_manager, probe):
        """`museum_9` shipped a `pragma_table_info` dump, which repair then spent three rounds
        rewriting into an invented answer; `crypto_7` shipped bare `PRAGMA table_info(...)`."""
        turns = [{"role": "assistant", "tool_calls": [call("execute_sql", sql=probe)]}]
        agent, _ = agent_with(prompt_manager, turns, max_turns=1)
        _, (sql, trace) = await drain(agent)
        assert sql == "" and "recovered_from" not in trace

    async def test_failed_queries_are_not_recovered(self, prompt_manager):
        async def fails(sql: str) -> str:
            return "ERROR: no such table: nope"

        tools = [Tool("execute_sql", "run", {"sql": "sql"}, fails)]
        turns = [{"role": "assistant", "tool_calls": [call("execute_sql", sql="SELECT * FROM nope")]}]
        agent, _ = agent_with(prompt_manager, turns, tools=tools, max_turns=1)
        _, (sql, trace) = await drain(agent)
        assert sql == ""
        assert "recovered_from" not in trace

    async def test_a_submitted_query_wins_over_the_recovered_one(self, prompt_manager):
        turns = [
            {"role": "assistant", "tool_calls": [call("execute_sql", sql="SELECT 1")]},
            {"role": "assistant", "tool_calls": [call(SUBMIT, sql="SELECT 2")]},
        ]
        agent, _ = agent_with(prompt_manager, turns, max_turns=5)
        _, (sql, trace) = await drain(agent)
        assert (sql, trace["termination"]) == ("SELECT 2", "submit")
        assert "recovered_from" not in trace

    async def test_prose_answer_with_no_sql_falls_back(self, prompt_manager):
        """`no_tool_call` with unparseable content used to yield empty SQL."""
        turns = [
            {"role": "assistant", "tool_calls": [call("execute_sql", sql="SELECT 1")]},
            {"role": "assistant", "content": "I could not determine the query."},
        ]
        agent, _ = agent_with(prompt_manager, turns, max_turns=5)
        _, (sql, trace) = await drain(agent)
        assert (sql, trace["recovered_from"]) == ("SELECT 1", "no_tool_call")


class TestToolHandling:
    async def test_failing_tool_becomes_a_message_and_the_loop_continues(self, prompt_manager):
        agent, llm = agent_with(prompt_manager, [
            {"role": "assistant", "tool_calls": [call("bad_tool", x="1")]},
            {"role": "assistant", "tool_calls": [call(SUBMIT, sql="SELECT 4")]},
        ])
        _, (sql, trace) = await drain(agent)
        assert sql == "SELECT 4"  # recovered
        assert "ERROR: tool exploded" in message_text(llm.seen[-1][-1])

    async def test_giving_up_in_prose_with_nothing_to_ship_is_pushed_back(self, prompt_manager):
        """One measured run answered "I cannot find a column named FVS" at turn 9 of 20, having
        executed nothing, and shipped no SQL at all - which scores below a wrong answer."""
        agent, llm = agent_with(prompt_manager, [
            {"role": "assistant", "content": "I cannot find a column named FVS."},
            {"role": "assistant", "tool_calls": [call(SUBMIT, sql="SELECT best")]},
        ], tools=[Tool(SUBMIT, "submit", {"sql": "sql"}, _echo)], max_turns=4)
        _, (sql, trace) = await drain(agent)
        assert (sql, trace["termination"]) == ("SELECT best", "submit")
        assert "No SQL yet" in message_text(llm.seen[-1][-1])

    async def test_unknown_tool_is_reported_not_raised(self, prompt_manager):
        agent, llm = agent_with(prompt_manager, [
            {"role": "assistant", "tool_calls": [call("no_such_tool")]},
            {"role": "assistant", "tool_calls": [call(SUBMIT, sql="SELECT 5")]},
        ])
        _, (sql, _) = await drain(agent)
        assert sql == "SELECT 5"
        assert "unknown tool" in message_text(llm.seen[-1][-1])

    async def test_prose_sql_that_does_not_run_is_discarded(self, prompt_manager):
        """One measured run answered in prose with a query that parsed but named invented
        columns, and it shipped without ever executing."""
        async def boom(sql: str) -> str:
            return "ERROR: no such column: nope"

        agent, _ = agent_with(prompt_manager, [
            {"role": "assistant", "tool_calls": [call("execute_sql", sql="SELECT ok")]},
            {"role": "assistant", "content": "SELECT nope FROM t"},
        ], tools=[Tool(SUBMIT, "submit", {"sql": "sql"}, _echo),
                  Tool("execute_sql", "run", {"sql": "sql"}, boom)])
        _, (sql, trace) = await drain(agent)
        assert sql == "" and trace["termination"] == "no_tool_call"

    async def test_the_fallback_prefers_a_nominated_query_over_the_last_executed(
            self, prompt_manager):
        """Ending a turn short of submitting once fell back to the `SELECT COUNT(*)` the agent
        had run to sanity-check the answer it meant to give."""
        agent, _ = agent_with(prompt_manager, [
            {"role": "assistant", "tool_calls": [call(REVIEW, sql="SELECT answer")]},
            {"role": "assistant", "tool_calls": [call("execute_sql", sql="SELECT COUNT(*)")]},
        ], tools=[Tool(SUBMIT, "submit", {"sql": "sql"}, _echo),
                  Tool("execute_sql", "run", {"sql": "sql"}, _echo),
                  Tool(REVIEW, "review", {"sql": "sql"}, _clean)], max_turns=2)
        _, (sql, trace) = await drain(agent)
        assert sql == "SELECT answer" and trace["recovered_from"] == "max_turns"

    async def test_a_review_finding_buys_turns_and_a_repeat_does_not(self, prompt_manager):
        """One measured run was cut off at the cap mid-fix and shipped the flawed query. A
        finding already paid for buys nothing, so a fix the agent cannot make still ends."""
        agent, _ = agent_with(prompt_manager, [
            *[{"role": "assistant", "tool_calls": [call(REVIEW, sql="SELECT a")]}] * 2,
            {"role": "assistant", "tool_calls": [call(SUBMIT, sql="SELECT a")]},
        ], tools=[Tool(SUBMIT, "submit", {"sql": "sql"}, _echo),
                  Tool(REVIEW, "review", {"sql": "sql"}, _no_answer)], max_turns=2)
        _, (sql, trace) = await drain(agent)
        assert (sql, trace["termination"], trace["limit"], trace["granted"]) \
            == ("SELECT a", "submit", 4, 1)

    async def test_the_last_turn_asks_for_the_tool_it_offers(self, prompt_manager):
        """One measured run was told to review on its final turn, where only submitting is
        offered, so it was refused and never submitted at all."""
        agent, llm = agent_with(prompt_manager, [
            {"role": "assistant", "tool_calls": [call("execute_sql", sql="SELECT a")]},
            {"role": "assistant", "tool_calls": [call(SUBMIT, sql="SELECT a")]},
        ], tools=[Tool(SUBMIT, "submit", {"sql": "sql"}, _echo),
                  Tool("execute_sql", "run", {"sql": "sql"}, _echo),
                  Tool(REVIEW, "review", {"sql": "sql"}, _no_answer)], max_turns=2)
        _, (sql, _) = await drain(agent)
        assert sql == "SELECT a" and f"call {SUBMIT} now" in message_text(llm.seen[-1][-1])

    async def test_submit_opens_only_after_a_review(self, prompt_manager):
        """One measured run reviewed at turn 29 of 30, was told the query returned nothing, and
        submitted it anyway because that was the only turn left."""
        agent, _ = agent_with(prompt_manager, [
            {"role": "assistant", "tool_calls": [call(SUBMIT, sql="SELECT unreviewed")]},
            {"role": "assistant", "tool_calls": [call(REVIEW, sql="SELECT answer")]},
            {"role": "assistant", "tool_calls": [call(SUBMIT, sql="SELECT answer")]},
        ], tools=[Tool(SUBMIT, "submit", {"sql": "sql"}, _echo),
                  Tool(REVIEW, "review", {"sql": "sql"}, _echo)], max_turns=4)
        _, (sql, trace) = await drain(agent)
        assert trace["tool_calls"][0]["result"] == f"ERROR: call {REVIEW} first."
        assert sql == "SELECT answer" and trace["termination"] == "submit"

    async def test_the_cap_still_lets_an_unreviewed_query_out(self, prompt_manager):
        """The final turn must always have an exit, or the gate costs the whole budget."""
        agent, _ = agent_with(prompt_manager, [
            {"role": "assistant", "tool_calls": [call(SUBMIT, sql="SELECT last")]},
        ], tools=[Tool(SUBMIT, "submit", {"sql": "sql"}, _echo),
                  Tool(REVIEW, "review", {"sql": "sql"}, _echo)], max_turns=1)
        _, (sql, trace) = await drain(agent)
        assert sql == "SELECT last" and trace["termination"] == "submit"

    async def test_a_reformatted_repeat_is_still_a_repeat(self, prompt_manager):
        """One measured run re-ran the same four-CTE query seven times, reformatted each time,
        and the byte-identical guard caught none of them."""
        agent, _ = agent_with(prompt_manager, [
            {"role": "assistant", "tool_calls": [call("execute_sql", sql="SELECT 1 FROM t")]},
            {"role": "assistant", "tool_calls": [call("execute_sql", sql="select 1\n  FROM  t")]},
            {"role": "assistant", "tool_calls": [call(SUBMIT, sql="SELECT 1 FROM t")]},
        ])
        _, (_, trace) = await drain(agent)
        assert "identical call already made" in trace["tool_calls"][1]["result"]

    async def test_malformed_arguments_do_not_crash_the_run(self, prompt_manager):
        agent, _ = agent_with(prompt_manager, [
            {"role": "assistant",
             "tool_calls": [{"id": "x", "function": {"name": "execute_sql", "arguments": "{oops"}}]},
            {"role": "assistant", "tool_calls": [call(SUBMIT, sql="SELECT 6")]},
        ])
        _, (sql, _) = await drain(agent)
        assert sql == "SELECT 6"

    async def test_tool_results_are_fed_back_with_the_turn_budget(self, prompt_manager):
        """The budget only reaches the model here: the system prompt is cached and the
        final-turn nudge arrives too late to plan against."""
        agent, llm = agent_with(prompt_manager, [
            {"role": "assistant", "tool_calls": [call("execute_sql", sql="SELECT 7")]},
            {"role": "assistant", "tool_calls": [call(SUBMIT, sql="SELECT 7")]},
        ])
        await drain(agent)
        tool_msgs = [m for m in llm.seen[-1] if m.get("role") == "tool"]
        assert tool_msgs and message_text(tool_msgs[0]) == "SELECT 7\n[turn 1/12]"


class TestObservability:
    async def test_streams_tokens_and_emits_one_event_per_tool_call(self, prompt_manager):
        agent, _ = agent_with(prompt_manager, [
            {"role": "assistant", "content": "thinking",
             "tool_calls": [call("execute_sql", sql="SELECT 1")]},
            {"role": "assistant", "tool_calls": [call(SUBMIT, sql="SELECT 1")]},
        ])
        items, _ = await drain(agent)
        assert any(isinstance(i, TokenDelta) for i in items)
        events = [i for i in items if isinstance(i, PipelineEvent)]
        assert len(events) == 2
        assert events[0].data["tool"] == "execute_sql"

    async def test_trace_records_every_tool_call(self, prompt_manager):
        agent, _ = agent_with(prompt_manager, [
            {"role": "assistant", "tool_calls": [call("execute_sql", sql="SELECT 1")]},
            {"role": "assistant", "tool_calls": [call(SUBMIT, sql="SELECT 1")]},
        ])
        _, (_, trace) = await drain(agent)
        assert [c["name"] for c in trace["tool_calls"]] == ["execute_sql", SUBMIT]
        assert trace["tool_calls"][0]["sql"] == "SELECT 1"

    async def test_seed_is_forwarded_for_candidate_diversity(self, prompt_manager):
        agent, llm = agent_with(prompt_manager, [{"role": "assistant", "content": "SELECT 1"}])
        await drain(agent, seed=4242)
        assert llm.seeds == [4242]

    async def test_stream_failure_surfaces_as_an_error_event(self, prompt_manager):
        class Boom(StubLLM):
            async def stream_chat_messages(self, *a, **kw):
                raise RuntimeError("stream died")
                yield

        agent = SQLAgent(Boom([]), prompt_manager, [])
        items, (sql, trace) = await drain(agent)
        assert sql == "" and trace["termination"] == "error"
        assert any(isinstance(i, PipelineEvent) and i.status == "error" for i in items)
