"""Tests for text2sql.pipeline.tools — the agent's tool layer.

Each tool is an adapter over a pipeline capability, so these check the adapter's
contract (guards, error shape, enablement) rather than the underlying feature.
"""

from __future__ import annotations

import pytest

from text2sql.pipeline.examples import ExampleStore
from text2sql.pipeline.tools import (
    MAX_HITS,
    MAX_TERMS,
    MAX_TOOL_ROWS,
    REVIEW,
    SUBMIT,
    build_tools,
)
from text2sql.profiler.knowledge import DatabaseKnowledge, KnowledgeEntry
from text2sql.schema.loader import SchemaLoader


@pytest.fixture
def tools(db_conn):
    def build(mode="retrieval", knowledge=None, store=None, question="", linked=None, loader=None):
        return {t.name: t for t in build_tools(
            db_conn, loader or SchemaLoader(db_conn), knowledge or DatabaseKnowledge(),
            store or ExampleStore(), tools=mode, question=question, linked=linked)}
    return build


class TestToolSet:
    @pytest.mark.parametrize(
        ("mode", "expected"),
        [("schema_preloaded", {"execute_sql", REVIEW, SUBMIT}),
         ("retrieval", {"execute_sql", REVIEW, SUBMIT, "describe_table", "describe_columns",
                        "search_columns"})],
    )
    def test_enabled_tools_per_mode(self, tools, mode, expected):
        assert set(tools(mode)) == expected

    def test_knowledge_tool_needs_knowledge_and_an_undefined_question(self, tools):
        """A benchmark question carries its own definitions block, so there is nothing left to
        look up; offering the tool anyway drew the agent away from the terms it was given."""
        kb = DatabaseKnowledge({0: KnowledgeEntry(0, "DTI", definition="debt / income")})
        assert "search_knowledge" not in tools("retrieval")
        assert "search_knowledge" in tools("retrieval", knowledge=kb)
        assert "search_knowledge" not in tools(
            "retrieval", knowledge=kb, question="Rank them.\n<definitions>\n- DTI: ...")

    def test_example_tool_is_tiered_like_the_other_retrieval_tools(self, tools, tmp_path):
        """`schema_preloaded` has no lookup tools, so a configured scenarios file must not
        leak one in."""
        f = tmp_path / "scenarios.md"
        f.write_text("## Churn\nCustomers with no order in 90 days.\n")
        store = ExampleStore(str(f))
        assert store.headings  # otherwise the tool is never built and this proves nothing
        assert "lookup_example" not in tools("schema_preloaded", store=store)
        assert "lookup_example" in tools("retrieval", store=store)

    def test_spec_is_a_valid_function_schema(self, tools):
        spec = tools()["execute_sql"].spec()
        assert spec["type"] == "function"
        assert spec["function"]["name"] == "execute_sql"
        assert spec["function"]["parameters"]["required"] == ["sql"]


class TestExecuteSql:
    async def test_returns_rows(self, tools):
        out = await tools()["execute_sql"].run(sql="SELECT COUNT(*) AS cnt FROM users")
        assert "cnt" in out and "5" in out

    async def test_error_is_returned_not_raised(self, tools):
        assert "ERROR" in await tools()["execute_sql"].run(sql="SELECT * FROM nope")

    async def test_unlimited_select_is_capped(self, tools, db_conn, monkeypatch):
        """An exploratory SELECT must not pull an entire table into the context."""
        seen = []
        original = db_conn.execute_safe
        monkeypatch.setattr(db_conn, "execute_safe",
                            lambda sql, *a, **k: (seen.append(sql), original(sql, *a, **k))[1])
        await tools()["execute_sql"].run(sql="SELECT * FROM users")
        assert f"LIMIT {MAX_TOOL_ROWS}" in seen[0].upper()

    async def test_the_cap_does_not_rewrite_the_query(self, tools, db_conn, monkeypatch):
        """The cap re-serialized the parse, which turns `json_extract(x, p)` into sqlite's
        `x -> p` — quoted JSON, so a correct filter matched nothing and the agent was shown
        `"value"` for what it had asked for bare."""
        seen = []
        original = db_conn.execute_safe
        monkeypatch.setattr(db_conn, "execute_safe",
                            lambda sql, *a, **k: (seen.append(sql), original(sql, *a, **k))[1])
        sent = "SELECT json_extract(name, '$.a') AS a FROM users"
        await tools()["execute_sql"].run(sql=sent)
        assert seen[0].startswith(sent) and "->" not in seen[0]

    async def test_existing_limit_is_kept(self, tools, db_conn, monkeypatch):
        seen = []
        original = db_conn.execute_safe
        monkeypatch.setattr(db_conn, "execute_safe",
                            lambda sql, *a, **k: (seen.append(sql), original(sql, *a, **k))[1])
        await tools()["execute_sql"].run(sql="SELECT * FROM users LIMIT 2")
        assert "LIMIT 2" in seen[0].upper()


class TestNonQueryStatements:
    """A rolled-back CREATE INDEX returns nothing; calling that "0 rows" reads as failure
    and has pushed the agent to replace correct DDL with a SELECT."""

    async def test_ddl_reports_success_not_zero_rows(self, tools):
        out = await tools()["execute_sql"].run(sql="CREATE INDEX IF NOT EXISTS ix ON users (name)")
        assert "executed successfully" in out
        assert "0 rows" not in out

    async def test_a_name_it_created_is_reported_as_rolled_back(self, tools):
        """Half the write instances in one measured run burned turns verifying a write in a
        later call: the rollback had discarded it, and `no such column` reads as a failed ALTER."""
        t = tools()["execute_sql"]
        await t.run(sql="ALTER TABLE users ADD COLUMN score REAL")
        out = await t.run(sql="SELECT score FROM users")
        assert "ERROR" in out and "rolled back" in out and "same call" in out

    async def test_an_unrelated_missing_name_gets_no_such_hint(self, tools):
        out = await tools()["execute_sql"].run(sql="SELECT nope FROM users")
        assert "ERROR" in out and "rolled back" not in out

    async def test_a_select_with_no_matches_still_reports_zero_rows(self, tools):
        out = await tools()["execute_sql"].run(sql="SELECT id FROM users WHERE 1 = 0")
        assert out == "0 rows."


class TestRetrievalTools:
    async def test_describe_table_omits_value_stats(self, tools):
        """A retrieval tool that returns as many bytes as preloading defeats its purpose:
        one measured table went 6,287 -> 1,537 chars. JSON paths are the exception below."""
        out = await tools()["describe_table"].run(table="users")
        assert "email" in out
        assert "values:" not in out and "range:" not in out and "distinct" not in out

    async def test_describe_unknown_table_names_the_closest_ones(self, tools):
        out = await tools()["describe_table"].run(table="ghosts")
        assert "No such table" in out and "users" in out

    @pytest.mark.parametrize("name", ["Users", "USERS", " users ", '"users"', "user"])
    async def test_a_table_name_resolves_however_the_agent_writes_it(self, tools, name):
        """SQL identifiers are case-insensitive and the agent copies names out of prose;
        every one of these used to cost a turn and return nothing."""
        assert "email" in await tools()["describe_table"].run(table=name)

    @staticmethod
    def _profiled(db_conn, db_profile, **kw):
        return {t.name: t for t in build_tools(
            db_conn, SchemaLoader(db_conn, db_profile), DatabaseKnowledge(), ExampleStore(),
            tools="retrieval", **kw)}

    async def test_describe_table_carries_no_meanings(self, db_conn, db_profile):
        """Breadth must not stand in for depth: short descriptions here had the agent skip
        describe_columns entirely, so it never saw a value distribution. With linking off every
        column keeps its type and nothing is demoted."""
        out = await self._profiled(db_conn, db_profile)["describe_table"].run(table="users")
        assert out.splitlines()[1:4] == ["id[INTEGER]", "name[TEXT]", "email[TEXT]"]
        # No meanings, no `more:` line, and no FKs — the prompt's join map already has those.
        assert ": " not in out and "more:" not in out and "FK:" not in out

    async def test_describe_table_types_the_linked_and_names_the_rest(self, db_conn, db_profile):
        """Linking misses ~1 gold column in 8, so the ones it dropped must stay visible by name
        — a name is enough to follow up with describe_columns."""
        out = await self._profiled(
            db_conn, db_profile, linked={"users": ["email"]})["describe_table"].run(table="users")
        assert out.splitlines()[1] == "email[TEXT]"
        more = out.split("more: ")[1]
        assert more.startswith("id, name, ") and "[" not in more

    async def test_a_padded_key_is_flagged_on_both_rungs(self, db_conn, db_profile):
        """A space-padded text key joins to zero rows against an unpadded one and the type
        does not show it; one measured run spent ten turns finding that by probing."""
        col = db_profile.tables["users"].columns["email"]
        col.min_value, col.max_value = "a@x.com   ", "z@x.com   "
        built = self._profiled(db_conn, db_profile)
        assert "email[TEXT space-padded]" in await built["describe_table"].run(table="users")
        described = await built["describe_columns"].run(fields="users.email")
        assert "users.email[space-padded]" in described

    async def test_describe_columns_details_only_what_was_asked(self, db_conn, db_profile):
        """The depth rung: long descriptions and value stats for the named fields, so the agent
        need not dump a whole table to learn what one column contains. Table case and spacing
        are forgiven here for the same reason they are in describe_table."""
        built = {t.name: t for t in build_tools(
            db_conn, SchemaLoader(db_conn, db_profile), DatabaseKnowledge(), ExampleStore(),
            tools="retrieval")}
        out = await built["describe_columns"].run(fields="users.age, USERS. name")
        assert out.splitlines() == ["users.age", "users.name"]

    async def test_describe_columns_addresses_a_profiled_json_path(self, db_conn, db_profile):
        """A JSON pseudo-column is itself dotted, so splitting the reference on its last dot
        put the path into the table name and lost the field."""
        built = {t.name: t for t in build_tools(
            db_conn, SchemaLoader(db_conn, db_profile), DatabaseKnowledge(), ExampleStore(),
            tools="retrieval")}
        out = await built["describe_columns"].run(fields="orders.meta.channel")
        assert out.splitlines()[0].startswith("orders.meta.channel")

    async def test_a_json_parent_brings_its_leaves(self, db_conn, db_profile):
        """The parent's own blurb only lists its leaf names; the meanings are on the leaves, so
        asking for the parent cost a turn and returned nothing usable."""
        built = {t.name: t for t in build_tools(
            db_conn, SchemaLoader(db_conn, db_profile), DatabaseKnowledge(), ExampleStore(),
            tools="retrieval")}
        out = await built["describe_columns"].run(fields="orders.meta")
        assert [ln.split(":")[0] for ln in out.splitlines()] == [
            "orders.meta", "orders.meta.channel", "orders.meta.ship.express",
            "orders.meta.weight", "orders.meta.scores"]

    async def test_a_json_path_carries_its_values_and_a_plain_column_does_not(
            self, db_conn, db_profile):
        """`credit_8` spent 10 of 19 turns running `SELECT propfinancialdata` to learn which
        keys a JSON blob holds and what they contain — every tool named the path and none
        showed its contents. Ordinary columns keep no stats: that trade is what makes these
        tools cheaper than preloading."""
        built = self._profiled(db_conn, db_profile)
        table = await built["describe_table"].run(table="orders")
        assert "meta.channel[JSON field]: values: 'web', 'store'" in table
        assert "amount[REAL]\n" in table + "\n"          # a plain column, still bare
        cols = await built["describe_columns"].run(fields="orders.meta.channel")
        assert "values: 'web', 'store'" in cols

    async def test_an_array_leaf_shows_an_example_not_a_range(self, db_conn, db_profile):
        """An array leaf's min/max are two whole arrays compared as text, so a range implies
        bounds that do not exist — solar's `irradiance_types` read
        `[0.7, 285.1, ...]-[998.4, 975.3, ...]`. A small domain lists the arrays, a large one
        shows one example; neither claims a range."""
        out = await self._profiled(db_conn, db_profile)["describe_table"].run(table="orders")
        line = next(x for x in out.splitlines() if "meta.scores" in x)
        assert "range:" not in line and "[" in line.split(": ", 1)[1]

    async def test_describe_columns_reports_what_it_could_not_resolve(self, tools):
        out = await tools()["describe_columns"].run(fields="users.ghost")
        assert "No such column" in out

    async def test_search_ranks_and_caps_its_hits(self, tools):
        """It was a boolean filter over prose descriptions: one measured term returned 50
        unranked names, none of them the column meant."""
        out = await tools()["search_columns"].run(term="email address")
        assert out.splitlines()[0].startswith("users.email")
        assert len(out.splitlines()) <= MAX_HITS + 1

    async def test_search_columns(self, tools):
        assert "users.email" in await tools()["search_columns"].run(term="email")

    async def test_a_search_repeating_earlier_hits_says_so(self, tools):
        """One run spent 9 of its 29 searches rephrasing "budget" for a column that does not
        exist, and each rephrasing returned the same two fields and cost a turn."""
        t = tools()["search_columns"]
        assert "orders.amount" in await t.run(term="amount")
        assert await t.run(term="amount spent") == "Same fields as 'amount'."

    async def test_search_finds_profiled_json_paths(self, db_conn, db_profile):
        """A JSON field the profiler discovered is invisible to the live schema."""
        found = {t.name: t for t in build_tools(
            db_conn, SchemaLoader(db_conn, db_profile), DatabaseKnowledge(),
            ExampleStore(), tools="retrieval")}
        assert "orders.meta.channel" in await found["search_columns"].run(term="channel")

    async def test_search_columns_appends_the_terms_its_words_define(self, tools):
        """`search_knowledge` was called on 0.4% of tool calls while `search_columns` ran in
        249/270 instances — both take a concept, so the definition rides along for free."""
        kb = DatabaseKnowledge({0: KnowledgeEntry(0, "DTI", definition="debt / income")})
        t = tools("retrieval", knowledge=kb)["search_columns"]
        first = await t.run(term="DTI")
        assert "debt / income" in first
        # Once told, never repeated: the tool runs ~4x per instance.
        assert "debt / income" not in await t.run(term="DTI ratio")

    async def test_terms_the_prompt_already_inlined_are_not_repeated(self, db_conn):
        """The agent system prompt inlines refuted entries; without seeding, the first
        `search_columns` printed them a second time."""
        kb = DatabaseKnowledge({0: KnowledgeEntry(0, "DTI", definition="debt / income")})
        from text2sql.pipeline.tools import build_tools
        from text2sql.schema.loader import SchemaLoader

        def build(shown):
            return {t.name: t for t in build_tools(
                db_conn, SchemaLoader(db_conn), kb, ExampleStore(),
                tools="retrieval", shown=shown)}
        assert "debt / income" in await build(())["search_columns"].run(term="DTI")
        assert "debt / income" not in await build({0})["search_columns"].run(term="DTI")

    async def test_a_told_entry_cannot_return_as_someone_elses_child(self, db_conn):
        """Deduping before `with_children` let a parent drag a shown entry back in — measured
        on `credit_10`, where entry 21's child is the entry the prompt had inlined."""
        from text2sql.pipeline.tools import build_tools
        from text2sql.schema.loader import SchemaLoader
        kb = DatabaseKnowledge({
            0: KnowledgeEntry(0, "Net Worth", definition="assets - debts"),
            1: KnowledgeEntry(1, "Solvent", definition="net worth > 0", children_knowledge=[0]),
        })
        tools = {t.name: t for t in build_tools(
            db_conn, SchemaLoader(db_conn), kb, ExampleStore(), tools="retrieval", shown={0})}
        out = await tools["search_columns"].run(term="solvent")
        assert "Solvent" in out and "assets - debts" not in out

    async def test_a_term_the_base_does_not_name_is_reported_as_a_miss(self, tools):
        """`cybermarket_8`'s question left "frequently changing connection parameters"
        unquantified; ranking definitions returned an unrelated cohort that did quantify it,
        and the one added conjunct was the whole difference from gold. A term the base does
        not name has no answer, and saying so is the answer."""
        kb = DatabaseKnowledge({0: KnowledgeEntry(
            0, "High-Risk Connection Pattern", definition="connpatscore > 90")})
        t = tools("retrieval", knowledge=kb)
        out = await t["search_knowledge"].run(term="frequently changing parameters")
        assert "No definition matched" in out and "connpatscore" not in out
        assert "High-Risk Connection Pattern" in out  # the closest names, never a definition
        # The ride-along is the same lookup, so it stays silent too.
        assert "connpatscore" not in await t["search_columns"].run(term="connection parameters")

    async def test_search_knowledge_still_answers_a_term_already_shown(self, db_conn):
        """The ride-along never repeats; an explicit lookup always answers."""
        from text2sql.pipeline.tools import build_tools
        from text2sql.schema.loader import SchemaLoader
        kb = DatabaseKnowledge({0: KnowledgeEntry(0, "DTI", definition="debt / income")})
        tools = {t.name: t for t in build_tools(
            db_conn, SchemaLoader(db_conn), kb, ExampleStore(), tools="retrieval", shown={0})}
        assert "debt / income" in await tools["search_knowledge"].run(term="DTI")

    async def test_search_columns_is_unchanged_without_knowledge(self, tools):
        out = await tools("retrieval")["search_columns"].run(term="name")
        assert "users.name" in out and "Definition" not in out

    async def test_search_knowledge_follows_dependencies(self, tools):
        kb = DatabaseKnowledge({
            0: KnowledgeEntry(0, "DTI", definition="debt / income", children_knowledge=[1]),
            1: KnowledgeEntry(1, "Income", definition="salary + bonus"),
        })
        out = await tools("retrieval", knowledge=kb)["search_knowledge"].run(term="dti")
        assert "DTI" in out and "Income" in out  # the child came along

    async def test_a_named_term_wins_over_prose_matches(self, tools):
        """Whole-word counting returned ten entries for an exact term name, because prose
        mentions of its words outscored the term itself."""
        kb = DatabaseKnowledge({i: KnowledgeEntry(i, name, f"about {name.lower()}")
                                for i, name in enumerate(
                                    ["Payment History Quality", "Payment Ratio", "History Depth",
                                     "Quality Index", "Payment Mix"])})
        out = await tools("retrieval", knowledge=kb)["search_knowledge"].run(
            term="Payment History Quality")
        assert out.strip().startswith("- Payment History Quality")
        assert out.count("\n- ") < MAX_TERMS

    async def test_a_miss_offers_the_nearest_names(self, tools):
        """Never a definition the ranker doubts — the agent would compute it — and never the
        alphabetical dump of every term, which was not aimable."""
        kb = DatabaseKnowledge({i: KnowledgeEntry(i, n) for i, n in enumerate(
            ["Peer Deviation", "Account Maturity", "High Spoofing Risk"])})
        out = await tools("retrieval", knowledge=kb)["search_knowledge"].run(term="UPDR")
        assert "No definition matched" in out and "Peer Deviation" in out
        assert "Definition:" not in out

    @pytest.mark.parametrize("term", ["*", "", "  ", "?"])
    async def test_the_index_cannot_be_dumped(self, tools, term):
        """A wildcard made one run enumerate the base a term per turn, spending 15 of its 30;
        the prompt already lists the names closest to the question."""
        kb = DatabaseKnowledge({0: KnowledgeEntry(0, "DTI", definition="debt / income")})
        out = await tools("retrieval", knowledge=kb)["search_knowledge"].run(term=term)
        assert out == "Name the term to define."

    async def test_a_miss_lists_what_exists(self, tools):
        """`search_knowledge("DTI")` once answered "No matching domain knowledge." while
        `Debt-to-Income Ratio (DTI)` existed — unaimable in retrieval mode, where the agent
        cannot see the terms."""
        kb = DatabaseKnowledge({0: KnowledgeEntry(0, "Debt-to-Income Ratio (DTI)")})
        out = await tools("retrieval", knowledge=kb)["search_knowledge"].run(term="leverage")
        assert "Debt-to-Income Ratio (DTI)" in out

    async def test_submit_normalizes_and_nothing_else(self, tools):
        """The terminator: whatever it returns is the answer, so it must be bare SQL, and the
        string it ships has to be the string review checked."""
        out = await tools()[SUBMIT].run(sql="```sql\nSELECT COUNT(*) AS n FROM users\n```")
        assert "SELECT COUNT(*) AS n" in out and "```" not in out
        assert out == await tools()[SUBMIT].run(sql=out)  # idempotent

    async def test_review_reports_an_actionable_finding_with_its_fix(self, tools):
        """Review is where the cascade reaches the agent, and it reports every time it is
        called: the budget the submit gate carried was what made a second look impossible."""
        review = tools("retrieval")[REVIEW]
        sql = "SELECT name FROM users, orders"
        out = await review.run(sql=sql)
        assert "[join]" in out and "ON" in out  # the fix, not just the complaint
        assert out == await review.run(sql=sql)

    async def test_review_shows_the_rows_an_answerless_query_returned(self, tools):
        """One measured run restructured a verified query, never re-ran it, and shipped a
        single NULL; the finding alone does not show that."""
        review = tools(question="How old is everyone?")[REVIEW]
        out = await review.run(sql="SELECT AVG(age) AS a FROM users WHERE age > 999")
        assert "only NULLs" in out and "It returned:" in out and "NULL" in out

    async def test_review_is_silent_on_a_clean_query(self, tools):
        out = await tools()[REVIEW].run(sql="SELECT name FROM users")
        assert out == "No static issues found."
