"""Tests for text2sql.schema — SchemaLoader rendering and SchemaLinker linking.

Covers schema text at each detail level, the focused schema, the linking modes,
literal grounding, and hallucinated-name filtering.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from text2sql.profiler.knowledge import KnowledgeEntry
from text2sql.profiler.stats import ColumnProfile, DatabaseProfile, StatsProfiler
from text2sql.schema.linker import SchemaLinker, VariantSpec
from text2sql.schema.loader import SchemaLoader, _stats, type_text

# ── SchemaLoader ─────────────────────────────────────────────────


class TestColumnValueStats:
    """One column line carries either the value domain or its bounds. Both draw on
    `top_k_values`: the `e.g.` examples used to come from a second list, `sample_values`,
    which duplicated it while costing every cached profile 20 strings per column."""

    @staticmethod
    def _profile(name, **kw):
        return ColumnProfile(table_name="t", column_name=name, column_type="TEXT",
                             row_count=1000, **kw)

    def test_low_cardinality_lists_values_and_high_cardinality_gives_range_examples(self):
        status = _stats(self._profile(
            "status", distinct_count=2, min_value="active", max_value="closed",
            top_k_values=[{"value": "active", "count": 9}, {"value": "closed", "count": 1}]))
        signup = _stats(self._profile(
            "signup", distinct_count=500, min_value="2020-01-01", max_value="2024-12-31",
            top_k_values=[{"value": "2021-06-01", "count": 4}, {"value": "2022-07-02", "count": 3},
                          {"value": "2023-08-03", "count": 2}, {"value": "2024-09-04", "count": 1}]))
        assert status[0] == "values: 'active', 'closed'"
        assert signup[0] == ("range: 2020-01-01-2024-12-31 "
                             "e.g. '2021-06-01', '2022-07-02', '2023-08-03'")  # capped at three

    def test_range_renders_without_examples_when_top_k_is_empty(self):
        """`profile_exact_top_k` can return nothing; the `e.g.` clause must vanish, not print bare."""
        assert _stats(self._profile("signup", distinct_count=500, min_value="2020-01-01",
                                    max_value="2024-12-31"))[0] == "range: 2020-01-01-2024-12-31"

    def test_a_padded_column_says_so_in_its_type(self):
        """A fixed-width type does not show padding and an equality join against an unpadded
        key returns nothing."""
        assert type_text(self._profile("k", min_value="A1  ", max_value="A9  ")) == "TEXT space-padded"
        assert type_text(self._profile("k", min_value="A1", max_value="A9  ")) == "TEXT"
        assert type_text(None, "INTEGER") == "INTEGER"


class TestSchemaLoader:
    @pytest.mark.parametrize("detail", ["name", "short", "long", "full"])
    def test_every_detail_level_renders_the_tables_with_keys_and_stats(
            self, schema_loader, detail):
        result = schema_loader.format_schema(detail=detail)
        assert "users" in result and "orders" in result
        assert "FK: orders.user_id -> users.id" in result
        assert "(5 rows)" in result  # the row count, whatever the description level

    def test_format_specific_tables(self, schema_loader):
        result = schema_loader.format_schema(tables=["users"])
        assert "users" in result and "orders" not in result

    def test_nonexistent_table(self, schema_loader):
        assert schema_loader.format_schema(tables=["nonexistent"]) == ""

    def test_the_table_list_carries_the_join_map(self, schema_loader):
        """Retrieval mode starts from this line, and without the edges the only way to learn
        how two tables connect is to describe them both."""
        out = schema_loader.format_table_list(["users", "orders"])
        assert out.startswith("Tables: users, orders")
        assert "orders.user_id -> users.id" in out
        # Labelled partial: 40 of the 166 golds that join use an edge the schema never declares.
        assert "Declared joins (partial" in out
        assert "\n" not in schema_loader.format_table_list(["users"])  # no edge, no line

    def test_get_table_names(self, db_conn):
        assert set(SchemaLoader(db_conn).get_table_names()) == {"users", "orders"}

    def test_a_narrowed_schema_drops_the_columns_it_excludes(self, schema_loader):
        """The one contract that makes linking on and off comparable: a column outside
        `fields` must be absent, not relegated to a names-only line. Generation cannot use
        a column it never saw, and a run that names every column reduces nothing."""
        result = schema_loader.format_schema(fields={"users": ["city"]})
        assert "users.city (TEXT" in result
        assert "values: 'New York'" in result  # categorical top-k from the sample
        assert "email" not in result and "age" not in result
        assert "Table: orders" not in result

    def test_unlinked_tables_are_named_only_for_a_model_that_can_inspect_them(
            self, schema_loader):
        """Regression: the "inspect before use" footer rendered for single-shot generation,
        which can only invent columns for the names."""
        linked = {"users": ["city"]}
        assert "Other tables" not in schema_loader.format_schema(fields=linked)
        explorable = schema_loader.format_schema(fields=linked, explorable=True)
        assert "Other tables in this database" in explorable and "orders" in explorable
        assert "Table: orders" not in explorable

    def test_json_fields_render_under_their_parent_at_every_level(
        self, db_conn, db_profile, db_summary
    ):
        """A profiled JSON path is a column the query can filter on, so it renders wherever
        its parent does - linking prompts included."""
        loader = SchemaLoader(db_conn, profile=db_profile, summary=db_summary)
        lines = loader.format_schema(fields={"orders": ["meta"]}).splitlines()
        parent = next(i for i, ln in enumerate(lines) if ln.startswith("  orders.meta "))
        fields = [ln for ln in lines[parent + 1:] if ln.lstrip().startswith("orders.meta.")]
        assert any("orders.meta.channel" in ln for ln in fields)
        assert any("orders.meta.ship.express" in ln for ln in fields)
        # Indented one level deeper than the parent column, with its own description.
        assert fields[0].startswith("    ") and "long orders.meta.channel" in fields[0]
        assert "orders.meta.channel" in loader.format_schema(detail="short")

    def test_full_detail_carries_both_summaries_deduped(self, db_conn, db_profile, db_summary):
        loader = SchemaLoader(db_conn, profile=db_profile, summary=db_summary)
        line = next(ln for ln in loader.format_schema(fields={"users": ["city"]}).splitlines()
                    if "users.city" in ln)
        assert "short users.city" in line and "long users.city" in line

    def test_value_hint_range_for_numeric(self, db_conn):
        """A wide column shows min-max rather than an unusable top-k list."""
        profile = StatsProfiler(db_conn).profile_database(selection={"users": ["age"]})
        profile.tables["users"].columns["age"].distinct_count = 90
        rendered = SchemaLoader(db_conn, profile=profile).format_schema(fields={"users": ["age"]})
        assert "range: 25-35" in rendered and "90 distinct" in rendered

    def test_profile_restricts_tables_and_columns(self, db_conn):
        """A filtered profile scopes the rendered schema to its tables/columns only."""
        # Profile only users.{id, name} — orders and users.email are out of scope.
        profile = StatsProfiler(db_conn).profile_database(selection={"users": ["id", "name"]})
        assert isinstance(profile, DatabaseProfile)
        result = SchemaLoader(db_conn, profile=profile).format_schema()
        assert "users" in result and "id" in result and "name" in result
        assert "orders" not in result  # table not in the profile
        assert "email" not in result   # column not in the profile


# ── SchemaLinker: question-value extraction ──────────────────────


class TestQuestionValues:
    def test_extracts_quoted_and_date_values(self, make_linker):
        linker = make_linker()
        assert "Acme Corp" in linker._question_literals("Orders from 'Acme Corp' in 2024?")
        vals = linker._question_literals("Between 2024-01-15 and 2024-12-31")
        assert {"2024-01-15", "2024-12-31"} <= set(vals)

    def test_capitalized_prose_is_never_probed(self, make_linker):
        """Regression: capitalized words were treated as candidate values.

        Over the 15 credit questions that heuristic produced 111 probes, matched 21
        fields and got zero of them right — every hit a false positive, which is what
        dragged value-linking precision to 0.29. Domain terms read exactly like proper
        nouns ('Net Worth', 'Financial Vulnerability Score'), so no stoplist saves it.
        """
        vals = make_linker()._question_literals(
            "Can you show the highest Net Worth? Include their IDs")
        assert vals == []

    def test_short_integers_are_counts_not_values(self, make_linker):
        """Regression: 'top 10' probed the index and matched 17 count columns at once.

        Every small integer is a stored value in some count column, so a LIMIT
        argument used to link most of the schema.
        """
        linker = make_linker()
        assert "10" not in linker._question_literals("Show the top 10 customers")
        assert "2024" in linker._question_literals("Orders in 2024")
        assert "9.99" in linker._question_literals("Orders over 9.99")


# ── SchemaLinker: SQL field extraction and merging ───────────────


class TestFieldExtraction:
    def test_extract_fields_qualified(self, make_linker):
        fields = make_linker().extract_fields("SELECT u.name, u.age FROM users u")
        assert {"name", "age"} <= set(fields["users"])

    def test_extract_fields_unqualified_single_table(self, make_linker):
        # Single-table queries reference columns without a table prefix; these must
        # still be attributed to the FROM table (previously dropped -> 0 columns).
        sql = ("SELECT region, COUNT(DISTINCT user_id) AS n FROM mir_amplitude "
               "WHERE platform = 'facebook' GROUP BY region ORDER BY n DESC LIMIT 5")
        fields = make_linker().extract_fields(sql)
        assert {"region", "user_id", "platform"}.issubset(set(fields["mir_amplitude"]))

    def test_extract_fields_excludes_cte_names(self, make_linker):
        """A CTE is not a table, whether referenced by name or through an alias.

        Regression on the aliased form: the alias was mapped only *after* the CTE was
        skipped, so `FROM ranked_quality rq` left `rq` unresolved and `rq.col` was scored
        as its own table. On LiveSQLBench that made 615 of 2423 gold entries (25%)
        unmatchable phantoms, understating every linker's recall by 13-23 points.
        """
        linker = make_linker()
        fields = linker.extract_fields("WITH t AS (SELECT id FROM users) SELECT t.id FROM t")
        assert "t" not in fields  # CTE referenced by its own name
        assert "users" in fields

        aliased = linker.extract_fields(
            "WITH ranked AS (SELECT u.id, u.city FROM users u) "
            "SELECT rq.city FROM ranked rq WHERE rq.id = 1")
        assert "rq" not in aliased and "ranked" not in aliased
        # the real column survives — it is also referenced inside the CTE body
        assert "city" in aliased["users"]

    def test_extract_fields_regex_fallback(self):
        fields = SchemaLinker._extract_fields_regex(
            "SELECT users.name, orders.amount FROM users JOIN orders")
        assert "orders" in fields and "name" in fields["users"]
        assert "FROM" not in SchemaLinker._extract_fields_regex("SELECT FROM.col")

    def test_the_fallback_reports_real_tables_only(self):
        """Regression: a truncated reply parsed to `['cr', 'bt', '0', '100', '10']` — aliases
        and numeric literals — so every consumer that filters on real tables got nothing."""
        fields = SchemaLinker._extract_fields_regex(
            "WITH base AS (SELECT cr.x FROM core_record cr JOIN banks AS b ON b.id = cr.id "
            "WHERE cr.rate > 100.0) SELECT s.x FROM base s WHERE CAST(SUM(CASE WH")
        assert set(fields) == {"core_record", "banks"}
        assert fields["core_record"] == ["id", "rate", "x"]

    def test_merge_adds_columns_and_tables(self):
        target = {"users": {"id", "name"}}
        SchemaLinker._merge(target, {"users": {"email"}, "orders": {"amount"}})
        assert target == {"users": {"id", "name", "email"}, "orders": {"amount"}}

    def test_variants_are_the_product_of_the_two_axes(self, make_linker):
        """Replaces an opaque `passes: 1-5` count into a fixed matrix: the two axes the
        count hid are now selected directly, and the passes are their product."""
        spec = VariantSpec(scopes=("full", "focused"), descriptions=("short", "long"))
        got = list(make_linker()._variants(spec, {"users": ["city"]}))
        assert [label for label, _, _ in got] == [
            "full_short", "full_long", "focused_short", "focused_long"]
        assert [(fields is None, detail) for _, fields, detail in got] == [
            (True, "short"), (True, "long"), (False, "short"), (False, "long")]

    def test_a_single_cell_is_one_pass(self, make_linker):
        spec = VariantSpec(scopes=("focused",), descriptions=("long",))
        assert len(list(make_linker()._variants(spec, {"users": ["city"]}))) == 1


# ── SchemaLinker: linking modes ──────────────────────────────────


class TestLinkModes:
    async def test_direct_mode(self, make_linker, mock_llm):
        mock_llm.chat.return_value = '[{"table": "users", "columns": ["name", "city"]}]'
        result = await make_linker(mode="direct").link("Where do users live?")
        assert "name" in result["users"]

    async def test_reversed_mode(self, make_linker, mock_llm):
        mock_llm.chat_for_sql.return_value = "SELECT u.name, u.city FROM users u"
        result = await make_linker(mode="reversed").link("Where do users live?")
        assert "users" in result

    async def test_value_mode_matches_question_values(self, make_linker, value_index):
        # Alice appears in users.name and New York in users.city.
        result = await make_linker(mode="value", value_index=value_index).link(
            "Orders from 'Alice' in 'New York'")
        assert {"name", "city"} <= set(result["users"])

    async def test_value_mode_also_matches_column_names(self, make_linker):
        """Value matching only ever finds filter columns, so names are matched too.

        Regression: credit_1 names `totassets`/`totliabs` in its question but no stored
        value resembles them, so a values-only linker scored 0.0 column precision.
        """
        result = await make_linker(mode="value").link("What is the email of each user?")
        assert "email" in result["users"]

    async def test_selected_modes_are_unioned(self, make_linker, mock_llm, value_index):
        """Replaces the old hardcoded `hybrid`: any subset of modes now composes."""
        mock_llm.chat.return_value = '[{"table": "users", "columns": ["age"]}]'
        result = await make_linker(mode=["direct", "value"], value_index=value_index).link(
            "Orders from 'Alice'")
        # `age` is unrelated to the question, so only direct can have proposed it;
        # `name` holds the literal 'Alice', so only value matching finds it.
        assert {"age", "name"} <= set(result["users"])

    @pytest.mark.parametrize("level, expected", [
        ("terms", ("Repeat Buyer", "buys twice")),
        ("full", ("Repeat Buyer", "orders >= 2")),
        ("off", ()),
    ])
    def test_the_knowledge_level_decides_what_reaches_the_link_prompt(
        self, make_linker, level, expected
    ):
        """`off` passes no entries, so the "Domain terms" heading goes with them."""
        linker = make_linker(mode="direct", knowledge=[
            KnowledgeEntry(0, "Repeat Buyer", "buys twice", "orders >= 2")])
        prompt = linker.prompt_manager.render(
            "schema_link_direct", schema="s", question="q",
            **linker._knowledge_for(level))
        assert all(e in prompt for e in expected)
        if level == "off":
            assert "Domain terms" not in prompt and "Repeat Buyer" not in prompt
        if level == "terms":
            assert "Definition:" not in prompt


class TestNameFiltering:
    """Hallucinated names are dropped, not fatal (they used to raise)."""

    def test_drops_unknown_table_and_column(self, make_linker):
        result = make_linker()._drop_unknown({
            "users": {"city", "not_a_column"},
            "not_a_table": {"whatever"},
        })
        assert result == {"users": ["city"]}

    def test_canonicalizes_case(self, make_linker):
        """A real name in the wrong case is kept and spelled the schema's way."""
        assert make_linker()._drop_unknown({"USERS": {"CITY"}}) == {"users": ["city"]}

    def test_keeps_json_field_paths(self, make_linker):
        """A dotted name is a field inside a real JSON column, not a hallucination."""
        result = make_linker()._drop_unknown({"orders": {"meta.ship.express", "nope.field"}})
        assert result == {"orders": ["meta.ship.express"]}


# ── ValueIndex: literal -> the fields that hold it ────────────────


class TestValueIndex:
    def test_finds_the_field_holding_a_literal(self, value_index):
        matches = value_index.fields_containing("Alice")
        assert ("users", "name") in {(m.table, m.column) for m in matches}

    def test_tolerates_case_and_partial_values(self, value_index):
        """Shingled LSH is an approximate match — the point of using it over equality."""
        assert value_index.fields_containing("alice")               # wrong case
        assert value_index.fields_containing("New York City")       # superstring

    def test_resemblance_below_the_threshold_is_not_a_match(self, value_index):
        """A short string's transposition falls under 0.5 trigram Jaccard, by design —
        the threshold is what keeps the index from linking arbitrary fields."""
        assert value_index.fields_containing("New Yrok") == []
        assert value_index.fields_containing("New York")

    def test_unknown_literal_matches_nothing(self, value_index):
        assert value_index.fields_containing("Ouagadougou") == []

    def test_one_match_per_field(self, value_index):
        """A field with many similar values must not crowd out the other fields."""
        keys = [(m.table, m.column) for m in value_index.fields_containing("Alice")]
        assert len(keys) == len(set(keys))

    def test_fields_for_unions_over_literals(self, value_index):
        fields = value_index.fields_for(["Alice", "New York"])
        assert {"name", "city"} <= fields["users"]

    def test_ignores_a_literal_that_matches_most_of_the_schema(self, value_index):
        """Regression: '10' from "top 10 customers" matched 17 of credit's 120 fields.

        A literal that appears everywhere describes the database, not the question, so
        matching it links most of the schema.
        """
        everywhere = next(iter(value_index.values))
        value_index.values = {f"t{i}.c": list(value_index.values[everywhere])
                              for i in range(20)}
        value_index._lsh = None
        probe = value_index.values["t0.c"][0]
        assert len(value_index.fields_containing(probe)) == 20  # matches every field
        assert value_index.fields_for([probe]) == {}            # so none are linked


# ── SchemaLinker: focused schema ─────────────────────────────────


class TestFocusedSchema:
    def test_focuses_on_question_wording_and_literals(self, make_linker, value_index):
        linker = make_linker(value_index=value_index)
        focused = linker._candidate_fields("Which city is 'Alice' in?")
        assert "city" in focused["users"] and "name" in focused["users"]

    def test_focus_is_a_strict_subset(self, make_linker, value_index):
        """A focused schema that is not smaller than the full one buys nothing."""
        linker = make_linker(value_index=value_index)
        focused = linker._candidate_fields("Which city is 'Alice' in?")
        assert sum(len(c) for c in focused.values()) < sum(
            len(t["columns"]) for t in linker.schema_loader._schema["tables"].values())

    def test_a_rare_token_outranks_one_shared_across_the_table(self, make_linker):
        """Regression: plain overlap gave every column of a table the same score.

        Each column's text carries the table name, so `credit_1` tied all 16 columns of
        `expenses_and_assets` and the top-K cut fell inside that tie — 40 columns linked.
        BM25 term weighting puts the column the question actually names first. Its siblings
        still make the candidate set, because "user" is a subword of the table name and so
        lifts the whole table; that is deliberate — measured over 237 questions it raises
        recall@50 rather than lowering it, and this stage is scored on recall.
        """
        linker = make_linker()
        ranked = linker._lexical_ranking(linker._tokens("What is the email of each user?"), "short")
        assert ranked[0] == ("users", "email")

    def test_a_question_word_matches_inside_a_column_name(self, make_linker):
        """Regression: `_tokens` cannot split `totassets`, so "assets" scored nothing.

        Cryptic unpunctuated names are most of a real schema; matching a question word as
        a subword took recall@50 from 0.768 to 0.811 over 237 LiveSQLBench questions.
        """
        fields = make_linker()._candidate_fields("Show me every user mail address")
        assert "email" in fields.get("users", set())

    def test_a_compound_identifier_reaches_the_column_it_names(self, make_linker):
        """Regression: splitting `CreatedAt` into "created"/"at" left the column it names
        scoring no better than any other column whose prose mentions "created".

        A LiveSQLBench HINT states its formula in the schema's own names, so the exact
        reference is the strongest evidence the question carries.
        """
        from text2sql.schema import lexical

        assert "modindex" in lexical.token_list("MCS = ModIndex * (1 + SSM)")
        fields = make_linker()._candidate_fields("Order each user by CreatedAt")
        assert "created_at" in fields["orders"]

    def test_a_named_column_is_admitted_past_the_budget(self, make_linker):
        """A question word that *is* a column name is the least ambiguous evidence there is
        — 60% of what it admits is gold against 15% for the ranking — so it enters whatever
        the cut made of it. Both columns here cannot fit a budget of one."""
        fields = make_linker(top_k=1)._candidate_fields("Report amount and created_at")
        assert {"amount", "created_at"} <= fields["orders"]

    def test_both_ends_of_a_foreign_key_one_hop_out_are_added(self, make_linker):
        """A gold query joins a *path*, and the bridge table is named by neither end: 21 of
        the 101 gold columns this stage missed were a bridge table's keys. Adding the two
        columns an edge is declared on — not the neighbour's other columns — took table
        recall to its ceiling over 262 LiveSQLBench questions for ~4 columns.
        """
        fields = make_linker(top_k=1)._candidate_fields("Which city?")
        assert "user_id" in fields["orders"] and "id" in fields["users"]
        assert "amount" not in fields.get("orders", set())  # the edge, not the neighbour

    def test_keys_of_a_reached_table_are_promoted(self, make_linker):
        """43% of gold columns are join keys the question never names."""
        fields = make_linker()._candidate_fields("What is the amount of each order?")
        assert "amount" in fields["orders"]
        assert {"id", "user_id"} <= fields["orders"]  # PK and FK, unnamed by the question

    def test_value_top_k_bounds_the_candidate_set(self, make_linker):
        linker = make_linker(top_k=2)
        fields = linker._candidate_fields("Which city is the user email in?")
        assert 0 < sum(len(c) for c in fields.values()) <= 2 + len(linker._index().keys)

    def test_the_budget_is_a_share_of_the_schema_over_a_floor(self, make_linker, monkeypatch):
        """A fixed cut is a benchmark artifact: 50 of 127 columns reduces by 61%, but 50 of
        2000 is 97.5% and far too tight to hold the ~7 columns a query needs."""
        from text2sql.schema import linker as linker_module

        linker = make_linker(top_k=50)
        columns = sum(len(t["columns"]) for t in linker.schema_loader._schema["tables"].values())
        assert linker._budget() == 50  # share of a narrow schema below the floor
        monkeypatch.setattr(linker_module, "_TOP_K_RATIO", 0.5)
        linker.top_k = 2
        assert linker._budget() == round(0.5 * columns) > 2

    def test_column_documents_are_built_once(self, make_linker):
        """They depend on the schema, not the question — rebuilding them per question is
        what made the old candidate set O(schema) work three times per request."""
        linker = make_linker()
        linker._candidate_fields("Which city is Alice in?")
        built = linker._documents["short"]
        linker._candidate_fields("What is the email of each user?")
        assert linker._documents["short"] is built

    async def test_the_candidate_set_is_computed_once_per_link(self, make_linker):
        """Regression: `value` mode built it, then each focused variant rebuilt it — three
        identical O(schema) passes per request, and it is the stage's most expensive step."""
        linker = make_linker(mode=["direct", "reversed", "value"],
                             direct=VariantSpec(scopes=("focused",)),
                             reversed_=VariantSpec(scopes=("focused",)))
        with patch.object(linker, "_candidate_fields", wraps=linker._candidate_fields) as spy:
            await linker.link("Which city is Alice in?")
        assert spy.call_count == 1

    async def test_no_mode_needing_candidates_skips_the_ranking(self, make_linker):
        """A full-schema direct run must not pay for a candidate set nothing reads."""
        linker = make_linker(mode="direct", direct=VariantSpec(scopes=("full",)))
        with patch.object(linker, "_candidate_fields") as spy:
            await linker.link("Which city is Alice in?")
        assert spy.call_count == 0

    async def test_key_promotion_does_not_touch_the_linked_result(self, make_linker):
        """Promotion widens the *candidate* set only; the LLM's answer is returned as given,
        so no FK expansion leaks into what generation is told to use."""
        linker = make_linker(mode="direct", direct=VariantSpec(scopes=("focused",)))
        linker.llm.chat.return_value = '[{"table": "orders", "columns": ["amount"]}]'
        assert await linker.link("What is the amount of each order?") == {"orders": ["amount"]}

    def test_ties_are_ordered_deterministically(self):
        """Ties fell back to dict order, which comes from set iteration — so a ranking, and
        with it the linked schema, changed between processes for the same question."""
        from text2sql.schema import lexical

        docs, index = lexical.documents({("t", c): "same words here" for c in "dcba"})
        assert lexical.bm25({"words"}, docs, index, subword=False) == [
            ("t", "a"), ("t", "b"), ("t", "c"), ("t", "d")]
        assert lexical.fuse([[("t", "b")], [("t", "a")]]) == [("t", "a"), ("t", "b")]

    def test_short_and_long_rankings_are_fused(self, make_linker):
        """`short` ranks names, `long` ranks vocabulary, and they miss different columns."""
        linker = make_linker()
        assert linker._fuse([[("users", "a")], [("users", "b")]]) == [
            ("users", "a"), ("users", "b")]
        assert linker._fuse([[("users", "b")], [("users", "b"), ("users", "a")]])[0] == (
            "users", "b")  # agreed-on field wins

    def test_renders_only_focused_columns(self, schema_loader):
        text = schema_loader.format_schema(fields={"users": ["city"]}, detail="short")
        assert "users.city" in text and "users.email" not in text
        assert "orders" not in text.split("FK:")[0]  # other tables excluded entirely

    def test_focused_schema_keeps_foreign_keys(self, schema_loader):
        """A focused schema still has to be joinable, so FK lines survive the filter."""
        text = schema_loader.format_schema(fields={"orders": ["amount"]}, detail="short")
        assert "FK:" in text


# ── SchemaLinker: the linked set is exactly what was linked ──────


class TestNoExpansion:
    """Each mode returns what it produced — nothing is inferred from the FK graph.

    Regression for a relational-closure pass that appended FK neighbours no linking
    method had proposed, which made the modes incomparable when benchmarked.
    """

    async def test_direct_mode_returns_only_the_llm_s_answer(self, make_linker, mock_llm):
        mock_llm.chat.return_value = '[{"table": "users", "columns": ["name"]}]'
        assert await make_linker(mode="direct").link("Q") == {"users": ["name"]}

    async def test_reversed_mode_returns_only_the_query_s_fields(self, make_linker, mock_llm):
        mock_llm.chat_for_sql.return_value = "SELECT u.name FROM users u"
        assert await make_linker(mode="reversed").link("Q") == {"users": ["name"]}

    def test_no_fk_neighbour_is_added(self, make_linker):
        """`orders` references `users`, but linking `users` alone must not pull it in."""
        assert make_linker()._drop_unknown({"users": {"name"}}) == {"users": ["name"]}
