"""Tests for text2sql.profiler — StatsProfiler, ProfileCache, and dataclass roundtrips."""

from __future__ import annotations

import json
import re

import pytest

from text2sql.profiler.cache import ProfileCache
from text2sql.profiler.stats import (
    ColumnProfile,
    DatabaseProfile,
    StatsProfiler,
    TableProfile,
    dotted,
    entries,
    flat,
    group,
)


def save_profile(cache: ProfileCache, key: str, profile: DatabaseProfile) -> None:
    """Cache a profile the way core.py does."""
    cache.save(key, "profile", profile.to_flat(key), profile.table_meta())


def load_profile(cache: ProfileCache, key: str) -> DatabaseProfile:
    return DatabaseProfile.from_flat(cache.load(key, "profile"), "sqlite")


class TestStatsProfiler:
    def test_profile_all_tables(self, db_conn):
        profiler = StatsProfiler(db_conn, top_k=5, sample_size=100)
        profile = profiler.profile_database()
        assert "users" in profile.tables
        assert "orders" in profile.tables
        assert profile.dialect == "sqlite"

    def test_row_counts(self, db_profile):
        assert db_profile.tables["users"].row_count == 5
        assert db_profile.tables["orders"].row_count == 5

    def test_null_and_distinct_counts(self, db_profile):
        email = db_profile.tables["users"].columns["email"]
        name = db_profile.tables["users"].columns["name"]
        assert (email.null_count, email.non_null_count) == (1, 4)
        assert (name.null_count, name.distinct_count) == (0, 5)

    def test_value_statistics(self, db_profile):
        users = db_profile.tables["users"].columns
        assert "New York" in [v["value"] for v in users["city"].top_k_values]
        assert users["age"].min_value is not None and users["age"].max_value is not None
        assert users["name"].top_k_values

    def test_profile_specific_tables(self, db_conn):
        profiler = StatsProfiler(db_conn)
        profile = profiler.profile_database(selection={"users": ["id", "name"]})
        assert "users" in profile.tables
        assert "orders" not in profile.tables
        assert set(profile.tables["users"].columns) == {"id", "name"}

    def test_empty_column_list_means_whole_table(self, db_conn):
        """``{"users": []}`` selects the table, matching Text2SQL._filter's semantics."""
        profile = StatsProfiler(db_conn).profile_database(selection={"users": []})
        assert set(profile.tables["users"].columns) == {"id", "name", "email", "age", "city"}

    def test_iter_profile_yields_per_table(self, db_conn):
        profiler = StatsProfiler(db_conn, top_k=5, sample_size=100)
        steps = list(profiler.iter_profile())
        tables = [step[1] for step in steps]
        assert set(tables) == {"users", "orders"}
        # Each step reports (profile, table, done, total) with a stable total.
        assert all(step[3] == len(tables) for step in steps)
        assert [step[2] for step in steps] == list(range(1, len(tables) + 1))


class TestStatsProfilerQueries:
    """The single-pass design: ~2 queries per table, dialect-portable, with fallbacks."""

    def _spy(self, db):
        from unittest.mock import patch
        calls: list[str] = []
        orig = db.execute_safe

        def wrapper(sql, params=None):
            calls.append(sql)
            return orig(sql, params)

        return patch.object(db, "execute_safe", side_effect=wrapper), calls

    def test_two_queries_per_table_by_default(self, db_conn):
        patcher, calls = self._spy(db_conn)
        with patcher:
            StatsProfiler(db_conn, top_k=5, sample_size=100)._profile_table("users")
        # One aggregate scan + one shared sample — independent of column count.
        assert len(calls) == 2

    def test_exact_top_k_adds_per_column_queries(self, db_conn):
        patcher, calls = self._spy(db_conn)
        with patcher:
            StatsProfiler(db_conn, top_k=5, sample_size=100, exact_top_k=True)._profile_table("users")
        # aggregate + sample + one GROUP BY per column (users has 5 columns).
        assert len(calls) == 2 + 5

    def test_sample_aggregates_uses_limit_subquery(self, db_conn):
        patcher, calls = self._spy(db_conn)
        with patcher:
            StatsProfiler(db_conn, sample_size=50, sample_aggregates=True)._profile_table("users")
        agg_sql = calls[0]
        assert "__s" in agg_sql and "LIMIT 50" in agg_sql

    def test_sample_based_top_k_matches_full_table_when_sample_covers_it(self, db_conn):
        # sample_size (100) > row count (5) → sample is the whole table → exact top-k.
        prof = StatsProfiler(db_conn, top_k=5, sample_size=100)._profile_table("users")
        cities = [v["value"] for v in prof.columns["city"].top_k_values]
        assert "New York" in cities

    def test_aggregate_falls_back_to_per_column(self, db_conn):
        """A failing multi-column aggregate degrades to one query per column, not a crash."""
        profiler = StatsProfiler(db_conn, top_k=5, sample_size=100)
        real = profiler._run_aggregate

        def flaky(table, cols):
            if len(cols) > 1:
                return None, None  # simulate the batch query failing
            return real(table, cols)

        from unittest.mock import patch
        with patch.object(profiler, "_run_aggregate", side_effect=flaky):
            prof = profiler._profile_table("users")
        # Per-column fallback still populates exact stats.
        assert prof.columns["email"].null_count == 1
        assert prof.columns["name"].distinct_count == 5

    def test_row_get_is_case_insensitive(self):
        # Snowflake upper-cases unquoted aliases; lookups must still resolve.
        assert StatsProfiler._row_get({"C0_NN": 4}, "c0_nn") == 4
        assert StatsProfiler._row_get({"n_rows": 5}, "n_rows") == 5
        assert StatsProfiler._row_get({}, "missing") is None


class TestApproxCountDistinct:
    """db.approx_count_distinct emits portable SQL per dialect, exact fallback elsewhere."""

    def test_sqlite_exact(self, db_conn):
        assert db_conn.approx_count_distinct('"c"') == 'COUNT(DISTINCT "c")'

    @pytest.mark.parametrize("dialect, expected", [
        ("awsathena", "approx_distinct(\"c\")"),
        ("snowflake", "APPROX_COUNT_DISTINCT(\"c\")"),
        ("redshift", "APPROXIMATE COUNT(DISTINCT \"c\")"),
        ("postgresql", 'COUNT(DISTINCT "c")'),
    ], ids=["athena", "snowflake", "redshift", "postgres"])
    def test_dialect_fragments(self, db_conn, dialect, expected):
        from unittest.mock import PropertyMock, patch
        with patch.object(type(db_conn), "dialect_name", new_callable=PropertyMock, return_value=dialect):
            assert db_conn.approx_count_distinct('"c"') == expected


class TestColumnProfileEnglish:
    def test_basic_output(self, db_profile):
        english = db_profile.tables["users"].columns["city"].to_english()
        assert "city" in english.lower()

    def test_nullable_column(self):
        col = ColumnProfile(table_name="t", column_name="email", column_type="TEXT",
                            row_count=10, null_count=3, distinct_count=7)
        assert "3 NULL" in col.to_english()

    def test_constant_length(self):
        col = ColumnProfile(table_name="t", column_name="code", column_type="TEXT",
                            row_count=5, null_count=0, distinct_count=5,
                            min_length=3, max_length=3, is_constant_length=True)
        assert "3 characters" in col.to_english()

    def test_numeric_column(self):
        col = ColumnProfile(table_name="t", column_name="id", column_type="INT",
                            row_count=100, null_count=0, distinct_count=100,
                            is_always_numeric=True)
        assert "number" in col.to_english().lower()

    def test_patterns(self):
        col = ColumnProfile(table_name="t", column_name="date", column_type="TEXT",
                            row_count=5, null_count=0, distinct_count=5,
                            common_patterns=["DDDD-DD-DD"])
        assert "DDDD-DD-DD" in col.to_english()


class TestColumnProfileRoundtrip:
    def test_roundtrip(self):
        original = ColumnProfile(
            table_name="users", column_name="email", column_type="TEXT",
            row_count=100, null_count=5, non_null_count=95, distinct_count=90,
            min_value="a@b.com", max_value="z@y.com",
            top_k_values=[{"value": "test@test.com", "count": 3}],
            min_length=5, max_length=50,
            is_always_numeric=False, is_constant_length=False,
            common_patterns=["A@A.AAA"],
        )
        restored = ColumnProfile.from_dict(original.to_dict())
        assert restored.table_name == original.table_name
        assert restored.null_count == original.null_count
        assert restored.top_k_values == original.top_k_values
        assert restored.common_patterns == original.common_patterns


class TestFlatFormat:
    """The shared ``db|table|column`` key format."""

    def test_flat_roundtrip(self):
        dp = DatabaseProfile(dialect="sqlite")
        dp.tables["t"] = TableProfile("t", 42, {
            "c1": ColumnProfile("t", "c1", "INT", row_count=42, distinct_count=42)})
        doc = {"columns": dp.to_flat("db"), "meta": dp.table_meta()}
        assert "db|t|c1" in doc["columns"]
        restored = DatabaseProfile.from_flat(doc, "sqlite")
        assert restored.tables["t"].row_count == 42
        assert restored.tables["t"].columns["c1"].distinct_count == 42

    def test_table_without_columns_survives(self):
        dp = DatabaseProfile(dialect="sqlite")
        dp.tables["t"] = TableProfile("t", 10)
        restored = DatabaseProfile.from_flat(
            {"columns": dp.to_flat("db"), "meta": dp.table_meta()})
        assert restored.tables["t"].row_count == 10

    def test_group_ignores_db_prefix(self):
        """The shipped files use the domain name, the cache uses the file stem."""
        assert group({"credit|t|c": "x"}) == group({"credit_template|t|c": "x"})

    def test_group_accepts_bare_map_or_envelope(self):
        bare = {"db|t|c": "x"}
        assert group(bare) == group({"version": 3, "columns": bare})
        assert entries(bare) == entries({"columns": bare}) == bare

    def test_group_expands_nested_field_meanings(self):
        """The shipped JSONB form becomes dotted sibling columns."""
        cols = group({"credit|t|blob": {
            "column_meaning": "parent",
            "fields_meaning": {"a": "A", "nest": {"b": "B"}},
        }})["t"]
        assert cols == {"blob": "parent", "blob.a": "A", "blob.nest.b": "B"}

    def test_group_skips_malformed_keys(self):
        assert group({"nokey": "x"}) == {}

    def test_flat_and_dotted(self):
        assert flat("db", "t", "c") == "db|t|c"
        assert dotted({"a": {"b": 1}, "c": 2}) == {"a.b": 1, "c": 2}


class TestJsonFieldProfiling:
    """JSON columns are harvested into ordinary dotted columns."""

    @staticmethod
    def _profiler(db_conn):
        return StatsProfiler(db_conn, top_k=5, sample_size=100)

    def test_detects_nested_paths(self, db_conn):
        fields = self._profiler(db_conn)._json_fields([
            '{"own": "Rent", "mort": {"bal": 100}}',
            '{"own": "Own", "mort": {"bal": 200}}',
        ])
        assert set(fields) == {"own", "mort.bal"}
        assert fields["mort.bal"] == ["100", "200"]

    def test_numeric_paths_get_numeric_bounds(self, db_conn):
        """A JSON value is extracted as text, so plain min/max compare lexicographically and
        name neither end: credit's `propvalue` profiled as 1000162-997368 for a true
        389-1999880, and that fabricated range reached the generation prompt."""
        cp = self._profiler(db_conn)._field_profile(
            "t", "meta.weight", 5, ["9.5", "1200.0", "340.0", "87.25", "15.75"])
        assert (cp.min_value, cp.max_value) == ("9.5", "1200.0")

    def test_text_paths_keep_lexicographic_bounds(self, db_conn):
        """Only numbers mis-sort as text; a code or an ISO date is ordered by its spelling."""
        cp = self._profiler(db_conn)._field_profile(
            "t", "meta.day", 3, ["2025-03-01", "2025-01-10", "2025-02-20"])
        assert (cp.min_value, cp.max_value) == ("2025-01-10", "2025-03-01")

    def test_ignores_non_json(self, db_conn):
        assert self._profiler(db_conn)._json_fields(["a", "b"]) == {}
        assert self._profiler(db_conn)._json_fields(["[1, 2]"]) == {}
        assert self._profiler(db_conn)._json_fields([]) == {}

    def test_caps_field_count(self, db_conn, monkeypatch):
        import json as _json

        import text2sql.profiler.stats as stats_mod
        monkeypatch.setattr(stats_mod, "_MAX_JSON_FIELDS", 3)
        blob = _json.dumps({f"k{i}": i for i in range(10)})
        assert len(self._profiler(db_conn)._json_fields([blob])) == 3

    def test_field_profile_stats(self, db_conn):
        cp = self._profiler(db_conn)._field_profile("t", "blob.x", 10, ["a", "a", "b"])
        assert (cp.distinct_count, cp.non_null_count, cp.null_count) == (2, 3, 7)
        assert cp.top_k_values[0] == {"value": "a", "count": 2}

    def test_end_to_end_dotted_column(self, db_conn):
        """A JSON column in the shared fixture DB profiles into dotted columns."""
        profile = self._profiler(db_conn).profile_database()
        cols = profile.tables["orders"].columns
        assert "meta.channel" in cols
        assert "meta.ship.express" in cols
        assert cols["meta.channel"].column_type == "JSON field"


class TestProfileCache:
    @staticmethod
    def _col(table, name, **kw):
        return ColumnProfile(table_name=table, column_name=name, column_type="TEXT", **kw)

    def _profile(self, **tables):
        """``_profile(users=["id", "email"])`` → a profile with those columns."""
        dp = DatabaseProfile(dialect="sqlite")
        for table, cols in tables.items():
            dp.tables[table] = TableProfile(table, len(cols), {
                c: self._col(table, c, row_count=5, distinct_count=5) for c in cols})
        return dp

    def test_cache_key_uses_filename_stem(self):
        assert ProfileCache.cache_key("sqlite:///test.db") == "test"

    def test_cache_key_ignores_path(self):
        assert ProfileCache.cache_key("sqlite:///data/credit/credit_template.sqlite") == \
            ProfileCache.cache_key(
                "sqlite:////app/data/credit/credit_template.sqlite")

    def test_cache_key_differs(self):
        assert ProfileCache.cache_key(
            "sqlite:///a.db") != ProfileCache.cache_key("sqlite:///b.db")

    def test_local_save_load_profile(self, tmp_path, db_profile):
        cache = ProfileCache(str(tmp_path / "cache"))
        save_profile(cache, "k", db_profile)
        assert set(load_profile(cache, "k").tables) == set(db_profile.tables)

    def test_writes_one_file_per_kind(self, tmp_path, db_profile):
        cache = ProfileCache(str(tmp_path / "cache"))
        save_profile(cache, "k", db_profile)
        cache.save("k", "meaning_base_long", {"k|users|email": "the address"})
        assert (tmp_path / "cache" / "k_profile.json").exists()
        assert (tmp_path / "cache" / "k_meaning_base_long.json").exists()
        assert group(cache.load("k", "meaning_base_long"))["users"]["email"] == "the address"

    def test_replace_rewrites_the_whole_document(self, tmp_path):
        """The KB re-derives names each run, so entries must replace rather than accumulate."""
        cache = ProfileCache(str(tmp_path / "cache"))
        cache.save("k", "kb", {"k|users|Old Rule": {"id": 0, "knowledge": "Old Rule"}})
        cache.save("k", "kb", {"k|users|New Rule": {"id": 0, "knowledge": "New Rule"}}, replace=True)
        assert list(group(cache.load("k", "kb"))["users"]) == ["New Rule"]

    def test_load_missing_returns_empty(self, tmp_path):
        cache = ProfileCache(str(tmp_path / "cache"))
        assert cache.load("x", "profile") == {}
        assert cache.cached_tables("x") == {}

    def test_save_upserts_tables(self, tmp_path):
        cache = ProfileCache(str(tmp_path / "cache"))
        save_profile(cache, "k", self._profile(a=["c1"]))
        save_profile(cache, "k", self._profile(b=["c2"]))
        # Partial runs accumulate into one file rather than overwriting.
        assert set(load_profile(cache, "k").tables) == {"a", "b"}
        assert set(cache.cached_tables("k")) == {"a", "b"}

    def test_cached_columns(self, tmp_path, db_profile):
        cache = ProfileCache(str(tmp_path / "cache"))
        save_profile(cache, "k", db_profile)
        cols = cache.cached_columns("k")
        assert set(cols) == set(db_profile.tables)
        assert "email" in cols["users"]

    def test_delete_columns_keeps_table(self, tmp_path, db_profile):
        cache = ProfileCache(str(tmp_path / "cache"))
        save_profile(cache, "k", db_profile)
        cache.delete("k", "users", ["email"])
        loaded = load_profile(cache, "k")
        assert "users" in loaded.tables  # table survives — still has columns
        assert "email" not in loaded.tables["users"].columns

    def test_delete_column_takes_its_json_fields(self, tmp_path):
        cache = ProfileCache(str(tmp_path / "cache"))
        save_profile(cache, "k", self._profile(a=["blob", "blob.x", "other"]))
        cache.delete("k", "a", ["blob"])
        assert set(load_profile(cache, "k").tables["a"].columns) == {"other"}

    def test_delete_last_column_drops_table(self, tmp_path):
        cache = ProfileCache(str(tmp_path / "cache"))
        save_profile(cache, "k", self._profile(a=["c1"]))
        cache.delete("k", "a", ["c1"])
        assert "a" not in load_profile(cache, "k").tables
        assert cache.cached_tables("k") == {}

    def test_delete_whole_table(self, tmp_path):
        cache = ProfileCache(str(tmp_path / "cache"))
        save_profile(cache, "k", self._profile(a=["c1"], b=["c2"]))
        cache.delete("k", "a")
        assert set(load_profile(cache, "k").tables) == {"b"}
        assert set(cache.cached_tables("k")) == {"b"}

    def test_delete_spans_every_kind(self, tmp_path):
        cache = ProfileCache(str(tmp_path / "cache"))
        save_profile(cache, "k", self._profile(a=["c1"]))
        cache.save("k", "meaning_base_long", {"k|a|c1": "desc"})
        cache.save("k", "kb", {"k|a|Some Rule": {"id": 0, "knowledge": "Some Rule"}})
        cache.delete("k", "a")
        assert all(cache.load("k", kind).get("columns") == {}
                   for kind in ("profile", "meaning_base_long", "kb"))

    def test_save_merges_columns_within_table(self, tmp_path):
        """Re-profiling a column subset of a cached table preserves its other columns."""
        cache = ProfileCache(str(tmp_path / "cache"))
        save_profile(cache, "k", self._profile(users=["id", "email", "city"]))

        # A single-column refresh sends a profile with only that column.
        refreshed = DatabaseProfile(dialect="sqlite")
        refreshed.tables["users"] = TableProfile("users", 5, {
            "email": self._col("users", "email", row_count=5, null_count=2)})
        save_profile(cache, "k", refreshed)

        loaded = load_profile(cache, "k")
        # Siblings survive the partial re-profile; the refreshed column is updated.
        assert set(loaded.tables["users"].columns) == {"id", "email", "city"}
        assert loaded.tables["users"].columns["email"].null_count == 2

    def test_load_rejects_old_version(self, tmp_path):
        import json
        cache = ProfileCache(str(tmp_path / "cache"))
        (tmp_path / "cache" / "k_profile.json").write_text(
            json.dumps({"version": 0, "columns": {"k|a|c": {}}}))
        # A stale cache-format version is invalidated (treated as a miss).
        assert cache.load("k", "profile") == {}
        assert cache.cached_tables("k") == {}


class TestProfilerHelpers:
    @pytest.mark.parametrize("value, expected", [
        ("123", True), ("45.67", True), ("-89", True), ("1,234", True),
        ("abc", False), ("12a", False), ("", False),
    ], ids=["int", "float", "neg", "comma", "alpha", "mixed", "empty"])
    def test_is_numeric(self, value, expected):
        assert StatsProfiler._is_numeric(value) is expected

    def test_detect_patterns_dates(self):
        patterns = StatsProfiler._detect_patterns(["2024-01-15", "2024-02-20"])
        assert "DDDD-DD-DD" in patterns

    def test_quote_identifier_sqlite(self, db_conn):
        assert db_conn.quote_identifier("name") == '"name"'


# ── S3 cache backend (mocked boto3) ─────────────────────────────


class TestProfileCacheS3:
    """Test S3 cache backend with mocked boto3."""

    def test_s3_init(self):
        from text2sql.profiler.cache import ProfileCache
        cache = ProfileCache("s3://bucket/prefix")
        assert cache._is_s3 is True
        assert cache._s3_bucket == "bucket"
        assert cache._s3_prefix == "prefix"
        assert cache._local_dir is None

    def test_write_s3(self):
        from unittest.mock import MagicMock, patch

        from text2sql.profiler.cache import ProfileCache
        cache = ProfileCache.__new__(ProfileCache)
        cache._is_s3 = True
        cache._s3_bucket = "bucket"
        cache._s3_prefix = "pfx"
        cache._local_dir = None

        mock_s3 = MagicMock()
        with patch.object(cache, "_s3", return_value=mock_s3):
            cache._write("data.json", {"key": "value"})
            mock_s3.put_object.assert_called_once()
            call_kwargs = mock_s3.put_object.call_args[1]
            assert call_kwargs["Bucket"] == "bucket"
            assert call_kwargs["Key"] == "pfx/data.json"

    def test_read_s3_success(self):
        import json
        from unittest.mock import MagicMock, patch

        from text2sql.profiler.cache import ProfileCache
        cache = ProfileCache.__new__(ProfileCache)
        cache._is_s3 = True
        cache._s3_bucket = "bucket"
        cache._s3_prefix = ""
        cache._local_dir = None

        mock_s3 = MagicMock()
        body_data = json.dumps({"tables": {}}).encode()
        mock_s3.get_object.return_value = {
            "Body": MagicMock(read=MagicMock(return_value=body_data)),
        }
        with patch.object(cache, "_s3", return_value=mock_s3):
            result = cache._read("data.json")
            assert result == {"tables": {}}

    def test_read_s3_not_found(self):
        from unittest.mock import MagicMock, patch

        from text2sql.profiler.cache import ProfileCache
        cache = ProfileCache.__new__(ProfileCache)
        cache._is_s3 = True
        cache._s3_bucket = "bucket"
        cache._s3_prefix = ""
        cache._local_dir = None

        mock_s3 = MagicMock()

        class NoSuchKeyError(Exception):
            pass

        mock_s3.exceptions.NoSuchKey = NoSuchKeyError
        mock_s3.get_object.side_effect = NoSuchKeyError("NoSuchKey")

        with patch.object(cache, "_s3", return_value=mock_s3):
            result = cache._read("missing.json")
            assert result is None


class TestKnowledgeGenerator:
    """Ids and child links are assigned deterministically after the LLM returns."""

    @staticmethod
    def _generator(rows_by_table, prompt_manager):
        from unittest.mock import AsyncMock, MagicMock

        from text2sql.profiler.knowledge import KnowledgeGenerator

        llm = MagicMock()
        llm.chat = AsyncMock(side_effect=lambda p, **kw: json.dumps(
            rows_by_table.get("_links", []) if "Terms already defined" in p
            else rows_by_table.get(m.group(1), []) if (m := re.search(r"^Table: (.+)$", p, re.M))
            else []))
        return KnowledgeGenerator(llm, prompt_manager)

    @staticmethod
    def _profile():
        dp = DatabaseProfile(dialect="sqlite")
        for t in ("a", "b"):
            dp.tables[t] = TableProfile(t, 1, {
                "c": ColumnProfile(t, "c", "INT", top_k_values=[{"value": "1", "count": 1}])})
        return dp

    async def test_prompt_tabulates_real_top_values(self, prompt_manager):
        """A key named "values" would resolve to ``dict.values`` in Jinja, not the samples."""
        gen = self._generator({"a": [], "b": []}, prompt_manager)
        await gen.generate(self._profile(), lambda t, c: "a | description\nsplit over lines")
        prompt = next(c.args[0] for c in gen.llm.chat.await_args_list if "| Description |" in c.args[0])
        assert "built-in method" not in prompt
        assert "| c | INT | a / description split over lines | 1 |" in prompt
        # The rules are the cached system block, so they are not re-sent in the user message.
        assert '"knowledge"' in gen.llm.chat.await_args.kwargs["system"]

    async def test_a_padded_column_is_marked_so_its_literals_are_trimmed(self, prompt_manager):
        """The table cell strips the stored padding, so `envhaz = 'High'` matched nothing on
        every padded column and the cohort selected no rows."""
        profile = self._profile()
        profile.tables["a"].columns["c"].min_value = "High     "
        profile.tables["a"].columns["c"].max_value = "Low      "
        gen = self._generator({"a": [], "b": []}, prompt_manager)
        await gen.generate(profile, lambda t, c: "")
        prompt = next(c.args[0] for c in gen.llm.chat.await_args_list if "Table: a" in c.args[0])
        assert "| c | INT space-padded |" in prompt
        assert "TRIM(col)" in gen.llm.chat.await_args.kwargs["system"]

    async def test_ids_and_children_resolved(self, prompt_manager):
        """Ids follow table order; a child reference resolves by exact name or a unique partial."""
        rows = {
            "a": [{"knowledge": "Annual Income", "description": "d", "definition": "c * 12"},
                  {"knowledge": "Household Income", "definition": "c + c"},
                  {"knowledge": "Debt Burden", "definition": "c / income",
                   "children": ["annual income", "Missing"]}],
            "b": [{"knowledge": "Stress Flag", "definition": "c > 0",
                   "children": ["household", "income"]}],
        }
        dk = await self._generator(rows, prompt_manager).generate(self._profile(), lambda t, c: "")
        assert [e.knowledge for e in dk.entries.values()] == [
            "Annual Income", "Household Income", "Debt Burden", "Stress Flag"]
        assert dk.entries[0].children_knowledge == -1  # none declared
        assert dk.entries[2].children_knowledge == [0]  # exact (case-insensitive); unknown dropped
        assert dk.entries[3].children_knowledge == [1]  # unique partial; "income" matches two → dropped
        assert dk.entries[3].tables == ("b",)

    async def test_a_second_pass_relates_two_tables_and_builds_on_the_terms(self, prompt_manager):
        """One call per table can only ever produce single-table entries: 12% of the corpus KB
        spans tables and 34% defines a term over other terms, and a per-table call reached
        0.3% and 0.8% of those."""
        rows = {
            "a": [{"knowledge": "Local Rate", "definition": "c / 2"}],
            "b": [{"knowledge": "Other Rate", "definition": "c * 3"}],
            "_links": [
                {"knowledge": "Joint Load", "definition": "a.c / b.c",
                 "children": ["Local Rate"]},
                {"knowledge": "Healthy", "definition": "Local Rate > Other Rate",
                 "children": ["Local Rate", "Other Rate"]},
            ],
        }
        gen = self._generator(rows, prompt_manager)
        dk = await gen.generate(self._profile(), lambda t, c: "", joins="a.c -> b.c")
        by_name = {e.knowledge: e for e in dk.entries.values()}
        assert by_name["Joint Load"].tables == ("a", "b")   # owned by both, so either refreshes it
        assert by_name["Healthy"].tables == ()             # names only terms, so no table owns it
        assert by_name["Local Rate"].tables == ("a",)
        # A term built only on grounded terms survives grounding, and keeps its children.
        assert by_name["Healthy"].children
        links = next(c.args[0] for c in gen.llm.chat.await_args_list if "Terms already" in c.args[0])
        assert "a.c -> b.c" in links and "Local Rate" in links

    async def test_a_single_table_database_skips_the_second_pass(self, prompt_manager):
        gen = self._generator({"a": [{"knowledge": "K", "definition": "c > 0"}]}, prompt_manager)
        dp = DatabaseProfile(dialect="sqlite")
        dp.tables["a"] = TableProfile("a", 1, {"c": ColumnProfile("a", "c", "INT")})
        await gen.generate(dp, lambda t, c: "")
        assert gen.llm.chat.await_count == 1

    async def test_re_profiling_one_table_still_regenerates_the_database_wide_entries(
            self, prompt_manager):
        """Cross-table and term-built entries belong to the database, not to one table, so a
        partial re-profile must regenerate them rather than drop them unreplaced."""
        rows = {"a": [{"knowledge": "Local", "definition": "c > 0"}],
                "_links": [{"knowledge": "Joint", "definition": "a.c / b.c > 2"}]}
        gen = self._generator(rows, prompt_manager)
        dk = await gen.generate(self._profile(), lambda t, c: "", only=["a"])
        assert {e.knowledge for e in dk.entries.values()} == {"Local", "Joint"}

    async def test_only_restricts_tables(self, prompt_manager):
        rows = {"a": [{"knowledge": "K", "definition": "c > 0"}],
                "b": [{"knowledge": "Other", "definition": "c < 0"}]}
        dk = await self._generator(rows, prompt_manager).generate(
            self._profile(), lambda t, c: "", only=["a"])
        assert [e.knowledge for e in dk.entries.values()] == ["K"]

    async def test_unparseable_response_is_skipped(self, prompt_manager):
        from unittest.mock import AsyncMock, MagicMock

        from text2sql.profiler.knowledge import KnowledgeGenerator

        llm = MagicMock()
        llm.chat = AsyncMock(return_value="not json at all")
        dk = await KnowledgeGenerator(llm, prompt_manager).generate(self._profile(), lambda t, c: "")
        assert dk.entries == {}

    def test_roundtrip_through_cache(self, tmp_path, prompt_manager):
        from text2sql.profiler.knowledge import DatabaseKnowledge, KnowledgeEntry
        dk = DatabaseKnowledge({0: KnowledgeEntry(0, "K", "d", "def", "domain_knowledge", [1],
                                                  tables=("a", "b"))})
        cache = ProfileCache(str(tmp_path / "cache"))
        cache.save("k", "kb", dk.to_flat("k"))
        restored = DatabaseKnowledge.from_flat(cache.load("k", "kb"))
        assert restored.entries[0].definition == "def"
        assert restored.entries[0].tables == ("a", "b")

    def test_from_jsonl_matches_shipped_format(self):
        from text2sql.profiler.knowledge import DatabaseKnowledge
        dk = DatabaseKnowledge.from_jsonl(
            '{"id": 0, "knowledge": "DTI", "description": "d", "definition": "= debincratio",'
            ' "type": "calculation_knowledge", "children_knowledge": -1}\n\n')
        assert dk.entries[0].knowledge == "DTI"
        assert dk.entries[0].children == []


class TestKnowledgePlainText:
    """The shipped KB writes its formulas in LaTeX; a prompt needs them as SQL."""

    @pytest.mark.parametrize("raw, expected", [
        # the whole point: a definition the model can read as a SQL expression
        (r"DTI = \frac{Total Debt}{Monthly Income}", "DTI = (Total Debt) / (Monthly Income)"),
        (r"$\text{CLSF} > 2.0$", "CLSF > 2.0"),
        (r"a \times b \cdot c", "a * b * c"),
        (r"\left(1 + \frac{x}{100}\right)", "(1 + (x) / (100))"),
        (r"TAF = \frac{tierstep}{\sqrt{membdays}}", "TAF = (tierstep) / (SQRT(membdays))"),
        (r"Years\_Since\_Install \geq 2", "Years_Since_Install >= 2"),
        (r"1{,}000{,}000 \times 10^{-2}", "1,000,000 * POWER(10, -2)"),
        # bars: a wrapped column is an absolute value, a bare relation is a row count
        (r"SNQI = \text{Snr} - |\text{NoiseFloorDbm}|", "SNQI = Snr - ABS(NoiseFloorDbm)"),
        (r"|x \in \{High, Medium\}|", "COUNT(x in (High, Medium))"),
        # operators written as Unicode, which carries no backslash to detect
        ("ECAC = SSF × (1 + TIE × 0.5) ÷ 2 ≥ 1", "ECAC = SSF * (1 + TIE * 0.5) / 2 >= 1"),
        # `\text` written with one backslash: JSON parses it as a TAB, losing the command head
        ("LRS = \text{LHI} \\times 2", "LRS = LHI * 2"),
        # prose with no maths is returned untouched, so generated KBs are unaffected
        ("Total assets minus liabilities.", "Total assets minus liabilities."),
    ])
    def test_latex_maths_is_rewritten_as_sql(self, raw, expected):
        from text2sql.profiler.knowledge import plain_text
        assert plain_text(raw) == expected

    def test_bars_count_a_relation_and_measure_a_column(self):
        """Reading a bare identifier as a relation turned `|macdtrail|` into COUNT() inside a
        formula gold computes with ABS, and lost the instance."""
        from text2sql.profiler.knowledge import plain_text

        assert plain_text(r"S = |macdtrail| + \frac{\sum score_i}{|assessments|}",
                          {"assessments"}) \
            == "S = ABS(macdtrail) + (SUM score_i) / (COUNT(assessments))"
        # A name the formula sums over is counted even where no table carries it.
        assert plain_text(r"\frac{\sum_{i \in sensitivities} w_i}{|sensitivities|}") \
            == "(SUM_i in sensitivities w_i) / (COUNT(sensitivities))"

    @pytest.mark.parametrize("rowbreak", ["\\\\", "\\"])  # the shipped files use both
    def test_cases_block_becomes_a_case_expression(self, rowbreak):
        r"""`3 & \text{if S = 'High'}` rows flatten to noise unless rewritten as CASE arms."""
        from text2sql.profiler.knowledge import plain_text
        raw = (r"DSI = \text{VolGB} \times \begin{cases} 3 & \text{if S = 'High'} " + rowbreak
               + r" 1 & \text{otherwise} \end{cases}")
        assert plain_text(raw) == "DSI = VolGB * CASE WHEN S = 'High' THEN 3 ELSE 1 END"

    def test_rewriting_leaves_no_markup_and_is_idempotent(self):
        r"""One pass must reach a fixed point; a residual `\frac` or brace reaches the model."""
        from text2sql.profiler.knowledge import plain_text
        once = plain_text(r"CPI = \begin{cases} 100 - PS, & \text{if at risk} \\ "
                          r"50 - \frac{TS}{10}, & \text{otherwise} \end{cases}")
        assert not {"\\", "{", "}", "$"} & set(once)
        assert plain_text(once) == once

    def test_entries_are_flattened_on_load(self):
        """`from_jsonl` is the only door the dataset KB comes through."""
        from text2sql.profiler.knowledge import DatabaseKnowledge
        dk = DatabaseKnowledge.from_jsonl(
            r'{"id": 0, "knowledge": "DTI", "description": "$\\text{ratio}$",'
            r' "definition": "= \\frac{a}{b}"}')
        assert dk.entries[0].description == "ratio"
        assert dk.entries[0].definition == "= (a) / (b)"


class TestKnowledgeContradiction:
    """A definition that equates a formula with a stored column is a claim about the data."""

    @staticmethod
    def _mismatches(disagree):
        return lambda table, predicate: 1 if disagree else 0

    def test_a_refuted_equality_is_rewritten_not_annotated(self):
        """`credit`'s KB asserts `totassets - totliabs = networth`, false in all 1000 rows.
        The false claim has to *go*: leaving it beside a correction states both at once."""
        from text2sql.profiler.knowledge import verified_definition

        definition = ("Net Worth = Total Assets - Total Liabilities = "
                      "totassets - totliabs = networth")
        tables = {"ea": {"totassets", "totliabs", "networth"}}
        assert verified_definition(definition, tables, self._mismatches(True)) == (
            "Net Worth = Total Assets - Total Liabilities = totassets - totliabs != networth")
        assert verified_definition(definition, tables, self._mismatches(False)) == definition

    def test_only_sql_expressions_over_one_table_are_compared(self):
        """`Total Assets` is prose, `tierstep <` is a fragment that passed the word test and
        produced `ABS((statustag) - (tierstep <))` — 37% of the queries this once issued."""
        from text2sql.profiler.knowledge import verified_definition

        for definition, tables in [
            ("FSI = 0.3 * (1 - debincratio)", {"t": {"fsi"}}),      # no column-only side pair
            # `<=` contains `=`, so splitting yields the fragment `tierstep <`
            ("statustag = tierstep <= 5", {"t": {"statustag", "tierstep"}}),
            ("a = b", {"t": {"a"}}),                                # b is not in this table
        ]:
            assert verified_definition(definition, tables, self._mismatches(True)) == definition

    def test_an_empty_side_is_dropped(self):
        """`tottravmm == keytravmm` split into three sides, one empty, and rejoined as
        `tottravmm =  != keytravmm`."""
        from text2sql.profiler.knowledge import verified_definition

        assert verified_definition("tottravmm == keytravmm", {"t": {"tottravmm", "keytravmm"}},
                                   self._mismatches(True)) == "tottravmm != keytravmm"

    def test_verified_rewrites_the_whole_base(self):
        from text2sql.profiler.knowledge import DatabaseKnowledge, KnowledgeEntry

        kb = DatabaseKnowledge({0: KnowledgeEntry(0, "Net Worth", "", "networth = a - b")})
        out = kb.verified({"t": {"networth", "a", "b"}}, self._mismatches(True))
        assert out.entries[0].definition == "networth != a - b"


class TestKnowledgeIsRelational:
    """A one-column definition is that column's own description, which the meaning base already
    carries; and a bare formula claims nothing the data can check."""

    def test_a_lone_column_name_is_not_knowledge(self):
        """A `value_illustration` run produced `Total Assets (TOTASSETS): totassets` three times
        over — redundant with `*_meaning_base_long.json` and unverifiable."""
        from text2sql.profiler.knowledge import DatabaseKnowledge, KnowledgeEntry, relates

        assert relates("totassets - totliabs") and relates("peercorr >= 0.8")
        assert not relates("totassets") and not relates("") and not relates("  ")
        kb = DatabaseKnowledge(
            {0: KnowledgeEntry(0, "Net Worth (NETWORTH)", "", "totassets - totliabs"),
             1: KnowledgeEntry(1, "Total Assets (TOTASSETS)", "", "totassets")})
        assert set(kb.ground({"totassets", "totliabs", "networth"}).entries) == {0}

    def test_a_bare_formula_becomes_the_equality_the_data_can_check(self):
        """`Net Worth (NETWORTH) = totassets - totliabs` was written without the `= networth`
        side, so the claim credit's data refutes in all 1000 rows went unchecked."""
        from text2sql.profiler.knowledge import asserted

        cols = {"totassets", "totliabs", "networth", "liqassets"}
        assert asserted("totassets - totliabs", "Net Worth (NETWORTH)", cols) == \
            "networth = totassets - totliabs"
        # The acronym is 8 characters; a {2,6} cap silently skipped it.
        assert asserted("liqassets / totassets", "Asset Liquidity Ratio (ALR)", cols) == \
            "liqassets / totassets"                      # ALR is not a column
        assert asserted("networth = totassets - totliabs", "Net Worth (NETWORTH)", cols) == \
            "networth = totassets - totliabs"            # already an equality
        assert asserted("totassets > 0", "Solvent (TOTASSETS)", cols) == "totassets > 0"


class TestKnowledgeMeasured:
    """Every entry carries what the data said about it. Nothing is dropped: a relation the
    agent can see is refuted is a dead end it will not spend a turn rediscovering, and the
    two shapes that looked broken were the most useful ones in the base."""

    @staticmethod
    def _counts(**by_predicate):
        """``count(table, predicate)`` from a mapping; a missing key cannot run."""
        return lambda table, predicate: by_predicate.get(predicate)

    def _verify(self, entry, columns, **counts):
        from dataclasses import replace

        from text2sql.profiler.knowledge import DatabaseKnowledge
        kb = DatabaseKnowledge({0: replace(entry, tables=("t",))})
        return kb.verified({"t": columns}, self._counts(**counts)).entries[0]

    @pytest.mark.parametrize(("hits", "misses", "note", "kept"), [
        (24, 76, "Selects 24% of rows.", True),          # precise, which is the point
        (1, 99, "Selects 1% of rows.", True),            # rare is not wrong
        (100, 0, "", False),      # restates the enum
        (0, 100, "", False),      # threshold never reached
        (9996, 4, "", False),     # rounds to 100%
    ])
    def test_a_cohort_keeps_its_definition_only_where_it_selects(self, hits, misses, note, kept):
        """The share is prevalence, not truth: a low one means precise, so only the extremes say
        nothing. One measured KB called 91% of rows a "significant" network. A dropped
        definition gets no prose either: saying why cost 161 lines in one measured run."""
        from text2sql.profiler.knowledge import KnowledgeEntry

        out = self._verify(KnowledgeEntry(0, "Strong Signal", "Score over 90.", "behansc > 90"),
                           {"behansc"},
                           **{"behansc > 90": hits, "NOT (behansc > 90)": misses})
        assert out.description == f"Score over 90. {note}".strip()
        assert bool(out.definition) is kept and out.knowledge == "Strong Signal"

    def test_a_formula_naming_its_result_is_kept_and_flagged(self):
        """`annualexpenses = mthexp * 12` names the derived value; no column holds it. Dropping
        it lost the shape the shipped bases use, and `search_knowledge` its only record."""
        from text2sql.profiler.knowledge import KnowledgeEntry

        out = self._verify(KnowledgeEntry(0, "Annual Expenses (AE)", "Yearly outgoings.",
                                          "annualexpenses = mthexp * 12"), {"mthexp"})
        assert out.definition == "annualexpenses = mthexp * 12"
        assert out.description.endswith("No column stores this; compute the formula.")

    def test_a_json_path_is_rewritten_to_sql(self):
        """`chaninvdatablock.onlineuse` is how the profile names the path and what the model
        copies; no database accepts it, and one gold query filters on exactly that field."""
        from text2sql.profiler.knowledge import KnowledgeEntry

        out = self._verify(
            KnowledgeEntry(0, "Digital First", "", "chaninvdatablock.onlineuse = 'High'"),
            {"chaninvdatablock", "chaninvdatablock.onlineuse"},
            **{"json_extract(chaninvdatablock, '$.onlineuse') = 'High'": 30,
               "NOT (json_extract(chaninvdatablock, '$.onlineuse') = 'High')": 70})
        assert out.definition == "json_extract(chaninvdatablock, '$.onlineuse') = 'High'"
        assert out.description == "Selects 30% of rows."

    @pytest.mark.parametrize("definition", [
        "othertable.x > 1",                 # a condition on a column this table lacks
        "(produsescore + chanusescore) / 2",  # a bare formula on columns it lacks
        "invdetreg REFERENCES compref",     # not SQL at all
    ])
    def test_what_the_data_cannot_evaluate_loses_its_definition(self, definition):
        """A formula naming a column its own table lacks would give an agent copying it a SQL
        error instead of an answer."""
        from text2sql.profiler.knowledge import KnowledgeEntry

        out = self._verify(KnowledgeEntry(0, "Broken", "Some term.", definition), {"behansc"})
        assert out.description == "Some term."
        assert not out.definition and out.knowledge == "Broken"  # the term stays findable

    @pytest.mark.parametrize("definition", [
        "SUM(CASE WHEN behansc > 90 THEN 1 ELSE 0 END) * 100.0 / COUNT(behansc)",
        "clkpos = ROW_NUMBER() OVER (PARTITION BY behansc ORDER BY behansc)",
    ])
    def test_an_aggregate_keeps_its_definition_unmeasured(self, definition):
        """SQLite refuses an aggregate or a window in WHERE, so the probe returned no count and
        every such entry lost its definition: one profiling run kept 0 of them across 18
        databases while the shipped corpus carries 16."""
        from text2sql.profiler.knowledge import KnowledgeEntry

        out = self._verify(KnowledgeEntry(0, "Rate", "A ratio.", definition), {"behansc"})
        assert out.definition == definition and out.description == "A ratio."

    def test_a_scalar_max_is_still_measured(self):
        """SQLite's multi-argument MAX is not an aggregate; it runs in WHERE and has a share."""
        from text2sql.profiler.knowledge import KnowledgeEntry

        d = "MAX(j1tempval, j2tempval) > 75.0"
        out = self._verify(KnowledgeEntry(0, "Hot Joint", "", d), {"j1tempval", "j2tempval"},
                           **{d: 42, f"NOT ({d})": 58})
        assert out.definition == d and out.description == "Selects 42% of rows."

    def test_a_refuted_equality_is_not_also_annotated(self):
        from text2sql.profiler.knowledge import KnowledgeEntry

        out = self._verify(KnowledgeEntry(0, "Net Worth", "", "networth = a - b"),
                           {"networth", "a", "b"},
                           **{"(networth) IS NOT NULL AND (a - b) IS NOT NULL "
                              "AND ABS((networth) - (a - b)) > 0.001": 1000})
        assert out.definition == "networth != a - b" and out.description == ""
        assert out.refuted  # only the flip sets it, so `refutes` decodes the pair

    def test_the_models_own_inequality_is_not_a_refutation(self):
        """A cohort writing `amlresult != 'Pass'` is the model's own condition, not the data
        contradicting it, and `refuted` is the channel that reaches a prompt uninvited."""
        from text2sql.profiler.knowledge import KnowledgeEntry

        out = self._verify(KnowledgeEntry(0, "Flagged", "", "amlresult != 'Pass'"), {"amlresult"})
        assert not out.refuted and out.refutes is None


class TestKnowledgeSelect:
    """Only the entries the linked columns need reach the generation prompt."""

    @staticmethod
    def _kb(*specs):
        from text2sql.profiler.knowledge import DatabaseKnowledge, KnowledgeEntry
        return DatabaseKnowledge({
            i: KnowledgeEntry(i, f"K{i}", "", definition=d, children_knowledge=c)
            for i, (d, c) in enumerate(specs)})

    def test_matches_linked_columns(self):
        kb = self._kb(("uses credutil", -1), ("uses something_else", -1))
        assert [e.id for e in kb.select({"credutil"})] == [0]

    def test_whole_word_match_only(self):
        assert self._kb(("uses credutilization", -1)).select({"credutil"}) == []

    def test_pulls_in_children(self):
        kb = self._kb(("uses credutil", [1]), ("base formula", -1))
        assert {e.id for e in kb.select({"credutil"})} == {0, 1}

    def test_child_closure_reaches_two_hops(self):
        kb = self._kb(("uses credutil", [1]), ("mid", [2]), ("base", -1))
        assert {e.id for e in kb.select({"credutil"})} == {0, 1, 2}

    def test_ranks_by_overlap_and_caps(self):
        kb = self._kb(("a", -1), ("a and b", -1), ("a and b and c", -1))
        assert [e.id for e in kb.select({"a", "b", "c"}, limit=2)] == [2, 1]

    def test_a_term_the_question_names_outranks_a_wider_overlap(self):
        """The cap made selection scope-dependent: a term the question asks about by name was
        cut by entries mentioning more of an unlinked schema's columns, so the same prompt slot
        carried different terms with linking on and off."""
        kb = self._kb(("a", -1), ("a and b", -1), ("a and b and c", -1))
        assert [e.id for e in kb.select({"a", "b", "c"}, "about K0", limit=2)] == [0, 2]

    def test_empty_names_selects_nothing(self):
        assert self._kb(("a", -1)).select(set()) == []

    def test_the_generation_prompt_omits_definitions_at_the_terms_level(self, prompt_manager):
        """`terms` renders `name: description`; only `full` spends the definition's tokens."""
        entries = list(self._kb(("the formula", -1)).entries.values())
        rendered = {full: prompt_manager.render(
            "generate_sql", schema="S", question="Q", dialect="sqlite",
            knowledge=entries, knowledge_full=full) for full in (False, True)}
        assert "the formula" not in rendered[False]
        assert "the formula" in rendered[True]

    def test_merge_offsets_ids_and_children(self):
        left, right = self._kb(("x", -1)), self._kb(("y", [1]), ("z", -1))
        merged = left.merge(right)
        assert set(merged.entries) == {0, 1, 2}
        assert merged.entries[1].children_knowledge == [2]

    def test_without_drops_a_table_and_prunes_links_to_it(self):
        from text2sql.profiler.knowledge import DatabaseKnowledge, KnowledgeEntry
        kb = DatabaseKnowledge({
            0: KnowledgeEntry(0, "A", children_knowledge=[1], tables=("t1",)),
            1: KnowledgeEntry(1, "B", tables=("t2",)),
            2: KnowledgeEntry(2, "Both", tables=("t1", "t2")),
        })
        left = kb.without({"t2"})
        assert set(left.entries) == {0}  # "Both" is dropped by either of its tables
        assert left.entries[0].children_knowledge == -1

    def test_ground_drops_a_cross_table_entry_that_only_restates_a_join(self):
        """The links prompt is given the join map so it knows which tables relate; copying it
        back is the foreign key map, not knowledge, and a same-table identity is not a join."""
        from text2sql.profiler.knowledge import DatabaseKnowledge, KnowledgeEntry
        kb = DatabaseKnowledge({
            0: KnowledgeEntry(0, "Join", definition="a.k = b.k", tables=("a", "b")),
            1: KnowledgeEntry(1, "Nulls", definition="a.k IS NOT NULL AND b.k IS NOT NULL",
                              tables=("a", "b")),
            2: KnowledgeEntry(2, "Cohort", definition="a.k = 'Y' AND b.k = 1", tables=("a", "b")),
            3: KnowledgeEntry(3, "Identity", definition="k = j", tables=("a",)),
        })
        assert set(kb.ground({"k", "j"}).entries) == {2, 3}

    def test_ground_drops_an_unquoted_text_literal(self):
        """A cohort written `sancresult = Fail` parses as a column comparison and errors on
        every row, so it is not knowledge the agent can use."""
        kb = self._kb(("sancresult = 'Fail'", -1), ("pepresult = Fail", -1))
        assert set(kb.ground({"sancresult", "pepresult"}).entries) == {0}

    def test_ground_drops_invented_columns(self):
        """Small models invent plausible names; those entries can never be selected."""
        kb = self._kb(("uses credutil", -1), ("uses assets.total_assets", -1))
        grounded = kb.ground({"credutil"})
        assert set(grounded.entries) == {0}

    def test_ground_keeps_entries_built_on_grounded_ones(self):
        from text2sql.profiler.knowledge import DatabaseKnowledge, KnowledgeEntry
        kb = DatabaseKnowledge({
            0: KnowledgeEntry(0, "DTI", definition="uses credutil"),
            1: KnowledgeEntry(1, "Over-Extended", definition="DTI > 0.43"),
            2: KnowledgeEntry(2, "Stressed", definition="Over-Extended and falling"),
            3: KnowledgeEntry(3, "Bogus", definition="uses made_up_col"),
        })
        assert set(kb.ground({"credutil"}).entries) == {0, 1, 2}

    def test_ground_prunes_dangling_children(self):
        kb = self._kb(("uses credutil", [1]), ("uses invented_col", -1))
        assert kb.ground({"credutil"}).entries[0].children_knowledge == -1


# ── MinHash value index ──────────────────────────────────────────


class TestValueIndexBuild:
    """ValueIndex over the values profiling already collected."""

    def test_indexes_the_profiles_values(self, db_profile):
        from text2sql.profiler.minhash import ValueIndex
        index = ValueIndex.from_profile(db_profile)
        assert "Alice" in index.values["users.name"]
        assert "orders.amount" in index.values

    def test_builds_without_touching_the_database(self, db_profile, db_conn):
        """Sampling every column cost one query each — unusable on Athena/Snowflake.

        The profiler has already read these values, so the index must reuse them.
        """
        from unittest.mock import patch

        from text2sql.profiler.minhash import ValueIndex
        with patch.object(db_conn, "execute", side_effect=AssertionError("queried the DB")):
            assert ValueIndex.from_profile(db_profile).values

    def test_skips_columns_with_no_values(self, db_profile):
        from text2sql.profiler.minhash import ValueIndex
        db_profile.tables["users"].columns["name"].top_k_values = []
        assert "users.name" not in ValueIndex.from_profile(db_profile).values

    def test_value_match_str(self):
        from text2sql.profiler.minhash import ValueMatch
        s = str(ValueMatch("users", "name", "Alice", 0.75))
        assert "users.name" in s and "Alice" in s and "0.750" in s

    def test_shingles_handle_short_and_empty_values(self):
        from text2sql.profiler.minhash import shingles, sketch
        assert shingles("ab") == [b"ab"]      # shorter than the shingle size
        assert shingles("   ") == []
        assert sketch("") is None

    def test_value_index_is_not_a_cached_artifact(self):
        """It derives from the profile, so persisting it would be a second copy."""
        from text2sql.profiler.cache import KINDS
        assert "value_index" not in KINDS


# ── ProfileSummarizer ────────────────────────────────────────────


class TestProfileSummarizer:
    """Column descriptions from a mocked LLM."""

    @staticmethod
    def _json_for_all_columns(short="A column description", long="long desc"):
        """A JSON blob keyed by every sample-DB column; parse picks each table's own keys."""
        cols = ["id", "name", "email", "age", "city", "user_id", "amount", "created_at"]
        return json.dumps({c: {"short": short, "long": long} for c in cols})

    @pytest.fixture
    def summarize(self, db_profile, mock_llm, prompt_manager):
        """Factory: run a summarizer over the sample profile with a canned response."""
        from text2sql.profiler.summarizer import ProfileSummarizer

        async def _run(*, one_call_per_table=True, **kw):
            mock_llm.chat.return_value = self._json_for_all_columns()
            summarizer = ProfileSummarizer(mock_llm, prompt_manager,
                                           one_call_per_table=one_call_per_table)
            return await summarizer.summarize_database(db_profile, **kw)

        return _run

    async def test_summarize_database(self, summarize):
        summary = await summarize()
        assert summary.columns["users"]["name"].short_summary == "A column description"
        assert summary.columns["users"]["name"].long_summary == "long desc"

    async def test_call_granularity(self, summarize, db_profile, mock_llm):
        await summarize(one_call_per_table=True)
        assert mock_llm.chat.await_count == len(db_profile.tables)  # one call per table

        mock_llm.chat.reset_mock()
        await summarize(one_call_per_table=False)
        assert mock_llm.chat.await_count == sum(
            len(tp.columns) for tp in db_profile.tables.values())  # one call per column

    async def test_summarize_without_long(self, summarize):
        summary = await summarize(generate_long=False)
        assert summary.columns["users"]["name"].long_summary == ""

    def test_summary_roundtrip(self):
        from text2sql.profiler.summarizer import ColumnSummary, DatabaseSummary
        ds = DatabaseSummary(columns={"t": {"c": ColumnSummary("t", "c", "short", "long")}})
        restored = DatabaseSummary.from_flat(ds.to_flat("db", long=False), ds.to_flat("db", long=True))
        assert restored.get_short("t", "c") == "short"
        assert restored.get_long("t", "c") == "long"

    def test_describe_falls_back_to_the_other_kind(self):
        """With profile_summary='short' only the short file exists — consumers still get text."""
        from text2sql.profiler.summarizer import ColumnSummary, DatabaseSummary
        ds = DatabaseSummary(columns={"t": {"c": ColumnSummary("t", "c", short_summary="short")}})
        assert ds.get_long("t", "c") == ds.get_short("t", "c") == "short"

    def test_get_missing(self):
        from text2sql.profiler.summarizer import DatabaseSummary
        ds = DatabaseSummary()
        assert ds.get_short("missing", "col") == ds.get_long("missing", "col") == ""

    @pytest.mark.parametrize("short, long, wanted, unwanted", [
        (True, True, ("short", "long"), ()),
        (True, False, ("short",), ("long",)),
        (False, True, ("long",), ("short",)),
    ], ids=["both", "short_only", "long_only"])
    def test_prompt_mentions_only_requested_kinds(self, prompt_manager, short, long,
                                                  wanted, unwanted):
        """A key we don't want must not appear at all — telling a small model to return
        an empty string for it invites it to invent one."""
        prompt = prompt_manager.render(
            "summarize_rules", generate_short=short, generate_long=long)
        assert all(f'"{k}"' in prompt for k in wanted)
        assert not any(f'"{k}"' in prompt for k in unwanted)
        assert "empty string" not in prompt
