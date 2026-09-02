"""Shared test fixtures for the text2sql test suite.

Provides reusable fixtures for database connections, mock LLM clients,
prompt managers, profiling data, and configuration objects. All fixtures
that are needed by more than one test file belong here.
"""

from __future__ import annotations

import json
import os
import sqlite3
from unittest.mock import AsyncMock, MagicMock

import pytest

from text2sql.config import Settings
from text2sql.core import Text2SQL
from text2sql.db import DatabaseConnection
from text2sql.pipeline.examples import ExampleStore
from text2sql.profiler.stats import DatabaseProfile, TableProfile
from text2sql.profiler.summarizer import DatabaseSummary
from text2sql.prompts.manager import PromptManager
from text2sql.schema.linker import SchemaLinker
from text2sql.schema.loader import SchemaLoader


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Ensure local environment variables and .env file don't break tests."""
    for key in list(os.environ.keys()):
        if key.startswith("TEXT2SQL_"):
            monkeypatch.delenv(key, raising=False)
    # Prevent pydantic from loading the local .env file
    if hasattr(Settings, "model_config"):
        monkeypatch.setattr(Settings, "model_config", {**Settings.model_config, "env_file": None})


# ── Database fixtures ────────────────────────────────────────────


@pytest.fixture
def sample_db(tmp_path):
    """Create a small SQLite database for testing.

    ``users`` holds 5 rows with one NULL email; ``orders`` holds 5 rows keyed to it, whose
    ``meta`` is a JSON object so nested-field profiling has something to find.
    """
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT,
            age INTEGER,
            city TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            user_id INTEGER REFERENCES users(id),
            amount REAL NOT NULL,
            created_at TEXT,
            meta TEXT
        )
    """)
    # Insert sample data
    users = [
        (1, "Alice", "alice@example.com", 30, "New York"),
        (2, "Bob", "bob@example.com", 25, "London"),
        (3, "Charlie", None, 35, "New York"),
        (4, "Diana", "diana@example.com", 28, "London"),
        (5, "Eve", "eve@example.com", 32, "Paris"),
    ]
    cursor.executemany("INSERT INTO users VALUES (?, ?, ?, ?, ?)", users)

    def meta(channel, express, weight):
        # `scores` is an array leaf: 2 of the 18 shipped databases have one.
        return json.dumps({"channel": channel, "ship": {"express": express}, "weight": weight,
                           "scores": [weight, 1.0]})

    orders = [
        (1, 1, 99.99, "2025-01-15", meta("web", True, 9.5)),
        (2, 1, 149.50, "2025-02-20", meta("web", False, 1200.0)),
        (3, 2, 75.00, "2025-01-10", meta("store", False, 340.0)),
        (4, 3, 200.00, "2025-03-01", meta("web", True, 87.25)),
        (5, 5, 50.00, "2025-01-25", meta("store", True, 15.75)),
    ]
    cursor.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?)", orders)
    conn.commit()
    conn.close()
    return f"sqlite:///{db_path}"


@pytest.fixture
def db_conn(sample_db):
    """Yield a managed DatabaseConnection, closed automatically after the test."""
    with DatabaseConnection(sample_db) as db:
        yield db


@pytest.fixture
def db_schema(db_conn):
    """Pre-loaded schema dict from the sample database."""
    return db_conn.get_schema()


# ── Profiler fixtures ────────────────────────────────────────────


@pytest.fixture
def db_profile(db_conn):
    """Pre-built DatabaseProfile from the sample database."""
    from text2sql.profiler.stats import StatsProfiler

    profiler = StatsProfiler(db_conn, top_k=5, sample_size=100)
    return profiler.profile_database()


@pytest.fixture
def empty_summary():
    """Empty DatabaseSummary (no LLM calls needed)."""
    return DatabaseSummary()


@pytest.fixture
def db_summary(db_profile):
    """DatabaseSummary with a short + long description for every profiled column."""
    from text2sql.profiler.summarizer import ColumnSummary

    return DatabaseSummary(columns={
        t: {c: ColumnSummary(t, c, f"short {t}.{c}", f"long {t}.{c}") for c in tp.columns}
        for t, tp in db_profile.tables.items()
    })


# ── Prompt fixtures ──────────────────────────────────────────────


@pytest.fixture
def prompt_manager():
    """Default v1 PromptManager."""
    return PromptManager(version="v1")


# ── LLM mock fixtures ───────────────────────────────────────────


@pytest.fixture
def mock_llm():
    """MagicMock-based LLMClient returning a simple SELECT COUNT, overridable per test via
    ``mock_llm.chat.return_value``.
    """
    from unittest.mock import AsyncMock

    llm = MagicMock()
    llm.chat = AsyncMock(return_value="SELECT COUNT(*) AS cnt FROM users")
    llm.chat_for_sql = AsyncMock(return_value="SELECT COUNT(*) AS cnt FROM users")
    llm.chat_messages = AsyncMock(return_value="SELECT COUNT(*) AS cnt FROM users")
    llm.usage.summary.return_value = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "total_cost_usd": 0.0,
        "num_calls": 0,
    }
    return llm


# ── ExampleStore fixtures ────────────────────────────────────────


@pytest.fixture
def scenarios_file(tmp_path):
    """Temporary scenarios.md file with 3 sections."""
    md = tmp_path / "scenarios.md"
    md.write_text(
        "## Revenue\n"
        "Total revenue = net + tax\n"
        "\n"
        "## Churn\n"
        "Churn rate = lost / total\n"
        "\n"
        "## Active Users\n"
        "Active = logged in within 30 days\n"
    )
    return str(md)


@pytest.fixture
def example_store(scenarios_file):
    """ExampleStore pre-loaded with test scenarios."""
    return ExampleStore(scenarios_file)


# ── Schema fixtures ──────────────────────────────────────────────


@pytest.fixture
def schema_loader(db_conn, db_profile, empty_summary):
    """SchemaLoader over the sample DB with stats but no LLM descriptions."""
    return SchemaLoader(db_conn, profile=db_profile, summary=empty_summary)


@pytest.fixture
def make_linker(schema_loader, mock_llm, prompt_manager):
    """Factory: ``make_linker(mode="direct")`` or ``make_linker(mode=["direct", "value"])``."""
    def _make(mode="reversed", **kw):
        modes = [mode] if isinstance(mode, str) else list(mode)
        return SchemaLinker(schema_loader, mock_llm, prompt_manager, modes=modes, **kw)
    return _make


@pytest.fixture
def value_index(db_profile):
    """A `ValueIndex` over the sample DB's profile, so literal lookups hit real values."""
    from text2sql.profiler.minhash import ValueIndex
    return ValueIndex.from_profile(db_profile)


# ── Engine fixtures ──────────────────────────────────────────────


@pytest.fixture
def make_engine(sample_db):
    """Factory for a `Text2SQL` with `_profile`/`_summary` pre-primed, so `ask()` does not
    spend its profiling stage on real LLM calls.
    """
    def _make(*, tables=("users",), **kw):
        engine = Text2SQL(db_uri=sample_db, **kw)
        engine._profile = DatabaseProfile(dialect="sqlite", tables={
            t: TableProfile(table_name=t, row_count=5) for t in tables})
        engine._summary = DatabaseSummary()
        return engine
    return _make


def message_text(message):
    """A message's text, whether plain or wrapped as a cache-marked content block."""
    content = message["content"]
    return content if isinstance(content, str) else \
        "".join(b.get("text", "") for b in content)


def stub_agent(sql="SELECT 1", *, seen=None, trace=None, dead=False):
    """A stand-in for `SQLAgent` that appends each candidate's `schema_text` to `seen`;
    `dead=True` yields nothing, the "generation produced no candidates" case.
    """
    class StubAgent:
        async def generate(self, question, schema_text, **kwargs):
            if seen is not None:
                seen.append(schema_text)
            if dead:
                return
            yield (sql, trace if trace is not None else
                   {"turns": 1, "tool_calls": [], "termination": "submit"})

    return StubAgent()


def stub_repair(collect):
    """A stand-in for `SQLRepair` that appends to `collect` and rewrites the SQL."""
    class StubRepair:
        def __init__(self, *a, **kw):
            pass

        async def repair(self, sql, question, schema_text):
            collect.append((sql, schema_text))
            return "REWRITTEN", []

    return StubRepair


def mock_sql_llm(engine, sql):
    """Point every LLMClient entry point on `engine` at one canned SQL string."""
    for name in ("chat", "chat_for_sql", "chat_messages"):
        setattr(engine.llm, name, AsyncMock(return_value=sql))

    async def stream(*args, **kwargs):
        yield sql, False

    engine.llm.stream_chat_messages = stream
    return engine


# ── Config fixtures ──────────────────────────────────────────────


@pytest.fixture
def sample_yaml_config(tmp_path):
    """Temporary YAML config file for Settings.from_yaml() tests."""
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text(
        "sqlalchemy:\n"
        "  db_uri: sqlite:///from_yaml.db\n"
        "litellm:\n"
        "  model: gpt-4o\n"
        "  temperature: 0.5\n"
        "sql_generation:\n"
        "  num_candidates: 3\n"
        "  strategy: diverse\n"
        "  agent:\n"          # nested subsection → flat agent_* fields
        "    max_turns: 10\n"
        "    mode: retrieval\n"
        "verification:\n"
        "  repair: false\n"
    )
    return str(yaml_file)
