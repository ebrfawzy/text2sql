# 🔍 text2sql-toolkit

> Natural-language to SQL with automatic database profiling, agentic generation, deterministic repair, and real-time streaming. Works over any SQLAlchemy database and 100+ LLMs via [LiteLLM](https://github.com/BerriAI/litellm).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/Docker-2496ED.svg?logo=docker&logoColor=white)](https://docs.docker.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LiteLLM](https://img.shields.io/badge/LiteLLM-100%2B%20models-6f42c1.svg?logo=litellm&logoColor=white)](https://github.com/BerriAI/litellm)
[![uv](https://img.shields.io/badge/uv-package%20manager-blueviolet.svg?logo=uv&logoColor=white)](https://docs.astral.sh/uv/)
[![Code Style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg?logo=ruff&logoColor=white)](https://github.com/astral-sh/ruff)

This is the research artifact of an MSc dissertation at the University of Bath, *Profiling Instead of Documentation: Deterministic Retrieval and Data-Verified Context for Cost-Efficient Text-to-SQL*. The write-up is not published yet; the code and every run behind the numbers below are here.

## Results

All figures are execution accuracy on **LiveSQLBench-Base-Lite-SQLite** (270 tasks, 18 databases, a third of them writing to the database), scored by the **benchmark authors' own evaluation scripts, vendored unmodified**. DeepSeek v4 Flash unless stated.

| | Configuration |  EX  | Correct | Cost / task |
| --- | --- | ---: | ---: | ---: |
| **E3** | **agent + `value` linking** | **54.07%** | **146** | **$0.0128** |
| E2 | agent, no linking | 53.70% | 145 | $0.0121 |
| E4 | single-shot + `value` linking | 52.59% | 142 | $0.0270 |
| E5 | single-shot, no linking | 50.37% | 136 | $0.0247 |
| E1 | official baseline, same model | 43.70% | 118 | $0.0186 |
| E7 | E4 on a weaker model (Haiku 4.5) | 41.85% | 113 | $0.0202 |
| E6 | E2 with the benchmark's knowledge withheld | 13.70% | 37 | $0.0247 |

54.07% at $0.0128 per task would place **second of fourteen** on the public leaderboard for this benchmark, above an entry costing fifty-four times as much. The figure is self-scored with the vendored evaluator, so read it as "would rank", not "ranks".

**The headline finding is a negative one.** The four architecture variants (E2 to E5) sit within 3.7 points of each other and no pair of them differs by more than chance: each change moves 20 to 25 individual answers, but roughly half move each way. Laid out instance by instance the four return an *identical* verdict on **225 of 270** — 107 that none of them passes, 118 that all of them pass. What does move accuracy is not a design choice: withholding the benchmark's supplied knowledge costs 40 points, and changing the model costs 11.

Two things fall out of that, and they are the parts worth reusing:

- **Report gains and losses, not the net.** A contrast that wins 13 and loses 12, and one that wins 13 and loses none, both read as small positive numbers. Only the second is evidence.
- **Validate a checker before trusting it.** The 17-checker repair cascade is held to firing on *none* of the benchmark's own 270 reference queries. It fires on one, and that one genuinely fails to execute.

`results/` holds the per-instance record behind every row of this table: the verdict, the generated SQL, token counts and the full message history of each stage.

**Jump to:**

- [Results](#results)
- [Features](#features)
- [Pipeline](#pipeline)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Integrations](#integrations)
- [REST API](#rest-api)
- [Configuration](#configuration)
- [Benchmarking](#benchmarking)
- [Deployment](#deployment)
- [Development](#development)

## Features

Most Text-to-SQL tools assume you already know your schema. This one profiles it for you first, then runs a multi-stage, fully observable pipeline.

- **Auto database profiling:** column stats, LLM-written descriptions and a data-verified knowledge base, cached to disk/S3 (no hand-written docs needed).
- **Agentic generation:** a tool loop that explores the database, statically checks its own SQL, retrieves domain knowledge, and submits an answer — no agent framework, so tokens, cost and every tool call are recorded.
- **Composable schema linking:** any combination of `direct` / `reversed` / `value`, unioned, over a recall-first candidate set (BM25 + key promotion + value matching + join closure).
- **Deterministic repair:** 17-checker `sqlglot` cascade (syntax → logic → quality) catches what the LLM misses, and gates the agent's own submission.
- **Confidence-aware selection:** executes candidates, votes on consensus, and asks the LLM to adjudicate ties.
- **Real-time streaming:** progress events via async generator, SSE (`POST /ask`), or a Rich CLI display.
- **Universal:** any SQLAlchemy database + 100+ LLMs through LiteLLM, no lock-in.
- **SQL as a function:** stateless core, ships in Docker for Lambda / Cloud Run / on-prem.
- **Config-driven:** every stage and agent feature toggles on/off; zero fine-tuning required.

## Pipeline

```mermaid
flowchart LR
    Q([Question]) --> P[Profile] --> L[Schema Link] --> G["Generate<br/>direct · agent"] --> R["Repair<br/>17 checkers"] --> S[Select] --> O([SQL + Results])
    C[("Profile Cache<br/>local dir or s3://")] -.-> P
    P & L & G & R & S -. events .-> E([📡 Stream])
```

| Stage           | What it does                                                                                                      |
| --------------- | ----------------------------------------------------------------------------------------------------------------- |
| **Profile**     | Column stats, LLM descriptions and a knowledge base, cached to disk/S3 (skipped on cache hit).                    |
| **Schema Link** | Trims schema to relevant tables: any union of `direct` / `reversed` / `value`.                                    |
| **Generate**    | N candidates via a ReAct tool loop or direct prompting, diversified by seed, schema order and reasoning strategy. |
| **Repair**      | 17-checker `sqlglot` cascade (syntax → logic → quality) fixes what the LLM misses.                                |
| **Select**      | Executes candidates, picks by `single` / `majority` / `confidence` (LLM adjudication for ties).                   |

<details>
<summary><b>How config drives the flow</b> (detailed)</summary>

```mermaid
flowchart TD
    classDef stage fill:#2d2d3a,stroke:#bb86fc,stroke-width:2px,color:#fff;
    classDef opt fill:#1e1e1e,stroke:#03dac6,stroke-width:1px,stroke-dasharray:5 5,color:#fff;
    classDef db fill:#332940,stroke:#ffb86c,stroke-width:2px,color:#fff;
    classDef decision fill:#20303a,stroke:#64b5f6,color:#fff;
    classDef endNode fill:#1e3329,stroke:#4caf50,stroke-width:2px,color:#fff;

    subgraph PROF ["① Profiling"]
        direction TB
        PH{Cache hit?}
        PH -- miss --> PB
        PB["StatsProfiler + ProfileSummarizer + KnowledgeGenerator (LLM)"] --> C[("Profile Cache: profile · meaning_base_short<br/>meaning_base_long · kb<br/>local dir or s3://")]
        PH -- hit --> C
    end

    subgraph LINK ["② Schema Linking"]
        direction TB
        SLM{"schema_linking_modes (any combination)"}
        CAND["Candidate set: BM25 x3 + RRF · key promotion<br/>value match (MinHash) · named fields · join closure"]
        SLM -->|direct| D["LLM maps tables/columns"]
        SLM -->|reversed| RV["LLM writes SQL, sqlglot extracts fields"]
        SLM -->|value| VL["The candidate set itself"]
        CAND -. focused scope .-> D
        CAND -. focused scope .-> RV
        CAND --> VL
        D --> LT([Linked tables + columns, unioned])
        RV --> LT
        VL --> LT
    end

    subgraph GEN ["③ Generation (num_candidates)"]
        direction TB
        GC{generation_mode}
        GC -->|agent| AG["SQLAgent: ReAct tool loop"]
        GC -->|direct| PR["SQLGenerator: direct prompting"]
        AG -. tools .-> TOOLS["always: execute_sql · review_sql · submit_sql<br/>agent_mode=retrieval adds: search_columns<br/>describe_table · describe_columns<br/>search_knowledge · lookup_example"]
        AG -. ends on .-> FEAT["submit_sql · no tool call · agent_max_turns"]
    end

    subgraph REP ["④ Repair (per candidate)"]
        direction TB
        RC{use_repair}
        RC -->|true| CASC["sqlglot cascade:<br/>syntax · dry_run · join · order_by · time · null<br/>division · json_compare · as_stored · returning<br/>rename · rebuild · rowid_pk · precision · result<br/>+ naming · question (these read the question)"]
        CASC --> LOOP{"Issues left?<br/>(up to max_repair_retries)"}
        LOOP -- yes: LLM fix --> CASC
    end

    subgraph SELG ["⑤ Selection"]
        direction TB
        SEL{selection_mode}
        SEL -->|single| S1["First valid candidate"]
        SEL -->|majority| S2["Majority vote on results"]
        SEL -->|confidence| S3{Any agreement?}
        S3 -->|yes| FP["Largest cluster"]
        S3 -->|"all disagree"| ADJ["LLM adjudication"]
    end

    %% cross-stage wiring
    Q([User Question]) --> PH
    C --> SLC{use_schema_linking}
    SLC -- true --> SLM
    SLC -- false --> GPREP["Format schema + profile context"]
    LT --> GPREP
    GPREP --> GC
    AG --> RC
    PR --> RC
    RC -->|false| SEL
    LOOP -- clean --> SEL
    S1 --> OUT
    S2 --> OUT
    FP --> OUT
    ADJ --> OUT
    OUT([SQL + Results + Trace])

    STREAM([📡 PipelineEvent + TokenDelta stream])
    PH -.-> STREAM
    SLM -.-> STREAM
    AG -.-> STREAM
    PR -.-> STREAM
    CASC -.-> STREAM
    SEL -.-> STREAM

    class PB,D,RV,VL,CAND,GPREP,AG,PR,CASC,S1,S2,FP,ADJ stage
    class PH,SLC,SLM,GC,RC,LOOP,SEL,S3 decision
    class C db
    class TOOLS,FEAT,STREAM opt
    class Q,LT,OUT endNode
```

</details>

## Quick Start

```bash
git clone https://github.com/ebrfawzy/text2sql.git && cd text2sql
cp .env.example .env    # then add your LLM API key (e.g. OPENAI_API_KEY)
```

**Docker** (recommended):

```bash
docker compose run text2sql ask "How many users signed up in 2025?" --db sqlite:///example.db
docker compose run text2sql profile sqlite:///example.db
```

**Local with uv** (optional):

```bash
uv sync --extra all          # core only: `uv sync`
uv run text2sql ask "How many users signed up in 2025?" --db sqlite:///my.db
```

## Usage

`ask()` is an **async generator**: it streams `PipelineEvent` and `TokenDelta` items, then yields a final `Text2SQLResult`.

```python
import asyncio
from text2sql import Text2SQL
from text2sql.pipeline.events import collect_result

engine = Text2SQL(db_uri="sqlite:///my.db", model="gpt-4o-mini")

async def main():
    # Stream every event as it happens:
    async for item in engine.ask("How many orders last month?"):
        print(item)

    # ...or just get the final answer:
    result = await collect_result(engine.ask("Total revenue?"))
    print(result.sql, result.results, result.trace)

asyncio.run(main())
```

Pass any [config](#configuration) field as a kwarg:

```python
Text2SQL(db_uri="...", model="...", schema_linking_modes=["direct", "value"],
         selection_mode="confidence", num_candidates=3,
         generation_strategy="diverse")
```

## Integrations

- **Databases:** any SQLAlchemy driver.
- **LLMs:** any LiteLLM model string (set the matching provider env var).

```python
# Databases
Text2SQL(db_uri="postgresql://user:pass@localhost/mydb")
Text2SQL(db_uri="mysql+pymysql://user:pass@localhost/mydb")
Text2SQL(db_uri="snowflake://user:pass@account/db/schema")

# LLMs
Text2SQL(db_uri="...", model="gpt-4o")                             # OpenAI
Text2SQL(db_uri="...", model="anthropic/claude-sonnet-4-20250514") # Anthropic
Text2SQL(db_uri="...", model="gemini/gemini-2.5-pro")              # Google
Text2SQL(db_uri="...", model="ollama/llama3")                      # local
```

## REST API

POST endpoints stream Server-Sent Events. Serve it with:

```bash
# Docker (recommended)
docker compose run --service-ports text2sql serve --port 8000

# Local with uv (needs the `api` extra)
uv run text2sql serve --host 0.0.0.0 --port 8000
```

- Interactive OpenAPI docs list every endpoint and schema at **<http://localhost:8000/docs>**.
- Example call:

```bash
curl -N -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How many users?"}'
# → event: progress … / event: token … / event: result {"sql": "...", "results": [...]}
```

## Configuration

- Precedence (high to low): **constructor kwargs > env vars (`TEXT2SQL_` prefix) > `.env` > YAML (`--config`) > defaults.**
- Env var name = `TEXT2SQL_` + the field name uppercased (e.g. `TEXT2SQL_SCHEMA_LINKING_MODES`).
- Full commented reference: [`configs/config.yaml`](configs/config.yaml) and [`.env.example`](.env.example).

YAML sections follow the pipeline in execution order. `sql_generation.mode` is the fork
between the two pathways, so the agent's settings nest under `sql_generation.agent` and are
read only when `mode: agent`; `verification` holds the post-generation repair and selection
knobs that both pathways share. Many settings only matter in certain combinations — each
mode's block under `schema_linking` needs that mode selected, `selection_mode` needs
`num_candidates > 1` (which needs `strategy: diverse`), the agent block needs `mode: agent`. Those dependencies are declared
once in `FIELD_DEPENDS`, which the web UI uses to hide inert controls and which logs a
warning at startup naming any inert setting you explicitly set.

Most-used settings:

| Field                  | Default                | Notes                                                                                                       |
| ---------------------- | ---------------------- | ----------------------------------------------------------------------------------------------------------- |
| `model`                | `gpt-4o-mini`          | Any LiteLLM model string.                                                                                   |
| `db_uri`               | `sqlite:///example.db` | SQLAlchemy URI.                                                                                             |
| `num_candidates`       | `1`                    | 1–3; above 1 requires `generation_strategy: diverse` and enables `selection_mode`.                          |
| `generation_mode`      | `direct`               | `direct` (single-shot prompting) vs. `agent` (ReAct tool loop).                                             |
| `agent_mode`           | `retrieval`            | `schema_preloaded` (linked schema + execute/review/submit) / `retrieval` (table names only, pull the rest). |
| `agent_max_turns`      | `20`                   | Turn budget before the agent must answer; inert at `generation_mode: direct`.                               |
| `generation_strategy`  | `direct`               | `direct` / `decompose` / `query_plan` / `diverse`.                                                          |
| `use_repair`           | `true`                 | Deterministic repair cascade.                                                                               |
| `schema_linking_modes` | `[value]`              | Any combination of `direct` / `reversed` / `value`, unioned.                                                |
| `selection_mode`       | `single`               | `single` / `majority` / `confidence`; inert at `num_candidates: 1`.                                         |
| `reasoning_effort`     | `none`                 | `none` keeps `temperature`; any other value forces `temperature: 1`.                                        |
| `profile_cache_dir`    | `.cache/profiles`      | Local dir or `s3://bucket/prefix`.                                                                          |
| `event_verbosity`      | `verbose`              | `minimal` / `detailed` / `verbose`.                                                                         |

More capabilities:

- **Agent tools** (config-gated via `agent_mode`): `schema_preloaded` is execute + review + submit; `retrieval` prepends the retrieval tools as one progressive disclosure — `search_columns` (ranked fields), `describe_table` (the columns the question points at, the rest by name), `describe_columns` (long descriptions for named fields) and `search_knowledge` (which follows knowledge dependencies) — each ranking its results rather than dumping them, and `retrieval` withholds the schema and KB so the tools are the way in. `review_sql` runs the whole checker cascade and reports every finding with the rows the query returns; `submit_sql` only commits, and is withheld until a review has run.
- **Domain knowledge:** the profiler's generated KB reaches the agent through `search_knowledge`; pass `scenarios_file="scenarios.md"` to add hand-written business rules via `lookup_example` (both are `retrieval`-tier tools).
- **Custom prompts:** override bundled Jinja2 templates via `TEXT2SQL_PROMPT_<NAME>_PATH`, `TEXT2SQL_PROMPT_TEMPLATE_DIR`, or `TEXT2SQL_PROMPT_VERSION`.

## Benchmarking

Runs against [LiveSQLBench-Base-Lite-SQLite](https://huggingface.co/datasets/birdsql/livesqlbench-base-lite-sqlite).

> **Reference SQL and test cases are not in this repository.** The benchmark authors withhold
> them to prevent data leakage and supply them on request from `bird.bench25@gmail.com`. Save
> your copy under `data/` with a `local_` prefix (git-ignored) and point
> `TEXT2SQL_BENCHMARK_TESTCASES_JSONL` at it. Without it the benchmark runs and writes
> predictions, but cannot score them.
 Execution accuracy comes from the **official evaluation scripts, vendored unmodified** and run as-is over the run's `predictions.jsonl` — so the number is the published harness's number. Each run directory gets `predictions.jsonl` (official submission format), `run.json` (totals, per-category/difficulty/database breakdowns, per-instance traces, tokens and cost, Oracle@N and selection loss) and, when linking was measured, `linking.jsonl`.

```bash
docker compose run text2sql benchmark --dataset-folder ./data \
  --data-jsonl ./data/livesqlbench_data_sqlite.jsonl -o results/run1/
docker compose run text2sql benchmark --instance-id credit_4
docker compose run text2sql benchmark --max 20                 # quick smoke run

uv run text2sql eval results/run1              # re-score a finished run
uv run text2sql eval results/run1 --mode gold  # harness sanity check: must be 100%
uv run text2sql gold-check                     # the same check without a run directory
```

For ablations, run twice into different `-o` dirs (e.g. with/without profiling via `--config`) and compare reports.

**Stage-only runs.** `--stop-after` (setting `stop_after`, also available on `/ask`) halts the pipeline after `profiling`, `schema_linking` or `sql_generation`. Stopping at linking generates no SQL and runs no evaluator: `run.json` and `linking.jsonl` report the linked schema against the gold SQL — precision, recall, F1 and exact-match at both table and column level, plus the missing/extra names per question.

```bash
uv run text2sql benchmark -n 50 --stop-after schema_linking
TEXT2SQL_SCHEMA_LINKING_MODES='["direct","value"]' uv run text2sql benchmark -n 50 --stop-after schema_linking
```

## Development

```bash
uv sync --extra dev
uv run pytest                      # tests
uv run ruff check src/ tests/      # lint (line-length 120)
uv run mypy src/                   # types
```

Package layout: `core.py` (orchestrator) · `config.py` (settings) · `cli.py` · `db.py` · `llm.py` · `api/` (FastAPI/SSE) · `pipeline/` (agent, tools, repair, selector, generator, examples, events, tracer) · `profiler/` (stats, summarizer, knowledge, minhash, cache) · `schema/` (loader, linker, lexical) · `prompts/` (Jinja2 templates) · `benchmark/` (runner, results, vendored official evaluator).

## Acknowledgements & License

Synthesizes ideas from [Shkapenyuk et al. (2025)](https://arxiv.org/abs/2505.19988v2) (profiling), [DeepEye-SQL](https://arxiv.org/abs/2510.17586v3) (multi-mode linking + checker cascade), [TA-SQL](https://arxiv.org/abs/2405.15307v1) (task alignment), and [CHASE-SQL](https://arxiv.org/abs/2410.01943) (multi-strategy candidate diversity).

Distributed under the **MIT License**; see [LICENSE](LICENSE). The vendored LiveSQLBench
evaluation scripts and the dataset files under `data/` belong to the LiveSQLBench / BIRD
authors and carry their own terms.
