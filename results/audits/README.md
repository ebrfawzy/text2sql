# Tier 0 audits, run 2026-09-01

Zero LLM calls. Every arm uses `.cache/profiles-deepseek4flash` and the vendored official
evaluator. Reproduce with the commands in each section.

| id | What | Artifact | Headline |
| --- | --- | --- | --- |
| R1 | Linking scorecard restored from `983fb6f^` | `../E8_linking_scorecard/` | kept 48.9%, covered 85.5%, recall 95.9% |
| R2 | Gold SQL scored by the official evaluator | `R2_gold_check/`, `R2_gold_audit.txt` | **270/270 = 100.00%**; 3 degenerate golds |
| R3 | Repair checkers over golds and predictions | `R3_checkers_gold.txt`, `R3_checkers_agent_predictions.txt` | **1 actionable firing on 270 golds** |
| R4 | Generated KB vs the shipped base | `R4_kb_audit.txt` | cross-table 12.0% vs 12.8%; question terms 0.6% vs 100% |
| R5 | Linking budget curve and per-signal isolation | `R5_budget/`, `R5_signals/` | see below |
| R6 | Leaderboard snapshot | `R6_leaderboard_snapshot.md` + screenshot | **RESOLVED**: SQLite view captured. OntologyAgent 50.00% / $0.6942 confirmed; we would rank 2 of 14 |

## R2 - what the gold check does and does not prove

`uv run text2sql gold-check -o results/audits/R2_gold_check`

**270/270 (100.00%).** Read this narrowly. `gold_check` submits `sol_sql` as `pred_sqls`, so for the
178 reads `test_case_default` compares gold against itself and passing is **tautological**. What the
check genuinely proves: no gold errors, and no bespoke write test contradicts its own gold. It
**cannot** detect a gold that is self-consistent and wrong, which `data/ISSUE_livesqlbench_sqlite_gold.md`
documents 13 times over ~42 instances. **Do not quote 270 as the attainable ceiling.**

`--audit` reports 3 golds whose own output is degenerate:

| Instance | Finding |
| --- | --- |
| `credit_7` | all-NULL column `cohort_quarter` |
| `crypto_7` | all-NULL column `whale_activity` |
| `museum_M_5` | `preprocess_sql` raises `NOT NULL constraint failed: artifactsecurityaccess.secrecordregistry` |

`museum_M_5` still scores as passing, so its bespoke test passes **vacuously**. The scorer's ceiling
is 270; the honest ceiling is 269.

## R3 - the zero-false-positive bar holds

`uv run python scripts/gold_audit_checkers.py`

Over all 270 golds: **one actionable finding**, `dry_run` on `fake_M_4`, whose gold genuinely fails
to execute. Everything else is `info`, which never triggers a rewrite: `order_by` 11, `null` 9.

Over the shipped agent run's 270 predictions:

| Checker | Severity | Fired | Of which failed |
| --- | --- | --- | --- |
| `as_stored` | warning | 11 | **10** |
| `null` | info | 9 | 7 |
| `order_by` | info | 5 | 3 |
| `question` | info | 1 | 1 |
| `dry_run` | error | 1 | 0 |
| `rowid_pk` | error | 1 | 1 |

`as_stored` fires on **0 golds and 11 predictions, 10 of them failing**, against a 46% base failure
rate. The one that passes, `polar_7`, matches gold with or without the call.

## R4 - the KB is structurally right and lexically absent

`uv run python scripts/kb_audit.py .cache/profiles-deepseek4flash` and `--shipped`

| Shape | Generated (1,435) | Shipped (1,082) |
| --- | --- | --- |
| cross-table | **12.0%** | 12.8% |
| single-table | 73.8% | 37.0% |
| term-only | 14.2% | 50.3% |
| stored-result identity | 30.0% | 0% |
| term composite | 2.8% | 33.6% |
| refuted by the data | **5.3% (76 entries)** | 0% |
| unsound against the data | **0** | 0 |
| question terms named | **3 of 471 (0.6%)** | 471 of 471 (100%) |

Two findings. The **cross-table structural gap is closed** (12.0% against a 12.8% target), and
soundness is at the target (0 unsound). The **coverage gap is not**: 0.6% of the terms the questions
cite. 76 entries are refuted by the data, which is the falsification capability quantified across
all 18 databases.

## R5a - the budget curve

`TEXT2SQL_STOP_AFTER=schema_linking TEXT2SQL_SCHEMA_LINKING_MODES='["value"]' TEXT2SQL_VALUE_TOP_K=<k> uv run text2sql benchmark -o results/audits/R5_budget/k<k>`

The budget is `max(value_top_k, 0.45 x columns)`, so `k` is a floor and only bites above the ratio.

| `value_top_k` | col kept | col covered | col recall | tbl kept |
| --- | --- | --- | --- | --- |
| **30 (shipped)** | **48.9%** | **85.5%** | **95.9%** | 98.7% |
| 40 | 49.0% | 85.5% | 95.9% | 98.7% |
| 50 | 50.5% | 85.5% | 96.0% | 99.1% |
| 60 | 54.1% | 87.5% | 96.2% | 99.4% |
| 70 | 60.2% | 88.4% | 96.3% | 99.6% |
| 90 | 73.2% | 88.8% | 96.4% | 100.0% |
| 120 | 86.4% | 90.5% | 96.5% | 100.0% |

Coverage keeps rising with no plateau, so linking metrics alone argue for widening. The shipped
point buys 85.5% coverage for 48.9% of the schema; reaching 90.5% costs 86.4%. **k=30 reproduced
the restored R1 scorecard to every decimal**, which is a reproducibility check as well as a curve
point.

## R5b - per-signal isolation

Module attributes patched in a harness; `src/` unmodified. `value` mode, `value_top_k=30`.

| Arm | col kept | col covered | col recall | tbl kept | tbl covered |
| --- | --- | --- | --- | --- | --- |
| **full** | **48.9%** | **85.5%** | **95.9%** | 98.7% | 82.1% |
| no key promotion | 55.0% | **85.9%** | 96.0% | 99.3% | 82.1% |
| no named fields | 48.9% | 85.5% | 95.9% | 98.7% | 82.1% |
| no join closure | 46.5% | **79.2%** | 94.5% | 87.6% | 78.2% |
| levels: `short` only | 48.2% | 80.1% | 95.3% | 99.7% | 82.1% |
| levels: `short`+`long` | 48.2% | 82.6% | 95.6% | 99.6% | 82.1% |
| levels: `name` only | 47.0% | 78.8% | 95.0% | 96.8% | 82.1% |

Three results, two of which contradict the older notes and both of which were measured on a
different profile cache there.

1. **Join closure is the load-bearing signal.** Removing it costs 6.3 points of coverage and 11
   points of table reach. This confirms the earlier claim.
2. **Key promotion no longer buys coverage.** Removing it *raises* coverage by 0.4 points while
   keeping 6 points more schema. It is now a compression device, not a recall device: it spends
   budget slots on keys so the ranking keeps fewer columns overall. The older claim that dropping it
   "collapses the stage to 51.5 complete" does not hold on this cache.
3. **Named fields is a complete no-op.** Every metric is identical to `full`, to four decimals.
   Instrumenting it shows it is *not* broken: it fires on **258 of 270 questions and admits 1,135
   fields**. Everything it admits is already admitted by the ranking plus join closure. This is the
   signature the notes warn about, an unchanged number rather than a bad one, and here it is a
   genuine measured redundancy.

The three description levels fused by RRF are worth **+5.4 points of coverage over `short` alone**
and +2.9 over `short`+`long`, confirming that direction.

---

## Follow-on: what Tier 1-3 then measured

Full analysis is in the dissertation, Chapter 5.

| Run | Arm | EX | Billed | $/task |
| --- | --- | --- | --- | --- |
| `../E3_agent_linking` | agent + `value` linking | **54.07%** | $3.45 | $0.0128 |
| `../E2_agent` | agent, no linking | 53.70% | $3.27 | $0.0121 |
| `../R7_...value_linking_direct_*` | direct + `value` linking | 52.59% | $7.29 | $0.0270 |
| `../E5_direct` | direct, no linking | 50.37% | $6.66 | $0.0247 |
| `../E7_haiku_direct_linking` | R7's config on Haiku 4.5 | 41.85% | $5.46 | $0.0202 |
| `../R8_...retrieval_agent_no_kb_*` | agent, curated definitions withheld | **13.70%** | $6.66 | $0.0247 |

**WITHDRAWN: the repeated execution.** The second run of the agent configuration
(`R11_...`, deleted 2026-09-02) was found to be contaminated and is not byte-identical to the
execution it was meant to repeat. Every claim that rested on it is withdrawn, including the
"26 flips of 270" noise floor. **Do not cite a noise floor from this project.** Run-to-run
variance is unmeasured, and the dissertation states that as a threat to validity.

**What survives.** The four architectural contrasts are each non-significant on their own
(p = 0.11 to 1.00), and each changes 20 to 25 individual answers while moving the total by nine
at most, because the changes run in both directions. The two contrasts that are significant are
withdrawing the external knowledge (p < 0.0001) and changing the model (p = 0.0003), and both are
one-sided. Neither is a pipeline design choice.

---

## Status

**Measurement is closed.** These seven audits plus the nine run directories in `results/` are the
complete evidence base for the dissertation. No further runs will be made.

The one gap to be aware of when reading R8: it withholds the *curated* definitions while the
*generated* context remains, so its residual 13.70% cannot be attributed. The non-substitution
finding is secure; the value of the generated context in the absence of curation is not measured.
See the dissertation, Chapter 5.

