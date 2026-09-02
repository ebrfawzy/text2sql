# R6 - LiveSQLBench leaderboard snapshot: RESOLVED

Captured 2026-09-01 from <https://livesqlbench.ai/>, **SQLite** benchmark set, dataset
`Base-Lite-SQLite 2025-07`, 1 selected, **270 samples**, date range 2025-07 to 2026-09.
Leaderboard last updated 2026-03-02. Screenshot: `R6_leaderboard_2026-09-01.png`.

The site ranks **Base, SQLite and Large independently** and shows "only entries with results for all
selected SQLite datasets". This is the view that matches our corpus; the default Base view does not.

| Rank | Model | Organization | Success Rate | Cost / Task |
| --- | --- | --- | --- | --- |
| 1 | Agent Sentinel V1 (New) | Genloop | **68.15** | - |
| 2 | OntologyAgent (New) | TheAntHillAI | **50.00** | **$0.6942** |
| 3 | o3-mini | OpenAI | 42.59 | - |
| 4 | Claude 3.7 Sonnet | Anthropic | 41.11 | - |
| 5 | GPT-4o | OpenAI | 34.44 | - |
| 6 | Gemini 2.0 Flash | Google | 33.70 | - |
| 7 | DeepSeek R1-0528 | DeepSeek | 32.96 | - |
| 8 | QwQ-32B | Qwen | 31.48 | - |
| 9 | Qwen2.5 Coder 32B | Qwen | 22.22 | - |
| 10 | Codestral 22B | Mistral | 19.63 | - |
| 11 | Qwen2.5 Coder 7B | Qwen | 12.22 | - |
| 12 | Mixtral 8x7B Instruct | Mistral | 8.89 | - |
| 13 | Mistral 7B Instruct | Mistral | 4.44 | - |

**Where we sit.** R12 scores **54.07%** at **$0.0128 per task** on the same 270 samples, scored by
the vendored copy of the same official evaluator. That is **rank 2 of 14**, above OntologyAgent
(50.00%) and below Agent Sentinel V1 (68.15%), at **1/54th of OntologyAgent's cost per task**. It is
the only entry besides those two with a published cost figure.

**Three caveats that must travel with the number.**

1. **Two populations in one table.** Ranks 3-13 are bare models under a fixed harness; ranks 1-2 are
   agent systems. Comparing our pipeline to o3-mini or GPT-4o is system-versus-model. Only
   Agent Sentinel V1 and OntologyAgent are system-versus-system, and neither discloses its model.
2. **We are not verified.** The site marks BIRD-team-verified submissions with a badge, awarded after
   submitting a codebase for pipeline evaluation. Ours is self-scored with the vendored evaluator.
   Say "would rank", not "ranks", unless the run is submitted.
3. **DeepSeek R1-0528 bare scores 32.96%.** Our pipeline on DeepSeek v4 Flash scores 54.07%. Tempting
   as a "+21 points from the pipeline" claim, and it is not one: different model generation, so it
   bounds nothing.
