# 🚀 LiveSQLBench-Base-Lite
*A dynamic, **contamination‑free** benchmark for evaluating LLMs on complex, real‑world ****text‑to‑SQL**** tasks.*

[🌐 LiveSQLBench Website](https://livesqlbench.ai) • [🌐 BIRD-INTERACT Project Page](https://bird-interact.github.io/) • [📄 Paper](https://huggingface.co/papers/2510.05318) • [💻 LiveSQLBench GitHub](https://github.com/bird-bench/livesqlbench) • [💻 BIRD-INTERACT GitHub](https://github.com/bird-bench/BIRD-Interact)

Maintained by the **🦜 [BIRD Team @ HKU](https://bird-bench.github.io)** & **☁️ [Google Cloud](https://cloud.google.com/)**


## 📊 LiveSQLBench Overview

**LiveSQLBench** (BIRD-SQL Pro v0.5) is a **contamination-free**, **continuously evolving** benchmark designed to evaluate LLMs on **complex, real-world text-to-SQL tasks**, featuring **diverse real-world user queries**, including **Business Intelligence (BI)**, **CRUD operations**, and more. Each release will include **50 new, fully open-source DBs** curated by the BIRD team through expert collaboration and continuous improvement. It will cover a **wide range of database sizes**, from **end-user level** (around 127 columns) to **industrial level** (1340+ columns). Here are the features of the LiveSQLBench benchmark:

1.  **🗄️ Live Databases:**
    Constructed dynamically from extensive and regularly updated CSV datasets, with both base (user-end level) and large (industrial level) versions (1340+ columns each DB) to test scalability.

2.  **💬 Live User Queries and SQL:**
    Each task pairs unambiguous user queries with annotated, gold-standard SQL statements. The user queries are grounded in an external knowledge base, with medium to hard complexity solution SQL statements.

3.  **🧠 Contextual Reasoning (HKB):**
    Every DB includes a hierarchical knowledge base (HKB) where each knowledge may have dependencies to others, which requires the multi-hop reasoning ability. Two HKB formats are provided: (1) structured JSON format, and (2) unstructured Document format.

4.  **🔍 The First Full SQL Spectrum:**
    Supports not just SELECT (Business Intelligence) queries, but also CRUD (e.g., UPDATE, CREATE, and other database management operations) queries.

5.  **⚡ Automated Evaluation:**
    Support fast evaluation via PostgreSQL template & docker. Each question includes verifiable test cases for accurate, reproducible scoring. Soft EX metric is used to evaluate SELECT-ONLY tasks; customized test cases are designed for DBA tasks, such as CRUD (CREATE, READ, UPDATE, DELETE). 

6.  **🔄 Truly Live & Hidden Test:**
    New databases and tasks are added over time. Each release features both open development and hidden test phases. The hidden test set from each release becomes the open development set for the next release, ensuring continuous evolution and fair evaluation.


> 💡 LiveSQLBench's updating databases, tasks, and HKB support BIRD-Interact's conversational and agentic evaluation. BIRD-Interact evaluates LLMs' text-to-SQL ability in dynamic interactive settings with database and user simulation.

## 🎯 Current Release: LiveSQLBench-Base-Lite-SQLite
We are pleased to release a **SQLite version** of **LiveSQLBench-Base-Lite**, extending from PostgreSQL to SQLite dialect to **improve accessibility** as SQLite requires no server setup and runs locally. This release features **18 end-user level databases** with **270** tasks (180 SELECT-only, 90 Management tasks), **HKB-JSON** and **JSON operations in SQL** for trial.

> Beyond SQL and test case translation, we **carefully adapted 20+ user queries** to align with SQLite's database engine characteristics. For example, since SQLite doesn't support custom functions, we modified queries to either return specific scenario values or utilize views (e.g., `CREATE VIEW AS ...`) to maintain query complexity while ensuring compatibility.


## 💻 How to Use the Dataset
Download the dataset containing data file `livesqlbench_data_sqlite.jsonl` and DB metafiles (including schema, HKB, column meaning files) by:
```bash
huggingface-cli download --repo-type dataset --resume-download birdsql/livesqlbench-base-lite-sqlite --local-dir /local/path/livesqlbench-base-lite-sqlite
```
To prevent data leakage through automated crawling, please request access to the ground truth and test cases by emailing **[📧 bird.bench25@gmail.com](mailto:bird.bench25@gmail.com)** with the subject line `[livesqlbench-base-lite-SQLite GT&Test Cases]`. An automated response will provide these data fields.


And please refer to the BIRD-MiniDev [Github repo](https://github.com/bird-bench/mini_dev) for details of usage and evaluation based on this dataset.

## Sample Usage
You can load the dataset using the Hugging Face `datasets` library:

```python
from datasets import load_dataset

# Load the LiveSQLBench-Base-Lite-SQLite dataset
dataset = load_dataset("birdsql/livesqlbench-base-lite-sqlite", "livesqlbench")

# Access the development split
dev_data = dataset["dev"]

# Print the first example
print(dev_data[0])
```

## 📊 Performance on LiveSQLBench-Base-Lite
| Model | PostgreSQL | SQlite |
| :-------------------- | :--------- | :----- |
| o3-mini | 47.78 | 42.59 |
| Claude 3.7 Sonnet | 39.26 | 41.11 |
| GPT-4o | 34.44 | 34.44 |
| Gemini 2.0 Flash | 34.44 | 33.7 |
| DeepSeek R1-0528 | 38.14 | 32.96 |
| QwQ-32B | 31.48 | 31.48 |
| Qwen2.5 Coder 32B | 22.96 | 22.22 |
| Codestral 22B | 21.11 | 19.63 |
| Qwen2.5 Coder 7B | 12.22 | 12.22 |
| Mixtral 8x7B Instruct | 2.59 | 8.89 |
| Mistral 7B Instruct | 3.7 | 4.44 |


## 📁 Directory Structure
Each database has its own directory:

```
.
├── README.md
├── alien
│   ├── alien_column_meaning_base.json
│   ├── alien_kb.jsonl
│   ├── alien_schema.txt
│   ├── alien_tempalte.sqlite
...
├── livesqlbench_data_sqlite.jsonl
```

### 📂 Directory Contents:


*   `*_schema.txt`: Database schema.
*   `*_kb.jsonl`: Hierarchical knowledge base entries required to solve the user task.
    *   `id`: The unique identifier for the knowledge.
    *   `knowledge`: The name of the knowledge.
    *   `description`: The description of the knowledge.
    *   `definition`: The clear definition of the knowledge.
    *   `type`: The type of the knowledge.
    *   `children_knowledge`: A list of knowledge IDs that the current knowledge is dependent on. -1 means no children.
*   `*_column_meaning_base.json`: Explanation of database columns.


## 📋 Dataset Fields (`livesqlbench_data_sqlite.jsonl`):
*   **instance\_id**: Unique task identifier.
*   **selected\_database**: Associated database name.
*   **query**: Ambiguous user query.
*   **sol\_sql** 🔒: Ground truth SQL solution.
*   **external\_knowledge** 🔒: IDs of required external knowledge to solve the user task.
*   **preprocess\_sql**: SQL setup queries.
*   **clean\_up\_sql**: SQL queries to reset database state.
*   **test\_cases** 🔒: Test cases to validate the predicted corrected SQL.
*   **category**: "Query" (SELECT-only) or "Management" (CRUD).
*   **high\_level**: Boolean indicating whether the user query contains high-level description.
*   **conditions**: Indicates decimal/distinct conditions in the user query.
*   **difficulty\_tier**: Task difficulty (Simple, Moderate, Challenging). 
## 🔒 Accessing Complete Data
To avoid data leakage by auto-crawling, certain fields (e.g., `sol_sql`, `test_cases`, `external_knowledge`) are excluded from the public dataset. For the full dataset, please email: **[📧 bird.bench25@gmail.com](mailto:bird.bench25@gmail.com)** with subject tag `[livesqlbench-base-lite-SQLite GT&Test Cases]`, which will be sent automatically.



## 🔄 Stay Tuned!

Upcoming releases:

*   **🔄 LiveSQLBench-Base-Full:** 600 BI tasks, 200 management tasks, Document-based HKB.
*   **🔄 LiveSQLBench-Large-Lite:** Industrial-scale databases with 1340+ columns.
*   **🔄 LiveSQLBench-Large-Full:** Comprehensive large-scale datasets.

Want new dialects? Vote for new SQL dialects [🗳️ here](https://docs.google.com/forms/d/e/1FAIpQLSfEogmsA7LObI13KOoiojdnYfW28KEqvEVtC9hXaZJ8O9aCpQ/viewform?usp=header)!


## 📄 License:

cc-by-sa-4.0