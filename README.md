# FinAgentBench RFT Dataset & Training Pipeline

This repository builds Reinforcement Fine-Tuning (RFT) datasets from annotated finance QA signals, evaluates baseline models, and can launch/monitor an OpenAI RFT job using an nDCG grader.

---

## Repository Layout

```
.
├── base_eval.py                 # Baseline (non-trained) model evaluation on combined eval splits
├── chunk_ranking.py             # Produce per-company chunk-ranking raw_data
├── document_ranking.py          # Produce per-company document-ranking raw_data
├── grader.py                    # Convert raw_data → QRELS per-company files + combined splits
├── requirements.txt
├── standardize_filenames.py     # Normalize noisy JSONL input filenames under data/**
├── data/                        # Source filings + annotated relevance (messy filenames)
├── output/                      # Pipeline outputs (raw_data + rft_data)
└── rft/                         # RFT job helpers (grader, launcher, monitor)
    ├── grader/
    │   └── ndcg_grader.py       # Simple nDCG grader used by RFT jobs
    ├── monitor_rft.py           # TUI-like helper to inspect/cancel/monitor jobs
    └── run_rft.py               # Launch an OpenAI RFT job
```

---

## Environment

Create a `.env` in repo root:

```
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
```

Install dependencies:

```bash
pip install -r requirements.txt
```

> Tip: `base_eval.py` can use either/both providers; RFT launch requires a valid `OPENAI_API_KEY`.

---

## Source Data Directory (before processing)

Each company (ticker) folder looks like:

```
data/<ticker>/
├── 10-k.html
├── 10-q.html
├── 8-k.json
├── def14a.json
├── earnings.html
└── qa/
    ├── Analyst_Q&A/
    ├── Compensation/
    ├── Earnings_result/Financials/
    ├── Guidance/
    ├── Industry_&_Market/
    ├── Investor_Sentiment/
    ├── Macro_&_Economics/
    ├── Management_Commentary/
    ├── Operating_metric/
    └── Risks_&_Challenges/
```

Inside each `qa/<category>/` are annotated relevance `.jsonl` files — one per question. Raw filenames are inconsistent (e.g., `_annotated.jsonl`, `_annotateds.jsonl`, `_annotated_annotated.jsonl`, `annotatedjsonl`, etc.).

---

## Pipeline Overview

```
data/<ticker>/qa/**.jsonl
        │
        ├── standardize_filenames.py     (fix inconsistent filenames safely)
        │
        ├── document_ranking.py          (build per-company document-ranking JSONL)
        │         └─> output/raw_data/<ticker>/document-ranking/{train,eval}.jsonl + stats.json
        │
        ├── chunk_ranking.py             (build per-company chunk-ranking JSONL)
        │         └─> output/raw_data/<ticker>/chunk-ranking/{train,eval}.jsonl + stats.json
        │
        └── grader.py                    (convert raw_data → QRELS datasets + combined splits)
                  └─> output/rft_data/...
```

Expected final tree:

```
output
├── raw_data
│   └── <ticker>/
│       ├── document-ranking/{train.jsonl, eval.jsonl, stats.json}
│       └── chunk-ranking/{train.jsonl, eval.jsonl, stats.json}
└── rft_data
    ├── dataset/
    │   ├── document_ranking/{train,eval}/<ticker>.jsonl
    │   └── chunk_ranking/{train,eval}/<ticker>.jsonl
    └── combined_datasets/
        ├── document_ranking_train.jsonl
        ├── document_ranking_eval.jsonl
        ├── chunk_ranking_train.jsonl
        └── chunk_ranking_eval.jsonl
```

---

## Quickstart (end‑to‑end)

From repo root:

```bash
# 0) Install deps & set API keys in .env
pip install -r requirements.txt

# 1) Standardize filenames (safe; creates data_backup/ by default)
python standardize_filenames.py

# 2) Build per-company RAW datasets
python document_ranking.py
python chunk_ranking.py

# 3) Convert to QRELS + make combined train/eval splits
python grader.py

# 4) (Optional) Run baseline evals on combined eval sets
python base_eval.py
```

---

## Script Roles & Schemas

### `standardize_filenames.py`

**Goal:** Normalize every `.jsonl` filename under `data/**/qa/**` to a consistent pattern so downstream scripts can glob and parse reliably.

* Creates a `data_backup/` copy (configurable in `main()`).
* Fixes suffix variants: `_annotated.jsonl`, `_annotateds.jsonl`, `_annotated_annotated.jsonl`, `_annotated_annotateds.jsonl`, and the `annotatedjsonl` anomaly.
* Final target pattern:

```
relevance_results_{doc_type}_filter_q{question_num}_annotateds.jsonl
```

* Has a DRY‑RUN mode. Prints a summary of renamed / conflicts / parse failures and verifies conformity afterward.

> **Why this matters:** If filenames aren’t consistent, `document_ranking.py` / `chunk_ranking.py` won’t collect anything → `output/raw_data` stays empty → `grader.py` has nothing to convert.

---

### `document_ranking.py`

**Goal:** Build per‑company **document‑ranking** samples for train/eval.

* Aggregates per‑question signals **by document type** (`def14a`, `10k`, `10q`, `8k`, `earnings`).
* Emits per‑company JSONL to `output/raw_data/<ticker>/document-ranking/{train,eval}.jsonl` and writes `stats.json`.

**Typical record (one JSON line):**

```json
{
  "messages": [ ... ],
  "document_ranking": [0,1,2,3,4],
  "document_score_counts": {
    "10k":      {"2": X, "1": Y, "0": Z},
    "10q":      {"2": X, "1": Y, "0": Z},
    "8k":       {"2": X, "1": Y, "0": Z},
    "def14a":   {"2": X, "1": Y, "0": Z},
    "earnings": {"2": X, "1": Y, "0": Z}
  },
  "metadata": {
    "category": "...",
    "question_num": 7,
    "ticker": "aapl"
  }
}
```

---

### `chunk_ranking.py`

**Goal:** Build per‑company **chunk‑ranking** samples for train/eval.

* Uses annotated chunk‑level scores per question.
* Emits per‑company JSONL to `output/raw_data/<ticker>/chunk-ranking/{train,eval}.jsonl` and writes `stats.json`.

**Typical record:**

```json
{
  "messages": [ ... ],
  "chunk_scores": [0,0,2,0,1, ...],
  "metadata": {
    "category": "...",
    "question_num": 7,
    "ticker": "aapl"
  }
}
```

---

### `grader.py` (QRELS conversion + combined splits)

**Goal:** Convert `output/raw_data/**` into **QRELS** datasets usable by both baseline evaluation and RFT, and also produce combined splits.

**Outputs**

1. **Per‑company QRELS files**

```
output/rft_data/dataset/
├── document_ranking/{train,eval}/<ticker>.jsonl
└── chunk_ranking/{train,eval}/<ticker>.jsonl
```

2. **Combined splits** (balanced 80/20 across company×category — if enough samples):

```
output/rft_data/combined_datasets/
├── document_ranking_train.jsonl
├── document_ranking_eval.jsonl
├── chunk_ranking_train.jsonl
└── chunk_ranking_eval.jsonl
```

**QRELS format**

* **Document ranking** → `qrel` maps *document index* (`"0".."4"`) to graded relevance. Ties are handled by grouping doc types with identical `(count_2, count_1, count_0)` signatures; tied groups receive the same relevance score.
* **Chunk ranking** → `qrel` maps *chunk index* (`"0".."N-1"`) to its graded relevance.

**Example — document QRELS line:**

```json
{
  "messages": [ ... ],
  "qrel": {"0": 1, "1": 3, "2": 2, "3": 0, "4": 0},
  "metadata": {
    "task_type": "document_ranking",
    "num_documents": 5,
    "relevant_docs": 3,
    "category": "Guidance",
    "ticker": "aapl"
  }
}
```

**Example — chunk QRELS line:**

```json
{
  "messages": [ ... ],
  "qrel": {"0": 0, "1": 0, "2": 2, "3": 0, "4": 1, "5": 0},
  "metadata": {
    "task_type": "chunk_ranking",
    "num_chunks": 6,
    "positive_chunks": 2,
    "category": "Risks_&_Challenges",
    "ticker": "aapl"
  }
}
```

**Quality filters**

* Drop samples with **no learning signal**:

  * all‑zero `qrel`
  * all same non‑zero values (no preference signal)
* Deterministic tie‑handling for documents.

**Note on summary printing:** If you saw a `ZeroDivisionError` (`filter rate` when `total_samples==0`), guard the print with:

```python
rate = (self.stats["all_zero_filtered"]/self.stats["total_samples"]*100) if self.stats["total_samples"] else 0.0
print(f"Filter rate: {rate:.1f}%")
```

---

## Baseline Evaluation (`base_eval.py`)

Evaluate a (reasoning) baseline model against the **combined eval** splits using nDCG\@K, MAP\@K, MRR\@K.

**Inputs**

* `output/rft_data/combined_datasets/{document_ranking_eval.jsonl, chunk_ranking_eval.jsonl}`

**How it works**

1. Loads eval JSONL.
2. For each sample, sends the original prompt (`messages[0].content]`) to the model.
3. Tries to parse a ranking of indices from the response.
4. Computes metrics (`k=5` for documents, `k=10` for chunks).
5. Writes logs & summaries under `output/rft_data/baseline_results/`.

**Run**

```bash
python base_eval.py                   # defaults: both tasks, 20 samples, OpenAI o4-mini
```

**Artifacts**

* `output/rft_data/baseline_results/baseline_results.json`
* `output/rft_data/baseline_results/baseline_comparison.csv`
* `output/rft_data/baseline_results/logs/*.jsonl` (per-model per-task detailed logs)

**Gotchas**

* Ensure `.env` contains a valid `OPENAI_API_KEY` (and optionally `ANTHROPIC_API_KEY`).
* In `base_eval.py`, the conditional `if model_name == "o3" or "o4-mini":` is a Python truthiness trap — it always takes the first branch. Use `if model_name in {"o3", "o4-mini"}:` or separate conditionals.
* OpenAI parameters differ between reasoning vs non‑reasoning models. Keep `max_tokens` vs `max_completion_tokens` consistent with the target SDK version/model family.

---

## RFT: Training & Monitoring

The `rft/` folder contains three helpers to launch and manage an OpenAI **Reinforcement Fine‑Tuning** job with a custom Python grader.

### 1) Prepare `rft/data/` from combined datasets

`run_rft.py` expects two files:

```
rft/data/
├── train.jsonl
└── val.jsonl
```

You can **copy** or **symlink** from the combined datasets produced by `grader.py` (choose either document or chunk task):

**Document ranking:**

```bash
mkdir -p rft/data
cp output/rft_data/combined_datasets/document_ranking_train.jsonl rft/data/train.jsonl
cp output/rft_data/combined_datasets/document_ranking_eval.jsonl  rft/data/val.jsonl
```

**Chunk ranking (alternative):**

```bash
mkdir -p rft/data
cp output/rft_data/combined_datasets/chunk_ranking_train.jsonl rft/data/train.jsonl
cp output/rft_data/combined_datasets/chunk_ranking_eval.jsonl  rft/data/val.jsonl
```

> The grader provided (`rft/grader/ndcg_grader.py`) works for both tasks. It reads `item["qrel"]` and parses the model’s `output_text` as a list of indices.

### 2) Launch the RFT job

Edit `rft/run_rft.py` as needed:

* `MODEL_ID`: snapshot ID (e.g., `o4-mini-2025-04-16`).
* Hyperparameters under `"reinforcement" → "hyperparameters"` (`n_epochs`, `batch_size`, `compute_multiplier`, `reasoning_effort`, `eval_interval`, `eval_samples`).

Run:

```bash
python rft/run_rft.py
```

This will:

1. Upload `rft/data/{train,val}.jsonl`.
2. Read `rft/grader/ndcg_grader.py` and attach it as a Python grader.
3. Create a fine‑tune job with method `reinforcement`.
4. Poll for status and stream new events. JSON artifacts saved under `output/`.

### 3) Monitor and manage jobs

Interactive manager:

```bash
python rft/monitor_rft.py
```

Menu options:

1. **List jobs** (shows created time, status, job id)
2. **View job** (full details)
3. **View job + recent events**
4. **List events** for a job
5. **Cancel job**
6. **Monitor job with budget** (cancel if cost/time exceeds a specified threshold)

---

## Data & Grader Details

### nDCG grader (`rft/grader/ndcg_grader.py`)

* Parses indices from `sample["output_text"]` using a couple of robust regexes.
* Computes **nDCG\@5** for `document_ranking` and **nDCG\@10** for `chunk_ranking` using graded relevance and log₂ discount.
* Returns a float in `[0,1]` as the reward.

---

## One‑liner Recap

```bash
pip install -r requirements.txt && \
python standardize_filenames.py && \
python document_ranking.py && \
python chunk_ranking.py && \
python grader.py && \
mkdir -p rft/data && \
cp output/rft_data/combined_datasets/document_ranking_train.jsonl rft/data/train.jsonl && \
cp output/rft_data/combined_datasets/document_ranking_eval.jsonl  rft/data/val.jsonl && \
python rft/run_rft.py
```
