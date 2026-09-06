# Credit Risk Intelligence Platform

An end-to-end credit risk assessment platform built on the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) dataset. It combines a trained ML model, SHAP-based explainability, decision-tree-derived business rules, and a natural-language chatbot that lets analysts query the underlying data conversationally — all behind a single Docker Compose command.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Docker Compose                           │
│                                                                 │
│  ┌──────────┐    ┌───────────────────────────┐    ┌──────────┐  │
│  │  React   │───▶│     FastAPI Backend       │───▶│ Postgres │  │
│  │ Frontend │    │                           │    │  (data)  │  │
│  │ (Nginx)  │    │  /predict  /explain       │    └──────────┘  │
│  │ :5173    │    │  /eda      /rules         │         ▲        │
│  └──────────┘    │  /chat     /health        │         │        │
│                  │           :8000            │    CSV ingest    │
│                  └─────────┬─────────────────┘    (auto on      │
│                            │                       first boot)  │
│                     ┌──────┴──────┐                             │
│                     │  model.pkl  │                             │
│                     │  (LightGBM) │                             │
│                     └─────────────┘                             │
└─────────────────────────────────────────────────────────────────┘
```

### Component Breakdown

| Component | Path | Purpose |
|---|---|---|
| **Data loading** | `src/data/loader.py` | Loads Home Credit CSVs with memory optimization |
| **Preprocessing** | `src/data/preprocessor.py` | Anomaly fixes, day→year conversion, feature engineering, bureau/prev-app aggregates |
| **Training** | `src/ml/train.py` | Baseline (Logistic Regression) + LightGBM with 5-fold stratified CV |
| **Evaluation** | `src/ml/evaluate.py` | ROC-AUC, PR-AUC, KS statistic, F1-optimized threshold, risk banding |
| **Prediction** | `src/ml/predict.py` | Single-applicant scoring: probability → risk score (0-1000) → risk band |
| **Explainability** | `src/explainability/shap_explainer.py` | Per-prediction SHAP TreeExplainer with human-readable feature sentences |
| **Rule derivation** | `src/rules/rule_miner.py` | Surrogate decision tree that distils the model into IF-THEN rules |
| **Talk-to-data** | `src/talk_to_data/` | NL→SQL chatbot: rewrite → generate → validate → execute → answer |
| **API** | `api/` | FastAPI REST endpoints tying all modules together |
| **Frontend** | `frontend/` | React SPA with EDA dashboard, predict+explain, rules, and chat tabs |
| **Database** | `sql/schema.sql` | PostgreSQL schema for all 7 Home Credit tables |

---

## Quick Start (Docker — recommended)

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- An [OpenAI API key](https://platform.openai.com/api-keys) (for the chatbot)
- Home Credit CSVs downloaded from [Kaggle](https://www.kaggle.com/c/home-credit-default-risk/data)

### Steps

```bash
# 1. Clone the repo
git clone https://github.com/Jash9603/credit_risk_platform.git
cd credit_risk_platform

# 2. Drop the Kaggle CSVs into the data/ folder
#    (application_train.csv, bureau.csv, bureau_balance.csv,
#     previous_application.csv, POS_CASH_balance.csv,
#     credit_card_balance.csv, installments_payments.csv)

# 3. Create a .env file from the template
cp .env.example .env
#    Edit .env and set your OPENAI_API_KEY

# 4. Launch everything
docker compose up --build
```

On first boot, the system will automatically:
1. **Create the PostgreSQL database** and run `sql/schema.sql`
2. **Ingest all 7 CSV files** into Postgres (idempotent — skips tables already loaded)
3. **Start the API server** (warms model + data caches on startup)
4. **Serve the React frontend** via Nginx

Access the platform at:
- **Frontend**: http://localhost:5173
- **API docs**: http://localhost:8000/docs

### Local Development (without Docker)

```bash
# Backend
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt
# Start Postgres separately, then:
python -m src.data.ingest_db                      # Load CSVs into Postgres
uvicorn api.main:app --reload --port 8000

# Frontend
cd frontend && npm ci && npm run dev
```

---

## Model Selection Rationale

### Why LightGBM?

| Criterion | Logistic Regression | LightGBM ✅ |
|---|---|---|
| Handles missing values natively | ❌ Requires imputation | ✅ Uses missingness as a signal |
| Categorical features | Needs one-hot encoding | Native categorical splits |
| Non-linear interactions | Cannot capture | Tree-based, captures automatically |
| ROC-AUC on this dataset | **0.7572** | **0.7695** |
| Interpretability | High (coefficients) | Moderate (but compensated by SHAP) |

Logistic Regression was trained as a **baseline** to validate that the gradient-boosted model is actually earning its complexity. The 1.2pp AUC improvement confirms LightGBM captures non-linear patterns that a linear model misses — while SHAP explanations restore the per-prediction interpretability that matters in credit risk.

### Class Imbalance Strategy: `scale_pos_weight`

The dataset is ~92% non-default / ~8% default. We use **`scale_pos_weight = 11.39`** (the ratio of negatives to positives) rather than resampling (SMOTE, undersampling). The reasoning:

| Approach | Tradeoff |
|---|---|
| **SMOTE** | Synthesises minority samples — can create unrealistic data points in high-dimensional space, harder to reproduce |
| **Undersampling** | Discards majority-class data — wastes information |
| **`scale_pos_weight`** ✅ | Reweights the loss function to penalise misclassified defaults more heavily — uses all real data, fully deterministic, one parameter |

### Key Training Decisions

- **5-fold Stratified CV** with early stopping (patience=100 rounds) to prevent overfitting and determine the optimal number of boosting rounds (374 estimators selected)
- **F1-optimized threshold** (0.67) instead of the default 0.50 — `scale_pos_weight` shifts predicted probabilities upward, so the natural decision boundary is no longer at 0.5
- **Percentile-based risk bands** (Low: bottom 60%, Medium: next 30%, High: top 10%) instead of fixed cutoffs — stays meaningful regardless of probability calibration

---

## Preprocessing & Feature Engineering

### Data Cleaning
1. **Employment anomaly**: `DAYS_EMPLOYED = 365243` is a sentinel for "no current job" (retired/unemployed) — replaced with `NaN` and flagged via `IS_DAYS_EMPLOYED_ANOMALY`
2. **Invalid gender code**: `CODE_GENDER = "XNA"` (4 rows) replaced with `NaN`
3. **Days → Years**: All `DAYS_*` columns (negative day-offsets) converted to positive years for readability (e.g., `DAYS_BIRTH → AGE_YEARS`)

### Engineered Features
| Feature | Formula | Why |
|---|---|---|
| `CREDIT_INCOME_RATIO` | `AMT_CREDIT / AMT_INCOME_TOTAL` | Loan size relative to income — core affordability measure |
| `ANNUITY_INCOME_RATIO` | `AMT_ANNUITY / AMT_INCOME_TOTAL` | Monthly payment burden |
| `CREDIT_TERM` | `AMT_ANNUITY / AMT_CREDIT` | Effective repayment rate |
| `GOODS_CREDIT_RATIO` | `AMT_GOODS_PRICE / AMT_CREDIT` | Whether the loan exceeds the goods' value |
| `EMPLOYED_AGE_RATIO` | `YEARS_EMPLOYED / AGE_YEARS` | Employment stability relative to life stage |

### Bureau & Previous Application Aggregates
- **Bureau features**: `N_BUREAU_LOANS`, `N_BUREAU_ACTIVE`, `N_BUREAU_BAD_DEBT`, `AVG_BUREAU_CREDIT_SUM` — summarise the applicant's history with *other* lenders
- **Previous application features**: `N_PREV_APPLICATIONS`, `N_PREV_REFUSED`, `N_PREV_APPROVED`, `PREV_REFUSED_RATIO` — summarise the applicant's prior history with Home Credit itself
- Aggregates are **computed once and cached** to CSV (`artifacts/cache/`) — a single prediction shouldn't re-aggregate 3.4M rows

### Why NaNs Are Intentionally Preserved
Numeric columns keep their real `NaN` values — LightGBM handles missing values natively and often makes good use of "this value was missing" as a predictive signal. Median/mean imputation would destroy that information. Categorical columns are filled with `"Unknown"` since tree models need a concrete category to split on.

---

## Evaluation Metrics & Results

All metrics are computed on **out-of-fold predictions** from 5-fold CV (never on training data):

| Metric | Value | Why This Metric |
|---|---|---|
| **ROC-AUC** | **0.7695** | Standard rank-based discrimination metric |
| **PR-AUC** | **0.2557** | More informative than ROC-AUC under class imbalance — measures precision-recall tradeoff |
| **KS Statistic** | **0.402** | Credit-risk industry standard — max separation between TPR and FPR curves |
| **F1 Score** | **0.3169** | At the optimized threshold (0.67) |
| **Precision** | **0.26** | At threshold — conservative, catches more defaults at cost of false positives |
| **Recall** | **0.4059** | At threshold — captures ~41% of actual defaults |

### Confusion Matrix (at threshold = 0.67)

|  | Predicted No Default | Predicted Default |
|---|---|---|
| **Actual No Default** | 254,003 (TN) | 28,683 (FP) |
| **Actual Default** | 14,749 (FN) | 10,076 (TP) |

### Risk Band Distribution

| Band | Count | Range |
|---|---|---|
| Low | 184,506 (60%) | probability < 0.417 |
| Medium | 92,253 (30%) | 0.417 ≤ probability < 0.702 |
| High | 30,752 (10%) | probability ≥ 0.702 |

### Baseline Comparison

| Model | ROC-AUC |
|---|---|
| Logistic Regression (baseline) | 0.7572 |
| **LightGBM (deployed)** | **0.7695** |

---

## Prompt Engineering & Token Optimization

The talk-to-data chatbot converts natural language questions to SQL queries against the 7-table PostgreSQL database. Three design decisions keep it accurate and cost-efficient:

### 1. Schema Card (Context Grounding)

The SQL generation prompt includes a **schema card** — the complete column listing for all 7 tables, fetched dynamically from Postgres's `information_schema` (not hand-typed). This is supplemented with:

- **Exact categorical string values** (e.g., `'Higher education'`, not `'University'`) — prevents silent zero-row queries from guessed values
- **Known data quirks** (the `DAYS_EMPLOYED = 365243` sentinel, negative day-offset convention)
- **Join key mappings** across all tables

### 2. Query Rewriter Node (Token Optimization)

Multi-turn conversations are handled by a dedicated **query rewriter** that resolves follow-up references ("show me the same thing but by income type") into standalone questions *before* SQL generation.

**Why this matters for token cost**: Without this, every SQL generation call would need the entire growing conversation history appended alongside the schema-heavy prompt. With the rewriter, the history is read once by a small, schema-free prompt, and every downstream step only pays for one short standalone question. Token cost stays flat as conversations grow.

### 3. Defense-in-Depth SQL Validation

Generated SQL passes through an **AST-level validator** (using `sqlglot`) before execution:

- Enforces **SELECT-only** — rejects INSERT/UPDATE/DELETE/DDL regardless of what the LLM produces
- **Allowlists tables** — only the 7 known tables can be queried
- **Caps row count** (LIMIT 200) — prevents runaway queries on multi-million-row tables
- On validation or execution failure, the error is **fed back to the LLM** for a retry (up to 2 retries)

### 4. Answer Synthesis

Query results are passed to a separate **answer synthesis prompt** that formats numbers as humans read them (percentages, thousands separators, sensible rounding) and handles edge cases (empty results, single vs. multi-row, category-grouped vs. unique-ID rows).

### Prompt Design Principles

- **Few-shot examples** covering common query patterns (aggregations, joins, grouping, filtering, refusals)
- **Explicit "chain of thought" comments** in complex examples (e.g., "average count of X per Y needs count-then-average")
- **Strict output format** — "Output ONLY the SQL query, no explanation, no markdown fences"
- **Safe refusal path** — `NO_SQL:` prefix for unanswerable/destructive queries

---

## Rule Derivation Logic

Business rules are extracted using a **surrogate decision tree** approach:

1. The deployed LightGBM model's predicted probabilities are treated as the target
2. A shallow `DecisionTreeRegressor` (max_depth=4, min_samples_leaf=1000) is fit to approximate the LightGBM's behavior using only numeric features
3. Each leaf's path becomes an IF-THEN rule, measured against the *real* default labels

**Why a surrogate tree instead of direct LightGBM extraction?**
- LightGBM contains 374 trees with complex interactions — not human-readable
- A depth-4 surrogate produces ~16 rules a credit policy team can actually read and act on
- Rules describe *what the deployed model does*, not a second independent model that might disagree

**Why only numeric features?**
Categorical columns are excluded because splits like "organization type code ≤ 5.5" are meaningless to a business user, while "External Score 3 ≤ 0.477" is directly actionable.

### Sample Rules (sorted by default rate)

| Rule | Default Rate | Lift | Support |
|---|---|---|---|
| IF Ext Score 3 ≤ 0.477 AND Ext Score 2 ≤ 0.468 AND Ext Score 3 ≤ 0.266 AND Ext Score 2 ≤ 0.208 | **33.6%** | 4.16× | 4,378 |
| IF Ext Score 3 ≤ 0.477 AND Ext Score 2 ≤ 0.468 AND Ext Score 3 > 0.266 AND Ext Score 2 ≤ 0.160 | **22.8%** | 2.82× | 9,076 |
| IF Ext Score 3 > 0.477 AND Ext Score 2 > 0.498 AND Goods/Credit Ratio > 0.881 AND Ext Score 2 > 0.646 | **2.1%** | 0.25× | 30,501 |

The full 16-rule set is in `artifacts/rules.json` with conditions, support, observed default rate, and lift vs. the population average (8.07%).

---

## Project Structure

```
credit_risk_platform/
├── data/                           # Home Credit CSVs (not committed — mounted as Docker volume)
├── notebooks/
│   ├── eda.ipynb                   # Exploratory Data Analysis
│   └── eda.py                      # Converted .ipynb to .py
├── src/
│   ├── data/
│   │   ├── loader.py               # Load and join dataset tables
│   │   ├── preprocessor.py         # Cleaning, encoding, feature engineering
│   │   ├── build_schema.py         # Generate sql/schema.sql from CSVs
│   │   └── ingest_db.py            # CSV → PostgreSQL idempotent loader
│   ├── ml/
│   │   ├── train.py                # Model training pipeline
│   │   ├── predict.py              # Inference and scoring
│   │   └── evaluate.py             # ROC-AUC, PR-AUC, KS, risk bands
│   ├── explainability/
│   │   └── shap_explainer.py       # Per-prediction SHAP explanations
│   ├── rules/
│   │   └── rule_miner.py           # Surrogate tree → IF-THEN rules
│   ├── talk_to_data/
│   │   ├── nl_to_sql.py            # NL → SQL orchestration pipeline
│   │   ├── query_rewriter.py       # Multi-turn history → standalone question
│   │   ├── llm_client.py           # OpenAI API wrapper
│   │   ├── prompt_templates.py     # Versioned prompt templates
│   │   ├── schema_card.py          # Dynamic schema from Postgres
│   │   ├── sql_validator.py        # AST-level SQL safety (sqlglot)
│   │   └── query_runner.py         # Execute and return SQL results
│   └── utils/
│       ├── config.py               # Pydantic settings from .env
│       ├── logger.py               # Structured logging
│       ├── helpers.py              # Memory reduction, timing utilities
│       └── docker_utils.py         # Port readiness checks
├── api/
│   ├── main.py                     # FastAPI app with CORS, static mounts
│   ├── schemas.py                  # Pydantic request/response models
│   └── routers/                    # /eda, /predict, /explain, /rules, /chat, /health
├── frontend/                       # React + Vite frontend
├── sql/
│   └── schema.sql                  # PostgreSQL DDL for all 7 tables
├── models/
│   └── model.pkl                   # Saved LightGBM model + metadata
├── artifacts/
│   ├── metrics.json                # Evaluation metrics
│   ├── rules.json                  # Extracted business rules
│   └── eda/                        # Saved EDA chart PNGs
├── Dockerfile                      # API container (Python 3.11)
├── docker-compose.yml              # 3-service stack (db + api + web)
├── entrypoint.sh                   # Auto-ingest data, then start uvicorn
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variable template
└── .gitignore
```

---

## Known Limitations & Possible Improvements

### Limitations
- **No holdout test set evaluation**: Metrics are out-of-fold CV estimates on the training set. `application_test.csv` has no labels, so true out-of-time generalization cannot be measured
- **No hyperparameter tuning**: LightGBM uses sensible defaults with early stopping — a proper Optuna/Bayesian search would likely squeeze 1-3pp more AUC
- **Single model**: No ensemble (stacking, blending) — would improve accuracy but adds complexity
- **Schema card token cost**: The full schema card (~800 tokens) is injected on every SQL generation call. For very high-traffic production use, caching or compression would be needed
- **No probability calibration**: `scale_pos_weight` shifts raw probabilities — Platt scaling or isotonic regression would improve calibration for absolute probability interpretation

### Possible Improvements
- **Feature selection**: Use Boruta or recursive feature elimination to reduce the 100+ feature set — faster inference, potentially better generalization
- **Time-based features**: Aggregate temporal patterns from `bureau_balance`, `pos_cash_balance`, `credit_card_balance` (monthly snapshots over time) — currently unused
- **Model monitoring**: Add prediction drift detection (PSI on feature distributions) and automated retraining triggers
- **RAG-based chatbot**: Replace the schema card with a retrieval step that only surfaces relevant tables/columns per query — would reduce token usage for complex schemas
- **Caching SQL queries**: Frequently asked questions could be cached to avoid repeated LLM calls

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | *(required)* | OpenAI API key for the chatbot |
| `LLM_MODEL` | `gpt-4o-mini` | OpenAI model used for NL→SQL |
| `POSTGRES_HOST` | `localhost` | PostgreSQL host |
| `POSTGRES_PORT` | `5432` | PostgreSQL port |
| `POSTGRES_DB` | `credit_risk` | Database name |
| `POSTGRES_USER` | `credit_risk_app` | Database user |
| `POSTGRES_PASSWORD` | `change_me` | Database password |
| `API_PORT` | `8000` | FastAPI server port |
| `CORS_ORIGINS` | `http://localhost:5173` | Allowed CORS origins |

---

## License

This project was developed as a use case assignment submission. The dataset is provided by Home Credit under the Kaggle competition terms.
