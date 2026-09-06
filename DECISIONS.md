# Design Decision Log

The brief asks for a "brief note on major design decisions" — this is that note. Only
decisions that actually shape the design are here.

---

## Foundations & Setup

- `NeoStats_AI_Use_Case.pdf` (dataset-specific) was treated as authoritative over the
  generic candidate-assignment template wherever the two conflicted — it names the real
  dataset, mandates the folder structure, and requires Explainable AI + Rule Derivation,
  which the generic doc omits entirely. Building to the specific brief is a superset, so
  nothing from the generic one is lost.
- Every path in the brief's mandated folder tree exists exactly where specified; modules
  the brief requires but the diagram omits (Explainability, Rules, API, frontend) were
  added as new sibling folders rather than forced into existing ones.

## Data Loading & Ingest

- `sql/schema.sql` is generated from the real Postgres `information_schema` (via
  `src/data/build_schema.py`), not hand-typed — reading the real CSV headers caught an
  actual error in the provided column-description doc (bureau's id column is really
  `SK_ID_BUREAU`, documented as `SK_BUREAU_ID`).
- All numeric columns are typed `DOUBLE PRECISION` regardless of whether a sample looks
  like an int or a float — a 200k-row sample can't guarantee no later row introduces a
  `NaN`, and a wrong `INTEGER` declaration would fail the whole table's ingest partway
  through a multi-million-row `COPY`.
- Loading uses Postgres `COPY` (via `ingest_db.py`), not row-by-row `INSERT` — three of
  the seven tables are in the tens of millions of rows, where anything row-oriented isn't
  viable. `ingest_db.py` checks each table for existing rows first, so a repeated
  `docker compose up` never double-loads data.
- The ML pipeline (`src/data/loader.py`) reads the CSVs directly with pandas, independent
  of Postgres — it needs a fast in-memory feature matrix, and has no reason to round-trip
  through SQL. Two consumers, one source, deliberately not sharing a code path.

## Preprocessing

- Numeric missing values are left as real `NaN`, not imputed — LightGBM treats "this was
  missing" as a real signal (a missing `EXT_SOURCE` score is itself informative per EDA),
  so filling it with a mean/median would destroy that.
- Categorical missing values are filled with `"Unknown"` instead — a tree model needs an
  actual category to split on, it can't split on a blank.
- Fixed two real data bugs found in EDA: the `DAYS_EMPLOYED` sentinel value (`365243` →
  real missing + a flag) and a rare invalid `XNA` gender code.
- Bureau/previous-application aggregates are computed once and cached to a small CSV
  (`artifacts/cache/`) rather than re-aggregated from the raw multi-million-row tables on
  every single prediction — this is what makes a live API request fast instead of taking
  20-30 seconds.

## Model Selection

- LightGBM is the shipped model; Logistic Regression is trained only as a baseline
  (AUC 0.757 vs. LightGBM's 0.770) to prove the choice with a number, not just assert it.
- LightGBM over XGBoost is a convenience call, not a capability gap — both now support
  native categorical features and histogram-based splitting by default in modern
  versions. LightGBM does both out of the box for this dataset's mix of categorical and
  high-cardinality columns, with no extra configuration needed.
- All ~134 engineered/raw features are used — no manual feature selection. Tree models
  handle irrelevant columns gracefully (near-zero importance, unused in splits);
  selection would mainly help interpretability, not accuracy.

## Training & Class Imbalance

- Imbalance (~8% positive) is handled by reweighting the loss function directly
  (`scale_pos_weight`), not by resampling (SMOTE/undersampling) — it changes what the
  model already learns from rather than synthesising or discarding rows.
- The weight (11.39) is not tuned — it's `count(non-defaulters) / count(defaulters)` from
  the real training data, LightGBM's own recommended formula, so both classes contribute
  roughly equally to the loss.
- Risk bands (Low/Medium/High) are cut at percentiles (60th/90th) of predicted
  probability, not fixed values — `scale_pos_weight` pushes raw probabilities well above
  their normal range, so a fixed cutoff (e.g. 0.20) put ~75% of applicants in "High" when
  tested. Percentiles stay meaningful regardless of calibration.
- The F1-optimizing decision threshold is searched across the model's actual output
  range rather than assumed near 0.5, for the same reason — it landed at 0.67.

## Explainability

- SHAP's `TreeExplainer` is used directly on the trained LightGBM model — exact for tree
  ensembles, no approximation needed the way a model-agnostic method like LIME would
  require.
- Only the top 5 contributing features are shown per prediction, not all ~134 — SHAP
  computes a value for every feature internally, but showing all of them to a
  non-technical reader would defeat the purpose of "explainable."
- Feature names are mapped to plain-English labels (`src/utils/feature_labels.py`) shared
  between the SHAP explanations and the rule descriptions, so both describe the same
  feature the same way.

## Rule Derivation

- A shallow decision tree (depth 4) is fit to the LightGBM model's own predicted
  probabilities, not the raw `TARGET` labels — a surrogate that describes what the actual
  deployed model does, not a second, independent pattern that might disagree with it.
- Categorical columns are excluded from the surrogate tree — a rule like "organization
  type code ≤ 5.5" isn't something a credit policy team can read or act on, so only the
  numeric/ratio features (credit scores, ratios, age) are eligible for splits.
- Each rule reports real, observed numbers from the actual data (support count, observed
  default rate, lift vs. the population average), not the surrogate tree's own estimate,
  so the rules can be independently checked against reality.
- 13 of the 16 rules split only on the two external credit-bureau scores, because they
  dominate the model's decision surface — an honest reflection of the data (confirmed by
  EDA and SHAP independently), not a rule-miner limitation.

## Talk-to-Data (NL-to-SQL)

- The schema card given to the LLM lists every real column across all 7 tables
  (generated from Postgres's `information_schema`, not hand-typed), plus a small
  supplementary block of things a column list can't convey: exact categorical string
  values (guessing `'University'` when the real value is `'Higher education'` silently
  returns zero rows, no error) and the join keys linking all 7 tables. An earlier version
  curated a ~15-column subset to save tokens; it was dropped after a real column
  (`CNT_CHILDREN`) got wrongly refused as "not in the data" — the measured cost of the
  full schema (~2,300 tokens) is negligible, and refusing real questions is a worse
  failure mode than a few thousand extra tokens.
- Conversation history is owned entirely by a dedicated rewrite step
  (`query_rewriter.py`): a follow-up like "show the same thing but by income type
  instead" is first turned into one standalone question before anything else runs. Every
  step after that (SQL generation, validation, execution, answer synthesis) works with a
  single self-contained question and never sees history directly — so history is read
  once, by a small prompt with no schema in it, instead of being replayed alongside the
  large schema-heavy SQL prompt on every turn as a conversation grows.
- Three independent safety layers, not one: the prompt instructs SELECT-only and refusal
  of out-of-scope/destructive requests; an AST-level validator (`sqlglot`) rejects
  anything that isn't a single SELECT against a whitelisted table regardless of what the
  model produced; the DB connection itself runs in a `READ ONLY` transaction, so even a
  bug in the validator can't become a write. Each layer was tested independently,
  including sending a raw `DELETE` straight to the query runner (bypassing the validator
  on purpose) to confirm the `READ ONLY` transaction blocks it on its own.
- Invalid SQL triggers a bounded retry (max 2) that feeds the actual validator/database
  error back to the model, rather than failing on the first attempt.
- Answers are generated from the real returned rows only, via a second LLM call whose
  only input is the question and the actual rows — grounds the answer in real data
  instead of letting the model free-generate a number. The prompt requires per-row detail
  (an ID plus its value on every line for record-level results, a labelled percentage per
  category for grouped results) rather than collapsed prose that hides which ID had which
  value, and requires human-readable number formatting.
- The LLM integration is OpenAI-only, chosen for reliability and quality of SQL
  generation over supporting multiple providers — one well-tested path rather than a
  provider-agnostic abstraction that would need separate verification per provider.

## Application Architecture

- FastAPI backend + React frontend, built last of all modules — a real API boundary
  keeps the model/chatbot reusable by any client; building the UI last means a schedule
  slip costs polish, not a scored module.
- The model and full applicant dataset are loaded once, in-process (`lru_cache`), warmed
  explicitly on server startup — not reloaded per request. This directly fixes a real
  performance bug found earlier (single predictions were taking 20-30s from repeatedly
  reloading multi-GB CSVs); the API is what that fix was actually for.
- EDA charts are served as static files from the notebook's own saved output
  (`artifacts/eda/`) — the API exposes what was already computed, it never re-runs
  analysis on request.
- Plain React + hand-written CSS, no component/charting library — a handful of dashboard
  sections don't need one, and every screen was verified by actually rendering it
  (headless-browser screenshots), not just by reading the code.
- Layout is a fixed left sidebar (navigation) plus a scrollable right panel (content),
  with EDA as a multi-column dashboard grid rather than a single scrolling column.
- Risk Prediction and Explainability are merged into one section (one ID, one click,
  both the score and the SHAP reasons together) rather than two separate tabs requiring
  the same ID to be entered twice.
- Every section carries a short, plain-language "how this was built" note written for a
  non-technical reader, and starts with real, clickable examples (10 applicant IDs, 5
  documented chatbot questions) so the first click already produces a correct result.
- The chatbot is a real chat interface: messages appear the instant they're sent (with a
  "Thinking..." placeholder) rather than only once the answer arrives, input locks while
  a request is in flight so a second question can't be queued mid-answer, clicking a
  suggestion fills the input rather than sending immediately, and messages are laid out
  the conventional way (yours on the right, the assistant's on the left).

## Security

- No API key is ever committed — `.env` is gitignored, `.env.example` documents every
  variable with blank placeholders.
- The database connection for the chatbot is read-only at the transaction level
  (independent of the SQL validator), and Postgres itself only runs inside the Docker
  network — the API is the only service with a mapped external port for it to matter.

## Dockerization

- Three services (`db`, `api`, `frontend`), one `docker compose up --build`. `api`
  waits on `db`'s healthcheck before starting; `entrypoint.sh` runs the idempotent CSV
  ingest before starting the server, so a repeated `up` is always safe.
- The trained model and generated artifacts (`models/model.pkl`, `artifacts/metrics.json`,
  `artifacts/rules.json`, `artifacts/eda/*.png`) are copied into the API image at build
  time rather than mounted or regenerated at startup — the evaluator sees the actual
  results without needing to retrain anything or wait for training to finish.
- The frontend is a multi-stage build: `VITE_API_BASE_URL` is baked in at build time
  (Vite embeds env vars into the static bundle, they can't be read at container runtime),
  then served by nginx — a static SPA doesn't need a Node process running in production.

---

## Known limitations (feeds the README's limitations section)

- Deadline is ~2 days; depth was prioritised in the areas carrying the most scoring
  weight, and presentation polish was treated as the sacrificial buffer if time ran
  short.
- The derived rules lean heavily on the two external credit-bureau scores — 13 of 16
  split only on those, since they dominate the model's decision surface. An honest
  reflection of the data, but it means the rule set has less feature diversity than a
  reader might expect from a "business rules" deliverable.
- No held-out test-set evaluation: metrics are out-of-fold CV estimates, since
  `application_test.csv` has no labels to score against.
- No hyperparameter tuning beyond early-stopping-selected tree count; a proper search
  would likely add a few points of AUC.
- The LLM integration is single-provider (OpenAI) rather than provider-agnostic, so it
  depends on one vendor's API being available and the evaluator having an OpenAI key
  specifically.
