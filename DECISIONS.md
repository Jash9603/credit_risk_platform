# Design Decision Log

Running record of every non-obvious choice made building this platform, why it was made, what
was rejected, and how to defend it. The brief asks for a "brief note on major design decisions"
— this file is that note. It doubles as interview preparation: **the "Interview angle" line on
each entry is the answer to the question that decision invites.**

Format: newest decisions appended at the bottom of each topic section.

---

## Foundations & Setup

### D1. The specific brief governs, not the generic one
**Decision.** Two assignment PDFs were issued and they conflict. `NeoStats_AI_Use_Case.pdf`
(dataset-specific) is treated as authoritative over `NeoStats_Candidate_Assignment.pdf`
(generic template).

**Why.** The specific document names the actual dataset, mandates a folder structure, and adds
two modules the generic one omits entirely: **Explainable AI** and **Rule Derivation**. Building
only to the generic brief would silently drop two required modules. Building to the specific
brief is a superset — it satisfies both.

**Rejected.** Averaging the two, or building to the shorter one to save time.

**Interview angle.** *"You were given contradictory requirements — what did you do?"* Answer:
identified the conflict explicitly, chose the more specific and more recent artifact as
authoritative, verified the choice was a superset so nothing from the other document was lost,
and documented the reasoning rather than quietly picking one.

---

### D2. PostgreSQL as a compose service, not SQLite
**Decision.** The talk-to-data layer queries PostgreSQL 16 running as its own container.

**Why.**
1. The brief asks `docker-compose.yml` to *"orchestrate all services"* — with SQLite there is
   nothing to orchestrate, and the dockerisation criterion (10%) has little to assess.
2. NL-to-SQL generated against a real Postgres dialect is a more honest demonstration than
   SQLite's permissive subset. Window functions, `FILTER`, proper types all behave correctly.
3. Statement-level timeouts and read-only role grants are available as genuine safety controls,
   which strengthens the SQL-validation story.

**Cost accepted.** A one-time ~3–5 minute ingest on first `docker compose up`, mitigated by a
named volume so it only happens once, and an idempotent loader so re-runs are safe.

**Rejected.** SQLite — lower risk and faster to stand up, but it weakens two separately scored
criteria to save perhaps an hour.

**Interview angle.** *"Why not just use SQLite for a demo?"* Answer: the safety controls that
make an LLM-generated-SQL system trustworthy in a banking context — read-only roles, statement
timeouts, per-query row caps — are properties of the database, not the application. Choosing a
real RDBMS meant the guardrails could be enforced at the layer that actually owns them.

---

### D3. FastAPI + React, with React built last
**Decision.** A FastAPI backend serving a React (Vite) frontend, rather than Streamlit.

**Why.** A real API boundary means the ML model, the SHAP explainer, and the NL-to-SQL agent are
consumable by any client, not welded to one rendering framework. It also reflects how this would
actually ship inside a bank, where the scoring service is consumed by existing systems.

**Cost accepted.** Roughly 3 hours more than Streamlit, and a third container. Mitigated by
building React **last**, so a schedule overrun costs presentation polish rather than any scored
module.

**Rejected.** Streamlit — materially faster and explicitly blessed by the brief's "lightweight"
framing. A legitimate choice; the API boundary was judged worth the hours.

**Interview angle.** *"The brief said lightweight — why build two tiers?"* Answer: separating
the inference API from the presentation layer is what makes the model reusable and independently
testable, and it cost one afternoon. Be ready to concede honestly that Streamlit would have been
the correct call under a tighter deadline — showing you know when *not* to reach for the heavier
option is worth more than defending the heavier option unconditionally.

---

### D4. API keys: the evaluator brings their own, and the app still demos without one
**Decision.** No API key is ever committed. `.env` is gitignored; `.env.example` ships with
blank placeholders and comments explaining each variable. The LLM client is provider-agnostic,
selected by `LLM_PROVIDER` (`openai` | `groq` | `gemini`). A `DEMO_MODE` flag replays cached
question → SQL → answer traces for the documented query patterns with zero API calls.

**Why.** Three separate concerns, resolved together:
1. *Key safety.* A key in git is a security incident, and graders notice. `.env.example` exists
   precisely so the evaluator supplies their own credential — the brief mandates this file.
2. *Evaluator may have no key.* The talk-to-data module is 25% of the grade plus another 15%
   for prompt/hallucination control. If it cannot run on the evaluator's machine, 40% of the
   score rests on their willingness to go get a credential. `DEMO_MODE` removes that dependency
   entirely — the feature is demonstrable offline, and screenshots in the presentation PDF back
   it up.
3. *Provider lock-in.* If one provider is rate-limited or unavailable on the day, the env var
   switches to another without a code change.

**Rejected.** Shipping a live key in the repo (security failure, and the key would be revoked
before grading anyway). Also rejected: hardcoding a single provider, which would have been
simpler but leaves the evaluator stuck if they lack that specific vendor's key.

**Interview angle.** *"How do you handle secrets?"* Answer: env-var injection, never in source
control, `.env.example` as the contract describing what the deployment needs. Then the stronger
follow-up point — the interesting problem was not hiding the key, it was that a
credential-dependent feature becomes unverifiable to anyone without a credential, so the design
includes a degraded path that still proves the feature works.

**Addendum (revisited while designing the talk-to-data feature).** Considered committing a real,
rate-limited Groq free-tier key directly so `DEMO_MODE` could stay fully live instead of
replaying anything. Rejected: a key in a **public** GitHub repo is picked up by scraper bots
within minutes, gets rate-limited or the account suspended, and is *worse* than no live mode —
the evaluator would hit a dead key instead of a working replay. Resolution adopted: `DEMO_MODE`
replays **real recorded Groq responses** (actual API calls captured during development, not
hand-written text) for the 5 documented query patterns — so the automated/no-key evaluation path
still reflects genuine model behaviour. For an in-person live demo, arbitrary-question mode is
trivial to enable: export a personal key into `.env` on the machine doing the demo. No code
difference between the two paths — only which `.env` is loaded.

**Interview angle.** *"Isn't a cached demo mode faking it?"* Answer: no — the recordings are real
LLM outputs from real calls, just replayed to remove a live network/credential dependency for
graders who may not have a key. The live path is the same code, unlocked by any of three provider
keys; the recordings exist so a missing key degrades the demo instead of breaking it.

---

### D5. Superset of the mandated folder structure
**Decision.** Every path in the brief's mandated tree exists exactly where specified. Modules
the brief requires but omits from the diagram — Explainable AI, Rule Derivation, the API, the
frontend — are added as new siblings (`src/explainability/`, `src/rules/`, `api/`, `web/`)
rather than being forced into existing folders.

**Why.** The evaluator will likely diff the tree against the diagram. Every mandated file
resolves. The additions are traceable to explicit requirements elsewhere in the same document,
so they read as thoroughness rather than deviation.

**Interview angle.** *"You didn't follow the structure exactly."* Answer: the structure was
followed exactly and extended, because the diagram on page 3 does not contain a home for two
modules that pages 1–2 make mandatory. Preserving every specified path meant the deviation is
purely additive.

---

## Data Loading & Ingest

### D6. Two independent data paths from one source; schema inferred, not hand-typed
**Decision.** The CSVs are read twice, by two different consumers, for two different
purposes:
1. `src/data/loader.py` reads them directly with pandas for the ML pipeline, which needs a
   full feature matrix and has no reason to round-trip through SQL.
2. `src/data/ingest_db.py` streams them into PostgreSQL once, for the talk-to-data chatbot
   to query independently.

`application_test.csv` / `sample_submission.csv` are excluded from both paths — this
platform scores an applicant directly and returns a risk band; it does not produce a
Kaggle leaderboard submission, so there is nothing for the unlabelled test file to feed.

`sql/schema.sql` is **generated**, not hand-written (`src/data/build_schema.py`, run once).
It samples 200k rows of each real CSV to infer types rather than transcribing 300+ columns
across 7 tables by hand or trusting `HomeCredit_columns_description.csv` — which turned out
to have a real discrepancy: it names bureau's id column `SK_BUREAU_ID`, but the actual
Kaggle file header is `SK_ID_BUREAU`. Reading the real files caught this; the description
doc would not have.

**Typing rule.** ID columns → `BIGINT`, `TARGET` → `SMALLINT`, every other numeric column →
`DOUBLE PRECISION` regardless of whether pandas' sample sees `int64` or `float64` — and
every non-numeric column → `TEXT`. All identifiers are generated lowercase and unquoted.

**Why the blunt numeric typing.** Pandas upcasts an integer column with even one `NaN` to
`float64`; a 200k-row sample cannot guarantee no later row introduces a `NaN` a full scan
would have caught. Declaring a column `INTEGER` on the strength of a sample and then hitting
a `"2.0"` value during the full `COPY` fails the whole table's ingest. Collapsing every
numeric column to `DOUBLE PRECISION` removes that failure mode entirely; the cost is
cosmetic (`COUNT(*) FILTER (...)` returns `1.0` instead of `1`), not semantic — every
comparison, arithmetic operation, and aggregate the chatbot needs behaves identically.

**Why lowercase, unquoted identifiers.** Postgres folds *unquoted* identifiers to lowercase
automatically; quoting an identifier at `CREATE TABLE` time (e.g. `"SK_ID_CURR"`) instead
freezes its exact case and then requires every later reference — including every
LLM-generated query — to quote it identically or fail to resolve. Generating already-lowercase,
never-quoted DDL removes an entire class of case-sensitivity bugs before the NL-to-SQL layer is
even built.

**Why `COPY` over `INSERT`/`to_sql`.** Three of the seven tables are in the tens-of-millions
of rows (`installments_payments.csv` alone is ~13.6M rows). Row-by-row `INSERT` — including
pandas' default `to_sql` — would take an unacceptable amount of a two-day budget. `COPY`
streamed through the client connection (`cursor.copy_expert`, given a file handle) loads
the same file at native bulk-load speed and works identically whether Postgres is local or
in a container, because the stream goes through the client, not the server's filesystem.
The `COPY` column list is built from the CSV's own header, lower-cased — the file itself is
never rewritten, so mapping mismatched case between the source CSV and the target schema
costs zero extra I/O.

**Idempotency.** `ingest_table()` checks `SELECT 1 FROM {table} LIMIT 1` before loading and
skips any table that already has rows. A crashed or repeated `docker compose up` never
double-loads data, and `sql/schema.sql` is mounted at Postgres's
`/docker-entrypoint-initdb.d/`, so table creation itself is owned by Postgres's own
first-boot behaviour rather than application code — nothing in Python issues `CREATE TABLE`.

**Rejected.** Hand-typing the schema from the description CSV (slower and, as found, wrong
in at least one place); `INSERT`-based loads via SQLAlchemy `to_sql` (far too slow at this
row count); precise per-column integer/float typing (fragile against a sample that cannot
see the whole file).

**Interview angle.** *"Walk me through your data pipeline design."* Answer: two consumers,
one source, deliberately not sharing a code path — the ML pipeline optimizes for a fast
full-memory feature matrix, the chatbot's database optimizes for safe, queryable SQL. Then
the schema story: inferred from real data because the provided documentation was
demonstrably not fully reliable, typed defensively (favouring `DOUBLE PRECISION` broadly)
because a schema covering 300+ columns has to survive being generated once and never
revisited, and loaded with `COPY` because at this row count anything row-oriented is not a
viable option.

---

## Exploratory Data Analysis

*(to be filled in as decisions come up)*

## Preprocessing & Feature Engineering

*(to be filled in as decisions come up)*

## Model Training & Evaluation

*(to be filled in as decisions come up)*

## Explainability

*(to be filled in as decisions come up)*

## Rule Derivation

*(to be filled in as decisions come up)*

## Talk-to-Data (NL-to-SQL)

*(to be filled in as decisions come up)*

## Backend API

*(to be filled in as decisions come up)*

## Frontend

*(to be filled in as decisions come up)*

## Dockerization

*(to be filled in as decisions come up)*

## Documentation & Presentation

*(to be filled in as decisions come up)*

---

## Known limitations (accumulating — feeds the README's limitations section)

- Deadline is ~2 days; depth was prioritised in the areas carrying the most scoring weight,
  and presentation polish was treated as the sacrificial buffer if time ran short.
- *(more to be added as they are encountered — an honest limitations section scores better than
  a silent one, and interviewers probe it directly)*
