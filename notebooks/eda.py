# %% [markdown]
# # EDA — Home Credit Default Risk
#
# Looking at who applies for these loans, what shape the data is in, and a few patterns
# that matter for spotting risky applicants. Feeds into the model in the next phase.
#
# Quick glossary for financial terms used below:
# - **Applicant / borrower** — the person applying for the loan.
# - **Default** — the applicant didn't pay back properly (missed payments).
# - **Annuity** — the fixed amount paid back each month (like an EMI).
# - **Credit bureau** — an outside agency that tracks a person's loans across *all*
#   lenders, not just this one (similar to a CIBIL score in India).
# - **EXT_SOURCE** — a credit score from one of these outside bureaus, already
#   calculated for us. We just get the number, not the formula behind it.

# %%
import sys
from pathlib import Path

# lets this run whether launched from the repo root or from inside notebooks/
if Path.cwd().name == "notebooks":
    sys.path.insert(0, str(Path.cwd().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data.loader import load_application_train, load_table
from src.utils.config import settings

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 110
CHART_DIR = settings.artifacts_path / "eda"
CHART_DIR.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    """Small helper so every chart lands in the same place with the same naming."""
    fig.tight_layout()
    fig.savefig(CHART_DIR / name, bbox_inches="tight")
    plt.show()


# %% [markdown]
# ## 1. Dataset summary
#
# One row per loan application. `TARGET = 1` means they had trouble repaying,
# `TARGET = 0` means they paid fine — this is what we want to predict. The other tables
# (bureau, previous applications, etc.) just hold extra history on the same people.

# %%
app = load_application_train()
print(f"application_train: {app.shape[0]:,} rows, {app.shape[1]} columns")
print(f"Memory after downcasting: {app.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
app.head(3)

# %% [markdown]
# ## 2. Feature categorization
#
# 122 columns, grouped by what they describe so it's easier to reason about.

# %%
# Columns that map to exactly one category — plain data, easy to scan or extend.
EXACT_CATEGORY = {
    "SK_ID_CURR": "identifier",
    "TARGET": "target",
    "FLAG_MOBIL": "contact_flags", "FLAG_EMP_PHONE": "contact_flags",
    "FLAG_WORK_PHONE": "contact_flags", "FLAG_CONT_MOBILE": "contact_flags",
    "FLAG_PHONE": "contact_flags", "FLAG_EMAIL": "contact_flags",
    "CODE_GENDER": "demographics", "CNT_CHILDREN": "demographics",
    "CNT_FAM_MEMBERS": "demographics", "NAME_FAMILY_STATUS": "demographics",
    "NAME_EDUCATION_TYPE": "demographics", "DAYS_BIRTH": "demographics",
    "NAME_TYPE_SUITE": "demographics",
    "AMT_INCOME_TOTAL": "financials", "AMT_CREDIT": "financials",
    "AMT_ANNUITY": "financials", "AMT_GOODS_PRICE": "financials",
    "NAME_INCOME_TYPE": "financials",
    "NAME_HOUSING_TYPE": "employment_housing", "DAYS_EMPLOYED": "employment_housing",
    "OCCUPATION_TYPE": "employment_housing", "ORGANIZATION_TYPE": "employment_housing",
    "FLAG_OWN_CAR": "employment_housing", "FLAG_OWN_REALTY": "employment_housing",
    "OWN_CAR_AGE": "employment_housing",
    "WEEKDAY_APPR_PROCESS_START": "application_metadata",
    "HOUR_APPR_PROCESS_START": "application_metadata",
    "DAYS_REGISTRATION": "application_metadata", "DAYS_ID_PUBLISH": "application_metadata",
    "DAYS_LAST_PHONE_CHANGE": "application_metadata",
}

# The rest genuinely need a pattern, not an exact name — e.g. all 20 FLAG_DOCUMENT_*
# columns share one rule instead of 20 dict entries.
PATTERN_RULES = [
    (lambda c: c.startswith("FLAG_DOCUMENT_"), "document_flags"),
    (lambda c: c.startswith("EXT_SOURCE"), "external_credit_scores"),
    (lambda c: c.endswith(("_AVG", "_MODE", "_MEDI")), "building_info"),
    (lambda c: c.startswith(("REG_", "LIVE_", "REGION_")), "region_flags"),
    (lambda c: "SOCIAL_CIRCLE" in c, "social_circle"),
    (lambda c: c.startswith("AMT_REQ_CREDIT_BUREAU"), "bureau_enquiries"),
]


def categorize_column(col: str) -> str:
    if col in EXACT_CATEGORY:
        return EXACT_CATEGORY[col]
    for matches, category in PATTERN_RULES:
        if matches(col):
            return category
    return "other"


categories = pd.Series([categorize_column(c) for c in app.columns]).value_counts()
print(categories)

# %% [markdown]
# `building_info` (apartment size, floor count, etc.) is the biggest group — makes sense,
# it's really the same few building details repeated 3 ways (average / mode / median).
# Good to remember for the missing-data check next.

# %% [markdown]
# ## 3. Data quality — missing values
#
# Applicants don't have to fill in every field, so some columns are optional by design.
# The question is whether the missing pattern makes sense, not just how much is missing.

# %%
missing_pct = (app.isna().mean() * 100).sort_values(ascending=False)
missing_pct = missing_pct[missing_pct > 0]
print(f"{len(missing_pct)} of {app.shape[1]} columns have at least one missing value")
missing_pct.head(20)

# %%
top_missing = missing_pct.head(20)
fig, ax = plt.subplots(figsize=(8, 7))
top_missing.iloc[::-1].plot(kind="barh", ax=ax, color="#c0392b")
ax.set_xlabel("% missing")
ax.set_title("Top 20 columns by missing data")
save(fig, "01_missing_values_top20.png")

# %% [markdown]
# Mostly building/apartment columns, missing on 50%+ of rows — makes sense, renters or
# people living with family just don't have that info to give. Not a data error.
# `OWN_CAR_AGE` missing ~66% of the time also checks out: most people don't own a car.

# %% [markdown]
# ### The DAYS_EMPLOYED trap
#
# `DAYS_EMPLOYED` should just say "how many days ago this person started their current
# job" (as a negative number). But some rows show a huge number instead — a bug worth
# catching before it reaches the model.

# %%
anomaly_value = 365243
anomaly_share = (app["DAYS_EMPLOYED"] == anomaly_value).mean() * 100
print(f"DAYS_EMPLOYED == {anomaly_value}: {anomaly_share:.1f}% of applicants")
print(app["DAYS_EMPLOYED"].describe())

# %% [markdown]
# About 18% of rows show exactly `365243` days employed — that's not a real number, it's
# a placeholder for "this person has no current job" (retired or unemployed). If we leave
# it as-is, it throws off any average involving this column. So instead of deleting these
# rows, we mark them with their own flag — "no current job" turns out to be useful
# information on its own (see insight #5 below).

# %%
app["IS_DAYS_EMPLOYED_ANOMALY"] = (app["DAYS_EMPLOYED"] == anomaly_value).astype(int)

# %% [markdown]
# ## 4. Target imbalance
#
# How common is a default in this data? Important for how we train the model later.

# %%
target_counts = app["TARGET"].value_counts(normalize=True) * 100
print(target_counts)

fig, ax = plt.subplots(figsize=(5, 5))
labels = ["Paid on time (0)", "Had trouble paying (1)"]
ax.pie(target_counts.sort_index(), labels=labels, autopct="%1.1f%%",
       colors=["#2e8b57", "#c0392b"], startangle=90)
ax.set_title("Target distribution")
save(fig, "02_target_imbalance.png")

# %% [markdown]
# Only about 8% of applicants had repayment trouble. That's a real imbalance — a lazy
# model could just guess "will repay" every time and still look 92% accurate while being
# useless. This is why we need special handling for imbalance later when training the model.

# %% [markdown]
# ## 5. Insight #1 — outside credit scores are our strongest signal
#
# `EXT_SOURCE_1/2/3` = credit scores from outside bureaus (see glossary above). We don't
# know the exact formula, just how well the number tells the two groups apart.

# %%
ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, col in zip(axes, ext_cols):
    app.loc[app["TARGET"] == 0, col].dropna().plot.kde(ax=ax, label="Paid on time (0)", color="#2e8b57")
    app.loc[app["TARGET"] == 1, col].dropna().plot.kde(ax=ax, label="Had trouble paying (1)", color="#c0392b")
    ax.set_title(col)
    ax.set_xlabel("Score")
    ax.set_xlim(0, 1)
    ax.legend()
save(fig, "03_ext_source_by_target.png")

print(app.groupby("TARGET")[ext_cols].mean())

# %% [markdown]
# **Insight:** all three scores are clearly lower, on average, for applicants who
# defaulted. These three columns alone likely matter more than most of the rest of the
# application form put together.

# %% [markdown]
# ## 6. Insight #2 — younger applicants default more often
#
# `DAYS_BIRTH` is age stored as negative days, so converting to years first.

# %%
app["AGE_YEARS"] = -app["DAYS_BIRTH"] / 365.25
age_bins = [18, 25, 35, 45, 55, 65, 100]
age_labels = ["18-25", "26-35", "36-45", "46-55", "56-65", "65+"]
app["AGE_GROUP"] = pd.cut(app["AGE_YEARS"], bins=age_bins, labels=age_labels)

default_by_age = app.groupby("AGE_GROUP", observed=True)["TARGET"].mean() * 100

fig, ax = plt.subplots(figsize=(7, 4.5))
default_by_age.plot(kind="bar", ax=ax, color="#c0392b")
ax.set_ylabel("% had trouble paying")
ax.set_xlabel("Age group")
ax.set_title("Payment trouble by age group")
ax.tick_params(axis="x", rotation=0)
save(fig, "04_default_rate_by_age.png")

print(default_by_age)

# %% [markdown]
# **Insight:** default rate drops steadily with age — the 18-25 group defaults at roughly
# double the rate of the 56-65 group. Makes sense: more income and credit history builds
# up over time. Simple, easy-to-explain risk factor.

# %% [markdown]
# ## 7. Insight #3 — job type and education both separate risk cleanly

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

income_default = app.groupby("NAME_INCOME_TYPE")["TARGET"].mean().sort_values(ascending=False) * 100
income_default.plot(kind="barh", ax=axes[0], color="#e67e22")
axes[0].set_xlabel("% had trouble paying")
axes[0].set_title("Payment trouble by income type")

edu_default = app.groupby("NAME_EDUCATION_TYPE")["TARGET"].mean().sort_values(ascending=False) * 100
edu_default.plot(kind="barh", ax=axes[1], color="#8e44ad")
axes[1].set_xlabel("% had trouble paying")
axes[1].set_title("Payment trouble by education level")

save(fig, "05_income_education_default_rate.png")

print(income_default)
print()
print(edu_default)

# %% [markdown]
# **Insight:** "Maternity leave" and "Unemployed" applicants default far more than
# "Working" or "State servant" applicants — some segments run 3-4x the average rate.
# Education shows a clean pattern too: lower education level = roughly double the default
# rate of someone with an academic degree. Both are simple, explainable risk factors.

# %% [markdown]
# ## 8. Insight #4 — bigger loans relative to income are riskier
#
# Comparing the loan amount to yearly income — e.g. a ratio of 3 means the loan is 3x
# what the person earns in a year.

# %%
app["CREDIT_INCOME_RATIO"] = app["AMT_CREDIT"] / app["AMT_INCOME_TOTAL"]
# a few extreme outliers (very low reported income) distort the plot — clip for display only
ratio_clipped = app["CREDIT_INCOME_RATIO"].clip(upper=app["CREDIT_INCOME_RATIO"].quantile(0.99))

fig, ax = plt.subplots(figsize=(6, 5))
sns.boxplot(x=app["TARGET"], y=ratio_clipped, ax=ax, palette=["#2e8b57", "#c0392b"])
ax.set_xticklabels(["Paid on time", "Had trouble paying"])
ax.set_ylabel("Loan amount ÷ yearly income")
ax.set_title("Loan-to-income ratio by outcome")
save(fig, "06_credit_income_ratio_by_target.png")

print(app.groupby("TARGET")["CREDIT_INCOME_RATIO"].median())

# %% [markdown]
# **Insight:** applicants who defaulted were typically borrowing a bigger multiple of
# their income. Real signal, but not a strong one on its own — better used alongside the
# credit score and job-type features than as a standalone rule.

# %% [markdown]
# ## 9. Insight #5 — the "no job" flag from earlier is itself predictive
#
# Earlier we flagged the DAYS_EMPLOYED placeholder value as a data-quality issue. Turns
# out that flag alone tells us something useful about risk.

# %%
anomaly_default = app.groupby("IS_DAYS_EMPLOYED_ANOMALY")["TARGET"].mean() * 100

fig, ax = plt.subplots(figsize=(5.5, 4.5))
anomaly_default.plot(kind="bar", ax=ax, color=["#2e8b57", "#c0392b"])
ax.set_xticklabels(["Has a current job", "No current job (retired/unemployed)"], rotation=15, ha="right")
ax.set_ylabel("% had trouble paying")
ax.set_title("Payment trouble: employment status anomaly")
save(fig, "07_days_employed_anomaly_default_rate.png")

print(anomaly_default)

# %% [markdown]
# **Insight:** applicants with no current job (mostly retirees) actually default *less*
# than employed applicants — pension income is steady, a job can be lost. A data-quality
# fix turned into one of the more useful features.

# %% [markdown]
# ## 10. Insight #6 — credit bureau history (loans outside Home Credit)
#
# `bureau.csv` holds records of the applicant's *other* loans, reported by the credit
# bureau (think: their full credit report, not just what they owe us). Checking if a
# messier history elsewhere shows up as risk here too.

# %%
bureau = load_table("bureau")

bureau_summary = bureau.groupby("SK_ID_CURR").agg(
    N_PREVIOUS_CREDITS=("SK_ID_BUREAU", "count"),
    N_BAD_DEBT_OR_OVERDUE=("CREDIT_ACTIVE", lambda s: (s.isin(["Bad debt"])).sum()),
).reset_index()

app_bureau = app[["SK_ID_CURR", "TARGET"]].merge(bureau_summary, on="SK_ID_CURR", how="left")
app_bureau["N_PREVIOUS_CREDITS"] = app_bureau["N_PREVIOUS_CREDITS"].fillna(0)
app_bureau["HAS_BUREAU_HISTORY"] = (app_bureau["N_PREVIOUS_CREDITS"] > 0).astype(int)

history_default = app_bureau.groupby("HAS_BUREAU_HISTORY")["TARGET"].mean() * 100

fig, ax = plt.subplots(figsize=(5.5, 4.5))
history_default.plot(kind="bar", ax=ax, color=["#c0392b", "#2e8b57"])
ax.set_xticklabels(["No bureau history", "Has bureau history"], rotation=0)
ax.set_ylabel("% had trouble paying")
ax.set_title("Payment trouble vs. having any external credit history")
save(fig, "08_bureau_history_default_rate.png")

print(history_default)
print(f"\nApplicants with zero bureau records: {(app_bureau['N_PREVIOUS_CREDITS'] == 0).mean() * 100:.1f}%")

# %% [markdown]
# **Insight:** applicants with *no* credit history on file default *more* than those with
# some track record — "unknown risk" isn't the same as "low risk." First-time borrowers
# are worth treating as their own segment rather than just filling in a placeholder value.

# %% [markdown]
# ## Summary — the 6 insights, in plain terms
#
# 1. **Only ~8% of applicants had trouble paying** — real imbalance, needs explicit
#    handling in modeling (not something normal accuracy metrics will catch).
# 2. **External bureau scores (EXT_SOURCE_1/2/3) are the strongest single predictor** —
#    lower score, more payment trouble, consistently across all three.
# 3. **Younger applicants have more payment trouble than older ones** — age is a simple,
#    explainable risk factor.
# 4. **Income type and education level both separate risk cleanly** — unemployed/maternity
#    leave and lower education segments run well above the average trouble rate.
# 5. **Loan-to-income ratio matters, but only as one signal among several** — real but
#    modest effect on its own.
# 6. **Employment-status data quality issue turned into a feature** — the DAYS_EMPLOYED
#    placeholder value actually flags a lower-risk group (retirees), and having *zero*
#    credit bureau history is itself a mild risk signal, not a neutral "unknown."
#
# All of this feeds directly into feature engineering and gives the rule-derivation step
# a head start — most of these are already exactly the kind of plain-English "if X then
# higher risk" statement a credit policy team would recognize.
