"""Distil the LightGBM model into a small set of plain-language IF-THEN rules.

A shallow decision tree is fit not to the real TARGET labels, but to the LightGBM model's
own predicted probabilities. That makes it a *surrogate* that approximates what the
actual deployed model does — a description of the real model's behaviour, not a second,
independent pattern-finder that might disagree with it.

Categorical columns are left out of the surrogate tree on purpose: a rule like
"organization type code <= 5.5" isn't something a credit policy team could read or act
on, whereas the numeric/ratio features (credit scores, ratios, age) split into thresholds
a human can use directly.

Run: `python -m src.rules.rule_miner`
"""
import json

import joblib
from sklearn.tree import DecisionTreeRegressor, _tree

from src.data.preprocessor import build_feature_matrix
from src.utils.config import settings
from src.utils.feature_labels import readable_name
from src.utils.logger import get_logger

logger = get_logger(__name__)

MAX_DEPTH = 4
MIN_SAMPLES_LEAF = 1000


def load_model_bundle():
    return joblib.load(settings.models_path / "model.pkl")


def fit_surrogate_tree(X, predicted_probability):
    numeric_cols = X.select_dtypes(exclude="category").columns.tolist()
    tree = DecisionTreeRegressor(
        max_depth=MAX_DEPTH, min_samples_leaf=MIN_SAMPLES_LEAF, random_state=42
    )
    tree.fit(X[numeric_cols], predicted_probability)
    return tree, numeric_cols


def extract_rules(tree, feature_names, X, y) -> list:
    """Walk the fitted tree's structure to recover the IF/THEN condition behind each
    leaf, then measure that segment against the real data — support (how many
    applicants), observed default rate, and lift versus the population average."""
    tree_ = tree.tree_
    base_rate = float(y.mean())

    leaf_conditions = {}

    def walk(node_id, conditions):
        if tree_.feature[node_id] != _tree.TREE_UNDEFINED:
            feature = feature_names[tree_.feature[node_id]]
            threshold = tree_.threshold[node_id]
            walk(tree_.children_left[node_id], conditions + [(feature, "<=", threshold)])
            walk(tree_.children_right[node_id], conditions + [(feature, ">", threshold)])
        else:
            leaf_conditions[node_id] = conditions

    walk(0, [])

    leaf_ids = tree.apply(X[feature_names])
    rules = []
    for leaf_id, conditions in leaf_conditions.items():
        mask = leaf_ids == leaf_id
        support = int(mask.sum())
        if support == 0:
            continue
        observed_rate = float(y[mask].mean())
        rules.append({
            "conditions": [
                {"feature": f, "op": op, "threshold": round(float(t), 3)}
                for f, op, t in conditions
            ],
            "support": support,
            "observed_default_rate": round(observed_rate, 4),
            "lift_vs_average": round(observed_rate / base_rate, 2) if base_rate else None,
        })

    rules.sort(key=lambda r: r["observed_default_rate"], reverse=True)
    return rules


def format_rule(rule: dict) -> str:
    condition_text = " AND ".join(
        f"{readable_name(c['feature'])} {c['op']} {c['threshold']:.2f}"
        for c in rule["conditions"]
    )
    return (
        f"IF {condition_text} "
        f"THEN observed default rate is {rule['observed_default_rate'] * 100:.1f}% "
        f"({rule['support']:,} applicants, {rule['lift_vs_average']}x the average rate)"
    )


def main():
    bundle = load_model_bundle()
    model = bundle["model"]

    matrix = build_feature_matrix()
    X = matrix[bundle["feature_names"]]
    y = matrix["TARGET"]

    predicted_probability = model.predict_proba(X)[:, 1]
    tree, numeric_cols = fit_surrogate_tree(X, predicted_probability)
    rules = extract_rules(tree, numeric_cols, X, y)

    settings.artifacts_path.mkdir(parents=True, exist_ok=True)
    with open(settings.artifacts_path / "rules.json", "w", encoding="utf-8") as f:
        json.dump(rules, f, indent=2)

    print(f"Extracted {len(rules)} rules\n")
    for rule in rules:
        print(format_rule(rule))


if __name__ == "__main__":
    main()
