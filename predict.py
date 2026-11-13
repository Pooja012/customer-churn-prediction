"""Predict churn risk for a single customer.

Usage:
    python predict.py                                     # uses the high-risk example
    python predict.py --file examples/low_risk_customer.json
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "model" / "churn_model.pkl"
DEFAULT_EXAMPLE = ROOT / "examples" / "high_risk_customer.json"
NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def load_bundle(path=MODEL_PATH):
    """Load the saved pipeline and decision threshold."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run notebooks/03_modeling.ipynb to create it."
        )
    return joblib.load(path)


def allowed_values(pipeline):
    """Return {column: [valid categories]} from the fitted encoder."""
    prep = pipeline.named_steps["prep"]
    encoder = prep.named_transformers_["cat"]
    cat_cols = [t for t in prep.transformers_ if t[0] == "cat"][0][2]
    return {
        col: [x.item() if hasattr(x, "item") else x for x in cats]
        for col, cats in zip(cat_cols, encoder.categories_)
    }


def validate(customer, pipeline):
    """Raise a clear error if the input is incomplete or has invalid values."""
    required = list(pipeline.feature_names_in_)
    missing = [c for c in required if c not in customer]
    if missing:
        raise ValueError(f"Missing fields: {missing}")

    problems = []
    for col in NUM_COLS:
        v = customer[col]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0:
            problems.append(f"{col} must be a non-negative number (got {v!r})")
    for col, allowed in allowed_values(pipeline).items():
        if customer[col] not in allowed:
            problems.append(f"{col}={customer[col]!r} is not one of {allowed}")
    if problems:
        raise ValueError("; ".join(problems))


def predict_churn(customer, bundle=None):
    """Return churn probability, the decision, and a risk level for one customer."""
    bundle = bundle or load_bundle()
    pipeline, threshold = bundle["pipeline"], bundle["threshold"]
    validate(customer, pipeline)

    X = pd.DataFrame([customer])[list(pipeline.feature_names_in_)]
    prob = float(pipeline.predict_proba(X)[0, 1])

    if prob >= threshold:
        risk = "High"
    elif prob >= threshold / 2:
        risk = "Medium"
    else:
        risk = "Low"

    return {
        "churn_probability": round(prob, 3),
        "threshold": threshold,
        "will_churn": prob >= threshold,
        "risk_level": risk,
    }


def main():
    parser = argparse.ArgumentParser(description="Predict customer churn risk.")
    parser.add_argument("--file", default=str(DEFAULT_EXAMPLE),
                        help="Path to a JSON file with the customer's details")
    args = parser.parse_args()

    try:
        with open(args.file) as f:
            customer = json.load(f)
        result = predict_churn(customer)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    verdict = "Likely to churn" if result["will_churn"] else "Likely to stay"
    print(f"Churn probability : {result['churn_probability']:.1%}")
    print(f"Decision threshold: {result['threshold']}")
    print(f"Prediction        : {verdict} ({result['risk_level']} risk)")


if __name__ == "__main__":
    main()