import json
from pathlib import Path

import pytest

from predict import predict_churn

EXAMPLES = Path(__file__).parent.parent / "examples"


def load(name):
    with open(EXAMPLES / name) as f:
        return json.load(f)


def test_output_format():
    result = predict_churn(load("high_risk_customer.json"))
    assert 0 <= result["churn_probability"] <= 1
    assert result["risk_level"] in {"Low", "Medium", "High"}


def test_high_risk_scores_above_low_risk():
    high = predict_churn(load("high_risk_customer.json"))
    low = predict_churn(load("low_risk_customer.json"))
    assert high["churn_probability"] > low["churn_probability"]


def test_invalid_category_is_rejected():
    customer = load("high_risk_customer.json")
    customer["Contract"] = "Monthly"
    with pytest.raises(ValueError):
        predict_churn(customer)


def test_missing_field_is_rejected():
    customer = load("high_risk_customer.json")
    del customer["tenure"]
    with pytest.raises(ValueError):
        predict_churn(customer)