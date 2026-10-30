
import numpy as np
import pandas as pd
import pytest

from core.churn_model import fit_churn_model


@pytest.fixture
def customer_data():
    rng = np.random.default_rng(17)
    rows = 300

    tenure = rng.integers(1, 72, rows)
    monthly_charges = rng.uniform(200, 1800, rows)
    support_calls = rng.poisson(2, rows)
    contract = rng.choice(
        ["monthly", "annual", "two_year"],
        rows,
        p=[0.55, 0.30, 0.15],
    )

    # Create a synthetic outcome with a real signal for testing.
    score = (
        -2.0
        + 0.035 * support_calls
        - 0.025 * tenure
        + 0.001 * monthly_charges
        + (contract == "monthly") * 1.2
    )
    probability = 1 / (1 + np.exp(-score))
    churn = rng.binomial(1, probability)

    # Ensure both classes are present in this generated test fixture.
    churn[0] = 0
    churn[1] = 1

    return pd.DataFrame({
        "Tenure": tenure,
        "Monthly_Charges": monthly_charges,
        "Support_Calls": support_calls,
        "Contract": contract,
        "Churn": churn,
    })


def test_trains_and_returns_metrics(customer_data):
    result = fit_churn_model(
        customer_data,
        target_column="Churn",
        random_state=42,
    )

    assert result["selected_algorithm"] == "logistic_regression"
    assert result["metrics"]["train_rows"] > 0
    assert result["metrics"]["test_rows"] > 0

    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        assert 0 <= result["metrics"][metric] <= 1

    assert len(result["predictions"]) == result["metrics"]["test_rows"]


def test_probabilities_are_valid(customer_data):
    result = fit_churn_model(customer_data, "Churn")

    probabilities = result["predictions"]["churn_probability"]

    assert probabilities.between(0, 1).all()


def test_handles_missing_predictor_values(customer_data):
    customer_data.loc[5, "Monthly_Charges"] = np.nan
    customer_data.loc[8, "Contract"] = None

    result = fit_churn_model(customer_data, "Churn")

    assert len(result["predictions"]) > 0


def test_accepts_yes_no_target(customer_data):
    customer_data["Churn_Label"] = customer_data["Churn"].map(
        {0: "No", 1: "Yes"}
    )

    result = fit_churn_model(
        customer_data,
        target_column="Churn_Label",
        feature_columns=[
            "Tenure",
            "Monthly_Charges",
            "Support_Calls",
            "Contract",
        ],
    )

    assert result["metrics"]["usable_rows"] == len(customer_data)


def test_rejects_missing_target(customer_data):
    with pytest.raises(ValueError):
        fit_churn_model(customer_data, "Unknown_Target")


def test_rejects_target_as_predictor(customer_data):
    with pytest.raises(ValueError):
        fit_churn_model(
            customer_data,
            target_column="Churn",
            feature_columns=["Tenure", "Churn"],
        )


def test_rejects_single_class_target(customer_data):
    customer_data["Churn"] = 0

    with pytest.raises(ValueError):
        fit_churn_model(customer_data, "Churn")


def test_rejects_too_few_rows():
    df = pd.DataFrame({
        "Tenure": range(10),
        "Churn": [0, 1] * 5,
    })

    with pytest.raises(ValueError):
        fit_churn_model(df, "Churn")
