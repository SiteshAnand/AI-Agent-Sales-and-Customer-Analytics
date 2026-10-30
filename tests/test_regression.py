
import numpy as np
import pandas as pd
import pytest

from core.regression import fit_regression


@pytest.fixture
def sales_data():
    """Create repeatable synthetic sales data for regression tests."""
    rng = np.random.default_rng(42)
    rows = 200

    df = pd.DataFrame({
        "Advertising_Spend": rng.uniform(100, 1000, rows),
        "Discount": rng.uniform(0, 20, rows),
        "Region": rng.choice(["North", "South", "West"], rows),
    })

    region_effect = df["Region"].map({
        "North": 100,
        "South": 50,
        "West": -20,
    })

    df["Sales"] = (
        500
        + 2.5 * df["Advertising_Spend"]
        - 12 * df["Discount"]
        + region_effect
        + rng.normal(0, 30, rows)
    )

    return df


def test_auto_selects_a_supported_model(sales_data):
    result = fit_regression(
        df=sales_data,
        target_column="Sales",
        feature_columns=[
            "Advertising_Spend",
            "Discount",
            "Region",
        ],
        algorithm="auto",
        random_state=42,
    )

    assert result["selected_algorithm"] in {
        "linear",
        "ridge",
        "lasso",
        "elastic_net",
    }


def test_returns_expected_metrics(sales_data):
    result = fit_regression(
        df=sales_data,
        target_column="Sales",
        random_state=42,
    )

    metrics = result["metrics"]

    assert metrics["train_rows"] > 0
    assert metrics["test_rows"] > 0
    assert metrics["usable_rows"] == len(sales_data)
    assert metrics["rows_dropped"] == 0
    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= 0
    assert np.isfinite(metrics["mae"])
    assert np.isfinite(metrics["rmse"])
    assert np.isfinite(metrics["r2"])


def test_predictions_match_test_row_count(sales_data):
    result = fit_regression(
        df=sales_data,
        target_column="Sales",
        random_state=42,
    )

    predictions = result["predictions"]

    assert len(predictions) == result["metrics"]["test_rows"]
    assert {"actual", "predicted", "residual"}.issubset(
        predictions.columns
    )
    assert np.isfinite(
        predictions[["actual", "predicted", "residual"]].to_numpy()
    ).all()


def test_handles_missing_predictor_values(sales_data):
    sales_data.loc[5, "Advertising_Spend"] = np.nan
    sales_data.loc[10, "Region"] = None

    result = fit_regression(
        df=sales_data,
        target_column="Sales",
        random_state=42,
    )

    assert len(result["predictions"]) > 0
    assert np.isfinite(result["metrics"]["rmse"])



def test_model_comparison_contains_all_algorithms(sales_data):
    result = fit_regression(
        df=sales_data,
        target_column="Sales",
        algorithm="auto",
        random_state=42,
    )

    comparison = result["model_comparison"]

    expected_algorithms = {
        "linear",
        "ridge",
        "lasso",
        "elastic_net",
    }

    # The regression engine returns model comparison as a dictionary.
    assert isinstance(comparison, dict)
    assert expected_algorithms.issubset(set(comparison.keys()))

    # Each model should have a finite cross-validation RMSE.
    for model_name in expected_algorithms:
        model_result = comparison[model_name]

        assert np.isfinite(model_result["cv_rmse_mean"])


def test_rejects_missing_target_column(sales_data):
    with pytest.raises((ValueError, KeyError)):
        fit_regression(
            df=sales_data,
            target_column="Missing_Target",
        )


def test_rejects_target_as_a_predictor(sales_data):
    with pytest.raises(ValueError):
        fit_regression(
            df=sales_data,
            target_column="Sales",
            feature_columns=["Advertising_Spend", "Sales"],
        )


def test_rejects_too_few_rows():
    df = pd.DataFrame({
        "Advertising_Spend": [100, 200, 300, 400, 500, 600],
        "Sales": [350, 600, 850, 1100, 1350, 1600],
    })

    with pytest.raises(ValueError):
        fit_regression(
            df=df,
            target_column="Sales",
        )
