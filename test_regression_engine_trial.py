
import numpy as np
import pandas as pd

from core.regression import fit_regression


def main():
    rng = np.random.default_rng(42)
    rows = 300

    data = pd.DataFrame(
        {
            "Advertising_Spend": rng.uniform(100, 1000, rows),
            "Discount": rng.uniform(0, 20, rows),
            "Region": rng.choice(["North", "South", "West"], rows),
        }
    )

    region_effect = data["Region"].map(
        {"North": 100, "South": 50, "West": -20}
    )

    data["Sales"] = (
        500
        + 2.5 * data["Advertising_Spend"]
        - 12 * data["Discount"]
        + region_effect
        + rng.normal(0, 30, rows)
    )

    # Exercise missing-value handling.
    data.loc[5, "Advertising_Spend"] = np.nan
    data.loc[10, "Region"] = None

    result = fit_regression(
        df=data,
        target_column="Sales",
        feature_columns=[
            "Advertising_Spend",
            "Discount",
            "Region",
        ],
        algorithm="auto",
        test_size=0.2,
        cv_folds=5,
        random_state=42,
    )

    print("\n--- REGRESSION ENGINE TEST ---")
    print("Selected model:", result["selected_algorithm"])
    print("Metrics:", result["metrics"])
    print("\nModel comparison:")
    print(pd.DataFrame(result["model_comparison"]).T)
    print("\nSample predictions:")
    print(result["predictions"].head())
    print("\nWarnings:", result["warnings"])

    assert result["selected_algorithm"] in {
        "linear", "ridge", "lasso", "elastic_net"
    }
    assert result["metrics"]["test_rows"] > 0
    assert np.isfinite(result["metrics"]["rmse"])
    assert result["metrics"]["rmse"] >= 0
    assert len(result["predictions"]) == result["metrics"]["test_rows"]
    assert "Region" in result["feature_columns"]

    print("\nPASS: Regression engine trial succeeded.")


if __name__ == "__main__":
    main()
