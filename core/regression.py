
"""Regression modelling utilities for the Sales Analytics Agent.

Supports:
- Linear Regression
- Ridge Regression
- Lasso Regression
- Elastic Net Regression
- Automatic model selection using cross-validation RMSE
- Numeric and categorical predictors
- Missing predictor values
- Random and time-ordered train/test validation
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    Ridge,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SUPPORTED_ALGORITHMS = ("linear", "ridge", "lasso", "elastic_net")


def _make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create preprocessing for numeric and categorical predictors."""
    numeric_columns = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [
        column for column in X.columns if column not in numeric_columns
    ]

    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(
            ("numeric", numeric_pipeline, numeric_columns)
        )

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent"),
                ),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]
        )
        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    if not transformers:
        raise ValueError("At least one usable predictor column is required.")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


def _make_model(algorithm: str):
    """Return a fresh estimator for the requested algorithm."""
    models = {
        "linear": LinearRegression(),
        "ridge": Ridge(alpha=1.0),
        "lasso": Lasso(alpha=0.1, max_iter=20000),
        "elastic_net": ElasticNet(
            alpha=0.1,
            l1_ratio=0.5,
            max_iter=20000,
        ),
    }

    if algorithm not in models:
        raise ValueError(
            f"Unsupported algorithm '{algorithm}'. "
            f"Choose one of: {', '.join(SUPPORTED_ALGORITHMS)}, or 'auto'."
        )

    return models[algorithm]


def _make_pipeline(X: pd.DataFrame, algorithm: str) -> Pipeline:
    """Combine preprocessing and regression into one pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", _make_preprocessor(X)),
            ("regressor", _make_model(algorithm)),
        ]
    )


def _rmse(actual, predicted) -> float:
    """Calculate root mean squared error."""
    return float(np.sqrt(mean_squared_error(actual, predicted)))


def _cross_validate_model(
    X: pd.DataFrame,
    y: pd.Series,
    algorithm: str,
    cv_folds: int,
    random_state: int,
) -> dict[str, Any]:
    """Evaluate a model with shuffled K-fold cross-validation."""
    model = _make_pipeline(X, algorithm)

    folds = min(max(2, int(cv_folds)), len(X))
    splitter = KFold(
        n_splits=folds,
        shuffle=True,
        random_state=random_state,
    )

    scores = cross_val_score(
        model,
        X,
        y,
        scoring="neg_root_mean_squared_error",
        cv=splitter,
        error_score="raise",
    )

    rmse_values = -scores

    return {
        "cv_rmse_mean": float(np.mean(rmse_values)),
        "cv_rmse_std": float(np.std(rmse_values)),
        "cv_folds": int(folds),
    }


def _feature_names_and_coefficients(
    fitted_pipeline: Pipeline,
) -> tuple[list[str], list[float] | None]:
    """Extract feature names and coefficients when supported."""
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    estimator = fitted_pipeline.named_steps["regressor"]

    try:
        names = preprocessor.get_feature_names_out().tolist()
    except (AttributeError, ValueError):
        names = []

    coefficients = getattr(estimator, "coef_", None)

    if coefficients is None:
        return names, None

    coefficients = np.asarray(coefficients).reshape(-1)

    return names, [float(value) for value in coefficients]


def fit_regression(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: list[str] | None = None,
    algorithm: str = "auto",
    test_size: float = 0.2,
    cv_folds: int = 5,
    random_state: int = 42,
    validation_strategy: str = "random",
) -> dict[str, Any]:
    """Fit a regression model and return metrics, predictions, and metadata.

    Parameters
    ----------
    df:
        Input dataset.
    target_column:
        Numeric column to predict.
    feature_columns:
        Optional list of predictor columns. Defaults to all columns except
        the target.
    algorithm:
        'auto', 'linear', 'ridge', 'lasso', or 'elastic_net'.
    test_size:
        Fraction of usable rows reserved for testing.
    cv_folds:
        Number of cross-validation folds used for model comparison.
    random_state:
        Seed used for reproducible random splitting and cross-validation.
    validation_strategy:
        'random' for a shuffled split, or 'time' for an ordered split.
        For time-based validation, rows must already be in chronological order.

    Returns
    -------
    dict
        Selected model, model comparison, metrics, predictions, coefficients,
        and validation metadata.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("The dataset contains no rows.")

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' was not found.")

    if feature_columns is None:
        feature_columns = [
            column for column in df.columns if column != target_column
        ]
    else:
        feature_columns = list(feature_columns)

    if target_column in feature_columns:
        raise ValueError("The target column cannot also be a predictor.")

    if not feature_columns:
        raise ValueError("At least one predictor column is required.")

    missing_features = [
        column for column in feature_columns if column not in df.columns
    ]
    if missing_features:
        raise ValueError(
            f"Predictor columns not found: {missing_features}"
        )

    if not 0 < test_size < 1:
        raise ValueError("test_size must be greater than 0 and less than 1.")

    if cv_folds < 2:
        raise ValueError("cv_folds must be at least 2.")

    if validation_strategy not in {"random", "time"}:
        raise ValueError(
            "validation_strategy must be either 'random' or 'time'."
        )

    algorithm = str(algorithm).strip().lower()

    if algorithm != "auto" and algorithm not in SUPPORTED_ALGORITHMS:
        raise ValueError(
            f"Unsupported algorithm '{algorithm}'. "
            f"Choose one of: {', '.join(SUPPORTED_ALGORITHMS)}, or 'auto'."
        )

    # Work on a copy; do not modify the user's original dataset.
    working = df[feature_columns + [target_column]].copy()

    # The target must be numeric. Invalid or missing target values are dropped.
    working[target_column] = pd.to_numeric(
        working[target_column], errors="coerce"
    )
    working[target_column] = working[target_column].replace(
        [np.inf, -np.inf], np.nan
    )
    working = working.dropna(subset=[target_column])

    if working.empty:
        raise ValueError("The target column has no usable numeric values.")

    # Remove predictor columns that contain no information at all.
    usable_features = [
        column
        for column in feature_columns
        if working[column].notna().any()
    ]

    if not usable_features:
        raise ValueError("No usable predictor columns remain.")

    feature_columns = usable_features

    X = working[feature_columns].copy()
    y = working[target_column].astype(float).copy()

    # Convert infinite numeric predictor values to missing values for imputation.
    numeric_columns = X.select_dtypes(include=["number"]).columns
    X[numeric_columns] = X[numeric_columns].replace(
        [np.inf, -np.inf], np.nan
    )

    # Ensure the training and test sets each contain at least two rows.
    usable_rows = len(working)
    test_rows_requested = int(np.ceil(usable_rows * test_size))
    test_rows_requested = max(1, test_rows_requested)
    train_rows_requested = usable_rows - test_rows_requested

    if usable_rows < 7 or train_rows_requested < 2 or test_rows_requested < 1:
        raise ValueError(
            "At least 7 usable rows are required for regression "
            "with a separate training and test set."
        )

    if validation_strategy == "random":
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
        )
    else:
        split_index = int(usable_rows * (1 - test_size))

        if split_index < 2 or usable_rows - split_index < 1:
            raise ValueError(
                "Not enough rows for the requested time-based split."
            )

        X_train = X.iloc[:split_index].copy()
        X_test = X.iloc[split_index:].copy()
        y_train = y.iloc[:split_index].copy()
        y_test = y.iloc[split_index:].copy()

    # Model comparison is based on the training set only, avoiding leakage
    # from the held-out test rows into model selection.
    comparison: dict[str, dict[str, Any]] = {}

    for model_name in SUPPORTED_ALGORITHMS:
        comparison[model_name] = _cross_validate_model(
            X=X_train,
            y=y_train,
            algorithm=model_name,
            cv_folds=cv_folds,
            random_state=random_state,
        )

    if algorithm == "auto":
        selected_algorithm = min(
            comparison,
            key=lambda name: comparison[name]["cv_rmse_mean"],
        )
    else:
        selected_algorithm = algorithm

    fitted_model = _make_pipeline(X_train, selected_algorithm)
    fitted_model.fit(X_train, y_train)

    predicted = np.asarray(fitted_model.predict(X_test), dtype=float)
    actual = y_test.to_numpy(dtype=float)
    residuals = actual - predicted

    if not (
        np.isfinite(predicted).all()
        and np.isfinite(actual).all()
        and np.isfinite(residuals).all()
    ):
        raise ValueError(
            "The model produced non-finite predictions or residuals."
        )

    predictions = pd.DataFrame(
        {
            "actual": actual,
            "predicted": predicted,
            "residual": residuals,
        },
        index=y_test.index,
    )
    predictions.index.name = "row_index"

    metrics = {
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "usable_rows": int(usable_rows),
        "rows_dropped": int(len(df) - usable_rows),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": _rmse(actual, predicted),
        "r2": float(r2_score(actual, predicted)),
    }

    feature_names, coefficients = _feature_names_and_coefficients(
        fitted_model
    )

    estimator = fitted_model.named_steps["regressor"]
    intercept = getattr(estimator, "intercept_", None)

    if intercept is not None:
        intercept = float(np.asarray(intercept).reshape(-1)[0])

    warnings: list[str] = []

    if metrics["r2"] < 0:
        warnings.append(
            "Test R² is negative; the model performs worse than predicting "
            "the test-set mean on this split."
        )

    if len(X_train) < 20:
        warnings.append(
            "The training sample is small; model metrics may be unstable."
        )

    return {
        "selected_algorithm": selected_algorithm,
        "algorithm": selected_algorithm,
        "target_column": target_column,
        "feature_columns": feature_columns,
        "validation_strategy": validation_strategy,
        "model": fitted_model,
        "fitted_model": fitted_model,
        "metrics": metrics,
        "predictions": predictions,
        "model_comparison": comparison,
        "feature_names": feature_names,
        "coefficients": coefficients,
        "intercept": intercept,
        "warnings": warnings,
    }
