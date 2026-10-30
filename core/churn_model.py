
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _normalise_target(series: pd.Series) -> pd.Series:
    """Convert common binary churn labels to 0/1."""
    if series.isna().any():
        raise ValueError("The churn target contains missing values.")

    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)

    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.notna().all():
        values = set(numeric.unique())
        if values == {0, 1}:
            return numeric.astype(int)

    labels = series.astype(str).str.strip().str.lower()

    mapping = {
        "1": 1,
        "true": 1,
        "yes": 1,
        "y": 1,
        "churn": 1,
        "churned": 1,
        "left": 1,
        "0": 0,
        "false": 0,
        "no": 0,
        "n": 0,
        "stay": 0,
        "stayed": 0,
        "retained": 0,
        "active": 0,
        "not churned": 0,
        "not_churned": 0,
    }

    mapped = labels.map(mapping)

    if mapped.isna().any():
        unsupported = sorted(labels[mapped.isna()].unique().tolist())
        raise ValueError(
            "The churn target must contain two recognised classes "
            f"(0/1, yes/no, churned/retained). Unsupported values: {unsupported}"
        )

    if set(mapped.unique()) != {0, 1}:
        raise ValueError("The churn target must contain both classes: 0 and 1.")

    return mapped.astype(int)


def _build_pipeline(X: pd.DataFrame) -> Pipeline:
    """Build a preprocessing and logistic-regression pipeline."""
    numeric_columns = X.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical_columns = [
        column
        for column in X.columns
        if column not in numeric_columns
    ]

    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    keep_empty_features=True,
                ),
            ),
            ("scaler", StandardScaler()),
        ])

        transformers.append(
            ("numeric", numeric_pipeline, numeric_columns)
        )

    if categorical_columns:
        categorical_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                    keep_empty_features=True,
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
            ),
        ])

        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    if not transformers:
        raise ValueError("At least one usable predictor is required.")

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    classifier = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


def fit_churn_model(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: list[str] | None = None,
    test_size: float = 0.2,
    cv_folds: int = 5,
    random_state: int = 42,
    probability_threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Train and evaluate a customer churn classification model.

    Target convention:
        1 = churned
        0 = retained

    Returns a fitted pipeline, test metrics, cross-validation metrics,
    test predictions, and model metadata.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("The input DataFrame is empty.")

    if not isinstance(target_column, str) or not target_column:
        raise ValueError("target_column must be a non-empty string.")

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' was not found.")

    if not 0 < test_size < 0.5:
        raise ValueError("test_size must be greater than 0 and less than 0.5.")

    if not isinstance(cv_folds, int) or cv_folds < 2:
        raise ValueError("cv_folds must be an integer of at least 2.")

    if not 0 < probability_threshold < 1:
        raise ValueError("probability_threshold must be between 0 and 1.")

    if feature_columns is None:
        feature_columns = [
            column for column in df.columns
            if column != target_column
        ]
    else:
        feature_columns = list(feature_columns)

    if not feature_columns:
        raise ValueError("At least one predictor column is required.")

    if len(feature_columns) != len(set(feature_columns)):
        raise ValueError("feature_columns must not contain duplicates.")

    if target_column in feature_columns:
        raise ValueError("The target column cannot also be a predictor.")

    missing_columns = [
        column for column in feature_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing predictor columns: {missing_columns}")

    data = df[feature_columns + [target_column]].copy()
    original_rows = len(data)

    # Avoid unsupported nested structures as predictor values.
    for column in feature_columns:
        if data[column].dtype == "object":
            nested_values = data[column].map(
                lambda value: isinstance(value, (list, dict, set, tuple))
            )
            if nested_values.any():
                raise ValueError(
                    f"Predictor '{column}' contains nested values."
                )

        if pd.api.types.is_numeric_dtype(data[column]):
            data[column] = data[column].replace(
                [np.inf, -np.inf], np.nan
            )

    data[target_column] = _normalise_target(data[target_column])

    # Drop rows where every predictor is missing.
    data = data.dropna(subset=feature_columns, how="all")

    rows_dropped = original_rows - len(data)

    if len(data) < 20:
        raise ValueError(
            "At least 20 usable rows are required for churn modelling."
        )

    if data[feature_columns].nunique(dropna=True).eq(0).all():
        raise ValueError("All predictors are constant or entirely missing.")

    X = data[feature_columns]
    y = data[target_column]

    class_counts = y.value_counts()

    if set(class_counts.index) != {0, 1}:
        raise ValueError("The target must contain both churn classes: 0 and 1.")

    if int(class_counts.min()) < 5:
        raise ValueError(
            "At least 5 examples of each churn class are required."
        )

    # Reserve a stratified test set so both classes appear in each split.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    train_class_counts = y_train.value_counts()

    folds = min(cv_folds, int(train_class_counts.min()))

    if folds < 2:
        raise ValueError(
            "Not enough examples in each training class for cross-validation."
        )

    cv = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=random_state,
    )

    model = _build_pipeline(X_train)

    # All preprocessing is inside the pipeline, so each CV fold fits its
    # imputers, scaler, and encoder using training data only.
    cv_results = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring={
            "roc_auc": "roc_auc",
            "f1": "f1",
            "precision": "precision",
            "recall": "recall",
        },
        error_score="raise",
    )

    model.fit(X_train, y_train)

    churn_probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (
        churn_probabilities >= probability_threshold
    ).astype(int)

    # ROC AUC and average precision measure ranking quality; thresholded
    # metrics describe classifications at the selected probability threshold.
    metrics = {
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "usable_rows": int(len(data)),
        "rows_dropped": int(rows_dropped),
        "churn_rate": float(y.mean()),
        "threshold": float(probability_threshold),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, churn_probabilities)),
        "average_precision": float(
            average_precision_score(y_test, churn_probabilities)
        ),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).tolist(),
    }

    cross_validation = {
        metric: {
            "mean": float(np.mean(cv_results[f"test_{metric}"])),
            "std": float(np.std(cv_results[f"test_{metric}"])),
            "folds": int(folds),
        }
        for metric in ["roc_auc", "f1", "precision", "recall"]
    }

    prediction_table = pd.DataFrame(
        {
            "actual": y_test,
            "predicted": predictions,
            "churn_probability": churn_probabilities,
        },
        index=y_test.index,
    )

    prediction_table["actual"] = prediction_table["actual"].astype(int)
    prediction_table["predicted"] = prediction_table["predicted"].astype(int)
    prediction_table["correct"] = (
        prediction_table["actual"] == prediction_table["predicted"]
    )

    warnings = [
        "Predictions indicate statistical risk, not certainty that a customer will churn.",
        "Validate performance on representative data before operational use.",
        "Thresholded metrics depend on probability_threshold and class balance.",
    ]

    return {
        "model": model,
        "target_column": target_column,
        "feature_columns": feature_columns,
        "selected_algorithm": "logistic_regression",
        "metrics": metrics,
        "cross_validation": cross_validation,
        "predictions": prediction_table,
        "probability_threshold": float(probability_threshold),
        "warnings": warnings,
    }
