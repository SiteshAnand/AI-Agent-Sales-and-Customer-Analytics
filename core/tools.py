
import pandas as pd

from core.analytics import (
    SUPPORTED_METRICS,
    numeric_summary,
    distinct_count,
    group_metric,
    filtered_metric,
    period_totals,
    percentage_change,
    data_profile,
)
from core.regression import fit_regression
from core.churn_model import fit_churn_model
from core.rfm import calculate_rfm


def _require_string(parameters, name):
    value = parameters.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Provide a valid '{name}' parameter.")
    return value.strip()


def _require_metric(parameters):
    metric = parameters.get("metric", "sum")
    if not isinstance(metric, str) or metric not in SUPPORTED_METRICS:
        raise ValueError(
            f"Unsupported aggregation metric. Choose from: "
            f"{', '.join(sorted(SUPPORTED_METRICS))}."
        )
    return metric


def _require_top_n(parameters):
    top_n = parameters.get("top_n", 10)
    if type(top_n) is not int or not 1 <= top_n <= 1000:
        raise ValueError("top_n must be an integer between 1 and 1000.")
    return top_n


def _require_column(df, name):
    column = _require_string({"column": name}, "column")
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in the dataset.")
    return column


def _require_columns(df, value, parameter_name="feature_columns"):
    if value is None:
        return None

    if not isinstance(value, list) or not value:
        raise ValueError(f"{parameter_name} must be a non-empty list of columns.")

    columns = []
    for column in value:
        if not isinstance(column, str) or column not in df.columns:
            raise ValueError(f"Column '{column}' does not exist in the dataset.")
        if column not in columns:
            columns.append(column)

    return columns


def _require_number(parameters, name, default, minimum, maximum):
    value = parameters.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number.")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")
    return value


def run_tool(tool_name: str, df: pd.DataFrame, parameters: dict) -> dict:
    """Run an approved analytics tool on the supplied dataset."""
    if not isinstance(parameters, dict):
        raise ValueError("Tool parameters must be a dictionary.")
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Dataset must be a pandas DataFrame.")
    if df.empty and tool_name != "percentage_change":
        raise ValueError("The dataset contains no rows.")

    if tool_name == "data_profile":
        return data_profile(df)

    if tool_name == "numeric_summary":
        return numeric_summary(df, _require_column(
            df, _require_string(parameters, "column")
        ))

    if tool_name == "distinct_count":
        return distinct_count(df, _require_column(
            df, _require_string(parameters, "column")
        ))

    if tool_name == "group_metric":
        group_column = _require_column(
            df, _require_string(parameters, "group_column")
        )
        value_column = _require_column(
            df, _require_string(parameters, "value_column")
        )
        metric = _require_metric(parameters)
        top_n = _require_top_n(parameters)
        sort_order = parameters.get("sort_order", "desc")
        if sort_order not in {"asc", "desc"}:
            raise ValueError("sort_order must be 'asc' or 'desc'.")

        return {
            "group_column": group_column,
            "value_column": value_column,
            "metric": metric,
            "sort_order": sort_order,
            "results": group_metric(
                df, group_column, value_column,
                metric=metric, top_n=top_n, sort_order=sort_order,
            ),
        }

    if tool_name == "filtered_metric":
        filter_column = _require_column(
            df, _require_string(parameters, "filter_column")
        )
        filter_value = _require_string(parameters, "filter_value")
        value_column = _require_column(
            df, _require_string(parameters, "value_column")
        )
        return filtered_metric(
            df, filter_column, filter_value, value_column,
            metric=_require_metric(parameters),
        )

    if tool_name == "period_totals":
        date_column = _require_column(
            df, _require_string(parameters, "date_column")
        )
        value_column = _require_column(
            df, _require_string(parameters, "value_column")
        )
        frequency = parameters.get("frequency", "ME")
        if frequency not in {"ME", "QE", "YE"}:
            raise ValueError("frequency must be ME, QE, or YE.")
        return {
            "date_column": date_column,
            "value_column": value_column,
            "frequency": frequency,
            "results": period_totals(
                df, date_column, value_column, frequency=frequency
            ),
        }

    if tool_name == "percentage_change":
        current = parameters.get("current")
        previous = parameters.get("previous")
        for name, value in (("current", current), ("previous", previous)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a number.")
        return {
            "current": current,
            "previous": previous,
            "percentage_change": percentage_change(current, previous),
        }

    if tool_name in {"regression", "fit_regression"}:
        target = _require_column(
            df, _require_string(parameters, "target_column")
        )
        features = _require_columns(df, parameters.get("feature_columns"))
        if features and target in features:
            raise ValueError("The target column cannot also be a feature.")

        algorithm = parameters.get("algorithm", "auto")
        if algorithm not in {"auto", "linear", "random_forest"}:
            raise ValueError(
                "algorithm must be auto, linear, or random_forest."
            )

        test_size = _require_number(parameters, "test_size", 0.2, 0.1, 0.5)
        cv_folds = parameters.get("cv_folds", 5)
        if type(cv_folds) is not int or not 2 <= cv_folds <= 10:
            raise ValueError("cv_folds must be an integer between 2 and 10.")

        validation_strategy = parameters.get(
            "validation_strategy", "random"
        )
        if validation_strategy not in {"random", "time"}:
            raise ValueError("validation_strategy must be random or time.")

        return fit_regression(
            df,
            target_column=target,
            feature_columns=features,
            algorithm=algorithm,
            test_size=test_size,
            cv_folds=cv_folds,
            random_state=parameters.get("random_state", 42),
            validation_strategy=validation_strategy,
        )

    if tool_name in {"churn_model", "fit_churn_model"}:
        target = _require_column(
            df, _require_string(parameters, "target_column")
        )
        features = _require_columns(df, parameters.get("feature_columns"))
        if features and target in features:
            raise ValueError("The target column cannot also be a feature.")

        test_size = _require_number(parameters, "test_size", 0.2, 0.1, 0.5)
        threshold = _require_number(
            parameters, "probability_threshold", 0.5, 0.0, 1.0
        )
        cv_folds = parameters.get("cv_folds", 5)
        if type(cv_folds) is not int or not 2 <= cv_folds <= 10:
            raise ValueError("cv_folds must be an integer between 2 and 10.")

        return fit_churn_model(
            df,
            target_column=target,
            feature_columns=features,
            test_size=test_size,
            cv_folds=cv_folds,
            random_state=parameters.get("random_state", 42),
            probability_threshold=threshold,
        )

    if tool_name in {"rfm", "calculate_rfm"}:
        customer_column = _require_column(
            df, _require_string(parameters, "customer_id_column")
        )
        date_column = _require_column(
            df, _require_string(parameters, "date_column")
        )
        value_column = _require_column(
            df, _require_string(parameters, "transaction_value_column")
        )

        snapshot_date = parameters.get("snapshot_date")
        if snapshot_date is not None and not isinstance(snapshot_date, str):
            raise ValueError("snapshot_date must be a date string or null.")

        quantiles = parameters.get("quantiles", 5)
        if type(quantiles) is not int or not 2 <= quantiles <= 10:
            raise ValueError("quantiles must be an integer between 2 and 10.")

        return calculate_rfm(
            df,
            customer_id_column=customer_column,
            date_column=date_column,
            transaction_value_column=value_column,
            snapshot_date=snapshot_date,
            quantiles=quantiles,
        )

    raise ValueError(f"Unsupported tool: {tool_name}")
