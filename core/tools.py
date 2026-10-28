
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


def _require_string(parameters, name):
    """Return a required, nonblank string parameter."""
    value = parameters.get(name)

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Provide a valid '{name}' parameter.")

    return value.strip()


def _require_metric(parameters):
    """Validate and return an aggregation metric."""
    metric = parameters.get("metric", "sum")

    if not isinstance(metric, str) or metric not in SUPPORTED_METRICS:
        raise ValueError(
            f"Unsupported aggregation metric. "
            f"Choose from: {', '.join(sorted(SUPPORTED_METRICS))}."
        )

    return metric


def _require_top_n(parameters):
    """Validate the maximum number of groups to return."""
    top_n = parameters.get("top_n", 10)

    if type(top_n) is not int or not 1 <= top_n <= 1000:
        raise ValueError("top_n must be an integer between 1 and 1000.")

    return top_n


def run_tool(
    tool_name: str,
    df: pd.DataFrame,
    parameters: dict,
) -> dict:
    """Run one approved analytics tool on the supplied dataset."""

    if not isinstance(parameters, dict):
        raise ValueError("Tool parameters must be a dictionary.")

    if not isinstance(df, pd.DataFrame):
        raise ValueError("Dataset must be a pandas DataFrame.")

    if tool_name == "data_profile":
        return data_profile(df)
    
    if tool_name == "numeric_summary":
        column = _require_string(parameters, "column")
        return numeric_summary(df, column)

    if tool_name == "distinct_count":
        column = _require_string(parameters, "column")
        return distinct_count(df, column)

    if tool_name == "group_metric":
        group_column = _require_string(parameters, "group_column")
        value_column = _require_string(parameters, "value_column")
        metric = _require_metric(parameters)
        top_n = _require_top_n(parameters)

        sort_order = parameters.get("sort_order", "desc")

        if sort_order not in {"asc", "desc"}:
            raise ValueError("sort_order must be 'asc' or 'desc'.")

        result = group_metric(
            df,
            group_column,
            value_column,
            metric=metric,
            top_n=top_n,
            sort_order=sort_order,
        )

        return {
            "group_column": group_column,
            "value_column": value_column,
            "metric": metric,
            "sort_order": sort_order,
            "results": result,
        }

    if tool_name == "filtered_metric":
        filter_column = _require_string(parameters, "filter_column")
        filter_value = _require_string(parameters, "filter_value")
        value_column = _require_string(parameters, "value_column")
        metric = _require_metric(parameters)

        return filtered_metric(
            df,
            filter_column,
            filter_value,
            value_column,
            metric=metric,
        )

    if tool_name == "period_totals":
        date_column = _require_string(parameters, "date_column")
        value_column = _require_string(parameters, "value_column")
        frequency = parameters.get("frequency", "ME")

        if frequency not in {"ME", "QE", "YE"}:
            raise ValueError("frequency must be ME, QE, or YE.")

        result = period_totals(
            df,
            date_column,
            value_column,
            frequency=frequency,
        )

        return {
            "date_column": date_column,
            "value_column": value_column,
            "frequency": frequency,
            "results": result,
        }

    if tool_name == "percentage_change":
        current = parameters.get("current")
        previous = parameters.get("previous")

        if (
            isinstance(current, bool)
            or not isinstance(current, (int, float))
            or isinstance(previous, bool)
            or not isinstance(previous, (int, float))
        ):
            raise ValueError("Current and previous must be numbers.")

        result = percentage_change(current, previous)

        return {
            "current": current,
            "previous": previous,
            "percentage_change": result,
        }

    raise ValueError(f"Unsupported tool: {tool_name}")
