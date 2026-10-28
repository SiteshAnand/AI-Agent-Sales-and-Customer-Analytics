
import pandas as pd


SUPPORTED_METRICS = {
    "sum", "mean", "median", "min", "max", "count", "std", "variance"
}


def _require_columns(df, *columns):
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def _numeric_series(df, column):
    _require_columns(df, column)
    values = pd.to_numeric(df[column], errors="coerce").dropna()

    if values.empty:
        raise ValueError(f"No usable numeric values in '{column}'.")

    return values


def _calculate_metric(values, metric):
    if metric not in SUPPORTED_METRICS:
        raise ValueError(
            f"Unsupported metric '{metric}'. "
            f"Supported metrics: {', '.join(sorted(SUPPORTED_METRICS))}."
        )

    operations = {
        "sum": values.sum,
        "mean": values.mean,
        "median": values.median,
        "min": values.min,
        "max": values.max,
        "count": values.count,
        "std": values.std,
        "variance": values.var,
    }

    result = operations[metric]()

    if pd.isna(result):
        return None

    return float(result)


def numeric_summary(df, column):
    values = _numeric_series(df, column)

    return {
        "column": column,
        "count": int(values.count()),
        "sum": float(values.sum()),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "min": float(values.min()),
        "max": float(values.max()),
        "std": _calculate_metric(values, "std"),
        "variance": _calculate_metric(values, "variance"),
    }


def distinct_count(df, column):
    _require_columns(df, column)
    values = df[column].dropna()

    values = values.map(
        lambda value: value.strip() if isinstance(value, str) else value
    )
    values = values[
        ~values.map(
            lambda value: isinstance(value, str) and value == ""
        )
    ]
    values = values.map(
        lambda value: value.casefold() if isinstance(value, str) else value
    )

    return {
        "column": column,
        "distinct_count": int(values.nunique(dropna=True)),
    }


def group_metric(
    df,
    group_column,
    value_column,
    metric="sum",
    top_n=10,
    sort_order="desc",
):
    _require_columns(df, group_column, value_column)

    if metric not in SUPPORTED_METRICS:
        raise ValueError(
            f"Unsupported metric '{metric}'. "
            f"Supported metrics: {', '.join(sorted(SUPPORTED_METRICS))}."
        )

    if type(top_n) is not int or not 1 <= top_n <= 1000:
        raise ValueError("top_n must be an integer between 1 and 1000.")

    if sort_order not in {"asc", "desc"}:
        raise ValueError("sort_order must be 'asc' or 'desc'.")

    data = df[[group_column, value_column]].copy()

    data[value_column] = pd.to_numeric(
        data[value_column], errors="coerce"
    )

    data[group_column] = data[group_column].map(
        lambda value: value.strip() if isinstance(value, str) else value
    )

    data = data.dropna(subset=[group_column, value_column])
    data = data[
        ~data[group_column].map(
            lambda value: isinstance(value, str) and value == ""
        )
    ]

    if data.empty:
        raise ValueError("No valid rows for this calculation.")

    grouped = data.groupby(group_column, dropna=True)[value_column]

    if metric == "count":
        result = grouped.count()
    elif metric == "variance":
        result = grouped.var()
    elif metric == "std":
        result = grouped.std()
    else:
        result = grouped.agg(metric)

    # Business convention: a group with one observation has zero
    # within-group variation, so preserve it instead of dropping it.
    if metric in {"std", "variance"}:
        result = result.fillna(0.0)

    result = result.sort_values(
        ascending=(sort_order == "asc"),
        kind="stable",
    ).head(top_n)

    return [
        {"group": str(key), "value": float(value)}
        for key, value in result.items()
    ]


def period_totals(df, date_column, value_column, frequency="ME"):
    _require_columns(df, date_column, value_column)

    if frequency not in {"ME", "QE", "YE"}:
        raise ValueError("frequency must be ME, QE, or YE.")

    data = df[[date_column, value_column]].copy()
    data[date_column] = pd.to_datetime(
        data[date_column], errors="coerce"
    )
    data[value_column] = pd.to_numeric(
        data[value_column], errors="coerce"
    )
    data = data.dropna(subset=[date_column, value_column])

    if data.empty:
        raise ValueError("No valid dates and numeric values found.")

    result = (
        data.set_index(date_column)[value_column]
        .resample(frequency)
        .sum()
    )

    return [
        {"period": date.strftime("%Y-%m-%d"), "value": float(value)}
        for date, value in result.items()
    ]


def percentage_change(current, previous):
    if previous == 0:
        return None

    return float((current - previous) / abs(previous) * 100)


def filtered_metric(
    df,
    filter_column,
    filter_value,
    value_column,
    metric="sum",
):
    _require_columns(df, filter_column, value_column)

    if metric not in SUPPORTED_METRICS:
        raise ValueError(
            f"Unsupported aggregation metric '{metric}'. "
            f"Supported metrics: {', '.join(sorted(SUPPORTED_METRICS))}."
        )

    if not isinstance(filter_value, str) or not filter_value.strip():
        raise ValueError("A nonblank filter value is required.")

    data = df[[filter_column, value_column]].copy()

    matches = (
        data[filter_column].astype("string").str.strip().str.casefold()
        == filter_value.strip().casefold()
    )
    data = data.loc[matches].copy()

    if data.empty:
        raise ValueError(
            f"No records found where {filter_column} matches '{filter_value}'."
        )

    values = pd.to_numeric(
        data[value_column], errors="coerce"
    ).dropna()

    if values.empty:
        raise ValueError(
            f"No usable numeric values in '{value_column}' "
            f"for {filter_column}='{filter_value}'."
        )

    return {
        "filter_column": filter_column,
        "filter_value": filter_value.strip(),
        "value_column": value_column,
        "metric": metric,
        "matched_rows": int(len(data)),
        "valid_numeric_values": int(values.count()),
        "result": _calculate_metric(values, metric),
    }

def data_profile(df):
    """Summarise dataset structure and basic data-quality indicators."""
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Dataset must be a pandas DataFrame.")

    columns = []

    for column in df.columns:
        series = df[column]
        non_null = series.dropna()

        # Convert examples to strings so results are easy to display
        # and safe to pass through a JSON-based LLM workflow.
        sample_values = [
            str(value)[:100]
            for value in non_null.drop_duplicates().head(3).tolist()
        ]

        columns.append({
            "column": str(column),
            "data_type": str(series.dtype),
            "non_null_count": int(series.notna().sum()),
            "missing_count": int(series.isna().sum()),
            "missing_percent": (
                round(float(series.isna().mean() * 100), 2)
                if len(series) else 0.0
            ),
            "distinct_count": int(non_null.nunique()),
            "sample_values": sample_values,
        })

    return {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "duplicate_rows": int(df.duplicated().sum()),
        "columns": columns,
    }

def data_profile(df):
    """Summarise dataset structure and basic data-quality indicators."""
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Dataset must be a pandas DataFrame.")

    columns = []

    for column in df.columns:
        series = df[column]
        non_null = series.dropna()

        columns.append({
            "column": str(column),
            "data_type": str(series.dtype),
            "non_null_count": int(series.notna().sum()),
            "missing_count": int(series.isna().sum()),
            "missing_percent": (
                round(float(series.isna().mean() * 100), 2)
                if len(df) else 0.0
            ),
            "distinct_count": int(non_null.nunique()),
            "sample_values": [
                str(value)[:100]
                for value in non_null.drop_duplicates().head(3).tolist()
            ],
        })

    return {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "duplicate_rows": int(df.duplicated().sum()),
        "columns": columns,
    }
