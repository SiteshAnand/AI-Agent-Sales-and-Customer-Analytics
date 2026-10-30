
import json
import math

import numpy as np
import pandas as pd


def _safe_value(value, depth=0):
    """Convert analytical results into JSON-safe, displayable values."""
    if depth > 8:
        return "[Maximum result nesting reached]"

    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, pd.Series):
        return _safe_value(value.to_dict(), depth + 1)

    if isinstance(value, pd.DataFrame):
        preview = value.head(20)
        return {
            "type": "dataframe",
            "row_count": int(len(value)),
            "column_count": int(len(value.columns)),
            "columns": [str(column) for column in value.columns],
            "preview": _safe_value(
                preview.to_dict(orient="records"), depth + 1
            ),
            "preview_truncated": len(value) > len(preview),
        }

    if isinstance(value, dict):
        safe = {}
        for key, item in value.items():
            # Never expose fitted model objects directly.
            if str(key).casefold() in {
                "model", "estimator", "fitted_model", "pipeline"
            } and hasattr(item, "predict"):
                safe[str(key)] = _model_summary(item)
            else:
                safe[str(key)] = _safe_value(item, depth + 1)
        return safe

    if isinstance(value, (list, tuple, set)):
        return [_safe_value(item, depth + 1) for item in value]

    # Handle a fitted estimator passed directly.
    if hasattr(value, "predict") and hasattr(value, "get_params"):
        return _model_summary(value)

    # Support simple NumPy arrays without exposing arbitrary object internals.
    if isinstance(value, np.ndarray):
        return _safe_value(value.tolist(), depth + 1)

    # Last resort: a readable label, never a raw object representation.
    return str(value)


def _model_summary(model):
    """Return safe metadata for a fitted model, not the model itself."""
    summary = {
        "type": "fitted_model",
        "model_class": type(model).__name__,
    }

    if hasattr(model, "n_features_in_"):
        summary["feature_count"] = int(model.n_features_in_)

    if hasattr(model, "feature_names_in_"):
        summary["feature_names"] = [
            str(name) for name in model.feature_names_in_
        ]

    if hasattr(model, "coef_"):
        summary["coefficients"] = _safe_value(model.coef_)

    if hasattr(model, "intercept_"):
        summary["intercept"] = _safe_value(model.intercept_)

    return summary


def format_result(result) -> str:
    """Format a dictionary result as safe, readable JSON."""
    if not isinstance(result, dict):
        raise TypeError("Result must be a dictionary.")

    return json.dumps(
        _safe_value(result),
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )

def format_error(message: str) -> dict:
    """Return a consistent error response."""
    return {
        "status": "error",
        "message": str(message),
    }


def format_success(tool_name: str, result) -> dict:
    """Return a successful response containing only safe display values."""
    if not isinstance(result, dict):
        raise TypeError("Result must be a dictionary.")

    return {
        "status": "success",
        "tool": tool_name,
        "result": _safe_value(result),
    }


def format_data_profile(result: dict) -> str:
    """Create a management-friendly Markdown report from a data profile."""
    if not isinstance(result, dict):
        raise TypeError("Profile result must be a dictionary.")

    row_count = result.get("row_count", 0)
    column_count = result.get("column_count", 0)
    duplicate_rows = result.get("duplicate_rows", 0)
    columns = result.get("columns", [])

    lines = [
        "# Dataset Data Quality Report",
        "",
        "## 1. Executive Summary",
        "",
        f"- **Total rows:** {row_count:,}",
        f"- **Total columns:** {column_count:,}",
        f"- **Duplicate rows:** {duplicate_rows:,}",
    ]

    columns_with_missing = [
        col for col in columns if col.get("missing_count", 0) > 0
    ]

    lines.extend([
        f"- **Columns with missing values:** "
        f"{len(columns_with_missing):,} of {column_count:,}",
        "",
        "## 2. Data Quality Findings",
        "",
    ])

    if not columns_with_missing:
        lines.append("No missing values were detected.")
    else:
        lines.extend([
            "| Column | Missing values | Missing % |",
            "|---|---:|---:|",
        ])
        for col in sorted(
            columns_with_missing,
            key=lambda item: item.get("missing_percent", 0),
            reverse=True,
        ):
            lines.append(
                f"| {col.get('column', '')} "
                f"| {col.get('missing_count', 0):,} "
                f"| {col.get('missing_percent', 0):.2f}% |"
            )

    lines.extend([
        "",
        "## 3. Column-Level Profile",
        "",
        "| Column | Data type | Non-null | Distinct | Missing | Sample values |",
        "|---|---|---:|---:|---:|---|",
    ])

    for col in columns:
        samples = ", ".join(
            str(value).replace("|", "\\|").replace("\n", " ")
            for value in col.get("sample_values", [])
        )
        lines.append(
            f"| {col.get('column', '')} "
            f"| {col.get('data_type', '')} "
            f"| {col.get('non_null_count', 0):,} "
            f"| {col.get('distinct_count', 0):,} "
            f"| {col.get('missing_percent', 0):.2f}% "
            f"| {samples} |"
        )

    lines.extend([
        "",
        "## 4. Recommended Checks",
        "",
        "- Confirm whether missing values are expected for the relevant business process.",
        "- Treat phone numbers and other identifiers as identifiers, not numeric measures.",
        "- Confirm the grain of the data before summing customer lifetime values or sales targets.",
        "",
        "Recommendations are checks to perform, not confirmed data errors.",
    ])

    return "\n".join(lines)
