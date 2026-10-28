
import json


def format_result(result: dict) -> str:
    """Format a calculation result as readable, consistent JSON."""

    if not isinstance(result, dict):
        raise TypeError("Result must be a dictionary.")

    return json.dumps(
        result,
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


def format_success(tool_name: str, result: dict) -> dict:
    """Return a successful, traceable tool response."""

    if not isinstance(result, dict):
        raise TypeError("Result must be a dictionary.")

    return {
        "status": "success",
        "tool": tool_name,
        "result": result,
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
        "- Validate date columns before time-series or delivery-duration analysis.",
        "- Treat phone numbers and other identifiers as identifiers, not numeric measures.",
        "- Confirm the grain of the data before summing customer lifetime values or sales targets.",
        "",
        "Recommendations are checks to perform, not confirmed data errors.",
    ])

    return "\n".join(lines)
