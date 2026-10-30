
import re


def parse_query(question: str, columns: list[str]) -> dict:
    """Convert a supported sales question into a structured intent."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Please enter a question.")

    question = question.strip()

    if len(question) > 10000:
        raise ValueError("Question is too long (maximum 10,000 characters).")

    text = question.lower()

    # Match dataset column names mentioned in the question.
    matched_columns = [
        col for col in columns
        if re.search(
            r"(?<!\w)" + re.escape(str(col).lower()) + r"(?!\w)",
            text,
        )
    ]

    # Identify the requested calculation.
    if any(word in text for word in ["average", "mean"]):
        metric = "mean"
    elif any(word in text for word in ["count", "number of"]):
        metric = "count"
    elif any(word in text for word in ["maximum", "highest", "top", "largest"]):
        metric = "top"
    elif any(word in text for word in ["minimum", "lowest", "smallest"]):
        metric = "bottom"
    elif any(word in text for word in ["growth", "change", "increase", "decline"]):
        metric = "change"
    elif any(word in text for word in ["monthly", "by month", "month-wise"]):
        metric = "monthly"
    elif any(word in text for word in ["quarterly", "by quarter"]):
        metric = "quarterly"
    elif any(word in text for word in ["yearly", "annually", "by year"]):
        metric = "yearly"
    elif any(word in text for word in ["total", "sum", "sales", "revenue"]):
        metric = "sum"
    else:
        metric = "unknown"

    # Detect a likely grouping column.
    group_column = None
    for col in matched_columns:
        if str(col).lower() in {
            "region", "city", "state", "product", "category",
            "customer", "customer_id", "channel", "salesperson",
            "segment", "territory",
        }:
            group_column = col
            break

    return {
        "question": question,
        "metric": metric,
        "matched_columns": matched_columns,
        "group_column": group_column,
    }