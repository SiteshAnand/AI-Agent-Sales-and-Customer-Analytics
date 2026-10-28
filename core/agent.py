
import json
import re

import pandas as pd
from core.data_quality_audit import data_quality_audit
from config.settings import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    MAX_QUERY_LENGTH,
)
from core.tools import run_tool
from core.response import format_success, format_error
from core.rule_based_analysis import (
    dataset_overview,
    monthly_region_analysis,
    sales_decline_investigation,
)


# Only tools that can be safely selected by Gemini are listed here.
# percentage_change is intentionally excluded because its current/previous
# arguments could otherwise be invented by the model rather than calculated
# from the uploaded dataset.
ALLOWED_TOOLS = {
    "numeric_summary",
    "distinct_count",
    "group_metric",
    "filtered_metric",
    "period_totals",
    "data_profile",
}

SUPPORTED_METRICS = {
    "sum", "mean", "median", "min", "max", "count", "std", "variance"
}


def _get_client():
    """Create a Gemini client using local environment settings."""
    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError(
            "Gemini dependency is not installed. Run: "
            "pip install -r requirements.txt"
        ) from exc

    if not GEMINI_API_KEY:
        raise ValueError(
            "Gemini API key is missing. Check your local .env file."
        )

    if not GEMINI_MODEL:
        raise ValueError(
            "Gemini model is missing. Check GEMINI_MODEL in .env."
        )

    return genai.Client(api_key=GEMINI_API_KEY)


def _find_column(df, aliases):
    """Find a column by normalized name."""
    normalized = {
        re.sub(r"[^a-z0-9]+", "", str(column).casefold()): column
        for column in df.columns
    }

    for alias in aliases:
        key = re.sub(r"[^a-z0-9]+", "", alias.casefold())
        if key in normalized:
            return normalized[key]

    return None


def _try_direct_sales_ranking(question, df):
    """Calculate clear highest/lowest salesperson sales questions locally."""
    text = question.casefold()

    if not re.search(r"\bsales?\b|\brevenue\b", text):
        return None

    if any(word in text for word in (
        "monthly", "quarterly", "yearly", "trend", "growth",
        "change", "compare", "region", "zone", "product",
    )):
        return None

    person_column = _find_column(
        df,
        ["Sales Person", "Salesperson", "Sales_Person", "Sales Person Name"],
    )
    value_column = _find_column(
        df, ["VALUE", "Sales", "Sales Amount", "Revenue"]
    )

    if person_column is None or value_column is None:
        return None

    wants_lowest = bool(re.search(
        r"\b(lowest|least|minimum|smallest|bottom|worst)\b", text
    ))
    wants_highest = bool(re.search(
        r"\b(highest|most|maximum|largest|top|best)\b", text
    ))

    if wants_lowest == wants_highest:
        # Avoid guessing when no direction or conflicting directions exist.
        return None

    people = df[person_column].dropna().astype(str).str.strip()
    if not people[people != ""].any():
        return None

    number_words = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "twenty": 20,
    }

    count_pattern = (
        r"\b(?:top|bottom|lowest|highest|best|worst|least|most)\s+(\d{1,4})\b"
        r"|\b(\d{1,4})\s+(?:lowest|highest|best|worst|top|bottom)\b"
    )
    count_match = re.search(count_pattern, text)

    if not count_match:
        word_pattern = (
            r"\b(?:top|bottom|lowest|highest|best|worst|least|most)\s+"
            r"(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty)\b"
            r"|\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty)\s+"
            r"(?:lowest|highest|best|worst|top|bottom)\b"
        )
        count_match = re.search(word_pattern, text)

    requested_n = 1
    if count_match:
        token = next(
            (group for group in count_match.groups() if group is not None),
            None,
        )
        requested_n = int(token) if token.isdigit() else number_words[token]

    requested_n = max(1, min(requested_n, 1000))

    result = run_tool(
        "group_metric",
        df,
        {
            "group_column": person_column,
            "value_column": value_column,
            "metric": "sum",
            "top_n": requested_n,
            "sort_order": "asc" if wants_lowest else "desc",
        },
    )

    rows = result.get("results", [])
    if not rows:
        return format_error("No valid salesperson sales values were found.")

    output = format_success("group_metric", result)
    output["question"] = question
    output["explanation"] = (
        f"Salespeople ranked by total sales in "
        f"{'ascending' if wants_lowest else 'descending'} order; "
        f"showing {len(rows)} result(s)."
    )
    return output


def _build_prompt(question: str, df: pd.DataFrame) -> str:
    """Ask Gemini to map a question to an approved analytics tool."""
    columns = []

    for column in df.columns:
        sample_values = df[column].dropna().head(3).astype(str).tolist()
        columns.append({
            "name": str(column),
            "data_type": str(df[column].dtype),
            "sample_values": sample_values,
        })

    return f"""
You interpret business questions for a business analytics application.
Python performs every calculation. Never invent numerical results.

Approved tools and parameters:

1. numeric_summary
{{"column": "numeric column"}}

2. distinct_count
{{"column": "column"}}

3. group_metric
{{"group_column": "category column",
  "value_column": "numeric column",
  "metric": "sum|mean|median|min|max|count|std|variance",
  "top_n": 10,
  "sort_order": "asc|desc"}}

4. filtered_metric
{{"filter_column": "category column",
  "filter_value": "category value",
  "value_column": "numeric column",
  "metric": "sum|mean|median|min|max|count|std|variance"}}

5. period_totals
{{"date_column": "date column",
  "value_column": "numeric column",
  "frequency": "ME|QE|YE"}}

6. data_profile
{{}}
Use only existing dataset columns and valid category values.
If the question is ambiguous, unsupported, or requires missing data,
return tool="unsupported" and explain the limitation.
Do not invent figures, causes, columns, or category values.
Never generate SQL or Python code.
Never supply your own numeric inputs for a calculation.

Dataset columns and examples:
{json.dumps(columns, ensure_ascii=False)}

User question:
{question}

Return only a valid JSON object with keys tool, parameters, and explanation.
"""


def _validate_plan(plan: dict, df: pd.DataFrame) -> None:
    """Validate Gemini's proposed tool and column references."""
    if not isinstance(plan, dict):
        raise ValueError("Gemini returned an invalid response.")

    tool_name = plan.get("tool")

    if tool_name == "unsupported":
        explanation = plan.get(
            "explanation",
            "This question is not supported yet.",
        )
        raise ValueError(str(explanation))

    if tool_name not in ALLOWED_TOOLS:
        raise ValueError("The requested analytics operation is unsupported.")

    parameters = plan.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("The analytics parameters are invalid.")

    column_parameters = {
        "numeric_summary": ["column"],
        "distinct_count": ["column"],
        "group_metric": ["group_column", "value_column"],
        "filtered_metric": ["filter_column", "value_column"],
        "period_totals": ["date_column", "value_column"],
    }

    for parameter_name in column_parameters.get(tool_name, []):
        column_name = parameters.get(parameter_name)
        if not isinstance(column_name, str) or column_name not in df.columns:
            raise ValueError(
                f"Column '{column_name}' does not exist in the dataset."
            )

    if tool_name in {"group_metric", "filtered_metric"}:
        metric = parameters.get("metric", "sum")
        if not isinstance(metric, str) or metric not in SUPPORTED_METRICS:
            raise ValueError("Unsupported aggregation metric.")

    if tool_name == "group_metric":
        sort_order = parameters.get("sort_order", "desc")
        if sort_order not in {"asc", "desc"}:
            raise ValueError("sort_order must be 'asc' or 'desc'.")

        top_n = parameters.get("top_n", 10)
        if type(top_n) is not int or not 1 <= top_n <= 1000:
            raise ValueError("top_n must be an integer between 1 and 1000.")

    if tool_name == "filtered_metric":
        filter_value = parameters.get("filter_value")
        if not isinstance(filter_value, str) or not filter_value.strip():
            raise ValueError("A valid filter value is required.")

        filter_column = parameters["filter_column"]
        available_values = (
            df[filter_column]
            .dropna()
            .astype(str)
            .str.strip()
        )
        valid_values = {
            value.casefold()
            for value in available_values
            if value
        }

        if filter_value.strip().casefold() not in valid_values:
            raise ValueError(
                f"Filter value '{filter_value}' was not found in "
                f"column '{filter_column}'."
            )

    if tool_name == "period_totals":
        frequency = parameters.get("frequency", "ME")
        if frequency not in {"ME", "QE", "YE"}:
            raise ValueError("Frequency must be ME, QE, or YE.")


def ask_agent(question: str, df: pd.DataFrame) -> dict:
    """Interpret a business question and execute an approved analysis."""
    if not isinstance(question, str) or not question.strip():
        return format_error("Please enter a business question.")

    question = question.strip()

    if len(question) > MAX_QUERY_LENGTH:
        return format_error(
            f"Question must be no longer than {MAX_QUERY_LENGTH} characters."
        )

    if not isinstance(df, pd.DataFrame) or df.empty:
        return format_error("Please upload a dataset containing data.")

    text = question.casefold()

    overview_terms = (
        "what columns", "available columns", "dataset columns",
        "what analysis", "what business analysis",
        "what can you analyse", "what can you analyze", "capabilities",
    )
    if any(term in text for term in overview_terms):
        return dataset_overview(question, df)

    
    # Run the data quality audit locally without using Gemini.
    audit_terms = (
        "data quality",
        "quality audit",
        "audit the dataset",
        "data validation report",
        "check data quality",
    )

    if any(term in text for term in audit_terms):
        try:
            return {
                "status": "success",
                "tool": "data_quality_audit",
                "question": question,
                "explanation": "Read-only data quality audit completed.",
                "result": data_quality_audit(df),
            }
        except Exception as exc:
            return format_error(
                f"Could not complete the data quality audit: {exc}"
            )


    # Investigate sales declines before sending the question to Gemini.
    try:
        decline_result = sales_decline_investigation(question, df)
        if decline_result is not None:
            return decline_result
    except Exception as exc:
        return format_error(
            f"Could not complete sales decline investigation safely: {exc}"
        )

    # Existing deterministic monthly-by-region analysis.
    try:
        multi_step = monthly_region_analysis(question, df)
        if multi_step is not None:
            return multi_step
    except Exception as exc:
        return format_error(
            f"Could not complete monthly analysis safely: {exc}"
        )

    # Existing deterministic salesperson ranking.
    try:
        direct_result = _try_direct_sales_ranking(question, df)
        if direct_result is not None:
            return direct_result
    except Exception as exc:
        return format_error(str(exc))

    client = None
    try:
        client = _get_client()
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=_build_prompt(question, df),
            config={
                "response_mime_type": "application/json",
                "temperature": 0,
            },
        )

        if not response.text:
            raise ValueError("Gemini returned an empty response.")

        plan = json.loads(response.text)
        _validate_plan(plan, df)

        result = run_tool(plan["tool"], df, plan["parameters"])
        output = format_success(plan["tool"], result)
        output["question"] = question
        output["explanation"] = str(
            plan.get("explanation", "Calculation completed.")
        )
        return output

    except json.JSONDecodeError:
        return format_error(
            "Gemini returned an unreadable response. Please try again."
        )
    except Exception as exc:
        message = str(exc)

        if any(word in message.lower() for word in (
            "api key", "unauthorized", "permission denied",
        )):
            message = (
                "Gemini authentication failed. Check your API key "
                "and API access in your local configuration."
            )

        return format_error(message)
    finally:
        if client is not None:
            client.close()
