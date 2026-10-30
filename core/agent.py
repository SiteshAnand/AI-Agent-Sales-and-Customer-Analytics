
import json
import re

import pandas as pd

from config.settings import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    MAX_QUERY_LENGTH,
)
from core.data_quality_audit import data_quality_audit
from core.rule_based_analysis import (
    dataset_overview,
    monthly_region_analysis,
    sales_decline_investigation,
)
from core.tools import run_tool
from core.response import format_success, format_error
from core.forecasting import (
    forecast_product_sales,
    forecast_total_sales,
    is_product_forecast_question,
)
from core.market_basket import market_basket_analysis
from core.query_intelligence import split_multifold_question
from core.result_validation import combine_task_results, validate_result


ALLOWED_TOOLS = {
    "numeric_summary",
    "distinct_count",
    "group_metric",
    "filtered_metric",
    "period_totals",
    "data_profile",
    "regression",
    "churn_model",
    "rfm",
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
    """Answer clear highest/lowest salesperson questions locally."""
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
    """Ask Gemini to map a business question to an approved tool."""
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
{{{{"column": "numeric column"}}}}

2. distinct_count
{{{{"column": "column"}}}}

3. group_metric
{{{{"group_column": "category column",
  "value_column": "numeric column",
  "metric": "sum|mean|median|min|max|count|std|variance",
  "top_n": 10,
  "sort_order": "asc|desc"}}}}

4. filtered_metric
{{{{"filter_column": "category column",
  "filter_value": "category value",
  "value_column": "numeric column",
  "metric": "sum|mean|median|min|max|count|std|variance"}}}}

5. period_totals
{{{{"date_column": "date column",
  "value_column": "numeric column",
  "frequency": "ME|QE|YE"}}}}

6. data_profile
{{{{}}}}

7. regression
{{{{"target_column": "numeric target column",
  "feature_columns": ["feature column 1", "feature column 2"],
  "algorithm": "auto",
  "test_size": 0.2,
  "cv_folds": 5,
  "validation_strategy": "random"}}}}

8. churn_model
{{{{"target_column": "churn label column",
  "feature_columns": ["feature column 1", "feature column 2"],
  "test_size": 0.2,
  "cv_folds": 5,
  "probability_threshold": 0.5}}}}

9. rfm
{{{{"customer_id_column": "customer identifier column",
  "date_column": "transaction date column",
  "transaction_value_column": "transaction amount column",
  "quantiles": 5}}}}

Use only existing dataset columns and valid category values.
If the question is ambiguous, unsupported, or requires missing data,
return tool="unsupported" and explain the limitation.
Do not invent figures, causes, columns, or category values.
Never generate SQL or Python code.
Never supply your own numeric inputs for a calculation.

Use regression only when a numeric target and suitable input features exist.
Use churn_model only when the dataset contains a meaningful churn label.
Use rfm only when customer identifiers, transaction dates, and transaction values exist.
Never use an identifier as a numeric prediction feature.
Never claim that a model proves causation.

For regression and churn, feature_columns may be omitted to let the
analytics tool choose its default features, if supported.
Do not select a target column as an input feature.
Use regression for a numeric outcome and churn_model for a categorical
churn outcome. Do not confuse the two tasks.
Use rfm for transaction-level customer data, not one-row-per-customer
aggregates unless they include the required transaction information.

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
        "regression": ["target_column"],
        "churn_model": ["target_column"],
        "rfm": [
            "customer_id_column",
            "date_column",
            "transaction_value_column",
        ],
        "data_profile": [],
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
            df[filter_column].dropna().astype(str).str.strip()
        )
        valid_values = {
            value.casefold() for value in available_values if value
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

    if tool_name in {"regression", "churn_model"}:
        features = parameters.get("feature_columns")

        if features is not None:
            if not isinstance(features, list) or not features:
                raise ValueError(
                    "feature_columns must be a non-empty list of columns."
                )

            if len(features) != len(set(features)):
                raise ValueError("feature_columns must not contain duplicates.")

            for column in features:
                if not isinstance(column, str) or column not in df.columns:
                    raise ValueError(
                        f"Feature column '{column}' does not exist."
                    )

            if parameters["target_column"] in features:
                raise ValueError(
                    "The target column cannot also be a feature."
                )

        test_size = parameters.get("test_size", 0.2)
        if (
            isinstance(test_size, bool)
            or not isinstance(test_size, (int, float))
            or not 0.1 <= test_size <= 0.5
        ):
            raise ValueError("test_size must be between 0.1 and 0.5.")

        cv_folds = parameters.get("cv_folds", 5)
        if type(cv_folds) is not int or not 2 <= cv_folds <= 10:
            raise ValueError("cv_folds must be an integer between 2 and 10.")

    if tool_name == "regression":
        algorithm = parameters.get("algorithm", "auto")
        if not isinstance(algorithm, str) or not algorithm.strip():
            raise ValueError("algorithm must be a non-empty string.")

        strategy = parameters.get("validation_strategy", "random")
        if strategy not in {"random", "time"}:
            raise ValueError(
                "validation_strategy must be 'random' or 'time'."
            )

    if tool_name == "churn_model":
        threshold = parameters.get("probability_threshold", 0.5)
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not 0 <= threshold <= 1
        ):
            raise ValueError(
                "probability_threshold must be between 0 and 1."
            )

    if tool_name == "rfm":
        quantiles = parameters.get("quantiles", 5)
        if type(quantiles) is not int or not 2 <= quantiles <= 10:
            raise ValueError("quantiles must be an integer between 2 and 10.")

        snapshot_date = parameters.get("snapshot_date")
        if snapshot_date is not None and not isinstance(snapshot_date, str):
            raise ValueError("snapshot_date must be a date string or null.")



def _try_high_value_inactive_customers(question: str, df: pd.DataFrame):
    """Deterministically rank high-value customers with no recent purchases."""
    text = question.casefold()
    customer_intent = bool(re.search(r"\bcustomer(s)?\b|\bclient(s)?\b", text))
    inactivity_intent = any(term in text for term in (
        "not bought", "not bought recently", "did not buy", "did not bought", "did not bught", "didn't buy",
        "not purchased", "haven't purchased", "have not purchased",
        "inactive", "gone inactive", "no recent purchase", "not buying",
        "stopped buying", "not made recent", "recent time", "recently",
    ))
    value_intent = any(term in text for term in (
        "high sales", "highest sales", "high-value", "high value",
        "highest value", "sales value", "monetary", "valuable",
        "big spender", "top customer", "top customers",
    ))
    if not (customer_intent and inactivity_intent and value_intent):
        return None

    customer_col = _find_column(df, [
        "Customer_ID", "Customer ID", "Customer", "Customer_Name",
        "Customer Name", "Client_ID", "Client Name",
    ])
    date_col = _find_column(df, [
        "Order_Date", "Order Date", "Transaction_Date", "Transaction Date",
        "Purchase_Date", "Purchase Date", "Date",
    ])
    sales_col = _find_column(df, [
        "Net_Sales_INR", "Net Sales INR", "Net Sales", "Sales_Value",
        "Sales Value", "Sales_Amount", "Sales Amount", "Revenue",
        "Total_Amount_INR", "Gross_Sales_INR", "Sales",
    ])

    missing = [
        label for label, column in (
            ("customer identifier", customer_col),
            ("transaction date", date_col),
            ("sales value", sales_col),
        ) if column is None
    ]
    if missing:
        return format_error(
            "I can identify inactive high-value customers only when the "
            "dataset has a customer identifier, transaction date, and sales "
            f"value column. Missing: {', '.join(missing)}."
        )

    # Explicit day threshold wins; otherwise use a clear 90-day default.
    day_match = re.search(r"\b(\d{1,3})\s*days?\b", text)
    month_match = re.search(r"\b(\d{1,2})\s*months?\b", text)
    if day_match:
        threshold_days = int(day_match.group(1))
    elif month_match:
        threshold_days = int(month_match.group(1)) * 30
    else:
        threshold_days = 90

    if not 1 <= threshold_days <= 3650:
        return format_error("Choose an inactivity threshold between 1 and 3650 days.")

    data = df[[customer_col, date_col, sales_col]].copy()
    data[customer_col] = data[customer_col].astype("string").str.strip()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce", utc=True)
    data[sales_col] = pd.to_numeric(data[sales_col], errors="coerce")
    data = data.dropna(subset=[customer_col, date_col, sales_col])
    data = data[
        data[customer_col].ne("")
        & data[sales_col].map(lambda value: pd.notna(value) and abs(value) != float("inf"))
    ].copy()

    if data.empty:
        return format_error(
            "No valid customer transactions remain after validating IDs, dates, and sales values."
        )

    # Anchor recency to the latest transaction date in the uploaded dataset,
    # avoiding dependence on the computer's clock or stale sample data.
    snapshot_date = data[date_col].max().normalize() + pd.Timedelta(days=1)
    grouped = data.groupby(customer_col, dropna=True).agg(
        Last_Purchase=(date_col, "max"),
        Transaction_Count=(date_col, "size"),
        Total_Sales=(sales_col, "sum"),
    ).reset_index()
    grouped["Days_Inactive"] = (
        snapshot_date - grouped["Last_Purchase"].dt.normalize()
    ).dt.days

    # Only customers who meet the explicit inactivity threshold are eligible.
    inactive = grouped[grouped["Days_Inactive"] >= threshold_days].copy()
    if inactive.empty:
        result = {
            "threshold_days": threshold_days,
            "snapshot_date": snapshot_date.date().isoformat(),
            "customer_count": 0,
            "rows": [],
            "markdown": (
                f"No customers met the **{threshold_days}-day inactivity "
                f"threshold** as of {snapshot_date.date().isoformat()}."
            ),
        }
        output = format_success("inactive_high_value_customers", result)
        output["question"] = question
        output["explanation"] = "No customers met the inactivity threshold."
        return output

    # Rank by historical net sales; negative/return-heavy totals are retained
    # and surfaced transparently rather than silently removed.
    inactive = inactive.sort_values(
        ["Total_Sales", "Days_Inactive"],
        ascending=[False, False],
        kind="stable",
    ).head(20)

    rows = []
    for _, row in inactive.iterrows():
        rows.append({
            "Customer": str(row[customer_col]),
            "Last_Purchase": row["Last_Purchase"].date().isoformat(),
            "Days_Inactive": int(row["Days_Inactive"]),
            "Transactions": int(row["Transaction_Count"]),
            "Total_Sales": round(float(row["Total_Sales"]), 2),
        })

    lines = [
        f"### Top inactive high-value customers",
        "",
        f"**{len(rows)} customers shown** · Inactive for at least "
        f"**{threshold_days} days** · Snapshot date: "
        f"**{snapshot_date.date().isoformat()}**",
        "",
        "| Customer | Last purchase | Days inactive | Transactions | Historical sales |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['Customer']} | {row['Last_Purchase']} "
            f"| {row['Days_Inactive']:,} | {row['Transactions']:,} "
            f"| ₹{row['Total_Sales']:,.2f} |"
        )
    lines.extend([
        "",
        "*Ranked by total historical value in the selected sales column. "
        "Returns/negative transactions are included if present. Inactivity "
        "is measured against the latest transaction date in the uploaded data.*",
    ])

    result = {
        "threshold_days": threshold_days,
        "snapshot_date": snapshot_date.date().isoformat(),
        "customer_count": int(len(inactive)),
        "rows": rows,
        "markdown": "\n".join(lines),
        "sales_column_used": sales_col,
    }
    output = format_success("inactive_high_value_customers", result)
    output["question"] = question
    output["explanation"] = (
        f"Ranked inactive customers by historical sales, using a "
        f"{threshold_days}-day threshold."
    )
    return output

def _requested_horizon(question: str, default: int = 1) -> int:
    """Extract an explicit 1-3 month horizon from ordinary wording."""
    text = str(question).casefold()
    match = re.search(r"\b(1|2|3|one|two|three)\s*[- ]?(?:months?|month|mths?)\b", text)
    if match:
        words = {"one": 1, "two": 2, "three": 3}
        token = match.group(1)
        return words.get(token, int(token) if token.isdigit() else default)
    if "quarter" in text or "next quarter" in text:
        return 3
    if "next month" in text or "coming month" in text or "following month" in text:
        return 1
    return default


def _try_market_basket(question: str, df: pd.DataFrame):
    text = question.casefold()
    if not any(term in text for term in ("market basket", "association rule", "products bought together", "items bought together", "frequently purchased together", "bought together")):
        return None
    transaction_col = _find_column(df, ["Transaction_ID", "Transaction ID", "Order_ID", "Order ID", "Invoice", "InvoiceNo", "Basket_ID", "Receipt_ID"])
    product_col = _find_column(df, ["Product_Name", "Product Name", "Product", "Item", "Item_Name", "SKU", "Product_ID"])
    if transaction_col is None or product_col is None:
        return format_error("Market Basket Analysis needs a transaction/order ID column and a product/item column. The uploaded data does not contain both.")
    support_match = re.search(r"\b(?:min(?:imum)?\s+)?support\s*(?:=|of|at least)?\s*(0?\.\d+|1(?:\.0+)?)", text)
    confidence_match = re.search(r"\b(?:min(?:imum)?\s+)?confidence\s*(?:=|of|at least)?\s*(0?\.\d+|1(?:\.0+)?)", text)
    lift_match = re.search(r"\b(?:min(?:imum)?\s+)?lift\s*(?:=|of|at least)?\s*(\d+(?:\.\d+)?)", text)
    try:
        itemset_match = re.search(r"\b(?:up\s+to\s+)?([2-4])\s*(?:-?item|product)\s+(?:combinations|itemsets|rules)\b", text)
        max_itemset_size = int(itemset_match.group(1)) if itemset_match else 3
        result = market_basket_analysis(df, transaction_col, product_col,
            min_support=float(support_match.group(1)) if support_match else 0.01,
            min_confidence=float(confidence_match.group(1)) if confidence_match else 0.1,
            min_lift=float(lift_match.group(1)) if lift_match else 1.0,
            max_itemset_size=max_itemset_size)
        output = format_success("market_basket_analysis", result)
        output["question"] = question
        output["explanation"] = "Product association rules calculated from observed transactions, including multi-item rules within the configured search bound."
        return output
    except Exception as exc:
        return format_error(f"Could not complete Market Basket Analysis: {exc}")


def _split_multifold_question(question: str) -> list[str]:
    """Compatibility wrapper for the dedicated query-intelligence layer."""
    return split_multifold_question(question)


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

    # Handle clearly separated multi-part questions independently, then combine.
    parts = _split_multifold_question(question)
    if len(parts) > 1:
        results = []
        for part in parts:
            item = ask_agent(part, df)
            results.append({"question": part, "status": item.get("status", "error"),
                            "tool": item.get("tool"),
                            "result": item.get("result") if item.get("status") == "success" else None,
                            "message": item.get("message") if item.get("status") != "success" else item.get("explanation")})
        combined = combine_task_results(question, results)
        combined["result"]["validation"] = validate_result(combined)
        return combined

    # Route explicit Market Basket Analysis requests to deterministic code.
    basket_result = _try_market_basket(question, df)
    if basket_result is not None:
        return basket_result

    # Route explicit future product-sales questions to deterministic forecasting
    # before asking the LLM to select from its general-purpose tool list.
    if is_product_forecast_question(question):
        horizon = _requested_horizon(question, default=1)
        wants_product = bool(re.search(r"\b(product|sku|item)-?wise\b|\bby product\b|\bfor each product\b|\bproduct sales\b|\bsales by product\b", text))
        try:
            if wants_product:
                forecast_result = forecast_product_sales(df, horizon=horizon)
                tool_name = "product_sales_forecast"
                description = f"Product-wise monthly sales forecast for the next {horizon} month(s)."
            else:
                forecast_result = forecast_total_sales(df, horizon=horizon)
                tool_name = "total_sales_forecast"
                description = f"Total monthly sales forecast for the next {horizon} month(s)."
            output = format_success(tool_name, forecast_result)
            output["question"] = question
            output["explanation"] = description + " Estimates are based on historical data and are not guarantees."
            return output
        except Exception as exc:
            return format_error(f"Could not produce a reliable sales forecast: {exc}")

    try:
        inactive_customers = _try_high_value_inactive_customers(question, df)
        if inactive_customers is not None:
            return inactive_customers
    except Exception as exc:
        return format_error(
            f"Could not safely identify inactive high-value customers: {exc}"
        )

    overview_terms = (
        "what columns", "available columns", "dataset columns",
        "what analysis", "what business analysis",
        "what can you analyse", "what can you analyze", "capabilities",
    )
    if any(term in text for term in overview_terms):
        return dataset_overview(question, df)

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

    try:
        decline_result = sales_decline_investigation(question, df)
        if decline_result is not None:
            return decline_result
    except Exception as exc:
        return format_error(
            f"Could not complete sales decline investigation safely: {exc}"
        )

    try:
        multi_step = monthly_region_analysis(question, df)
        if multi_step is not None:
            return multi_step
    except Exception as exc:
        return format_error(
            f"Could not complete monthly analysis safely: {exc}"
        )

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
            try:
                client.close()
            except Exception:
                pass
