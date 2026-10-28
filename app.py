
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from core.agent import ask_agent
from core.data_loader import load_sales_file


# ---------------------------------------------------------
# APP CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Business Analysis & Customer Success Agent",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Business Analysis & Customer Success AI Agent")
st.caption(
    "Upload a business dataset and ask questions in plain English. "
    "Python performs the calculations; Gemini interprets your question."
)


# ---------------------------------------------------------
# PERMANENT DATASET STORAGE
# ---------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = PROJECT_DIR / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

METADATA_FILE = UPLOAD_DIR / "active_dataset.json"
SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


class LocalDatasetFile:
    """Adapt a locally saved file to the existing data loader."""

    def __init__(self, path):
        self.name = path.name
        self._content = path.read_bytes()

    def getvalue(self):
        return self._content


def load_saved_dataset():
    """Restore the active dataset from permanent local storage."""

    if not METADATA_FILE.exists():
        return None, None

    metadata = json.loads(
        METADATA_FILE.read_text(encoding="utf-8")
    )

    stored_filename = metadata.get("stored_filename", "")
    saved_path = (UPLOAD_DIR / stored_filename).resolve()

    if saved_path.parent != UPLOAD_DIR.resolve():
        raise ValueError("Invalid saved dataset path.")

    if not saved_path.is_file():
        raise FileNotFoundError(
            "The saved dataset file is missing. Please upload it again."
        )

    df = load_sales_file(LocalDatasetFile(saved_path))
    original_filename = metadata.get(
        "original_filename",
        stored_filename,
    )

    return df, original_filename


def save_uploaded_dataset(uploaded_file):
    """Validate and permanently save an uploaded dataset."""

    extension = Path(uploaded_file.name).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError("Upload a CSV, XLSX, or XLS file.")

    # Validate using the existing data loader.
    df = load_sales_file(uploaded_file)
    content = uploaded_file.getvalue()

    saved_path = UPLOAD_DIR / f"active_dataset{extension}"
    temporary_path = UPLOAD_DIR / f"upload_temp{extension}"

    try:
        temporary_path.write_bytes(content)
        temporary_path.replace(saved_path)

        # Remove previous active dataset files after saving.
        for old_path in UPLOAD_DIR.glob("active_dataset.*"):
            if old_path != saved_path:
                old_path.unlink(missing_ok=True)

        metadata = {
            "original_filename": uploaded_file.name,
            "stored_filename": saved_path.name,
        }

        METADATA_FILE.write_text(
            json.dumps(
                metadata,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        return df, uploaded_file.name

    finally:
        temporary_path.unlink(missing_ok=True)


def delete_saved_dataset():
    """Delete the saved dataset after explicit user action."""

    if METADATA_FILE.exists():
        try:
            metadata = json.loads(
                METADATA_FILE.read_text(encoding="utf-8")
            )

            stored_filename = metadata.get("stored_filename", "")
            saved_path = (UPLOAD_DIR / stored_filename).resolve()

            if saved_path.parent == UPLOAD_DIR.resolve():
                saved_path.unlink(missing_ok=True)

        finally:
            METADATA_FILE.unlink(missing_ok=True)

    for saved_path in UPLOAD_DIR.glob("active_dataset.*"):
        saved_path.unlink(missing_ok=True)


# ---------------------------------------------------------
# DISPLAY HELPERS
# ---------------------------------------------------------

def format_number(value):
    """Format numeric values for business-friendly display."""

    if isinstance(value, bool):
        return str(value)

    if isinstance(value, (int, float)):
        return f"{value:,.2f}".rstrip("0").rstrip(".")

    return str(value)

    
def format_data_quality_audit(report):
    """Convert data-quality audit results into a readable report."""

    if not isinstance(report, dict):
        return "The data quality audit returned an unexpected result."

    row_count = report.get("row_count", 0)
    column_count = report.get("column_count", 0)
    duplicate_rows = report.get("duplicate_rows", 0)
    missing_columns = report.get("missing_columns", [])
    findings = report.get("findings", [])
    finding_counts = report.get("finding_counts", {})

    if not isinstance(missing_columns, list):
        missing_columns = []

    if not isinstance(findings, list):
        findings = []

    if not isinstance(finding_counts, dict):
        finding_counts = {}

    lines = [
        "# Data Quality Audit Report",
        "",
        "## 1. Executive Summary",
        "",
        f"- **Total rows:** {row_count:,}",
        f"- **Total columns:** {column_count:,}",
        f"- **Duplicate rows:** {duplicate_rows:,}",
        f"- **Missing-value columns:** {len(missing_columns):,}",
        f"- **Total findings:** {len(findings):,}",
        "",
        "## 2. Finding Severity",
        "",
        f"- **High:** {finding_counts.get('high', 0)}",
        f"- **Medium:** {finding_counts.get('medium', 0)}",
        f"- **Low:** {finding_counts.get('low', 0)}",
        "",
        "## 3. Missing Values",
        "",
    ]

    if missing_columns:
        lines.extend([
            "| Column | Missing values | Missing (%) |",
            "|---|---:|---:|",
        ])

        for item in missing_columns:
            if not isinstance(item, dict):
                continue

            column = str(item.get("column", "Unknown"))
            count = item.get("missing_count", 0)
            percent = item.get("missing_percent", 0)

            lines.append(
                f"| {column.replace('|', '\\|')} "
                f"| {count} | {percent}% |"
            )
    else:
        lines.append("No missing-value columns were reported.")

    lines.extend([
        "",
        "## 4. Audit Findings",
        "",
    ])

    if findings:
        lines.extend([
            "| Severity | Check | Column | Count | Details |",
            "|---|---|---|---:|---|",
        ])

        for item in findings:
            if not isinstance(item, dict):
                continue

            severity = str(item.get("severity", "Info"))
            check = str(item.get("check", ""))
            column = str(item.get("column", "—"))
            count = item.get("count", 0)
            details = str(item.get("details", ""))

            # Escape Markdown table separators.
            check = check.replace("|", "\\|")
            column = column.replace("|", "\\|")
            details = details.replace("|", "\\|").replace("\n", " ")

            lines.append(
                f"| {severity} | {check} | {column} "
                f"| {count} | {details} |"
            )
    else:
        lines.append(
            "No issues were flagged by the configured audit checks."
        )

    audit_note = report.get("audit_note")
    if audit_note:
        lines.extend(["", f"**Note:** {audit_note}"])

    return "\n".join(lines)



def format_data_profile(profile):
    """Convert data-profile results into a management-friendly report."""

    if not isinstance(profile, dict):
        return "The data profile returned an unexpected result."

    row_count = profile.get("row_count", 0)
    column_count = profile.get("column_count", 0)
    duplicate_rows = profile.get("duplicate_rows", 0)
    columns = profile.get("columns", [])

    if not isinstance(columns, list):
        columns = []

    columns_with_missing = [
        col for col in columns
        if isinstance(col, dict)
        and col.get("missing_count", 0) > 0
    ]

    lines = [
        "# Dataset Data Quality Report",
        "",
        "## 1. Executive Summary",
        "",
        f"- **Total rows:** {row_count:,}",
        f"- **Total columns:** {column_count:,}",
        f"- **Duplicate rows:** {duplicate_rows:,}",
        (
            f"- **Columns with missing values:** "
            f"{len(columns_with_missing):,} of {column_count:,}"
        ),
        "",
        "## 2. Missing-Value Findings",
        "",
    ]

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
        (
            "| Column | Data type | Non-null | Distinct | "
            "Missing % | Sample values |"
        ),
        "|---|---|---:|---:|---:|---|",
    ])

    for col in columns:
        if not isinstance(col, dict):
            continue

        samples = ", ".join(
            str(value)
            .replace("|", "\\|")
            .replace("\n", " ")
            for value in col.get("sample_values", [])
        )

        column_name = str(col.get("column", "")).replace("|", "\\|")
        data_type = str(col.get("data_type", "")).replace("|", "\\|")

        lines.append(
            f"| {column_name} "
            f"| {data_type} "
            f"| {col.get('non_null_count', 0):,} "
            f"| {col.get('distinct_count', 0):,} "
            f"| {col.get('missing_percent', 0):.2f}% "
            f"| {samples} |"
        )

    lines.extend([
        "",
        "## 4. Recommended Checks",
        "",
        "- Confirm whether missing values are expected in each field.",
        "- Validate date fields before time-series or delivery-duration analysis.",
        "- Treat phone numbers and other identifiers as identifiers, not measures.",
        (
            "- Confirm the data grain before summing customer lifetime values "
            "or sales targets."
        ),
        "- Review duplicate records using the business definition of a duplicate.",
        "",
        (
            "*These recommendations identify checks to perform; they do not "
            "prove that a data error exists.*"
        ),
    ])

    return "\n".join(lines)


def create_answer(result):
    """Convert the agent's structured response into readable text."""

    if not isinstance(result, dict):
        return "**Analysis could not be completed.** The response was invalid."

    if result.get("status") != "success":
        return (
            "**Analysis could not be completed.**\n\n"
            + str(result.get("message", "An unexpected error occurred."))
        )

    explanation = result.get("explanation", "Analysis completed.")
    result_data = result.get("result", {})
    tool_name = result.get("tool", "")

        # Data quality audit: management-friendly report.
    if tool_name == "data_quality_audit":
        return format_data_quality_audit(result_data)

    # Data profile: management-friendly report.
    if tool_name == "data_profile":
        return format_data_profile(result_data)

    # Multi-step deterministic analyses.
    if tool_name in {
        "monthly_region_analysis",
        "dataset_overview",
    } and isinstance(result_data, dict):
        return result_data.get("markdown", explanation)

    if not isinstance(result_data, dict):
        return f"**{explanation}**\n\n{result_data}"

    # Filtered metric.
    if tool_name == "filtered_metric" or (
        "filter_value" in result_data
        and "result" in result_data
        and "value_column" in result_data
    ):
        metric = result_data.get("metric", "sum")
        value = format_number(result_data.get("result", 0))
        person = result_data.get("filter_value", "Selected entity")
        value_column = result_data.get("value_column", "Value")

        metric_labels = {
            "sum": "Total",
            "mean": "Average",
            "count": "Count",
        }

        label = metric_labels.get(metric, str(metric).title())

        return f"**{label} {value_column} for {person}: {value}**"

    # Numeric summary.
    if "column" in result_data and "sum" in result_data:
        column = result_data["column"]
        total = format_number(result_data["sum"])

        return (
            f"**Total {column}: {total}**\n\n"
            f"Calculated from {result_data.get('count', 0):,} "
            "valid numeric records."
        )

    # Unique-value count.
    if "column" in result_data and "distinct_count" in result_data:
        return (
            f"**Unique {result_data['column']}: "
            f"{result_data['distinct_count']:,}**"
        )

    # Grouped results and period totals.
    if isinstance(result_data.get("results"), list):
        rows = result_data["results"]

        if rows and all(
            isinstance(row, dict)
            and "group" in row
            and "value" in row
            for row in rows
        ):
            lines = [
                f"- **{row['group']}**: {format_number(row['value'])}"
                for row in rows
            ]

            return f"**{explanation}**\n\n" + "\n".join(lines)

        if rows and all(isinstance(row, dict) for row in rows):
            lines = [
                "- " + " | ".join(
                    f"{key}: {format_number(value)}"
                    for key, value in row.items()
                )
                for row in rows
            ]

            return f"**{explanation}**\n\n" + "\n".join(lines)

    # Percentage-change results.
    if "percentage_change" in result_data:
        change = result_data["percentage_change"]

        if change is None:
            change_text = (
                "Cannot calculate percentage change from a zero baseline."
            )
        else:
            change_text = f"{change:+,.2f}%"

        return (
            f"**{explanation}**\n\n"
            f"Percentage change: **{change_text}**"
        )

    # Fallback for other results.
    return (
        f"**{explanation}**\n\n"
        "```json\n"
        f"{json.dumps(result_data, indent=2, ensure_ascii=False, default=str)}\n"
        "```"
    )


# ---------------------------------------------------------
# SESSION STATE AND AUTOMATIC RESTORE
# ---------------------------------------------------------

if "sales_df" not in st.session_state:
    st.session_state.sales_df = None
    st.session_state.uploaded_filename = None

    try:
        saved_df, saved_filename = load_saved_dataset()
        st.session_state.sales_df = saved_df
        st.session_state.uploaded_filename = saved_filename

    except FileNotFoundError as exc:
        st.warning(str(exc))

    except Exception as exc:
        st.error(f"Could not restore the saved dataset: {exc}")

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------------------------------------------------
# SIDEBAR: UPLOAD AND DATASET MANAGEMENT
# ---------------------------------------------------------

with st.sidebar:
    st.header("Upload your dataset")

    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=["csv", "xlsx", "xls"],
    )

    if uploaded_file is not None:
        previous_upload_key = st.session_state.get("last_upload_key")
        current_upload_key = (
            uploaded_file.name,
            len(uploaded_file.getvalue()),
        )

        if current_upload_key != previous_upload_key:
            try:
                df, filename = save_uploaded_dataset(uploaded_file)

                st.session_state.sales_df = df
                st.session_state.uploaded_filename = filename
                st.session_state.messages = []
                st.session_state.last_upload_key = current_upload_key

                st.success(
                    "Dataset saved permanently and loaded successfully."
                )

            except Exception as exc:
                st.error(f"Could not save the dataset: {exc}")

    if st.session_state.sales_df is not None:
        df = st.session_state.sales_df

        st.subheader("Dataset overview")
        st.write(f"**File:** {st.session_state.uploaded_filename}")
        st.write(f"**Rows:** {len(df):,}")
        st.write(f"**Columns:** {len(df.columns):,}")

        with st.expander("View column names"):
            for column in df.columns:
                st.write(f"- {column}")

        with st.expander("Preview data"):
            st.dataframe(df.head(10), use_container_width=True)

        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.caption("The dataset is stored on this computer.")

        if st.button("Remove saved dataset", type="secondary"):
            delete_saved_dataset()

            st.session_state.sales_df = None
            st.session_state.uploaded_filename = None
            st.session_state.messages = []
            st.session_state.last_upload_key = None

            st.success("Saved dataset removed.")
            st.rerun()


# ---------------------------------------------------------
# CONVERSATION HISTORY
# ---------------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("chart_data") is not None:
            chart_df = message["chart_data"]

            if not chart_df.empty:
                st.bar_chart(chart_df)


# ---------------------------------------------------------
# QUESTION INPUT AND AGENT RESPONSE
# ---------------------------------------------------------

question = st.chat_input(
    "Ask a business question, e.g. total quantity sold"
)

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        chart_data = None

        if st.session_state.sales_df is None:
            answer_text = (
                "Please upload a CSV or Excel dataset in the sidebar first."
            )
            st.warning(answer_text)

        else:
            with st.spinner("Analysing your question..."):
                result = ask_agent(
                    question,
                    st.session_state.sales_df,
                )

            answer_text = create_answer(result)
        
            st.markdown(answer_text)

            # Generate charts for grouped results only.
            if (
                isinstance(result, dict)
                and result.get("status") == "success"
            ):
                result_data = result.get("result", {})

                if isinstance(result_data, dict):
                    rows = result_data.get("results", [])

                    if rows and all(
                        isinstance(row, dict)
                        and "group" in row
                        and "value" in row
                        for row in rows
                    ):
                        chart_df = pd.DataFrame(rows).set_index("group")
                        st.bar_chart(chart_df["value"])
                        chart_data = chart_df["value"].to_frame()

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer_text,
                "chart_data": chart_data,
            }
        )
