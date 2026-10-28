
import pandas as pd
import pytest

from core.agent import (
    _validate_plan,
    _try_direct_sales_ranking,
    ask_agent,
)


@pytest.fixture
def sample_sales_data():
    """A small dataset with known results for reliability testing."""
    return pd.DataFrame({
        "Sales Person": [
            "Amit",
            "Rahul",
            "Amit",
            "Priya",
            "Rahul",
            None,
        ],
        "Region": [
            "West",
            "North",
            "West",
            "South",
            "North",
            "West",
        ],
        "QUANTITY": [
            10,
            20,
            15,
            30,
            None,
            -5,
        ],
    })


def test_sample_data_has_expected_duplicate_names(sample_sales_data):
    """Repeated names must not be mistaken for unique people."""
    unique_names = sample_sales_data["Sales Person"].nunique(
        dropna=True
    )

    assert unique_names == 3
    assert len(sample_sales_data) == 6


def test_quantity_sum_handles_missing_and_negative_values(
    sample_sales_data,
):
    """The sum includes negative quantities and skips missing values."""
    quantity = pd.to_numeric(
        sample_sales_data["QUANTITY"],
        errors="coerce",
    )

    assert quantity.sum() == 70
    assert quantity.count() == 5


def test_region_totals_are_correct(sample_sales_data):
    """Known regional totals should be calculated correctly."""
    from core.analytics import group_metric

    result = group_metric(
        sample_sales_data,
        "Region",
        "QUANTITY",
        metric="sum",
        top_n=10,
    )

    actual = {item["group"]: item["value"] for item in result}

    assert actual["North"] == 20
    assert actual["South"] == 30
    assert actual["West"] == 20


def test_validate_plan_rejects_unknown_column(sample_sales_data):
    """The agent must reject a column that does not exist."""
    plan = {
        "tool": "numeric_summary",
        "parameters": {"column": "NON_EXISTENT_COLUMN"},
        "explanation": "Test invalid column",
    }

    with pytest.raises(ValueError, match="does not exist"):
        _validate_plan(plan, sample_sales_data)


def test_validate_plan_rejects_unsupported_tool(sample_sales_data):
    """Only approved analytics tools should be accepted."""
    plan = {
        "tool": "run_arbitrary_code",
        "parameters": {},
        "explanation": "Invalid tool",
    }

    with pytest.raises(ValueError, match="unsupported"):
        _validate_plan(plan, sample_sales_data)


def test_validate_plan_rejects_percentage_change_tool(
    sample_sales_data,
):
    """The model must not calculate change from invented input numbers."""
    plan = {
        "tool": "percentage_change",
        "parameters": {
            "current": 100,
            "previous": 80,
        },
        "explanation": "Attempt to supply arbitrary numbers",
    }

    with pytest.raises(ValueError, match="unsupported"):
        _validate_plan(plan, sample_sales_data)



def test_validate_plan_rejects_nonexistent_filter_value(
    sample_sales_data,
):
    """A filter must not silently accept a category absent from the data."""
    plan = {
        "tool": "filtered_metric",
        "parameters": {
            "filter_column": "Region",
            "filter_value": "East",
            "value_column": "QUANTITY",
            "metric": "sum",
        },
        "explanation": "Test a category absent from the dataset",
    }

    with pytest.raises(ValueError, match="not found"):
        _validate_plan(plan, sample_sales_data)


def test_empty_question_returns_error(sample_sales_data):
    """An empty question must not be sent to Gemini."""
    result = ask_agent("", sample_sales_data)

    assert result["status"] == "error"
    assert "question" in result["message"].lower()


def test_empty_dataset_returns_error():
    """An empty dataset must not be analysed."""
    result = ask_agent("total quantity sold", pd.DataFrame())

    assert result["status"] == "error"
    assert "dataset" in result["message"].lower()


def test_non_dataframe_input_returns_error():
    """Invalid dataset input must return a controlled error."""
    result = ask_agent("total quantity sold", None)

    assert result["status"] == "error"
    assert "dataset" in result["message"].lower()


def test_lowest_salesperson_is_ranked_correctly():
    """The lowest salesperson is ranked by their total sales."""
    df = pd.DataFrame({
        "Sales Person": [
            "Amit", "Amit",
            "Rahul", "Rahul",
            "Priya", "Priya",
        ],
        "VALUE": [100, 150, 500, 400, 200, 250],
    })

    result = _try_direct_sales_ranking(
        "Which salesperson has the lowest sales?",
        df,
    )

    assert result["status"] == "success"
    assert result["tool"] == "group_metric"
    assert result["result"]["results"][0]["group"] == "Amit"
    assert result["result"]["results"][0]["value"] == 250.0
