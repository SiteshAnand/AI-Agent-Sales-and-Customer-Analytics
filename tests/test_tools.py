
import pandas as pd
import pytest

from core.tools import run_tool


@pytest.fixture
def sales_data():
    return pd.DataFrame({
        "Date": ["2026-01-05", "2026-01-20", "2026-02-05"],
        "Region": ["North", "South", "North"],
        "Sales": [100.0, 200.0, 300.0],
    })


def test_numeric_summary_tool(sales_data):
    result = run_tool(
        "numeric_summary",
        sales_data,
        {"column": "Sales"},
    )

    assert result["sum"] == pytest.approx(600.0)


def test_group_metric_tool(sales_data):
    result = run_tool(
        "group_metric",
        sales_data,
        {
            "group_column": "Region",
            "value_column": "Sales",
            "metric": "sum",
        },
    )

    assert result["results"][0]["group"] == "North"
    assert result["results"][0]["value"] == pytest.approx(400.0)


def test_period_totals_tool(sales_data):
    result = run_tool(
        "period_totals",
        sales_data,
        {
            "date_column": "Date",
            "value_column": "Sales",
            "frequency": "ME",
        },
    )

    assert len(result["results"]) == 2


def test_percentage_change_tool():
    result = run_tool(
        "percentage_change",
        pd.DataFrame(),
        {"current": 120, "previous": 100},
    )

    assert result["percentage_change"] == pytest.approx(20.0)


def test_unsupported_tool_raises(sales_data):
    with pytest.raises(ValueError):
        run_tool("delete_data", sales_data, {})
       
def test_data_profile_tool(sales_data):
    result = run_tool("data_profile", sales_data, {})

    assert result["row_count"] == 3
    assert result["column_count"] == 3
    assert result["duplicate_rows"] == 0
    assert len(result["columns"]) == 3
