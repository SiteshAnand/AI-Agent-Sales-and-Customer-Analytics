
import pandas as pd
import pytest

from core.analytics import (
    numeric_summary,
    group_metric,
    percentage_change,
    period_totals,
    distinct_count,
)


@pytest.fixture
def sales_data():
    return pd.DataFrame({
        "Date": [
            "2026-01-05",
            "2026-01-20",
            "2026-02-05",
        ],
        "Region": ["North", "South", "North"],
        "Sales": [100.0, 200.0, 300.0],
    })


def test_numeric_summary(sales_data):
    result = numeric_summary(sales_data, "Sales")

    assert result["count"] == 3
    assert result["sum"] == pytest.approx(600.0)
    assert result["mean"] == pytest.approx(200.0)


def test_group_metric(sales_data):
    result = group_metric(
        sales_data, "Region", "Sales", metric="sum"
    )

    assert result[0]["group"] == "North"
    assert result[0]["value"] == pytest.approx(400.0)


def test_percentage_change():
    assert percentage_change(120, 100) == pytest.approx(20.0)
    assert percentage_change(80, 100) == pytest.approx(-20.0)
    assert percentage_change(100, 0) is None


def test_period_totals(sales_data):
    result = period_totals(sales_data, "Date", "Sales", "ME")

    assert len(result) == 2
    assert result[0]["value"] == pytest.approx(300.0)
    assert result[1]["value"] == pytest.approx(300.0)


def test_missing_column_raises(sales_data):
    with pytest.raises(ValueError):
        numeric_summary(sales_data, "Revenue")


def test_distinct_count_ignores_duplicate_names():
    df = pd.DataFrame({
        "Sales Person": ["Amit", "Rahul", "Amit", "Priya", "Rahul"]
    })

    result = distinct_count(df, "Sales Person")

    assert result["distinct_count"] == 3


def test_distinct_count_ignores_missing_and_blank_names():
    df = pd.DataFrame({
        "Sales Person": ["Amit", "Rahul", None, "", "   "]
    })

    result = distinct_count(df, "Sales Person")

    assert result["distinct_count"] == 2


def test_distinct_count_ignores_case_and_surrounding_spaces():
    df = pd.DataFrame({
        "Sales Person": ["Amit", " amit ", "AMIT", "Rahul"]
    })

    result = distinct_count(df, "Sales Person")

    assert result["distinct_count"] == 2


@pytest.mark.parametrize(
    "metric",
    [
        "sum",
        "mean",
        "median",
        "min",
        "max",
        "count",
        "std",
        "variance",
    ],
)
def test_group_metric_supports_all_metrics(metric):
    df = pd.DataFrame({
        "Region": ["North", "North", "South"],
        "Sales": [10.0, 30.0, 20.0],
    })

    result = group_metric(
        df,
        "Region",
        "Sales",
        metric=metric,
        top_n=10,
        sort_order="desc",
    )

    values = {row["group"]: row["value"] for row in result}

    if metric == "sum":
        assert values["North"] == pytest.approx(40.0)
        assert values["South"] == pytest.approx(20.0)
    elif metric in {"mean", "median"}:
        assert values["North"] == pytest.approx(20.0)
        assert values["South"] == pytest.approx(20.0)
    elif metric == "min":
        assert values["North"] == pytest.approx(10.0)
        assert values["South"] == pytest.approx(20.0)
    elif metric == "max":
        assert values["North"] == pytest.approx(30.0)
        assert values["South"] == pytest.approx(20.0)
    elif metric == "count":
        assert values["North"] == 2.0
        assert values["South"] == 1.0
    elif metric == "std":
        assert values["North"] == pytest.approx(14.1421356237)
        assert values["South"] == 0.0
    elif metric == "variance":
        assert values["North"] == pytest.approx(200.0)
        assert values["South"] == 0.0


def test_group_metric_lowest_first():
    df = pd.DataFrame({
        "Sales Person": ["Amit", "Rahul", "Priya"],
        "VALUE": [300, 100, 200],
    })

    result = group_metric(
        df,
        "Sales Person",
        "VALUE",
        metric="sum",
        sort_order="asc",
    )

    assert result[0]["group"] == "Rahul"
    assert result[0]["value"] == 100.0
    
def test_data_profile_reports_dataset_structure():
    from core.analytics import data_profile

    df = pd.DataFrame({
        "Region": ["North", "South", "North", None],
        "Sales": [100, 200, 100, None],
    })

    result = data_profile(df)

    assert result["row_count"] == 4
    assert result["column_count"] == 2
    assert result["duplicate_rows"] == 1

    region = next(
        column for column in result["columns"]
        if column["column"] == "Region"
    )
    assert region["missing_count"] == 1
    assert region["distinct_count"] == 2
    assert region["missing_percent"] == 25.0


def test_data_profile_handles_empty_dataframe():
    from core.analytics import data_profile

    result = data_profile(pd.DataFrame())

    assert result["row_count"] == 0
    assert result["column_count"] == 0
    assert result["duplicate_rows"] == 0
    assert result["columns"] == []

