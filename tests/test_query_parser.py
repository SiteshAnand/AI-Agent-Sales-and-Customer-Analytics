
import pytest

from core.query_parser import parse_query


COLUMNS = ["Date", "Region", "Sales", "Product"]


def test_parse_total_sales():
    result = parse_query("What are my total sales?", COLUMNS)

    assert result["metric"] == "sum"
    assert "Sales" in result["matched_columns"]


def test_parse_average_sales_by_region():
    result = parse_query("What is the average Sales by Region?", COLUMNS)

    assert result["metric"] == "mean"
    assert result["group_column"] == "Region"


def test_parse_monthly_sales():
    result = parse_query("Show monthly sales", COLUMNS)

    assert result["metric"] == "monthly"


def test_parse_top_products():
    result = parse_query("Show the highest Sales by Product", COLUMNS)

    assert result["metric"] == "top"
    assert result["group_column"] == "Product"


def test_empty_question_raises():
    with pytest.raises(ValueError):
        parse_query("   ", COLUMNS)


def test_unknown_question_is_not_guessed():
    result = parse_query("Tell me something interesting", COLUMNS)

    assert result["metric"] == "unknown"

def test_parser_accepts_long_query_within_configured_limit():
    from core.query_parser import parse_query
    question = 'show sales ' + ('for each region and product ' * 120)
    assert len(question) > 2000
    result = parse_query(question, ['Sales', 'Region', 'Product'])
    assert result['metric'] in {'sum', 'unknown', 'monthly', 'top', 'bottom', 'change', 'mean', 'count'}
