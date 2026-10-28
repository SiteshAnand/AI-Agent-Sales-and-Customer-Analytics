
import pandas as pd
import pytest

from core.rule_based_analysis import sales_decline_investigation


@pytest.fixture
def sales_data():
    return pd.DataFrame({
        "Month": [
            "Jan", "Jan",
            "Feb", "Feb",
            "Mar", "Mar",
            "Apr", "Apr",
        ],
        "Region": [
            "North", "South",
            "North", "South",
            "North", "South",
            "North", "South",
        ],
        "Product": [
            "Faucet", "Shower",
            "Faucet", "Shower",
            "Faucet", "Shower",
            "Faucet", "Shower",
        ],
        "Sales": [
            120000, 95000,
            90000, 110000,
            150000, 105000,
            80000, 130000,
        ],
        "Quantity": [
            120, 95,
            90, 110,
            150, 105,
            80, 130,
        ],
        "Customers": [
            35, 28,
            30, 32,
            42, 31,
            25, 38,
        ],
    })


def get_markdown(result):
    """Extract readable text from supported result formats."""
    if isinstance(result, dict):
        markdown = result.get("markdown")
        if markdown is not None:
            return str(markdown)
        return str(result)

    markdown = getattr(result, "markdown", None)
    if markdown is not None:
        return str(markdown)

    return str(result)


def test_march_to_april_sales_decline(sales_data):
    question = (
        "North's sales fell from 150000 in March to 80000 in April. "
        "Investigate the decline using product, quantity, and customer columns."
    )

    result = sales_decline_investigation(question, sales_data)

    assert result is not None

    markdown = get_markdown(result)

    assert "150,000" in markdown
    assert "80,000" in markdown
    assert "70,000" in markdown
    assert "46.67%" in markdown
    assert "Faucet" in markdown


def test_decline_includes_quantity_and_customer_changes(sales_data):
    question = (
        "Investigate North's sales decline from March to April "
        "using quantity and customers."
    )

    result = sales_decline_investigation(question, sales_data)

    assert result is not None

    markdown = get_markdown(result)

    assert "150" in markdown
    assert "80" in markdown
    assert "42" in markdown
    assert "25" in markdown


def test_unrelated_question_returns_none(sales_data):
    question = "What is the weather forecast for tomorrow?"

    result = sales_decline_investigation(question, sales_data)

    assert result is None
