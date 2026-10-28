import pandas as pd
import pytest

from core.agent import ask_agent


@pytest.fixture
def monthly_sales():
    return pd.DataFrame({
        "Month": ["Jan", "Jan", "Feb", "Feb", "Mar", "Mar", "Apr", "Apr"],
        "Region": ["North", "South"] * 4,
        "Product": ["Faucet", "Shower"] * 4,
        "Sales": [120000, 95000, 90000, 110000, 150000, 105000, 80000, 130000],
        "Quantity": [120, 95, 90, 110, 150, 105, 80, 130],
        "Customers": [35, 28, 30, 32, 42, 31, 25, 38],
    })


def test_general_dataset_question_works_without_gemini(monthly_sales, monkeypatch):
    monkeypatch.setattr("core.agent.GEMINI_API_KEY", "")
    result = ask_agent("What columns are available in my dataset, and what business analysis can you perform?", monthly_sales)
    assert result["status"] == "success"
    markdown = result["result"]["markdown"]
    for column in ("Month", "Region", "Product", "Sales", "Quantity", "Customers"):
        assert column in markdown
    assert "management recommendations" in markdown.lower()


def test_monthly_region_analysis_calculates_extremes_and_changes(monthly_sales, monkeypatch):
    monkeypatch.setattr("core.agent.GEMINI_API_KEY", "")
    result = ask_agent(
        "Compare monthly sales performance across North and South. Identify the strongest and weakest months for each region, calculate the month-to-month changes, and recommend three actions management should take.",
        monthly_sales,
    )
    assert result["status"] == "success"
    assert result["tool"] == "monthly_region_analysis"
    markdown = result["result"]["markdown"]
    assert "North" in markdown and "South" in markdown
    assert "Mar (150,000)" in markdown
    assert "Apr (80,000)" in markdown
    assert "Jan (95,000)" in markdown
    assert "Apr (130,000)" in markdown
    assert "Three management actions" in markdown
    assert "Feb → Mar" in markdown
    assert "+66.67%" in markdown


def test_monthly_analysis_handles_zero_baseline():
    from core.rule_based_analysis import monthly_region_analysis
    df = pd.DataFrame({"Month": ["Jan", "Feb"], "Region": ["North", "North"], "Sales": [0, 100]})
    result = monthly_region_analysis("Compare monthly sales by region and calculate month-to-month changes", df)
    assert result is not None
    assert "not calculable (zero baseline)" in result["result"]["markdown"]
