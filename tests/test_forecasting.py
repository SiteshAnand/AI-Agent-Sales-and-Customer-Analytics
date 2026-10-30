import pandas as pd
import pytest

from core.forecasting import forecast_product_sales, is_product_forecast_question
from core.agent import ask_agent


def make_data(months=8):
    dates = pd.date_range("2025-01-15", periods=months, freq="MS", tz="UTC")
    rows = []
    for i, date in enumerate(dates):
        rows.append({"Order_Date": date, "Product_Name": "A", "Net_Sales_INR": 100 + i * 10})
        rows.append({"Order_Date": date, "Product_Name": "B", "Net_Sales_INR": 200 + i * 5})
    return pd.DataFrame(rows)


def test_forecast_produces_two_months_per_product():
    result = forecast_product_sales(make_data(), horizon=2)
    assert result["products_forecast"] == 2
    assert len(result["forecast_rows"]) == 4
    assert result["forecast_months"] == ["2025-10", "2025-11"]
    assert all(row["Forecast_Sales"] >= 0 for row in result["forecast_rows"])


def test_seasonal_naive_used_with_24_months():
    result = forecast_product_sales(make_data(24), horizon=2)
    assert set(row["Method"] for row in result["forecast_rows"]) == {"seasonal_naive_12_months"}


def test_insufficient_product_history_is_reported():
    df = pd.DataFrame({
        "Order_Date": pd.to_datetime(["2025-01-01", "2025-02-01", "2025-03-01", "2025-03-01"], utc=True),
        "Product_Name": ["A", "A", "A", "New"],
        "Net_Sales_INR": [10, 12, 14, 99],
    })
    result = forecast_product_sales(df)
    assert result["products_forecast"] == 1
    assert result["insufficient_history"] == [{"product": "New", "months_available": 1}]


def test_missing_required_columns_gives_clear_error():
    with pytest.raises(ValueError, match="requires a date, product, and sales-value column"):
        forecast_product_sales(pd.DataFrame({"A": [1, 2]}))


def test_invalid_horizon_rejected():
    with pytest.raises(ValueError, match="from 1 to 12"):
        forecast_product_sales(make_data(), horizon=0)


@pytest.mark.parametrize("question", [
    "Forecast product-wise sales for the next two months based on historical data",
    "Can you predict product sales for next 2 months?",
    "Project revenue by product for the coming quarter",
])
def test_forecast_question_detection(question):
    assert is_product_forecast_question(question)


def test_non_forecast_question_not_detected():
    assert not is_product_forecast_question("Which products had the highest sales?")


def test_agent_routes_product_forecast_without_gemini():
    result = ask_agent("Forecast product-wise sales for the next two months based on historical data", make_data())
    assert result["status"] == "success"
    assert result["tool"] == "product_sales_forecast"
    assert len(result["result"]["forecast_rows"]) == 4


def test_total_sales_forecast_supports_one_to_three_months():
    from core.forecasting import forecast_total_sales
    df = make_data(12).drop(columns=['Product_Name'])
    result = forecast_total_sales(df, horizon=3)
    assert len(result['forecast_months']) == 3
    last_month = result['last_historical_month']
    from pandas import Period
    assert result['forecast_months'][0]['Forecast_Month'] == str(Period(last_month, freq='M') + 1)
    assert result['forecast_months'][2]['Forecast_Month'] == str(Period(last_month, freq='M') + 3)


def test_agent_routes_total_sales_forecast_without_product_column():
    df = make_data(12).drop(columns=['Product_Name'])
    result = ask_agent('forecast total sales for next 3 months', df)
    assert result['status'] == 'success'
    assert result['tool'] == 'total_sales_forecast'
    assert len(result['result']['forecast_months']) == 3
