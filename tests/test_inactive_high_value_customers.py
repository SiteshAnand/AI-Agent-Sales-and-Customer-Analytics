import pandas as pd

from core.agent import _try_high_value_inactive_customers


def sample_transactions():
    return pd.DataFrame({
        "Customer_ID": ["A", "A", "B", "B", "C", "D"],
        "Order_Date": [
            "2026-01-01", "2026-02-01",
            "2026-06-01", "2026-07-01",
            "2026-10-01", "2026-01-15",
        ],
        "Net_Sales_INR": [10000, 15000, 9000, 11000, 50000, 8000],
    })


def test_ranks_high_value_customers_who_are_inactive():
    result = _try_direct_question(
        "give me list of customers who have highest sales value but did not buy in recent time",
        sample_transactions(),
    )
    assert result["status"] == "success"
    assert result["tool"] == "inactive_high_value_customers"
    payload = result["result"]
    assert payload["threshold_days"] == 90
    assert payload["snapshot_date"] == "2026-10-02"
    assert [row["Customer"] for row in payload["rows"]] == ["A", "B", "D"]
    assert all(row["Days_Inactive"] >= 90 for row in payload["rows"])


def test_explicit_inactivity_threshold_is_respected():
    result = _try_direct_question(
        "show high value customers who have not purchased in 60 days",
        sample_transactions(),
    )
    assert result["status"] == "success"
    assert result["result"]["threshold_days"] == 60
    assert "60 days" in result["result"]["markdown"]


def test_missing_required_columns_returns_clear_error():
    result = _try_direct_question(
        "show high sales value customers who are inactive",
        pd.DataFrame({"Product": ["A"], "Value": [100]}),
    )
    assert result["status"] == "error"
    assert "customer identifier" in result["message"]


def test_unrelated_question_is_not_captured():
    result = _try_direct_question(
        "what is total sales by region",
        sample_transactions(),
    )
    assert result is None


def _try_direct_question(question, df):
    # Use the production helper so these tests exercise the same intent routing.
    return _try_high_value_inactive_customers(question, df)


def test_user_typo_did_not_bught_still_routes_correctly():
    result = _try_direct_question(
        "give me list of customer who have highest sales value but did not bught in recent time",
        sample_transactions(),
    )
    assert result["status"] == "success"
    assert result["tool"] == "inactive_high_value_customers"
