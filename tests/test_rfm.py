
import pandas as pd
import pytest

from core.rfm import calculate_rfm


@pytest.fixture
def transactions():
    return pd.DataFrame({
        "Customer_ID": [
            "A", "A", "A",
            "B", "B",
            "C",
            "D", "D", "D", "D",
        ],
        "Order_Date": [
            "2026-01-05", "2026-02-05", "2026-03-05",
            "2026-02-10", "2026-03-10",
            "2026-03-15",
            "2025-11-10", "2025-12-10",
            "2026-01-10", "2026-02-10",
        ],
        "Sales": [
            100, 150, 200,
            80, 120,
            60,
            50, 50, 50, 50,
        ],
    })


def test_calculates_customer_rfm(transactions):
    result = calculate_rfm(
        transactions,
        customer_id_column="Customer_ID",
        date_column="Order_Date",
        transaction_value_column="Sales",
        snapshot_date="2026-04-01",
    )

    rfm = result["rfm"]

    assert len(rfm) == 4
    assert set([
        "Recency",
        "Frequency",
        "Monetary",
        "R_Score",
        "F_Score",
        "M_Score",
        "RFM_Score",
        "Segment",
    ]).issubset(rfm.columns)


def test_aggregates_frequency_and_monetary(transactions):
    result = calculate_rfm(
        transactions,
        "Customer_ID",
        "Order_Date",
        "Sales",
        snapshot_date="2026-04-01",
    )

    rfm = result["rfm"].set_index("Customer_ID")

    assert rfm.loc["A", "Frequency"] == 3
    assert rfm.loc["A", "Monetary"] == 450
    assert rfm.loc["C", "Frequency"] == 1
    assert rfm.loc["C", "Monetary"] == 60


def test_recency_uses_snapshot_date(transactions):
    result = calculate_rfm(
        transactions,
        "Customer_ID",
        "Order_Date",
        "Sales",
        snapshot_date="2026-04-01",
    )

    rfm = result["rfm"].set_index("Customer_ID")

    assert rfm.loc["C", "Recency"] == 17
    assert rfm.loc["A", "Recency"] == 27


def test_scores_are_within_configured_range(transactions):
    result = calculate_rfm(
        transactions,
        "Customer_ID",
        "Order_Date",
        "Sales",
        snapshot_date="2026-04-01",
        quantiles=4,
    )

    rfm = result["rfm"]

    for column in ["R_Score", "F_Score", "M_Score"]:
        assert rfm[column].between(1, 4).all()


def test_segments_and_summary_are_returned(transactions):
    result = calculate_rfm(
        transactions,
        "Customer_ID",
        "Order_Date",
        "Sales",
        snapshot_date="2026-04-01",
    )

    assert result["rfm"]["Segment"].notna().all()
    assert not result["segment_summary"].empty
    assert result["segment_summary"]["Customers"].sum() == 4


def test_handles_invalid_transaction_rows(transactions):
    transactions.loc[len(transactions)] = [
        "E", "invalid-date", 100
    ]

    result = calculate_rfm(
        transactions,
        "Customer_ID",
        "Order_Date",
        "Sales",
        snapshot_date="2026-04-01",
    )

    assert result["metadata"]["dropped_rows"] == 1
    assert result["metadata"]["customer_count"] == 4


def test_rejects_missing_required_column(transactions):
    with pytest.raises(ValueError):
        calculate_rfm(
            transactions,
            "Customer_ID",
            "Missing_Date",
            "Sales",
        )


def test_rejects_snapshot_before_latest_transaction(transactions):
    with pytest.raises(ValueError):
        calculate_rfm(
            transactions,
            "Customer_ID",
            "Order_Date",
            "Sales",
            snapshot_date="2026-02-01",
        )
