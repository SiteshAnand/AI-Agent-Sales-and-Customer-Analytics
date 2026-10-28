
import pandas as pd
import pytest

from core.data_quality_audit import data_quality_audit


def test_audit_reports_row_and_column_counts():
    df = pd.DataFrame({
        "Order_ID": ["A001", "A002"],
        "Quantity_Sold": [2, 3],
    })

    result = data_quality_audit(df)

    assert result["row_count"] == 2
    assert result["column_count"] == 2


def test_audit_does_not_modify_input_dataframe():
    df = pd.DataFrame({
        "Order_ID": ["A001", "A002"],
        "Quantity_Sold": [2, 3],
    })
    original = df.copy(deep=True)

    data_quality_audit(df)

    pd.testing.assert_frame_equal(df, original)


def test_audit_detects_duplicate_rows():
    df = pd.DataFrame({
        "Order_ID": ["A001", "A001"],
        "Quantity_Sold": [2, 2],
    })

    result = data_quality_audit(df)

    assert result["duplicate_rows"] == 1


def test_audit_rejects_non_dataframe_input():
    with pytest.raises((TypeError, ValueError, AttributeError)):
        data_quality_audit(None)
