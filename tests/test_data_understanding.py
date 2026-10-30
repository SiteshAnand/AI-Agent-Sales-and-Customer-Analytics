import pandas as pd
import pytest
from core.data_understanding import infer_column_roles, profile_dataset


def test_infers_common_business_column_roles():
    df = pd.DataFrame({"Order_Date": ["2026-01-01"], "Net_Sales_INR": [12], "Customer_ID": ["C1"], "Product_Name": ["A"]})
    roles = infer_column_roles(df)
    assert roles["date"] == "Order_Date"
    assert roles["sales_value"] == "Net_Sales_INR"
    assert roles["customer_id"] == "Customer_ID"
    assert roles["product"] == "Product_Name"


def test_profile_reports_missing_duplicates_and_date_range():
    df = pd.DataFrame({"Order_Date": ["2026-01-01", None, "2026-01-01"], "Sales": [1, 2, 1]})
    result = profile_dataset(df)
    assert result["row_count"] == 3
    assert result["duplicate_rows"] == 1
    date_col = next(c for c in result["columns"] if c["name"] == "Order_Date")
    assert date_col["missing_count"] == 1
    assert date_col["min_date"].startswith("2026-01-01")


def test_profile_rejects_non_dataframe():
    with pytest.raises(TypeError):
        profile_dataset([])
