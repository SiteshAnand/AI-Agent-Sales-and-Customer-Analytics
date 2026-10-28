
import pandas as pd
import pytest

from core.data_quality import profile_data


def test_profile_data():
    df = pd.DataFrame({
        "Region": ["North", "South", "North"],
        "Sales": [100, None, 100],
    })

    result = profile_data(df)

    assert result["rows"] == 3
    assert result["columns_count"] == 2
    assert result["duplicate_rows"] == 1
    assert result["missing_values"]["Sales"] == 1
    assert "Sales" in result["columns"]


def test_empty_data_raises():
    df = pd.DataFrame()

    with pytest.raises(ValueError):
        profile_data(df)


def test_non_dataframe_raises():
    with pytest.raises(TypeError):
        profile_data([1, 2, 3])