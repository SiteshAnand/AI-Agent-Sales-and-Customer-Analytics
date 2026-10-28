import pandas as pd

from core.rule_based_analysis import sales_decline_investigation


def test_decline_investigation_separates_years():
    df = pd.DataFrame({
        "Month": [
            "March 2025", "April 2025",
            "March 2026", "April 2026",
        ],
        "Region": ["North"] * 4,
        "Product": ["Faucet"] * 4,
        "Sales": [150000, 80000, 200000, 210000],
        "Quantity": [150, 80, 200, 210],
        "Customers": [42, 25, 50, 55],
    })

    result = sales_decline_investigation(
        "Investigate North's sales decline from March 2025 to April 2025",
        df,
    )

    assert result is not None
    markdown = result["result"]["markdown"]
    assert "150,000" in markdown
    assert "80,000" in markdown
    assert "70,000" in markdown
    assert "46.67%" in markdown
