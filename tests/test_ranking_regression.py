import pandas as pd
from core.agent import _try_direct_sales_ranking


def sample():
    return pd.DataFrame({
        "Sales Person": ["Amit", "Amit", "Rahul", "Rahul", "Priya", "Priya", "Neha", "Neha"],
        "VALUE": [100, 150, 500, 400, 200, 250, 50, 60],
    })


def test_five_lowest_returns_requested_count_when_available():
    df = pd.concat([sample(), pd.DataFrame({"Sales Person": ["Karan", "Karan"], "VALUE": [80, 20]})], ignore_index=True)
    result = _try_direct_sales_ranking("Show me the five lowest sales people by sales", df)
    assert result["status"] == "success"
    assert len(result["result"]["results"]) == 5
    assert [r["group"] for r in result["result"]["results"]] == ["Karan", "Neha", "Amit", "Priya", "Rahul"]


def test_top_3_numeric_count_is_respected():
    result = _try_direct_sales_ranking("show top 3 salespeople by sales", sample())
    assert len(result["result"]["results"]) == 3
    assert [r["group"] for r in result["result"]["results"]] == ["Rahul", "Priya", "Amit"]


def test_singular_question_defaults_to_one():
    result = _try_direct_sales_ranking("Which salesperson has the lowest sales?", sample())
    assert len(result["result"]["results"]) == 1
    assert result["result"]["results"][0]["group"] == "Neha"
