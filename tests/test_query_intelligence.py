import pytest
from core.query_intelligence import split_multifold_question, plan_query


def test_splits_explicit_and_action_boundaries():
    q = "Compare monthly sales and identify declining regions and forecast next 3 months"
    assert split_multifold_question(q) == [
        "Compare monthly sales", "identify declining regions", "forecast next 3 months"
    ]


def test_does_not_split_plain_noun_conjunction():
    q = "Compare sales and revenue by region"
    assert split_multifold_question(q) == [q]


def test_splits_semicolon_and_newlines():
    assert len(split_multifold_question("What columns are available?; Check data quality")) == 2
    assert len(split_multifold_question("Calculate total sales\nFind top products")) == 2


def test_plan_is_transparent_and_sequential():
    plan = plan_query("Check data quality; Calculate total sales")
    assert plan["task_count"] == 2
    assert plan["execution_strategy"] == "sequential_with_partial_results"
    assert plan["tasks"][0]["task_id"] == 1


def test_invalid_input_rejected():
    with pytest.raises(ValueError):
        split_multifold_question(" ")
