import math
import pytest
from core.result_validation import validate_result, combine_task_results


def test_validates_success_payload():
    assert validate_result({"status": "success", "tool": "x", "result": {"value": 1}})["valid"]


def test_detects_non_finite_numeric_output():
    report = validate_result({"status": "success", "result": {"value": math.inf}})
    assert not report["valid"]
    assert "Non-finite" in report["errors"][0]


def test_combines_partial_success_without_hiding_failure():
    result = combine_task_results("two tasks", [
        {"question": "A", "status": "success", "result": {"x": 1}},
        {"question": "B", "status": "error", "message": "missing column"},
    ])
    assert result["status"] == "success"
    assert result["result"]["partial_result"] is True
    assert result["result"]["failed_tasks"] == 1


def test_empty_task_list_rejected():
    with pytest.raises(ValueError):
        combine_task_results("x", [])
