
import json
import pytest

from core.response import (
    format_result,
    format_error,
    format_success,
)


def test_format_result():
    result = {"total_sales": 500}

    formatted = format_result(result)

    assert json.loads(formatted) == result


def test_format_error():
    result = format_error("Column not found")

    assert result["status"] == "error"
    assert result["message"] == "Column not found"


def test_format_success():
    result = format_success(
        "numeric_summary",
        {"sum": 500},
    )

    assert result["status"] == "success"
    assert result["tool"] == "numeric_summary"
    assert result["result"]["sum"] == 500


def test_reject_non_dictionary_result():
    with pytest.raises(TypeError):
        format_result([1, 2, 3])