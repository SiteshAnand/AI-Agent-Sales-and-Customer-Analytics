from core.presentation import format_generic_business_result


def test_generic_result_never_dumps_json_or_code_fences():
    answer = format_generic_business_result("Analysis completed", {"result": 123, "debug": {"traceback": "secret"}})
    assert "123" in answer
    assert "```json" not in answer
    assert "traceback" not in answer


def test_generic_list_of_records_becomes_readable_table():
    answer = format_generic_business_result("Top products", {"products": [{"Product_Name": "Tap", "Net_Sales_INR": 1200}]})
    assert "| Product Name | Net Sales INR |" in answer
    assert "Tap" in answer
    assert "1,200" in answer


def test_generic_code_like_string_is_not_exposed():
    answer = format_generic_business_result("Completed", {"details": "import pandas as pd\nprint('raw')"})
    assert "import pandas" not in answer
    assert "technical payload hidden" in answer


def test_generic_large_record_list_is_capped():
    rows = [{"name": f"Item {i}"} for i in range(25)]
    answer = format_generic_business_result("Results", {"items": rows})
    assert "Showing 20 of 25 records" in answer
    assert "Item 24" not in answer
