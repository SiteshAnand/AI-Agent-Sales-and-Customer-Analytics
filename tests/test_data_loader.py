
import pandas as pd
import pytest

from core.data_loader import load_sales_file


class FakeUpload:
    """Simulate a file uploaded through the application."""

    def __init__(self, name, content):
        self.name = name
        self._content = content

    def getvalue(self):
        return self._content


def test_load_csv():
    content = b"Region,Sales\nNorth,100\nSouth,200\n"
    upload = FakeUpload("sales.csv", content)

    df = load_sales_file(upload)

    assert len(df) == 2
    assert list(df.columns) == ["Region", "Sales"]
    assert df["Sales"].sum() == 300


def test_reject_unsupported_file():
    upload = FakeUpload("sales.txt", b"some text")

    with pytest.raises(ValueError, match="Upload a CSV"):
        load_sales_file(upload)


def test_reject_empty_file():
    upload = FakeUpload("sales.csv", b"")

    with pytest.raises(ValueError, match="empty"):
        load_sales_file(upload)


def test_trim_column_names():
    content = b" Region , Sales \nNorth,100\n"
    upload = FakeUpload("sales.csv", content)

    df = load_sales_file(upload)

    assert list(df.columns) == ["Region", "Sales"]


def test_reject_duplicate_column_names():
    content = b"Region,Region\nNorth,South\n"
    upload = FakeUpload("sales.csv", content)

    with pytest.raises(ValueError, match="Duplicate column"):
        load_sales_file(upload)