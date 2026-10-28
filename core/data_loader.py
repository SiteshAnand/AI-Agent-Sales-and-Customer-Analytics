
from io import BytesIO
from pathlib import Path
import csv

import pandas as pd


from config.settings import MAX_UPLOAD_MB, MAX_ROWS

SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


def load_sales_file(uploaded_file) -> pd.DataFrame:
    """Load and validate a CSV or Excel file uploaded through Streamlit."""

    filename = getattr(uploaded_file, "name", "")
    extension = Path(filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError("Upload a CSV, XLSX, or XLS file.")

    content = uploaded_file.getvalue()

    if not content:
        raise ValueError("The uploaded file is empty.")

    max_bytes = MAX_UPLOAD_MB * 1024 * 1024

    if len(content) > max_bytes:
        raise ValueError(
            f"File exceeds the {MAX_UPLOAD_MB} MB upload limit."
        )

    try:
        if extension == ".csv":
            # Check the original CSV header before Pandas modifies it.
            try:
                text = content.decode("utf-8-sig")
            except UnicodeDecodeError:
                text = content.decode("utf-8")

            header = next(csv.reader(text.splitlines()), [])

            cleaned_header = [name.strip() for name in header]

            if any(not name for name in cleaned_header):
                raise ValueError("Blank column names are not allowed.")

            if len(cleaned_header) != len(set(cleaned_header)):
                raise ValueError("Duplicate column names are not allowed.")

            df = pd.read_csv(BytesIO(content))

        elif extension == ".xlsx":
            df = pd.read_excel(BytesIO(content), engine="openpyxl")

        else:
            df = pd.read_excel(BytesIO(content), engine="xlrd")

    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(
            "Could not read the file. Check its format and contents."
        ) from exc

    if df.empty:
        raise ValueError("The uploaded file contains no data.")

    if len(df) > MAX_ROWS:
        raise ValueError(
            f"File exceeds the limit of {MAX_ROWS:,} rows."
        )

    df.columns = [str(col).strip() for col in df.columns]

    if any(not col for col in df.columns):
        raise ValueError("Blank column names are not allowed.")

    if df.columns.duplicated().any():
        raise ValueError("Duplicate column names are not allowed.")

    return df