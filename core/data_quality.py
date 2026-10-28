
import pandas as pd


def profile_data(df: pd.DataFrame) -> dict:
    """Identify basic data quality issues in a DataFrame."""

    if df is None:
        raise ValueError("No dataset was provided.")

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("The dataset contains no rows.")

    return {
        "rows": int(len(df)),
        "columns_count": int(len(df.columns)),
        "columns": list(df.columns),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values": {
            str(col): int(count)
            for col, count in df.isna().sum().items()
        },
        "data_types": {
            str(col): str(dtype)
            for col, dtype in df.dtypes.items()
        },
        "constant_columns": [
            str(col)
            for col in df.columns
            if df[col].nunique(dropna=False) <= 1
        ],
    }