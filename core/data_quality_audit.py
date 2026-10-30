"""Read-only data quality audit, enriched with semantic dataset understanding."""
import pandas as pd
from core.data_understanding import infer_column_roles, profile_dataset


def data_quality_audit(df: pd.DataFrame) -> dict:
    """Return row/column quality statistics and semantic column-role hints."""
    if df is None:
        raise ValueError("No dataset was provided.")
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    profile = profile_dataset(df)
    return {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values": {str(column): int(count) for column, count in df.isna().sum().items()},
        "column_profile": profile["columns"],
        "inferred_roles": infer_column_roles(df),
        "notes": [
            "Inferred column roles are hints based on column names, not guaranteed business definitions.",
            "Confirm the transaction grain and meaning of monetary columns before interpreting model results.",
        ],
    }
