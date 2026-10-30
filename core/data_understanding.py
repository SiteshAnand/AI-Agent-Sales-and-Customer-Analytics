"""Dataset understanding and column-role discovery utilities."""
from __future__ import annotations
import re
import pandas as pd

ROLE_ALIASES = {
    "date": ["date", "order date", "transaction date", "purchase date", "invoice date"],
    "sales_value": ["sales", "sales amount", "net sales", "net sales inr", "revenue", "total amount", "gross sales", "value"],
    "customer_id": ["customer id", "customer_id", "customer", "client id", "client"],
    "product": ["product", "product name", "product id", "item", "sku", "item name"],
    "transaction_id": ["transaction id", "order id", "invoice", "invoice no", "basket id", "receipt id"],
    "region": ["region", "zone", "territory", "sales region"],
    "quantity": ["quantity", "quantity sold", "units", "units sold"],
}


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def infer_column_roles(df: pd.DataFrame) -> dict:
    """Map semantic roles to exact existing columns without changing the data."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Dataset must be a pandas DataFrame.")
    normalized = {normalize_name(col): col for col in df.columns}
    roles = {}
    for role, aliases in ROLE_ALIASES.items():
        for alias in aliases:
            key = normalize_name(alias)
            if key in normalized:
                roles[role] = str(normalized[key])
                break
    return roles


def profile_dataset(df: pd.DataFrame, sample_size: int = 3) -> dict:
    """Return a safe, concise profile used by the data-understanding layer."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Dataset must be a pandas DataFrame.")
    if type(sample_size) is not int or not 0 <= sample_size <= 10:
        raise ValueError("sample_size must be an integer from 0 to 10.")
    columns = []
    for col in df.columns:
        series = df[col]
        parsed_dates = pd.to_datetime(series, errors="coerce") if ("date" in str(col).casefold() or "time" in str(col).casefold()) else None
        info = {
            "name": str(col), "dtype": str(series.dtype),
            "non_null_count": int(series.notna().sum()),
            "missing_count": int(series.isna().sum()),
            "missing_percent": round(float(series.isna().mean() * 100), 3) if len(df) else 0.0,
            "distinct_count": int(series.nunique(dropna=True)),
            "sample_values": [str(v)[:120] for v in series.dropna().head(sample_size).tolist()],
        }
        if parsed_dates is not None:
            valid_dates = parsed_dates.dropna()
            info["valid_date_count"] = int(len(valid_dates))
            if not valid_dates.empty:
                info["min_date"] = valid_dates.min().isoformat()
                info["max_date"] = valid_dates.max().isoformat()
        columns.append(info)
    return {
        "row_count": int(len(df)), "column_count": int(len(df.columns)),
        "duplicate_rows": int(df.duplicated().sum()),
        "columns": columns, "inferred_roles": infer_column_roles(df),
    }
