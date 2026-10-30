
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def calculate_rfm(
    df: pd.DataFrame,
    customer_id_column: str,
    date_column: str,
    transaction_value_column: str,
    snapshot_date: str | pd.Timestamp | None = None,
    quantiles: int = 5,
) -> dict[str, Any]:
    """
    Calculate RFM metrics and customer segments.

    Recency: days since the customer's latest transaction.
    Frequency: number of valid transactions.
    Monetary: total transaction value.

    Returns a dictionary containing customer-level RFM data,
    segment summary, and processing metadata.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("The input DataFrame is empty.")

    required_columns = [
        customer_id_column,
        date_column,
        transaction_value_column,
    ]

    if len(set(required_columns)) != 3:
        raise ValueError("Customer, date, and value columns must be different.")

    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if not isinstance(quantiles, int) or not 2 <= quantiles <= 5:
        raise ValueError("quantiles must be an integer between 2 and 5.")

    data = df[required_columns].copy()
    original_rows = len(data)

    data[date_column] = pd.to_datetime(
        data[date_column],
        errors="coerce",
        utc=True,
    )

    data[transaction_value_column] = pd.to_numeric(
        data[transaction_value_column],
        errors="coerce",
    )

    data = data.dropna(subset=required_columns)

    if data.empty:
        raise ValueError("No valid customer transactions remain after cleaning.")

    # Transaction values should be finite. Negative values are retained
    # because some datasets record refunds or adjustments.
    finite_values = np.isfinite(data[transaction_value_column])
    data = data.loc[finite_values].copy()

    if data.empty:
        raise ValueError("No finite transaction values remain.")

    if snapshot_date is None:
        reference_date = data[date_column].max().normalize() + pd.Timedelta(days=1)
    else:
        reference_date = pd.Timestamp(snapshot_date)

        if reference_date.tzinfo is None:
            reference_date = reference_date.tz_localize("UTC")
        else:
            reference_date = reference_date.tz_convert("UTC")

        reference_date = reference_date.normalize()

    if data[date_column].max() >= reference_date:
        raise ValueError(
            "snapshot_date must be later than every transaction date."
        )

    if data[customer_id_column].astype(str).str.strip().eq("").any():
        data = data[
            data[customer_id_column].astype(str).str.strip() != ""
        ].copy()

    if data.empty:
        raise ValueError("No valid customer IDs remain.")

    rfm = (
        data.groupby(customer_id_column, dropna=True)
        .agg(
            Last_Purchase=(date_column, "max"),
            Frequency=(date_column, "size"),
            Monetary=(transaction_value_column, "sum"),
        )
        .reset_index()
    )

    rfm["Recency"] = (
        reference_date - rfm["Last_Purchase"].dt.normalize()
    ).dt.days

    if (rfm["Recency"] < 0).any():
        raise ValueError("Negative recency detected; check snapshot_date.")

    # Rank first so tied values can be assigned stable quantile scores.
    # Recency is reversed: fewer days since purchase means a better score.
    def assign_score(values: pd.Series, higher_is_better: bool) -> pd.Series:
        ranked = values.rank(method="first", ascending=True)

        if not higher_is_better:
            ranked = values.rank(method="first", ascending=False)

        scores = pd.qcut(
            ranked,
            q=min(quantiles, len(ranked)),
            labels=False,
            duplicates="drop",
        )

        return scores.astype(int) + 1

    rfm["R_Score"] = assign_score(rfm["Recency"], higher_is_better=False)
    rfm["F_Score"] = assign_score(rfm["Frequency"], higher_is_better=True)
    rfm["M_Score"] = assign_score(rfm["Monetary"], higher_is_better=True)

    rfm["RFM_Score"] = (
        rfm["R_Score"].astype(str)
        + rfm["F_Score"].astype(str)
        + rfm["M_Score"].astype(str)
    )

    # Transparent, rule-based segments; these are business heuristics,
    # not machine-learned customer classes.
    def segment_customer(row: pd.Series) -> str:
        r = row["R_Score"]
        f = row["F_Score"]
        m = row["M_Score"]

        if r >= 4 and f >= 4 and m >= 4:
            return "Champions"
        if r >= 3 and f >= 3:
            return "Loyal Customers"
        if r >= 4 and f <= 2:
            return "Recent Customers"
        if r <= 2 and f >= 3 and m >= 3:
            return "At Risk"
        if r <= 2 and f <= 2:
            return "Hibernating"
        if m >= 4:
            return "Big Spenders"
        return "Potential Loyalists"

    rfm["Segment"] = rfm.apply(segment_customer, axis=1)

    rfm = rfm.sort_values(
        ["R_Score", "F_Score", "M_Score"],
        ascending=False,
    ).reset_index(drop=True)

    segment_summary = (
        rfm.groupby("Segment", observed=True)
        .agg(
            Customers=(customer_id_column, "nunique"),
            Total_Monetary=("Monetary", "sum"),
            Average_Monetary=("Monetary", "mean"),
            Average_Recency=("Recency", "mean"),
            Average_Frequency=("Frequency", "mean"),
        )
        .sort_values("Customers", ascending=False)
    )

    return {
        "rfm": rfm,
        "segment_summary": segment_summary,
        "metadata": {
            "input_rows": int(original_rows),
            "valid_transaction_rows": int(len(data)),
            "dropped_rows": int(original_rows - len(data)),
            "customer_count": int(rfm[customer_id_column].nunique()),
            "snapshot_date": reference_date,
            "quantiles": quantiles,
        },
        "warnings": [
            "RFM segments are heuristic groupings, not predictions of future behaviour.",
            "Negative transaction values are retained and may affect monetary rankings.",
            "Use a consistent transaction definition and snapshot date when comparing periods.",
        ],
    }
