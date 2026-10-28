"""Read-only deterministic data quality checks for business datasets."""
from __future__ import annotations
from typing import Any
import pandas as pd

DATE_COLUMNS = ("Order_Date", "Delivery_Date", "Customer_Signup_Date")
NUMERIC_RULES = {
    "Quantity_Sold": ("positive", "Quantity should be greater than zero."),
    "Discount_Percent": ("percent", "Percentage should be between 0 and 100."),
    "Tax_Percent": ("percent", "Percentage should be between 0 and 100."),
    "Customer_Age": ("age", "Age should be between 0 and 120."),
    "Customer_Rating": ("rating", "Rating should be between 1 and 5."),
    "Unit_Price_INR": ("nonnegative", "Value should not be negative."),
    "Gross_Sales_INR": ("nonnegative", "Value should not be negative."),
    "Discount_Amount_INR": ("nonnegative", "Value should not be negative."),
    "Net_Sales_INR": ("nonnegative", "Value should not be negative."),
    "Tax_Amount_INR": ("nonnegative", "Value should not be negative."),
    "Total_Amount_INR": ("nonnegative", "Value should not be negative."),
}

def _rows(mask):
    return [str(int(i) + 1) for i in mask[mask].index[:5]]

def _finding(out, severity, check, mask_or_count, recommendation, examples=None):
    count = int(mask_or_count.sum()) if hasattr(mask_or_count, "sum") else int(mask_or_count)
    if count:
        out.append({"severity": severity, "check": check, "affected_rows": count,
                    "recommendation": recommendation, "examples": (examples or [])[:5]})

def data_quality_audit(df: pd.DataFrame) -> dict[str, Any]:
    """Return audit findings without mutating df or exposing cell values."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Dataset must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("Dataset must contain at least one row.")
    nrows, ncols = df.shape
    findings, missing = [], []

    for col in df.columns:
        count = int(df[col].isna().sum())
        if count:
            missing.append({"column": str(col), "missing_count": count,
                            "missing_percent": round(count / nrows * 100, 2)})
            recommendation = "Confirm whether missing values are valid for this field before imputing or deleting."
            if col == "Return_Reason":
                recommendation = "Cross-check order/return status; a reason may only be required for returned orders."
            elif col == "Campaign_Name":
                recommendation = "Confirm whether orders without a campaign are expected; do not assume missing means an error."
            _finding(findings, "review" if count / nrows >= .2 else "info",
                     f"Missing values in {col}", count, recommendation)

    duplicates = int(df.duplicated().sum())
    _finding(findings, "warning", "Exact duplicate rows", duplicates,
             "Review against the source system before removing duplicates.")

    if "Order_ID" in df:
        ids = df["Order_ID"].astype("string").str.strip()
        blank = ids.isna() | ids.eq("")
        _finding(findings, "warning", "Blank Order_ID values", blank,
                 "Populate an order identifier for each order record.", _rows(blank))
        repeated = ids.notna() & ids.ne("") & ids.duplicated(keep=False)
        _finding(findings, "warning", "Order_ID is not unique", repeated,
                 "Confirm data grain. If each row is an order, investigate repeats; if order-line grain, define an order-line key.",
                 _rows(repeated))

    parsed = {}
    for col in DATE_COLUMNS:
        if col not in df:
            continue
        raw = df[col]
        nonblank = raw.notna() & raw.astype("string").str.strip().ne("")
        dates = pd.to_datetime(raw, errors="coerce")
        parsed[col] = dates
        bad = nonblank & dates.isna()
        _finding(findings, "warning", f"Unparseable dates in {col}", bad,
                 "Inspect source formatting and convert valid dates to a consistent date type.", _rows(bad))
        blank = ~nonblank
        _finding(findings, "info", f"Blank dates in {col}", blank,
                 "Confirm whether this date is mandatory for every record.", _rows(blank))

    if {"Order_Date", "Delivery_Date"}.issubset(parsed):
        mask = parsed["Order_Date"].notna() & parsed["Delivery_Date"].notna() & (parsed["Delivery_Date"] < parsed["Order_Date"])
        _finding(findings, "warning", "Delivery_Date precedes Order_Date", mask,
                 "Check source records and date parsing; delivery normally follows order placement.", _rows(mask))

    for col, (rule, recommendation) in NUMERIC_RULES.items():
        if col not in df:
            continue
        raw = df[col]
        num = pd.to_numeric(raw, errors="coerce")
        nonblank = raw.notna() & raw.astype("string").str.strip().ne("")
        bad_numeric = nonblank & num.isna()
        _finding(findings, "warning", f"Non-numeric values in {col}", bad_numeric,
                 "Check source formatting and intended data type.", _rows(bad_numeric))
        if rule == "positive":
            bad = num.notna() & (num <= 0)
        elif rule == "percent":
            bad = num.notna() & ((num < 0) | (num > 100))
        elif rule == "age":
            bad = num.notna() & ((num < 0) | (num > 120))
        elif rule == "rating":
            bad = num.notna() & ((num < 1) | (num > 5))
        else:
            bad = num.notna() & (num < 0)
        _finding(findings, "warning", f"Out-of-range values in {col}", bad,
                 recommendation, _rows(bad))

    if "Customer_Phone" in df and pd.api.types.is_numeric_dtype(df["Customer_Phone"]):
        _finding(findings, "info", "Customer_Phone stored as numeric",
                 int(df["Customer_Phone"].notna().sum()),
                 "Store phone numbers as text to preserve leading zeros and avoid treating them as quantities.")

    # Candidate financial formulas; findings are review-only pending business-rule confirmation.
    formulas = [
        ("Gross_Sales_INR", {"Quantity_Sold", "Unit_Price_INR", "Gross_Sales_INR"},
         lambda d: d["Quantity_Sold"] * d["Unit_Price_INR"],
         "Gross sales versus quantity × unit price"),
        ("Discount_Amount_INR", {"Gross_Sales_INR", "Discount_Percent", "Discount_Amount_INR"},
         lambda d: d["Gross_Sales_INR"] * d["Discount_Percent"] / 100,
         "Discount amount versus gross sales × discount percentage"),
        ("Net_Sales_INR", {"Gross_Sales_INR", "Discount_Amount_INR", "Net_Sales_INR"},
         lambda d: d["Gross_Sales_INR"] - d["Discount_Amount_INR"],
         "Net sales versus gross sales − discount"),
        ("Tax_Amount_INR", {"Net_Sales_INR", "Tax_Percent", "Tax_Amount_INR"},
         lambda d: d["Net_Sales_INR"] * d["Tax_Percent"] / 100,
         "Tax amount versus net sales × tax percentage"),
        ("Total_Amount_INR", {"Net_Sales_INR", "Tax_Amount_INR", "Total_Amount_INR"},
         lambda d: d["Net_Sales_INR"] + d["Tax_Amount_INR"],
         "Total amount versus net sales + tax"),
    ]
    for target, required, expected_fn, label in formulas:
        if not required.issubset(df.columns):
            continue
        work = df[list(required)].copy()
        for col in required:
            work[col] = pd.to_numeric(work[col], errors="coerce")
        comparable = work[list(required)].notna().all(axis=1)
        if not comparable.any():
            continue
        mismatch = comparable & ((work[target] - expected_fn(work)).abs() > 0.02)
        _finding(findings, "review", f"Financial reconciliation: {label}", mismatch,
                 "Confirm documented pricing, rounding, tax, cancellation and refund rules before treating differences as errors.",
                 _rows(mismatch))

    if {"Customer_ID", "Customer_Lifetime_Value_INR"}.issubset(df.columns):
        clv = df[["Customer_ID", "Customer_Lifetime_Value_INR"]].copy()
        clv["Customer_Lifetime_Value_INR"] = pd.to_numeric(clv["Customer_Lifetime_Value_INR"], errors="coerce")
        clv = clv.dropna(subset=["Customer_ID", "Customer_Lifetime_Value_INR"])
        if not clv.empty:
            counts = clv.groupby("Customer_ID")["Customer_Lifetime_Value_INR"].nunique()
            inconsistent = counts[counts > 1].index
            mask = df["Customer_ID"].isin(inconsistent)
            _finding(findings, "review", "Customer lifetime value varies within customer", mask,
                     "Confirm whether CLV is a snapshot or should be constant per customer; avoid summing repeated customer-level values across order rows.",
                     _rows(mask))

    order = {"critical": 0, "warning": 1, "review": 2, "info": 3}
    findings.sort(key=lambda x: (order.get(x["severity"], 9), -x["affected_rows"], x["check"]))
    counts = {s: sum(f["severity"] == s for f in findings) for s in ("critical", "warning", "review", "info")}
    return {
        "row_count": int(nrows), "column_count": int(ncols),
        "duplicate_rows": duplicates, "missing_columns": missing,
        "findings": findings, "finding_counts": counts,
        "audit_note": "Read-only checks. Financial formulas are candidate rules and need confirmation against documented business logic.",
    }
