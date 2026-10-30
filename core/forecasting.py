"""Deterministic, transparent monthly product-sales forecasting.

This is a lightweight baseline forecaster, not a claim of guaranteed accuracy.
It uses seasonal-naive forecasts when at least 24 monthly observations exist,
otherwise a damped linear trend (6+ months) or recent mean (3-5 months).
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd


def _find_column(df: pd.DataFrame, aliases: list[str]):
    normalized = {
        re.sub(r"[^a-z0-9]+", "", str(col).casefold()): col
        for col in df.columns
    }
    for alias in aliases:
        key = re.sub(r"[^a-z0-9]+", "", alias.casefold())
        if key in normalized:
            return normalized[key]
    return None


def forecast_product_sales(df: pd.DataFrame, horizon: int = 2) -> dict:
    """Forecast monthly sales per product and return JSON-safe business results."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Upload a dataset containing transaction rows first.")
    if type(horizon) is not int or not 1 <= horizon <= 12:
        raise ValueError("Forecast horizon must be an integer from 1 to 12 months.")

    date_col = _find_column(df, ["Order_Date", "Order Date", "Transaction_Date", "Transaction Date", "Purchase_Date", "Purchase Date", "Date"])
    product_col = _find_column(df, ["Product_Name", "Product Name", "Product", "Product_ID", "Product ID", "Item_Name", "Item Name", "SKU"])
    sales_col = _find_column(df, ["Net_Sales_INR", "Net Sales INR", "Net Sales", "Sales_Value", "Sales Value", "Sales_Amount", "Sales Amount", "Revenue", "Total_Amount_INR", "Gross_Sales_INR", "Sales"])

    missing = [label for label, col in (("transaction date", date_col), ("product identifier/name", product_col), ("sales value", sales_col)) if col is None]
    if missing:
        raise ValueError("Product-wise forecasting requires a date, product, and sales-value column. Missing: " + ", ".join(missing) + ".")

    data = df[[date_col, product_col, sales_col]].copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce", utc=True)
    data[product_col] = data[product_col].astype("string").str.strip()
    data[sales_col] = pd.to_numeric(data[sales_col], errors="coerce")
    data = data.dropna(subset=[date_col, product_col, sales_col])
    data = data[data[product_col].ne("") & np.isfinite(data[sales_col])].copy()
    if data.empty:
        raise ValueError("No valid rows with dates, product names, and numeric sales values were found.")

    # Periods are timezone-free monthly buckets; snapshot is data-driven.
    data["_month"] = data[date_col].dt.tz_convert(None).dt.to_period("M")
    latest_month = data["_month"].max()
    first_month = data["_month"].min()
    if latest_month < first_month:
        raise ValueError("The date range could not be validated.")

    rows, insufficient = [], []
    methods = set()
    for product, product_rows in data.groupby(product_col, sort=True):
        monthly = product_rows.groupby("_month")[sales_col].sum().sort_index()
        start = monthly.index.min()
        # Missing months between first and last observed product transaction are
        # represented as zero transaction sales, rather than silently skipped.
        full_index = pd.period_range(start=start, end=latest_month, freq="M")
        series = monthly.reindex(full_index, fill_value=0.0).astype(float)
        n = len(series)
        if n < 3:
            insufficient.append({"product": str(product), "months_available": int(n)})
            continue

        if n >= 24:
            method = "seasonal_naive_12_months"
            preds = [float(series.iloc[-12 + (step % 12)]) for step in range(horizon)]
        elif n >= 6:
            method = "damped_linear_trend"
            window = series.tail(12).to_numpy(dtype=float)
            x = np.arange(len(window), dtype=float)
            slope, intercept = np.polyfit(x, window, 1)
            # Damp the trend to avoid extreme linear extrapolation.
            level = float(window[-1])
            preds = []
            for step in range(1, horizon + 1):
                damped_delta = slope * sum(0.8 ** k for k in range(step))
                preds.append(max(0.0, level + damped_delta))
        else:
            method = "recent_3_month_average"
            average = float(series.tail(3).mean())
            preds = [max(0.0, average)] * horizon
        methods.add(method)

        for step, prediction in enumerate(preds, start=1):
            target_month = latest_month + step
            rows.append({
                "Product": str(product),
                "Forecast_Month": str(target_month),
                "Forecast_Sales": round(max(0.0, float(prediction)), 2),
                "Historical_Months": int(n),
                "Method": method,
                "Historical_Total_Sales": round(float(product_rows[sales_col].sum()), 2),
            })

    if not rows:
        raise ValueError("No products have at least 3 months of usable sales history. More history is needed to produce a responsible forecast.")

    forecast = pd.DataFrame(rows)
    summary = (forecast.groupby("Forecast_Month", as_index=False)["Forecast_Sales"]
               .sum().rename(columns={"Forecast_Sales": "Total_Forecast_Sales"}))
    return {
        "date_column_used": str(date_col),
        "product_column_used": str(product_col),
        "sales_column_used": str(sales_col),
        "last_historical_month": str(latest_month),
        "forecast_months": [str(latest_month + step) for step in range(1, horizon + 1)],
        "products_forecast": int(forecast["Product"].nunique()),
        "forecast_rows": forecast.to_dict(orient="records"),
        "monthly_summary": summary.to_dict(orient="records"),
        "insufficient_history": insufficient,
        "methods_used": sorted(methods),
        "warnings": [
            "Forecasts are estimates based on historical monthly sales, not guarantees.",
            "Negative forecasts are floored at zero; historical negative sales/returns remain in the input and can affect the trend.",
            "Missing months within a product's observed history are treated as zero sales.",
        ],
    }



def is_product_forecast_question(question: str) -> bool:
    """Recognize future sales forecasts, including overall and product-wise sales."""
    text = str(question).casefold()

    future = bool(re.search(
        r"\b(forecast|forecasting|predict|prediction|project)\w*\b",
        text,
    ))
    sales = bool(re.search(
        r"\b(sales|revenue|turnover)\b",
        text,
    ))
    horizon = bool(
        re.search(r"\b(next|coming|future|upcoming)\b", text)
        or re.search(r"\b\d+\s*(months?|quarters?)\b", text)
        or re.search(r"\b(month|quarter|year)\b", text)
    )

    return future and sales and horizon

def forecast_total_sales(df: pd.DataFrame, horizon: int = 1) -> dict:
    """Forecast total monthly sales without requiring a product column."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Upload a dataset containing transaction rows first.")

    if type(horizon) is not int or not 1 <= horizon <= 12:
        raise ValueError("Forecast horizon must be an integer from 1 to 12 months.")

    date_col = _find_column(
        df,
        ["Order_Date", "Order Date", "Transaction_Date",
         "Transaction Date", "Purchase_Date", "Purchase Date", "Date"],
    )
    sales_col = _find_column(
        df,
        ["Net_Sales_INR", "Net Sales INR", "Net Sales",
         "Sales_Value", "Sales Value", "Sales_Amount",
         "Sales Amount", "Revenue", "Total_Amount_INR",
         "Gross_Sales_INR", "Sales"],
    )

    missing = [
        name for name, column in (
            ("transaction date", date_col),
            ("sales value", sales_col),
        ) if column is None
    ]
    if missing:
        raise ValueError(
            "Total-sales forecasting requires a date and sales-value column. "
            f"Missing: {', '.join(missing)}."
        )

    data = df[[date_col, sales_col]].copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce", utc=True)
    data[sales_col] = pd.to_numeric(data[sales_col], errors="coerce")
    data = data.dropna(subset=[date_col, sales_col])
    data = data[np.isfinite(data[sales_col])].copy()

    if data.empty:
        raise ValueError("No valid dated numeric sales values were found.")

    data["_month"] = data[date_col].dt.tz_convert(None).dt.to_period("M")
    monthly = data.groupby("_month")[sales_col].sum().sort_index()

    # Fill gaps between observed months so the model uses a continuous series.
    full_index = pd.period_range(
        start=monthly.index.min(),
        end=monthly.index.max(),
        freq="M",
    )
    series = monthly.reindex(full_index, fill_value=0.0).astype(float)

    if len(series) < 3:
        raise ValueError(
            "At least 3 months of historical sales are required for forecasting."
        )

    n = len(series)

    if n >= 24:
        method = "seasonal_naive_12_months"
        predictions = [
            float(series.iloc[-12 + (step % 12)])
            for step in range(horizon)
        ]
    elif n >= 6:
        method = "damped_linear_trend"
        window = series.tail(12).to_numpy(dtype=float)
        x = np.arange(len(window), dtype=float)
        slope, _ = np.polyfit(x, window, 1)
        level = float(window[-1])
        predictions = [
            max(
                0.0,
                level + slope * sum(0.8 ** k for k in range(step)),
            )
            for step in range(1, horizon + 1)
        ]
    else:
        method = "recent_3_month_average"
        average = float(series.tail(3).mean())
        predictions = [max(0.0, average)] * horizon

    last_month = series.index[-1]
    forecast_rows = []

    for step, prediction in enumerate(predictions, start=1):
        target_month = last_month + step
        forecast_rows.append({
            "Forecast_Month": str(target_month),
            "Forecast_Total_Sales": round(float(prediction), 2),
            "Currency": "INR",
            "Method": method,
            "Historical_Months": int(n),
        })

    return {
        "date_column_used": str(date_col),
        "sales_column_used": str(sales_col),
        "last_historical_month": str(last_month),
        "forecast_months": forecast_rows,
        "method": method,
        "warning": (
            "This is a baseline estimate from historical monthly totals, "
            "not a guaranteed outcome. If the latest month is incomplete, "
            "the forecast may be affected."
        ),
    }


