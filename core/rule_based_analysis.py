
"""Deterministic multi-step analyses for common business questions.

All reported figures are computed locally from the uploaded DataFrame.
Gemini is not needed for these supported workflows.
"""
from __future__ import annotations

import calendar
import re
from typing import Any

import pandas as pd


def _find_column(df: pd.DataFrame, aliases: list[str]):
    normalized = {
        re.sub(r"[^a-z0-9]+", "", str(c).casefold()): c
        for c in df.columns
    }
    for alias in aliases:
        key = re.sub(r"[^a-z0-9]+", "", alias.casefold())
        if key in normalized:
            return normalized[key]
    return None


def _money(value: float) -> str:
    return f"{value:,.2f}".rstrip("0").rstrip(".")


def _month_sort_key(value: Any):
    text = str(value).strip()
    for fmt in ("%b", "%B", "%Y-%m", "%Y-%m-%d", "%b %Y", "%B %Y"):
        try:
            parsed = pd.to_datetime(text, format=fmt, errors="raise")
            return (0, parsed.year, parsed.month, parsed.day)
        except (ValueError, TypeError):
            pass

    parsed = pd.to_datetime(text, errors="coerce")
    if not pd.isna(parsed):
        return (0, parsed.year, parsed.month, parsed.day)

    return (1, text.casefold())


def _month_lookup():
    lookup = {}
    for number in range(1, 13):
        lookup[calendar.month_name[number].casefold()] = number
        lookup[calendar.month_abbr[number].casefold()] = number
    return lookup


def _parse_month_year(value):
    """Return (month number, year or None) from a date/month label."""
    if pd.isna(value):
        return None, None

    text = str(value).strip()
    if not text:
        return None, None

    lookup = _month_lookup()
    year_match = re.search(r"\b((?:19|20)\d{2})\b", text)
    year = int(year_match.group(1)) if year_match else None

    parsed = pd.to_datetime(text, errors="coerce")
    if not pd.isna(parsed):
        # Only treat a year as known if the label explicitly contains one.
        return parsed.month, year

    month = lookup.get(text.casefold())
    return month, year


def _question_periods(question: str):
    """Extract month/year periods in the order mentioned in the question."""
    text = question.casefold()
    lookup = _month_lookup()

    month_pattern = "|".join(
        sorted(
            (re.escape(name) for name in lookup),
            key=len,
            reverse=True,
        )
    )

    pattern = (
        r"\b(" + month_pattern + r")\b"
        r"(?:\s+((?:19|20)\d{2}))?"
    )

    periods = []
    for match in re.finditer(pattern, text):
        month = lookup[match.group(1)]
        year = int(match.group(2)) if match.group(2) else None
        period = (month, year)

        if not periods or periods[-1] != period:
            periods.append(period)

    return periods


def dataset_overview(question: str, df: pd.DataFrame) -> dict:
    columns = []

    for col in df.columns:
        series = df[col]
        columns.append({
            "name": str(col),
            "type": str(series.dtype),
            "non_null": int(series.notna().sum()),
            "distinct": int(series.nunique(dropna=True)),
        })

    lines = [
        "**Dataset overview**",
        f"- Rows: {len(df):,}",
        f"- Columns: {len(df.columns):,}",
        "",
        "**Available columns**",
    ]

    lines.extend(
        f"- **{c['name']}** (`{c['type']}`): "
        f"{c['non_null']:,} nonblank values, "
        f"{c['distinct']:,} distinct values"
        for c in columns
    )

    lines += [
        "",
        "**Analyses available when the required columns exist**",
        "- Totals, averages, minimums/maximums, and counts by region, "
        "product, customer, channel, or another category",
        "- Rankings and comparisons between groups",
        "- Monthly/quarterly/yearly trends and period-over-period changes "
        "when a time field is available",
        "- Data-quality checks, missing values, duplicate checks, "
        "and descriptive summaries",
        "- Evidence-based management recommendations based on observed results",
    ]

    return {
        "status": "success",
        "tool": "dataset_overview",
        "question": question,
        "explanation": "Dataset columns and analysis options inspected locally.",
        "result": {
            "columns": columns,
            "markdown": "\n".join(lines),
        },
    }


def monthly_region_analysis(
    question: str,
    df: pd.DataFrame,
) -> dict | None:
    text = question.casefold()

    if not (
        "month" in text
        and any(w in text for w in (
            "region",
            "monthly performance",
            "month-to-month",
            "month to month",
            "trend",
            "compare",
        ))
    ):
        return None

    month_col = _find_column(
        df,
        ["Month", "Order Month", "Sales Month", "Date",
         "Order Date", "Transaction Date"],
    )
    region_col = _find_column(
        df, ["Region", "Territory", "Zone", "Market"]
    )
    sales_col = _find_column(
        df, ["Sales", "Revenue", "Sales Amount", "Net Sales", "Amount"]
    )

    if not all((month_col, region_col, sales_col)):
        return None

    data = df[[month_col, region_col, sales_col]].copy()
    data[month_col] = data[month_col].astype("string").str.strip()
    data[region_col] = data[region_col].astype("string").str.strip()
    data[sales_col] = pd.to_numeric(data[sales_col], errors="coerce")

    data = data.dropna(subset=[month_col, region_col, sales_col])
    data = data[(data[month_col] != "") & (data[region_col] != "")]

    if data.empty:
        return None

    grouped = (
        data.groupby([month_col, region_col], dropna=True)[sales_col]
        .sum()
        .reset_index()
    )

    month_values = sorted(
        grouped[month_col].unique().tolist(),
        key=_month_sort_key,
    )
    regions = sorted(
        grouped[region_col].unique().tolist(),
        key=lambda x: str(x).casefold(),
    )

    matrix = (
        grouped.pivot(
            index=month_col,
            columns=region_col,
            values=sales_col,
        )
        .reindex(month_values)
    )

    markdown = [
        "**Monthly sales comparison**",
        "",
        "| Month | " + " | ".join(str(r) for r in regions) + " |",
        "|" + "---|" * (len(regions) + 1),
    ]

    for month in month_values:
        vals = [
            matrix.loc[month, r]
            if r in matrix.columns
            else float("nan")
            for r in regions
        ]
        markdown.append(
            "| " + str(month) + " | "
            + " | ".join(
                "—" if pd.isna(v) else _money(float(v))
                for v in vals
            )
            + " |"
        )

    markdown += ["", "**Strongest and weakest months**"]
    trends = {}

    for region in regions:
        series = (
            matrix[region].dropna()
            if region in matrix.columns
            else pd.Series(dtype=float)
        )
        if series.empty:
            continue

        high, low = float(series.max()), float(series.min())
        highs = [str(k) for k, v in series.items() if float(v) == high]
        lows = [str(k) for k, v in series.items() if float(v) == low]

        markdown.append(
            f"- **{region}:** strongest: {', '.join(highs)} "
            f"({_money(high)}); weakest: {', '.join(lows)} "
            f"({_money(low)})."
        )

        changes = []
        ordered_series = matrix[region]

        for i in range(1, len(month_values)):
            prev_month = month_values[i - 1]
            current_month = month_values[i]
            previous = ordered_series.loc[prev_month]
            current = ordered_series.loc[current_month]

            if pd.isna(previous) or pd.isna(current):
                continue

            delta = float(current - previous)
            pct = (
                None if float(previous) == 0
                else delta / abs(float(previous)) * 100
            )

            changes.append({
                "from": str(prev_month),
                "to": str(current_month),
                "previous": float(previous),
                "current": float(current),
                "delta": delta,
                "percentage_change": pct,
            })

        trends[str(region)] = changes

    markdown += ["", "**Month-to-month changes**"]

    for region in regions:
        for item in trends.get(str(region), []):
            pct_text = (
                "not calculable (zero baseline)"
                if item["percentage_change"] is None
                else f"{item['percentage_change']:+.2f}%"
            )
            direction = (
                "up" if item["delta"] > 0
                else "down" if item["delta"] < 0
                else "unchanged"
            )

            markdown.append(
                f"- **{region}, {item['from']} → {item['to']}:** "
                f"{_money(item['delta'])} ({pct_text}; {direction})."
            )

    markdown += ["", "**Three management actions**"]

    totals = data.groupby(region_col)[sales_col].sum().sort_values(
        ascending=False
    )
    best_region = str(totals.index[0])
    weakest_region = str(totals.index[-1])

    markdown.append(
        f"1. Review the weaker total-sales region (**{weakest_region}**) "
        "by account, product, channel, and order volume before deciding "
        "on corrective action; the current dataset alone does not establish "
        "the cause."
    )
    markdown.append(
        "2. Investigate the largest month-to-month declines shown above: "
        "validate whether they reflect volume, price/mix, lost customers, "
        "stock availability, or incomplete reporting."
    )
    markdown.append(
        f"3. Study what coincided with the stronger performance of "
        f"**{best_region}** in its peak month(s), then test whether those "
        "practices can be replicated elsewhere; verify with account/product-"
        "level data before scaling."
    )
    markdown.append(
        "\n*Method: totals and changes were calculated locally from the "
        "uploaded rows. Recommendations are hypotheses to investigate, "
        "not proven causal findings.*"
    )

    return {
        "status": "success",
        "tool": "monthly_region_analysis",
        "question": question,
        "explanation": (
            "Monthly comparison, extrema, and period-over-period changes "
            "calculated locally."
        ),
        "result": {
            "monthly_totals": grouped.to_dict(orient="records"),
            "trends": trends,
            "markdown": "\n".join(markdown),
        },
    }


def sales_decline_investigation(
    question: str,
    df: pd.DataFrame,
) -> dict | None:
    """Investigate a sales decline without mixing different years."""
    text = question.casefold()

    decline_terms = (
        "decline", "declined", "drop", "dropped", "fell",
        "decrease", "decreased", "investigate", "explain the decline",
        "why sales",
    )
    if not any(term in text for term in decline_terms):
        return None

    if df is None or df.empty:
        return None

    month_col = _find_column(
        df, ["Month", "Order Month", "Sales Month", "Date", "Order Date"]
    )
    sales_col = _find_column(
        df, ["Sales", "Revenue", "Sales Amount", "Net Sales", "Amount"]
    )
    quantity_col = _find_column(
        df, ["Quantity", "Units Sold", "Qty", "Units"]
    )
    customer_col = _find_column(
        df, ["Customers", "Customer Count", "Customers Count"]
    )
    region_col = _find_column(
        df, ["Region", "Territory", "Zone", "Market"]
    )
    product_col = _find_column(
        df, ["Product", "Product Name", "Item", "SKU"]
    )

    if not month_col or not sales_col:
        return None

    periods = _question_periods(question)
    if len(periods) < 2:
        return None

    (old_month, old_year), (new_month, new_year) = periods[:2]

    data = df.copy()
    parsed_periods = data[month_col].map(_parse_month_year)
    data["_analysis_month"] = parsed_periods.map(lambda p: p[0])
    data["_analysis_year"] = parsed_periods.map(lambda p: p[1])

    data[sales_col] = pd.to_numeric(data[sales_col], errors="coerce")
    if quantity_col:
        data[quantity_col] = pd.to_numeric(
            data[quantity_col], errors="coerce"
        )
    if customer_col:
        data[customer_col] = pd.to_numeric(
            data[customer_col], errors="coerce"
        )

    selected_region = None
    if region_col:
        for region in data[region_col].dropna().astype(str).unique():
            if re.search(r"\b" + re.escape(region.casefold()) + r"\b", text):
                selected_region = region
                break

    if selected_region and region_col:
        data = data[
            data[region_col].astype(str).str.casefold()
            == selected_region.casefold()
        ]

    def select_period(month, year):
        mask = data["_analysis_month"] == month

        if year is not None:
            mask &= data["_analysis_year"] == year
        else:
            # Do not silently add together multiple years if the question
            # does not specify which year it means.
            years = data.loc[mask, "_analysis_year"].dropna().unique()
            if len(years) > 1:
                return data.iloc[0:0]

        return data.loc[mask]

    previous = select_period(old_month, old_year)
    current = select_period(new_month, new_year)

    if previous.empty or current.empty:
        return None

    old_sales = float(previous[sales_col].sum())
    new_sales = float(current[sales_col].sum())
    change = new_sales - old_sales

    if change >= 0:
        return None

    pct = change / abs(old_sales) * 100 if old_sales else None

    old_label = (
        f"{calendar.month_name[old_month]} {old_year}"
        if old_year else calendar.month_name[old_month]
    )
    new_label = (
        f"{calendar.month_name[new_month]} {new_year}"
        if new_year else calendar.month_name[new_month]
    )

    lines = [
        "**Sales decline investigation**",
        f"- Period: {old_label} → {new_label}",
        f"- Region: {selected_region or 'All regions'}",
        f"- Sales: {_money(old_sales)} → {_money(new_sales)}",
        f"- Change: {_money(change)} "
        + (f"({pct:.2f}%)" if pct is not None else
           "(percentage unavailable)"),
    ]

    for label, column in (
        ("Quantity", quantity_col),
        ("Customer field total", customer_col),
    ):
        if column:
            old_value = float(previous[column].sum())
            new_value = float(current[column].sum())
            delta = new_value - old_value
            change_pct = (
                delta / abs(old_value) * 100 if old_value else None
            )

            lines.append(
                f"- {label}: {_money(old_value)} → {_money(new_value)}; "
                f"change {_money(delta)}"
                + (
                    f" ({change_pct:+.2f}%)"
                    if change_pct is not None else ""
                )
            )

    if product_col:
        lines += ["", "**Product-level sales**"]
        old_products = previous.groupby(product_col)[sales_col].sum()
        new_products = current.groupby(product_col)[sales_col].sum()

        for product in sorted(
            set(old_products.index) | set(new_products.index),
            key=lambda value: str(value).casefold(),
        ):
            old_value = float(old_products.get(product, 0))
            new_value = float(new_products.get(product, 0))

            lines.append(
                f"- {product}: {_money(old_value)} → "
                f"{_money(new_value)}; "
                f"change {_money(new_value - old_value)}"
            )

    lines += [
        "",
        "**Evidence-based interpretation**",
        "- The sales, quantity and customer-field changes above are "
        "observed facts.",
        "- A matching fall in sales and quantity is consistent with "
        "lower sales volume.",
        "- The product comparison shows how recorded product sales changed.",
        "- These figures alone cannot establish price changes, customer churn, "
        "lost orders, stock problems or the underlying cause.",
        "- Verify order-level records, prices, customer identities and stock "
        "data before claiming a causal explanation.",
    ]

    return {
        "status": "success",
        "tool": "sales_decline_investigation",
        "question": question,
        "explanation": (
            "Year-aware sales decline measures and product breakdown "
            "calculated locally."
        ),
        "result": {"markdown": "\n".join(lines)},
    }


def regional_sales_decline_comparison(
    question: str,
    df: pd.DataFrame,
) -> dict | None:
    """Compare regional sales between two specified periods."""
    if df is None or df.empty:
        return None

    text = question.casefold()

    decline_terms = (
        "decline", "declined", "drop", "dropped", "fell", "fall",
        "decrease", "decreased", "largest sales decline", "sales decline",
    )
    region_terms = ("region", "regional", "by region", "each region")

    if not any(term in text for term in decline_terms):
        return None
    if not any(term in text for term in region_terms):
        return None

    month_col = _find_column(
        df, ["Month", "Order Month", "Sales Month", "Date", "Order Date"]
    )
    region_col = _find_column(
        df, ["Region", "Sales Region", "Territory", "Zone", "Market"]
    )
    sales_col = _find_column(
        df, ["Sales", "Revenue", "Net Sales", "Sales Amount", "Amount"]
    )

    if not all([month_col, region_col, sales_col]):
        return None

    periods = _question_periods(question)
    if len(periods) < 2:
        return None

    (old_month, old_year), (new_month, new_year) = periods[:2]

    working = df.copy()
    parsed_periods = working[month_col].map(_parse_month_year)
    working["_analysis_month"] = parsed_periods.map(lambda p: p[0])
    working["_analysis_year"] = parsed_periods.map(lambda p: p[1])
    working[sales_col] = pd.to_numeric(working[sales_col], errors="coerce")

    working = working.dropna(
        subset=["_analysis_month", region_col, sales_col]
    )

    def select_period(month, year):
        mask = working["_analysis_month"] == month

        if year is not None:
            mask &= working["_analysis_year"] == year
        else:
            years = working.loc[mask, "_analysis_year"].dropna().unique()
            if len(years) > 1:
                return working.iloc[0:0]

        return working.loc[mask]

    old_data = select_period(old_month, old_year)
    new_data = select_period(new_month, new_year)

    if old_data.empty or new_data.empty:
        return None

    old_totals = old_data.groupby(region_col)[sales_col].sum()
    new_totals = new_data.groupby(region_col)[sales_col].sum()

    common_regions = old_totals.index.intersection(new_totals.index)
    if len(common_regions) == 0:
        return None

    results = []
    for region in common_regions:
        before = float(old_totals.loc[region])
        after = float(new_totals.loc[region])
        change = after - before
        change_pct = (
            change / abs(before) * 100 if before != 0 else None
        )

        results.append({
            "region": str(region),
            "previous_sales": before,
            "current_sales": after,
            "change": change,
            "change_pct": change_pct,
        })

    declines = [item for item in results if item["change"] < 0]
    declines.sort(key=lambda item: item["change"])

    old_label = (
        f"{calendar.month_name[old_month]} {old_year}"
        if old_year else calendar.month_name[old_month]
    )
    new_label = (
        f"{calendar.month_name[new_month]} {new_year}"
        if new_year else calendar.month_name[new_month]
    )

    lines = [
        f"## Regional sales comparison: {old_label} to {new_label}",
        "",
        "| Region | Previous sales | Current sales | Change | Change % |",
        "|---|---:|---:|---:|---:|",
    ]

    for item in sorted(results, key=lambda row: row["change"]):
        pct = (
            f'{item["change_pct"]:+.2f}%'
            if item["change_pct"] is not None
            else "N/A (previous sales were zero)"
        )
        lines.append(
            f'| {item["region"]} '
            f'| {_money(item["previous_sales"])} '
            f'| {_money(item["current_sales"])} '
            f'| {_money(item["change"])} '
            f'| {pct} |'
        )

    if declines:
        largest = declines[0]
        largest_pct = (
            f'{largest["change_pct"]:.2f}%'
            if largest["change_pct"] is not None
            else "N/A (previous sales were zero)"
        )
        lines.extend([
            "",
            "**Largest regional decline**",
            f'- Region: {largest["region"]}',
            f'- Sales reduction: {_money(abs(largest["change"]))}',
            f'- Percentage change: {largest_pct}',
            "",
            "This identifies the largest recorded sales decrease between "
            "the selected periods. It does not establish the cause.",
        ])
    else:
        lines.extend([
            "",
            "No region recorded a sales decline between these periods.",
        ])

    return {
        "status": "success",
        "tool": "regional_sales_decline_comparison",
        "question": question,
        "markdown": "\n".join(lines),
        "regions": results,
        "largest_decline": declines[0] if declines else None,
        "previous_month": old_label,
        "current_month": new_label,
    }
