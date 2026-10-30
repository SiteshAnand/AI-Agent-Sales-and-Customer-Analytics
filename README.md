# Business Analysis & Customer Success AI Agent

Local Streamlit app for analysing uploaded CSV/Excel business datasets. Python performs calculations; Gemini maps natural-language questions to approved analytics tools and explains the result.

## Setup (Windows PowerShell)

1. Open this folder in VS Code.
2. Create/activate a virtual environment if desired.
3. Install dependencies: `python -m pip install -r requirements.txt`
4. Copy `.env.example` to `.env` and add your Gemini API key.
5. Start: `python -m streamlit run app.py`
6. Tests: `python -m pytest -q`

## Configuration

`config/settings.py` loads `.env`. Do not upload `.env` or share your API key.

## Current analytics tools

Numeric summaries, distinct counts, grouped metrics/rankings, filtered metrics, monthly/quarterly/yearly totals, and percentage change when current/previous values are supplied explicitly. Unsupported questions should be reported as unsupported rather than having figures invented.

## Ranking behaviour

Clear highest/lowest salesperson sales questions are calculated locally without calling Gemini. Requested counts such as “top 3” or “five lowest” are honoured, up to the available groups. Other question types use Gemini's validated tool plan.

## Deterministic workflows added in the reliability update

- Dataset-overview questions (columns, types, nonblank counts, distinct counts, and available analysis categories) are answered locally without requiring Gemini.
- Monthly sales comparisons by region calculate monthly totals, strongest/weakest periods, month-to-month absolute and percentage changes, and cautious management actions from the observed data.
- These workflows are covered by regression tests, including a zero-baseline percentage-change case.

The app is designed for reliable analysis of supported question patterns, not arbitrary unrestricted reasoning. Unsupported questions should fail clearly rather than invent results. Validate any recommendations against the business context before acting on them.


## Customer reactivation analysis

Questions such as “list customers with high sales value who have not bought recently” are handled locally when customer ID, transaction date, and sales-value columns are available. The default inactivity threshold is 90 days; a threshold such as “60 days” or “3 months” in the question overrides it. The snapshot date is anchored to the latest valid transaction date in the uploaded data. The response shows up to 20 qualifying customers, ranked by historical sales value, with last purchase, days inactive, transaction count, and sales value.

See `RELIABILITY_AUDIT.md` for the changes made and verification limits.


## Product-wise monthly sales forecasting

The agent supports questions such as “Forecast product-wise sales for the next two months based on historical data.” It aggregates the configured sales-value field by product and calendar month, then produces a transparent baseline forecast for each eligible product. Products with fewer than three monthly observations are reported as insufficient-history cases rather than assigned fabricated forecasts.

Forecast method selection is history-dependent: seasonal-naive (same month in the prior year) when at least 24 monthly observations are available; damped linear trend for 6–23 months; and a trailing-three-month average for 3–5 months. Forecasts are estimates, not guarantees. Missing months within a product's observed span are treated as zero sales; negative historical values are retained and may influence the trend, while future negative predictions are floored at zero. Validate these assumptions against the business's transaction/returns policy before operational use.

## Reliability upgrade notes

This build includes deterministic total-sales and product-wise monthly forecasts (1–3 month horizons from natural-language requests), pairwise Market Basket Analysis (support, confidence, lift), and independent execution of clearly separated multi-part questions using semicolons, new lines, or phrases such as “and also”. Existing RFM, churn classification, regression, data-quality, and rule-based sales analysis modules are retained.

### Market Basket input

Provide one row per item within a transaction, with a transaction/order/invoice ID column and a product/item column. The model deduplicates repeated product rows within a transaction and reports pairwise association rules. This is association, not causal evidence.

### Forecasting notes

Forecasts are baseline estimates based on monthly totals. Seasonal-naive forecasts are used with at least 24 months of history; a damped trend is used with 6–23 months; a recent three-month average is used with 3–5 months. Forecasts are estimates, not guarantees. Verify that the latest month is complete and that missing months legitimately mean zero sales before interpreting results.

### Tests

Run `python -m pytest -q` from the project root. Test dependencies are listed in `requirements.txt`. Keep your real `.env` file and private source data outside shared ZIPs; `.env.example` should contain placeholders only.

## Six-layer architecture

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full design. The current implementation separates:

1. Query intelligence (`core/query_intelligence.py`) — bounded multi-task planning.
2. Data understanding (`core/data_understanding.py`) — dataset profile and semantic role hints.
3. Deterministic analytics (`core/analytics.py`, `core/tools.py`, `core/rule_based_analysis.py`).
4. Modeling (`core/forecasting.py`, `core/rfm.py`, `core/churn_model.py`, `core/market_basket.py`, `core/regression.py`).
5. Validation (`core/result_validation.py` plus tool-specific checks).
6. Streamlit experience (`app.py`) — existing upload and response interface retained.

### Multi-item Market Basket Analysis

Market Basket Analysis supports bounded Apriori itemsets of 2–4 products. You can specify “3-item combinations” or “4-item rules” in a request. Defaults search up to 3 items. The search has a candidate safety limit; increase minimum support or reduce itemset size if the data has very high product cardinality.

### Long and multi-part questions

The default maximum question length is 10,000 characters. Clear task boundaries include semicolons, new lines, “then also”, “and also”, and “and” followed by a new action (such as “and identify…”). The agent processes tasks sequentially and reports partial failures instead of concealing them.
