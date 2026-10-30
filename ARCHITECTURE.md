# Business Analytics Agent — six-layer architecture

This release adds explicit boundaries around the existing agent rather than replacing the current app or analytics modules.

## 1. Query intelligence (`core/query_intelligence.py`)

- Splits questions only at clear separators or conjunctions followed by a new action verb.
- Produces a transparent sequential task plan, with a bounded maximum task count.
- Leaves calculations and tool execution to the existing agent/tool layer.
- The agent preserves partial task results and reports failures rather than hiding them.
- Maximum query length defaults to 10,000 characters and can be configured with `MAX_QUERY_LENGTH`.

## 2. Data understanding (`core/data_understanding.py`)

- Profiles column types, missingness, distinct values, duplicate rows, sample values, and date coverage for likely date columns.
- Infers common semantic roles (date, sales value, customer, product, transaction, region, quantity) from normalized column names.
- Role inference is explicitly a hint; the business meaning and transaction grain still require confirmation.
- The read-only data quality audit now includes the profile and role hints.

## 3. Deterministic analytics (`core/analytics.py`, `core/tools.py`, `core/rule_based_analysis.py`)

- Existing Python calculation tools remain the source of numeric results.
- The LLM selects or interprets tools; it is not trusted to invent computed figures.
- Tool inputs continue to be checked against actual dataset columns and allowed parameters.

## 4. Modeling (`core/forecasting.py`, `core/rfm.py`, `core/churn_model.py`, `core/market_basket.py`, `core/regression.py`)

- Forecasting supports total sales and product-wise sales, with a requested 1–3 month horizon and data-history-dependent baseline methods.
- RFM returns customer-level recency, frequency, monetary metrics, and segments.
- Churn uses an explicit target column and classification evaluation; a meaningful target must exist.
- Market Basket Analysis uses bounded Apriori search for rules involving itemsets of 2–4 items, with configurable support, confidence, lift, and a candidate safety limit.
- Regression remains a separate model path.

## 5. Validation (`core/result_validation.py` and tool-specific validation)

- Multi-task responses retain each task's status, successful output, or failure message.
- A recursive response-shape/numeric-finiteness check flags invalid numeric values.
- Domain correctness still depends on model-specific tests, assumptions, and suitable source data; generic validation cannot prove a forecast is accurate.

## 6. Streamlit experience (`app.py`)

- The existing Streamlit upload and answer presentation flow is retained.
- Existing feature routes remain in `core/agent.py`; the new modules are integrated into that route rather than creating a second app.
- End-to-end interaction with a real Gemini key and the user's actual data should be verified locally after extraction.

## Validation performed for this ZIP

- Full automated test suite: see `UPGRADE_REPORT.md` for the current count.
- Python compilation check across project source and tests.
- This environment did not perform a live production deployment or verify the user's Gemini credentials.
