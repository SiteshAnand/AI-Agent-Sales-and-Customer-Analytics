# Reliability audit and changes

## Changes in this build

- Added deterministic handling for requests asking for high-sales-value customers who have not purchased recently.
- Uses the latest valid transaction date in the uploaded dataset as the snapshot date, not the computer clock.
- Uses a default inactivity threshold of 90 days and supports explicit day/month thresholds in the question.
- Ranks qualifying customers by summed historical sales value and returns a top-20 business-ready Markdown table.
- Validates required customer, date, and sales columns; invalid or missing data returns a clear error instead of invented results.
- Adds regression tests for ranking, explicit thresholds, missing columns, unrelated questions, and the common typo “bught”.
- Changes RFM presentation in the Streamlit response formatter to show a concise segment summary instead of raw serialized JSON.
- Removes a UTF-8 BOM from one test file that prevented a basic AST syntax scan.

## Verification performed in this environment

- Parsed all Python files successfully after removing the BOM.
- `pytest -q`: 97 passed.
- Because Streamlit is not installed in this execution environment and network access prevented installing dependencies, the tests were run with a minimal temporary Streamlit `secrets` stub. This verifies the Python analytics/tests but is **not** an end-to-end Streamlit UI test.
- Exercised the new inactive-high-value-customer route against the included 100,000-row synthetic dataset. It returned a top-20 table, used 2026-10-24 as the snapshot date, and applied the default 90-day threshold.
- Gemini API calls and live Streamlit startup were not verified here. They must be checked in the user's local environment using their own `.env` API key.

## Important operational notes

- The ZIP intentionally excludes `.git`, local uploaded datasets, local environment folders, caches, and backup copies.
- No API key is included. Copy `.env.example` to `.env` locally and add your own Gemini key.
- This is a tested reliability improvement, not a guarantee of defect-free software. Review results on representative datasets before using for business-critical decisions.


## Follow-up update: product sales forecasting

Added `core/forecasting.py`, deterministic natural-language routing in `core/agent.py`, management-friendly rendering in `app.py`, and `tests/test_forecasting.py`. The forecast method is deliberately a transparent baseline and must not be represented as guaranteed or validated predictive performance. Verification in the build environment: 107 tests passed using a temporary Streamlit settings stub because Streamlit is not installed in that environment; Python compile checks passed. Live Streamlit/Gemini integration was not exercised in this environment.
