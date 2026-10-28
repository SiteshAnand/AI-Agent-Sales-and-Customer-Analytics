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
