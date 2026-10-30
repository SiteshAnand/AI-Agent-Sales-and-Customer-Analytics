# Business-friendly response presentation fix

## Changes
- Added `core/presentation.py` as a shared fallback presentation layer for unhandled result shapes.
- Replaced the user-facing raw JSON code-block fallback with readable Markdown summaries and capped tables.
- Suppressed known internal/debug fields and code-like string payloads from the generic fallback.
- Made technical-looking or excessively long error details user-friendly instead of exposing internals in chat.
- Fixed the RFM report's literal `\\n` separator so line breaks render correctly.
- Added four presentation regression tests.

## Verification
- `python -m py_compile app.py core/presentation.py` passed.
- `python -m pytest -q`: 136 passed.

## Scope and limitation
This is a central fallback-rendering hardening, not proof that every tool has a bespoke executive report. Existing specialized renderers remain in place. The app should still be smoke-tested in Streamlit with representative questions, especially forecast, rankings, RFM, multi-step answers, and errors.
