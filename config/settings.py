
import os
from dotenv import load_dotenv

load_dotenv()


def get_setting(name, default=""):
    """Read a setting from environment variables or Streamlit Secrets."""
    value = os.getenv(name)

    if value:
        return value

    try:
        import streamlit as st
        value = st.secrets.get(name, default)
    except Exception:
        value = default

    return value


GEMINI_API_KEY = get_setting("GEMINI_API_KEY")
GEMINI_MODEL = get_setting("GEMINI_MODEL", "gemini-3.5-flash-lite")

MAX_UPLOAD_MB = int(get_setting("MAX_UPLOAD_MB", "50"))
MAX_ROWS = int(get_setting("MAX_ROWS", "1000000"))
MAX_QUERY_LENGTH = int(get_setting("MAX_QUERY_LENGTH", "10000"))
QUERY_TIMEOUT_SECONDS = int(get_setting("QUERY_TIMEOUT_SECONDS", "30"))
