"""Central business-friendly presentation helpers for agent results."""

from numbers import Real


def _humanize_key(value):
    """Turn internal field names into readable labels."""
    import re
    label = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(value))
    label = label.replace("_", " ").strip()
    return label[:1].upper() + label[1:] if label else "Result"


def _business_value(value):
    """Format scalar values safely for business users."""
    if value is None:
        return "Not available"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, Real):
        return f"{value:,.2f}" if isinstance(value, float) else f"{value:,}"
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("{", "[", "import ", "def ", "class ", "```")):
            return "Detailed result available; technical payload hidden."
        return stripped
    return str(value)


def format_generic_business_result(explanation, data):
    """Render unhandled results as concise Markdown, never raw JSON/code."""
    lines = [f"**{_business_value(explanation or 'Analysis completed.')}**", ""]

    if isinstance(data, dict):
        hidden = {"debug", "traceback", "stack_trace", "raw_response", "prompt", "plan", "tool_call", "internal", "code"}
        visible = [(k, v) for k, v in data.items() if str(k).casefold() not in hidden]
        if not visible:
            return "**Analysis completed.**\n\nNo user-facing summary was available for this result."
        for key, value in visible:
            label = _humanize_key(key)
            if isinstance(value, list):
                if not value:
                    lines.append(f"- **{label}:** None")
                elif all(isinstance(row, dict) for row in value):
                    lines.extend([f"#### {label}", ""])
                    cols = list(dict.fromkeys(k for row in value[:20] for k in row.keys()))[:6]
                    if cols:
                        lines.append("| " + " | ".join(_humanize_key(c) for c in cols) + " |")
                        lines.append("|" + "|".join(["---"] * len(cols)) + "|")
                        for row in value[:20]:
                            lines.append("| " + " | ".join(_business_value(row.get(c)) for c in cols) + " |")
                        if len(value) > 20:
                            lines.extend(["", f"Showing 20 of {len(value):,} records to keep the answer readable."])
                    else:
                        lines.append("No displayable details were returned.")
                    lines.append("")
                else:
                    preview = ", ".join(_business_value(item) for item in value[:10])
                    lines.append(f"- **{label}:** {preview}" + (f" (and {len(value)-10:,} more)" if len(value) > 10 else ""))
            elif isinstance(value, dict):
                scalar_items = [(k, v) for k, v in value.items() if not isinstance(v, (dict, list))]
                if scalar_items:
                    lines.append(f"#### {label}")
                    for child_key, child_value in scalar_items[:12]:
                        lines.append(f"- **{_humanize_key(child_key)}:** {_business_value(child_value)}")
                else:
                    lines.append(f"- **{label}:** Details are available in the analysis result.")
            else:
                lines.append(f"- **{label}:** {_business_value(value)}")
        return "\n".join(lines).strip()

    if isinstance(data, list):
        if data and all(isinstance(row, dict) for row in data):
            return format_generic_business_result(explanation, {"Results": data})
        return "**Analysis completed.**\n\n" + ", ".join(_business_value(x) for x in data[:10])

    return "**Analysis completed.**\n\n" + _business_value(data)
