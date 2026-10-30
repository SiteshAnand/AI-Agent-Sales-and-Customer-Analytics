"""Post-execution validation for analytical outputs."""
from __future__ import annotations
import math
from numbers import Real


def validate_result(result: dict, expected_tool: str | None = None) -> dict:
    """Validate response shape and flag non-finite numeric values recursively.

    Returns a report rather than mutating the model/tool result. It does not
    claim domain correctness; that requires tool-specific checks and tests.
    """
    if not isinstance(result, dict):
        raise TypeError("Analytical result must be a dictionary.")
    errors, warnings = [], []
    status = result.get("status")
    if status not in {"success", "error", None}:
        errors.append("Unknown result status.")
    if status == "success" and "result" not in result:
        errors.append("Successful response is missing its result payload.")
    if expected_tool and result.get("tool") not in {expected_tool, "multi_analysis", None}:
        warnings.append("Returned tool name differs from the expected tool.")

    def walk(value, path="$", depth=0):
        if depth > 20:
            warnings.append(f"Maximum validation depth reached at {path}.")
            return
        if isinstance(value, dict):
            for key, child in value.items():
                walk(child, f"{path}.{key}", depth + 1)
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]", depth + 1)
        elif isinstance(value, Real) and not isinstance(value, bool):
            if not math.isfinite(float(value)):
                errors.append(f"Non-finite numeric value at {path}.")

    walk(result)
    return {"valid": not errors, "errors": errors, "warnings": warnings}


def combine_task_results(question: str, task_results: list[dict]) -> dict:
    """Combine task outcomes without hiding failures or fabricating missing results."""
    if not isinstance(task_results, list) or not task_results:
        raise ValueError("At least one task result is required.")
    successful = sum(item.get("status") == "success" for item in task_results)
    failed = len(task_results) - successful
    return {
        "status": "success" if successful else "error",
        "tool": "multi_analysis", "question": question,
        "explanation": f"Processed {len(task_results)} task(s): {successful} succeeded, {failed} failed.",
        "result": {"task_count": len(task_results), "successful_tasks": successful,
                   "failed_tasks": failed, "tasks": task_results,
                   "partial_result": successful > 0 and failed > 0},
    }
