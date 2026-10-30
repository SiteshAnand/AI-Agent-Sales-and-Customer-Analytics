"""Query planning utilities for basic, long, and multi-intent business questions.

This module deliberately uses conservative linguistic boundaries: it splits at
explicit separators and at conjunctions followed by a new action verb, rather
than splitting every occurrence of 'and'. Execution remains deterministic in
core.agent; this planner does not execute calculations or invent answers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict

MAX_PLANNED_TASKS = 8
MIN_TASK_LENGTH = 8
_ACTION_START = (
    r"identify|find|forecast|predict|calculate|show|compare|list|analyse|analyze|"
    r"rank|check|segment|train|build|evaluate|determine|summarise|summarize|"
    r"investigate|measure|profile|audit|report|display|explain|estimate|"
    r"calculate|tell\s+me|give\s+me"
)
_SPLIT_PATTERN = re.compile(
    rf"\s*(?:\n+|;|\bthen\s+also\b|\band\s+also\b|"
    rf"\band\s+(?=(?:{_ACTION_START})\b))\s*",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class QueryTask:
    task_id: int
    question: str
    depends_on: tuple[int, ...] = ()

    def to_dict(self) -> dict:
        value = asdict(self)
        value["depends_on"] = list(self.depends_on)
        return value


def split_multifold_question(question: str, max_tasks: int = MAX_PLANNED_TASKS) -> list[str]:
    """Split clear task boundaries while preserving conjunctions within phrases."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Please enter a question.")
    if type(max_tasks) is not int or max_tasks < 1:
        raise ValueError("max_tasks must be a positive integer.")

    cleaned = question.strip()
    pieces = [part.strip(" .\t\r") for part in _SPLIT_PATTERN.split(cleaned)]
    pieces = [part for part in pieces if part]
    if len(pieces) < 2 or any(len(part) < MIN_TASK_LENGTH for part in pieces):
        return [cleaned]
    # If the user supplies more tasks than supported, preserve the remaining
    # text as the final task rather than silently discarding it.
    if len(pieces) > max_tasks:
        pieces = pieces[: max_tasks - 1] + ["; ".join(pieces[max_tasks - 1 :])]
    return pieces


def plan_query(question: str, max_tasks: int = MAX_PLANNED_TASKS) -> dict:
    """Create a transparent task plan; execution order is currently sequential."""
    tasks = split_multifold_question(question, max_tasks=max_tasks)
    return {
        "original_question": question.strip(),
        "task_count": len(tasks),
        "tasks": [QueryTask(i + 1, task).to_dict() for i, task in enumerate(tasks)],
        "execution_strategy": "sequential_with_partial_results" if len(tasks) > 1 else "single_task",
    }
