from __future__ import annotations

import json
import re

from ..tools.schema import Tool, ToolCall


def extract_tag(text: str, tag: str) -> str | None:
    """Return the content of the first <tag>...</tag> block, or None if absent."""
    match = re.search(rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>", text, re.DOTALL)
    return match.group(1).strip() if match else None


def parse_generator_output(
    text: str,
) -> tuple[str, list[Tool], list[ToolCall]] | None:
    """
    Parse Generator LLM output into (question, available_tools, gold_calls).
    Returns None if any required block is missing or malformed.
    """
    question = extract_tag(text, "question")
    tools_raw = extract_tag(text, "available_tools")
    calls_raw = extract_tag(text, "tool_call_answer")

    if not question or not tools_raw or not calls_raw:
        return None

    try:
        tools = [Tool(**t) for t in json.loads(tools_raw)]
    except Exception:
        return None

    try:
        raw = json.loads(calls_raw)
        calls = [ToolCall(**c) for c in (raw if isinstance(raw, list) else [raw])]
    except Exception:
        return None

    return question, tools, calls


def parse_solver_output(text: str) -> list[ToolCall] | None:
    """
    Parse Solver LLM output into a list of ToolCall objects.
    Returns None if the block is absent or malformed.
    """
    calls_raw = extract_tag(text, "tool_call_answer")
    if not calls_raw:
        return None
    try:
        raw = json.loads(calls_raw)
        data = raw if isinstance(raw, list) else [raw]
        return [ToolCall(**c) for c in data]
    except Exception:
        return None
