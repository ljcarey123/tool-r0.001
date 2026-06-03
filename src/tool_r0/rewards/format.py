from __future__ import annotations

import json

from ..agents.parser import extract_tag
from ..tools.schema import Tool, ToolCall

# Generator format reward weights — from Tool-R0 paper Table 1 / Section 3.2:
#   r_fmt = 0.3·I_tag + 0.3·I_parse + 0.4·I_norm
# Each I_* is a binary 0/1 indicator; possible total scores: 0, 0.3, 0.6, 0.7, 1.0.
_W_TAG = 0.3
_W_PARSE = 0.3
_W_NORM = 0.4


def generator_format_reward(text: str) -> float:
    """
    Three binary criteria, weighted per the paper (0.3 / 0.3 / 0.4):

    I_tag  — all four required tags are present and extractable.
    I_parse — <available_tools> is a valid JSON list of well-formed Tool objects.
    I_norm  — <tool_call_answer> parses into at least one valid ToolCall object.
    """
    score = 0.0

    # I_tag
    required = ["think", "question", "available_tools", "tool_call_answer"]
    if all(f"<{t}>" in text and f"</{t}>" in text for t in required):
        score += _W_TAG

    # I_parse — tools block hydrates into Tool objects
    tools_raw = extract_tag(text, "available_tools")
    if tools_raw:
        try:
            data = json.loads(tools_raw)
            if isinstance(data, list) and data and all(
                isinstance(t, dict) and "name" in t for t in data
            ):
                [Tool(**t) for t in data]  # raises if schema is wrong
                score += _W_PARSE
        except Exception:
            pass

    # I_norm — answer block hydrates into ToolCall objects
    calls_raw = extract_tag(text, "tool_call_answer")
    if calls_raw:
        try:
            raw = json.loads(calls_raw)
            items = raw if isinstance(raw, list) else [raw]
            if items and all(isinstance(c, dict) and "name" in c for c in items):
                [ToolCall(**c) for c in items]  # raises if schema is wrong
                score += _W_NORM
        except Exception:
            pass

    return score


def solver_format_reward(text: str) -> float:
    """
    Binary reward (0 or 1) matching the ToolRL paper (same authors):
    returns 1.0 only if both the <think> tag and a parseable <tool_call_answer>
    containing at least one valid ToolCall are present.

    Tool-R0 describes this as "partial credit on parseability for special tokens"
    but gives no explicit weights; the ToolRL paper uses a strict binary check.
    """
    has_think = "<think>" in text and "</think>" in text

    calls_raw = extract_tag(text, "tool_call_answer")
    has_calls = False
    if calls_raw:
        try:
            raw = json.loads(calls_raw)
            items = raw if isinstance(raw, list) else [raw]
            if items and all(isinstance(c, dict) and "name" in c for c in items):
                [ToolCall(**c) for c in items]
                has_calls = True
        except Exception:
            pass

    return 1.0 if (has_think and has_calls) else 0.0
