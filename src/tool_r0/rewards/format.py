from __future__ import annotations

from ..agents.parser import extract_tag, is_placeholder, load_lenient, parse_calls, parse_tools

# r_fmt = 0.3·I_tag + 0.3·I_parse + 0.4·I_norm — Tool-R0 Appendix (solver format reward).
# The generator reuses the same weights; whether the paper weights the generator's
# indicators equally (range 0..3) is still to be confirmed against the PDF.
_W_TAG = 0.3
_W_PARSE = 0.3
_W_NORM = 0.4


def generator_format_reward(text: str) -> float:
    """
    I_tag   — all four required tags are present and extractable.
    I_parse — <available_tools> parses into a non-empty list of Tool objects.
    I_norm  — <tool_call_answer> normalises into at least one ToolCall.
    """
    score = 0.0

    required = ["think", "question", "available_tools", "tool_call_answer"]
    if all(f"<{t}>" in text and f"</{t}>" in text for t in required):
        score += _W_TAG

    tools_raw = extract_tag(text, "available_tools")
    if tools_raw and parse_tools(tools_raw) is not None:
        score += _W_PARSE

    calls_raw = extract_tag(text, "tool_call_answer")
    if calls_raw and parse_calls(calls_raw) is not None:
        score += _W_NORM

    return score


def solver_format_reward(text: str) -> float:
    """
    Partial credit per the Tool-R0 paper:
      I_tag   — <think> and <tool_call_answer> blocks are both present.
      I_parse — the <tool_call_answer> block parses (JSON / Python literal / fenced).
      I_norm  — it normalises into at least one ToolCall.
    A placeholder answer such as "[...]" scores zero overall.
    """
    calls_raw = extract_tag(text, "tool_call_answer")
    if calls_raw is not None and is_placeholder(calls_raw):
        return 0.0

    score = 0.0
    if "<think>" in text and "</think>" in text and calls_raw is not None:
        score += _W_TAG
    if calls_raw:
        try:
            load_lenient(calls_raw)
            score += _W_PARSE
        except ValueError:
            pass
        if parse_calls(calls_raw) is not None:
            score += _W_NORM
    return score
