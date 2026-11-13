from __future__ import annotations

import ast
import json
import re
import warnings
from typing import Any

from ..tools.schema import Tool, ToolCall


def extract_tag(text: str, tag: str) -> str | None:
    """Return the content of the LAST <tag>...</tag> block, or None if absent.

    Taking the last occurrence matches the reference repo's behaviour — models
    often emit an initial reasoning pass before producing a corrected final block.
    """
    matches = re.findall(rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>", text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    # Small models often stop right after the final block without closing it. Accept an
    # unclosed block that runs to the end of the text, unless another tag opens inside it.
    unclosed = re.search(rf"<{re.escape(tag)}>((?:(?!<[A-Za-z_]+>).)*)\Z", text, re.DOTALL)
    return unclosed.group(1).strip() if unclosed else None


def is_placeholder(text: str) -> bool:
    """True for template filler such as "...", "[...]" or "…" with no real content."""
    return not re.search(r"[A-Za-z0-9]", text)


def load_lenient(raw: str) -> Any:
    """Parse a JSON-ish block: strict JSON, Python-literal syntax, or either inside a
    markdown code fence (the paper's parser accepts all three), falling back to
    repairing a split list. Raises ValueError."""
    raw = raw.strip()
    fence = re.match(r"^```[a-zA-Z]*\s*(.*?)\s*```$", raw, re.DOTALL)
    if fence:
        raw = fence.group(1)
    for candidate in (raw, _merge_split_lists(raw)):
        for load in (json.loads, _literal_eval_quietly):
            try:
                data = load(candidate)
            except Exception:  # literal_eval also raises TypeError, e.g. `{{...}}` is a set of dicts
                continue
            if isinstance(data, tuple):  # `{a}, {b}` or `[{a}], [{b}]`: items with no outer list
                data = [x for part in data for x in (part if isinstance(part, list) else [part])]
            return data
    raise ValueError("unparseable block")


def _literal_eval_quietly(raw: str) -> Any:
    """ast.literal_eval without the SyntaxWarnings that model text such as "\\d" triggers."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.literal_eval(raw)


def _merge_split_lists(raw: str) -> str:
    """Repair a list of objects closed after each item: `[{a}], {b}]` or `[{a}], [{b}]`
    become `[{a}, {b}]`. Only used after strict parsing has failed."""
    return re.sub(r"\}\s*\]\s*,\s*\[?\s*\{", "}, {", raw)


def _unwrap_call(obj: Any) -> Any:
    """Normalise OpenAI-style wrappers to {"name": ..., "arguments": {...}}:
    {"type": "function", "function": {"name": ..., "arguments": "<json string>"}}."""
    if isinstance(obj, dict) and isinstance(obj.get("function"), dict) and "name" not in obj:
        obj = obj["function"]
    if isinstance(obj, dict):
        for key in ("arguments", "parameters"):
            if isinstance(obj.get(key), str):
                try:
                    obj = {**obj, key: load_lenient(obj[key])}
                except ValueError:
                    pass
    return obj


def parse_tools(raw: str) -> list[Tool] | None:
    """Parse an <available_tools> block into a non-empty list of Tools, or None."""
    try:
        data = load_lenient(raw)
        if not isinstance(data, list) or not data:
            return None
        return [Tool(**t) for t in data]
    except Exception:
        return None


def parse_calls(raw: str) -> list[ToolCall] | None:
    """Parse a <tool_call_answer> block into a non-empty list of ToolCalls, or None.
    A single call object (not wrapped in a list) is accepted."""
    if is_placeholder(raw):
        return None
    try:
        data = load_lenient(raw)
        items = data if isinstance(data, list) else [data]
        if not items:
            return None
        calls = [ToolCall(**_unwrap_call(c)) for c in items]
        # Python-literal parsing can yield values a tool call cannot carry, e.g. `...`
        json.dumps([c.parameters for c in calls])
        return calls
    except Exception:
        return None


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
    # Reject placeholder questions such as "..." copied from a format template.
    if not re.search(r"[A-Za-z]{2}", question):
        return None

    tools = parse_tools(tools_raw)
    calls = parse_calls(calls_raw)
    if tools is None or calls is None:
        return None
    return question, tools, calls


def parse_score(text: str) -> int | None:
    """First standalone digit 1–5 in a judge's reply, or None."""
    match = re.search(r"(?<!\d)[1-5](?!\d)", text)
    return int(match.group()) if match else None


def parse_solver_output(text: str) -> list[ToolCall] | None:
    """
    Parse Solver LLM output into a list of ToolCall objects.
    Returns None if the block is absent or malformed.
    """
    calls_raw = extract_tag(text, "tool_call_answer")
    if not calls_raw:
        return None
    return parse_calls(calls_raw)
