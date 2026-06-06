from __future__ import annotations

import logging

from ..data.models import GeneratedTask, TaskSpec
from ..tools.schema import Tool, ToolCall

logger = logging.getLogger(__name__)

# Berkeley Function Calling Leaderboard v3
# HuggingFace: https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard
# We use the "live_simple" split: single-turn, one function, structured JSON gold calls.
_DATASET_ID = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
_DEFAULT_SPLIT = "live_simple"


def load_bfcl(split: str = _DEFAULT_SPLIT, n: int | None = None) -> list[GeneratedTask]:
    """
    Load BFCL examples and convert to GeneratedTask objects.

    Each BFCL example has:
      - question: list of chat messages (we use the last user message)
      - function: list of OpenAI-style function schemas  →  available_tools
      - possible_answer: list of possible gold calls (we take the first)   →  gold_calls

    The function schema format is compatible with our Tool model:
      {"name": str, "description": str, "parameters": {"type": "object", "properties": {...}}}

    The gold call format uses "arguments" (OpenAI convention), which our ToolCall
    model already accepts via its _accept_arguments_alias validator.
    """
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError:
        logger.error("'datasets' package required. Run: pip install datasets")
        return []

    logger.info("Loading BFCL split '%s' from HuggingFace …", split)
    try:
        ds = load_dataset(_DATASET_ID, split=split, trust_remote_code=False)
    except Exception as exc:
        logger.error(
            "Failed to load BFCL dataset '%s' (split: %s): %s\n"
            "Check your internet connection or the split name. "
            "Available splits include: live_simple, live_multiple, live_parallel.",
            _DATASET_ID, split, exc,
        )
        return []

    if len(ds) == 0:
        logger.warning("BFCL split '%s' is empty.", split)
        return []

    logger.debug("BFCL first-row keys: %s", list(ds[0].keys()))

    examples: list[GeneratedTask] = []
    skipped = 0
    rows = ds if n is None else ds.select(range(min(n, len(ds))))

    for row in rows:
        task = _parse_row(row)
        if task is None:
            skipped += 1
        else:
            examples.append(task)

    if skipped:
        logger.warning("Skipped %d / %d BFCL rows that could not be parsed.", skipped, len(rows))
    logger.info("Loaded %d BFCL examples.", len(examples))
    return examples


def _parse_row(row: dict) -> GeneratedTask | None:
    try:
        # --- Question ---
        messages = row.get("question") or []
        if isinstance(messages, list) and messages:
            # Take the last user message
            user_msgs = [m for m in messages if isinstance(m, dict) and m.get("role") == "user"]
            question = user_msgs[-1].get("content", "") if user_msgs else str(messages[-1])
        else:
            question = str(messages)
        if not question:
            return None

        # --- Available tools ---
        funcs = row.get("function") or []
        if isinstance(funcs, str):
            import json
            funcs = json.loads(funcs)
        tools: list[Tool] = []
        for f in funcs:
            if not isinstance(f, dict):
                continue
            name = f.get("name", "")
            description = f.get("description", "")
            parameters = f.get("parameters", {"type": "object", "properties": {}, "required": []})
            if name:
                tools.append(Tool(name=name, description=description, parameters=parameters))
        if not tools:
            return None

        # --- Gold calls ---
        # possible_answer is a list of valid answers; take the first.
        # Each answer is a list of call dicts: {"name": str, "arguments": {...}}
        possible = row.get("possible_answer") or row.get("ground_truth") or []
        if isinstance(possible, str):
            import json
            possible = json.loads(possible)
        if not possible:
            return None
        first_answer = possible[0] if isinstance(possible[0], list) else possible
        gold_calls: list[ToolCall] = []
        for call in first_answer:
            if isinstance(call, dict) and call.get("name"):
                gold_calls.append(ToolCall(**call))  # accepts "arguments" via validator
        if not gold_calls:
            return None

        spec = TaskSpec(
            domain="bfcl",
            context="single-turn",
            n_tools=len(tools),
            n_calls=len(gold_calls),
        )
        return GeneratedTask(spec=spec, question=question, available_tools=tools, gold_calls=gold_calls)

    except Exception as exc:
        logger.debug("Failed to parse BFCL row: %s", exc)
        return None
