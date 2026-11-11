from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from ..data.models import GeneratedTask, TaskSpec
from ..rewards.accuracy import values_match
from ..tools.schema import Tool, ToolCall

logger = logging.getLogger(__name__)

# Berkeley Function Calling Leaderboard v3
# HuggingFace: https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard
# The repo holds raw JSON-lines files rather than a loadable dataset: questions in
# BFCL_v3_<split>.json and allowed answers in possible_answer/BFCL_v3_<split>.json.
_REPO_ID = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
_DEFAULT_SPLIT = "live_simple"


@dataclass
class BFCLTask(GeneratedTask):
    # BFCL ground truth: one {function_name: {param: [allowed values]}} per call. An empty
    # string among the allowed values marks the parameter as optional.
    ground_truth: list[dict[str, dict[str, list[Any]]]] = field(default_factory=list)


def load_bfcl(split: str = _DEFAULT_SPLIT, n: int | None = None) -> list[GeneratedTask]:
    """Load the first `n` BFCL examples of `split` as BFCLTasks.

    `gold_calls` holds one concrete answer (the first allowed value of each required
    parameter) for the shaped accuracy reward; `bfcl_match` checks the full ground truth.
    """
    from huggingface_hub import hf_hub_download

    logger.info("Loading BFCL split '%s' from HuggingFace …", split)
    try:
        questions = _read_jsonl(hf_hub_download(_REPO_ID, f"BFCL_v3_{split}.json", repo_type="dataset"))
        answers = _read_jsonl(
            hf_hub_download(_REPO_ID, f"possible_answer/BFCL_v3_{split}.json", repo_type="dataset")
        )
    except Exception as exc:
        logger.error(
            "Failed to download BFCL split '%s': %s\n"
            "Check your internet connection or the split name (e.g. live_simple, simple).",
            split, exc,
        )
        return []

    answers_by_id = {a["id"]: a for a in answers}
    rows = questions if n is None else questions[:n]
    examples: list[GeneratedTask] = []
    for row in rows:
        task = _parse_row(row, answers_by_id.get(row.get("id")))
        if task is not None:
            examples.append(task)

    if len(examples) < len(rows):
        logger.warning("Skipped %d / %d BFCL rows that could not be parsed.", len(rows) - len(examples), len(rows))
    logger.info("Loaded %d BFCL examples.", len(examples))
    return examples


def _read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _parse_row(row: dict, answer: dict | None) -> BFCLTask | None:
    try:
        # Questions are a list of turns, each a list of messages; use the last user message.
        messages = [m for turn in row.get("question") or [] for m in turn]
        user_msgs = [m for m in messages if isinstance(m, dict) and m.get("role") == "user"]
        question = user_msgs[-1].get("content", "") if user_msgs else ""

        tools = [
            Tool(name=f["name"], description=f.get("description", ""), **(
                {"parameters": f["parameters"]} if isinstance(f.get("parameters"), dict) else {}
            ))
            for f in row.get("function") or []
            if isinstance(f, dict) and f.get("name")
        ]

        ground_truth = (answer or {}).get("ground_truth") or []
        gold_calls = [_first_allowed_call(gt) for gt in ground_truth]
        if not question or not tools or not gold_calls:
            return None

        spec = TaskSpec(domain="bfcl", context="single-turn", n_tools=len(tools), n_calls=len(gold_calls))
        return BFCLTask(spec, question, tools, gold_calls, ground_truth=ground_truth)
    except Exception as exc:
        logger.debug("Failed to parse BFCL row %s: %s", row.get("id"), exc)
        return None


def _first_allowed_call(gt: dict[str, dict[str, list[Any]]]) -> ToolCall:
    """A concrete gold call: the first allowed value of every required parameter."""
    ((name, allowed),) = gt.items()
    return ToolCall(name=name, parameters={p: v[0] for p, v in allowed.items() if "" not in v and v})


def bfcl_match(predicted: list[ToolCall], ground_truth: list[dict[str, dict[str, list[Any]]]]) -> bool:
    """BFCL-style AST match: same number of calls, each matching a distinct ground-truth call."""
    if len(predicted) != len(ground_truth):
        return False
    unused = list(ground_truth)
    for call in predicted:
        hit = next((gt for gt in unused if _call_matches(call, gt)), None)
        if hit is None:
            return False
        unused.remove(hit)
    return True


def _call_matches(call: ToolCall, gt: dict[str, dict[str, list[Any]]]) -> bool:
    ((name, allowed),) = gt.items()
    if call.name != name or any(p not in allowed for p in call.parameters):
        return False
    for param, values in allowed.items():
        if param in call.parameters:
            if not any(_value_ok(call.parameters[param], v) for v in values if v != ""):
                return False
        elif "" not in values:  # a required parameter is missing
            return False
    return True


def _value_ok(pred: Any, allowed: Any) -> bool:
    """BFCL compares strings case-insensitively, ignoring spaces and some punctuation."""
    if isinstance(pred, str) and isinstance(allowed, str):
        return _standardise(pred) == _standardise(allowed)
    if isinstance(pred, list) and isinstance(allowed, list):
        return len(pred) == len(allowed) and all(_value_ok(p, a) for p, a in zip(pred, allowed))
    return values_match(pred, allowed)


def _standardise(s: str) -> str:
    return re.sub(r"[ ,./\-_*^]", "", s.lower())
