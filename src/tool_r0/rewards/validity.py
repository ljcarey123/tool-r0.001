from __future__ import annotations

import re

from ..data.models import GeneratedTask


def validity_reward(task: GeneratedTask) -> float:
    """
    Check internal consistency of a generated task. Averages three sub-scores
    per gold call:
      1/3 — tool name exists in the tool menu
      1/3 — all schema-required parameters are present in the call
      1/3 — every non-trivial string argument value appears word-boundary matched in the question

    Returns 0.0 if there are no gold calls.
    """
    if not task.gold_calls:
        return 0.0

    tool_map = {t.name: t for t in task.available_tools}
    question_lower = task.question.lower()
    per_call_scores: list[float] = []

    for call in task.gold_calls:
        score = 0.0

        # 1. Tool name in menu
        if call.name not in tool_map:
            per_call_scores.append(0.0)
            continue
        score += 1 / 3

        tool = tool_map[call.name]
        required_params = tool.parameters.get("required", [])
        properties = tool.parameters.get("properties", {})

        # 2. Required parameters present
        if all(p in call.parameters for p in required_params):
            score += 1 / 3

        # 3. Non-trivial string values appear in the question
        trivial_types = {"boolean", "integer", "number"}
        non_trivial_values = [
            str(v)
            for k, v in call.parameters.items()
            if isinstance(v, str)
            and len(str(v)) > 2
            and properties.get(k, {}).get("type") not in trivial_types
        ]
        if not non_trivial_values:
            score += 1 / 3  # nothing to check — full credit
        elif all(
            re.search(rf"\b{re.escape(v.lower())}\b", question_lower)
            for v in non_trivial_values
        ):
            score += 1 / 3

        per_call_scores.append(score)

    return sum(per_call_scores) / len(per_call_scores)
