from __future__ import annotations

import json
import math
from typing import Any

from ..config import Config
from ..tools.schema import ToolCall

# Per-pair weights — Tool-R0 paper, solver accuracy reward.
_W_NAME = 0.2
_W_KEY = 0.3
_W_VAL = 0.5

# Numeric strings longer than this are compared as strings, avoiding float artifacts
# on IDs, phone numbers, etc.
_MAX_NUMERIC_DIGITS = 15


def accuracy_reward(
    predicted: list[ToolCall],
    gold: list[ToolCall],
    config: Config,
) -> float:
    """
    Tool-R0 solver accuracy reward, in [0, 1].

    Each gold call is greedily matched to the best-scoring unused prediction, where
        s = 0.2·s_name + 0.3·s_key + 0.5·s_val
    (s_name: name equality; s_key: F1 over argument keys; s_val: fraction of matching
    values over the shared keys). Unmatched gold calls score 0. The mean pair score is
    scaled by 1 / (1 + α·max(0, |predicted| − |gold|)) to penalise spurious calls.
    """
    if not gold:
        return 0.0

    unused = list(range(len(predicted)))
    total = 0.0
    for g in gold:
        if not unused:
            break
        best = max(unused, key=lambda i: pair_score(predicted[i], g))
        total += pair_score(predicted[best], g)
        unused.remove(best)

    mean = total / len(gold)
    extra = max(0, len(predicted) - len(gold))
    return mean / (1.0 + config.extra_call_penalty * extra)


def pair_score(pred: ToolCall, gold: ToolCall) -> float:
    return (
        _W_NAME * float(pred.name == gold.name)
        + _W_KEY * _key_f1(pred.parameters, gold.parameters)
        + _W_VAL * _value_fraction(pred.parameters, gold.parameters)
    )


def _key_f1(pred: dict, gold: dict) -> float:
    pk, gk = set(pred), set(gold)
    if not pk and not gk:
        return 1.0  # zero-argument tool called with no arguments
    shared = len(pk & gk)
    if shared == 0:
        return 0.0
    precision, recall = shared / len(pk), shared / len(gk)
    return 2 * precision * recall / (precision + recall)


def _value_fraction(pred: dict, gold: dict) -> float:
    shared = set(pred) & set(gold)
    if not shared:
        return 1.0 if not pred and not gold else 0.0
    return sum(values_match(pred[k], gold[k]) for k in shared) / len(shared)


def values_match(a: Any, b: Any) -> bool:
    """Equal directly, after numeric coercion, ignoring whitespace, or as canonical JSON."""
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b  # Python treats True == 1; a boolean argument is not a number
    if a == b:
        return True
    na, nb = _as_number(a), _as_number(b)
    if na is not None and nb is not None:
        return math.isclose(na, nb, rel_tol=1e-9, abs_tol=1e-9)
    if isinstance(a, str) and isinstance(b, str):
        return "".join(a.split()) == "".join(b.split())
    try:
        return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    except (TypeError, ValueError):
        return False


def _as_number(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        if sum(ch.isdigit() for ch in s) > _MAX_NUMERIC_DIGITS:
            return None
        try:
            return float(s)
        except ValueError:
            return None
    return None
