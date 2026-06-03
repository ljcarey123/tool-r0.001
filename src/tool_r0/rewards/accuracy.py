from __future__ import annotations

from ..config import Config
from ..tools.schema import ToolCall


def accuracy_reward(
    predicted: list[ToolCall],
    gold: list[ToolCall],
    config: Config,
) -> float:
    """
    Fine-grained tool-call accuracy reward (from ToolRL paper, same authors).

    Greedily matches each gold call to the best unused predicted call by name,
    then computes:
      - name Jaccard across all calls
      - parameter-key Jaccard summed over matched pairs
      - parameter-value exact match summed over matched pairs

    Raw scores are normalised to [-3, 3], then a multiplicative penalty is
    applied for extra spurious calls.
    """
    if not gold:
        return 0.0

    pairs = _greedy_match(predicted, gold)

    r_name = _name_score(predicted, gold)
    r_keys = _key_score(pairs)
    r_values = _value_score(pairs)

    s_max = 1 + len(gold) + sum(len(g.parameters) for g in gold)
    normalised = 6.0 * (r_name + r_keys + r_values) / s_max - 3.0

    extra = max(0, len(predicted) - len(gold))
    return normalised * (1.0 / (1.0 + config.extra_call_penalty * extra))


# ---------------------------------------------------------------------------
# Sub-scorers
# ---------------------------------------------------------------------------

def _greedy_match(
    predicted: list[ToolCall], gold: list[ToolCall]
) -> list[tuple[ToolCall, ToolCall]]:
    """Match each gold call to the best unused predicted call sharing its name."""
    used: set[int] = set()
    pairs: list[tuple[ToolCall, ToolCall]] = []
    for g in gold:
        for i, p in enumerate(predicted):
            if i not in used and p.name == g.name:
                pairs.append((p, g))
                used.add(i)
                break
    return pairs


def _name_score(predicted: list[ToolCall], gold: list[ToolCall]) -> float:
    pred_names = {c.name for c in predicted}
    gold_names = {c.name for c in gold}
    union = pred_names | gold_names
    return len(pred_names & gold_names) / len(union) if union else 0.0


def _key_score(pairs: list[tuple[ToolCall, ToolCall]]) -> float:
    total = 0.0
    for pred, gold in pairs:
        pk, gk = set(pred.parameters), set(gold.parameters)
        union = pk | gk
        if union:
            total += len(pk & gk) / len(union)
    return total


def _value_score(pairs: list[tuple[ToolCall, ToolCall]]) -> float:
    total = 0.0
    for pred, gold in pairs:
        for key in gold.parameters:
            if key in pred.parameters and str(pred.parameters[key]) == str(gold.parameters[key]):
                total += 1.0
    return total
