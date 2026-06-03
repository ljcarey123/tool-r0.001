from __future__ import annotations

import math

from ..config import Config
from ..data.models import GeneratedTask
from ..tools.schema import ToolCall


def curriculum_reward(
    task: GeneratedTask,
    solver,  # SolverAgent — kept as Any to avoid circular import
    config: Config,
) -> float:
    """
    Estimate Solver's pass rate via Monte Carlo sampling, then apply a
    band-pass reward: 1.0 inside [p_low, p_high], Gaussian decay outside.

    This rewards tasks that are neither trivially easy nor completely unsolvable.
    """
    samples = solver.solve_mc(task, k=config.mc_samples)
    successes = sum(
        1
        for pred in samples
        if pred is not None and _exact_match(pred, task.gold_calls)
    )
    p_bar = successes / config.mc_samples
    return _bandpass(p_bar, config.p_low, config.p_high, config.sigma)


def _bandpass(p: float, low: float, high: float, sigma: float) -> float:
    """Flat-top band-pass with Gaussian tails."""
    if low <= p <= high:
        return 1.0
    distance = min(abs(p - low), abs(p - high))
    return math.exp(-(distance**2) / (2 * sigma**2))


def _exact_match(predicted: list[ToolCall], gold: list[ToolCall]) -> bool:
    if len(predicted) != len(gold):
        return False
    return {c.canonical() for c in predicted} == {c.canonical() for c in gold}
