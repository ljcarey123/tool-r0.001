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
    r_curr = r_diff + r_sem (Tool-R0 paper).

    r_diff — estimate the Solver's pass rate p̂ from K Monte Carlo samples and apply a
             band-pass: 1.0 inside [p_low, p_high], Gaussian decay outside, and 0 when
             no sample succeeds (p̂ < 1/K), so unsolvable tasks earn nothing.
    r_sem  — the Solver judges the task's coherence on a 1–5 scale, mapped to [0, 1].
    """
    samples = solver.solve_mc(task, k=config.mc_samples)
    successes = sum(
        1
        for pred in samples
        if pred is not None and exact_match(pred, task.gold_calls)
    )
    p_bar = successes / config.mc_samples
    r_diff = 0.0 if successes == 0 else _bandpass(p_bar, config.p_low, config.p_high, config.sigma)

    r_sem = 0.0
    if config.semantic_reward:
        score = solver.judge_coherence(task)
        r_sem = (score - 1) / 4 if score is not None else 0.0
    return r_diff + r_sem


def _bandpass(p: float, low: float, high: float, sigma: float) -> float:
    """Flat-top band-pass with Gaussian tails."""
    if low <= p <= high:
        return 1.0
    distance = min(abs(p - low), abs(p - high))
    return math.exp(-(distance**2) / (2 * sigma**2))


def exact_match(predicted: list[ToolCall], gold: list[ToolCall]) -> bool:
    if len(predicted) != len(gold):
        return False
    return {c.canonical() for c in predicted} == {c.canonical() for c in gold}
