from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from ..agents.solver import SolverAgent
from ..config import Config
from ..data.models import GeneratedTask
from ..rewards.accuracy import accuracy_reward
from ..agents.parser import parse_solver_output
from ..rewards.curriculum import exact_match
from ..tools.schema import ToolCall

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    n_examples: int
    format_rate: float   # fraction whose output parsed into at least one tool call
    exact_rate: float    # fraction judged fully correct by the benchmark's match rule
    accuracy: float      # mean shaped accuracy reward, 0..1 (partial credit per call)


def run_eval(
    solver: SolverAgent,
    examples: list[GeneratedTask],
    config: Config,
    n: int = 200,
    match: Callable[[GeneratedTask, list[ToolCall]], bool] | None = None,
) -> EvalResult:
    """
    Evaluate the solver greedily (temperature=0, single sample) over `examples[:n]`.

    `match` decides full correctness; it defaults to exact match against `gold_calls`
    (BFCL passes its own AST matcher, which accepts any of several allowed values).
    """
    match = match or (lambda task, predicted: exact_match(predicted, task.gold_calls))
    subset = examples[:n]
    format_hits = exact_hits = 0
    accuracy_total = 0.0

    logger.info("Evaluating %d examples", len(subset))
    for task, text in zip(subset, solver.generate_greedy_batch(subset)):
        predicted = parse_solver_output(text)

        if predicted is not None:
            format_hits += 1
            exact_hits += match(task, predicted)
            accuracy_total += accuracy_reward(predicted, task.gold_calls, config)

    n_eval = len(subset)
    return EvalResult(
        n_examples=n_eval,
        format_rate=format_hits / n_eval if n_eval else 0.0,
        exact_rate=exact_hits / n_eval if n_eval else 0.0,
        accuracy=accuracy_total / n_eval if n_eval else 0.0,
    )
