from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

from ..agents.solver import SolverAgent
from ..config import Config
from ..data.models import GeneratedTask
from ..rewards.accuracy import accuracy_reward
from ..rewards.curriculum import exact_match
from ..tools.schema import ToolCall

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    n_examples: int
    format_rate: float   # fraction whose output parsed into at least one tool call
    exact_rate: float    # fraction judged fully correct by the benchmark's match rule
    accuracy: float      # mean shaped accuracy reward, 0..1 (partial credit per call)
    parsed: list[bool] = field(default_factory=list)  # per example, for paired comparisons
    exact: list[bool] = field(default_factory=list)


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
    parsed: list[bool] = []
    exact: list[bool] = []
    accuracy_total = 0.0

    logger.info("Evaluating %d examples", len(subset))
    for task, text in zip(subset, solver.generate_greedy_batch(subset)):
        predicted = solver.parse(text)

        parsed.append(predicted is not None)
        exact.append(predicted is not None and match(task, predicted))
        if predicted is not None:
            accuracy_total += accuracy_reward(predicted, task.gold_calls, config)

    n_eval = len(subset)
    return EvalResult(
        n_examples=n_eval,
        format_rate=sum(parsed) / n_eval if n_eval else 0.0,
        exact_rate=sum(exact) / n_eval if n_eval else 0.0,
        accuracy=accuracy_total / n_eval if n_eval else 0.0,
        parsed=parsed,
        exact=exact,
    )
