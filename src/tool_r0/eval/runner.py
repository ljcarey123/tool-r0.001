from __future__ import annotations

import logging
from dataclasses import dataclass

from ..agents.solver import SolverAgent
from ..config import Config
from ..data.models import GeneratedTask
from ..rewards.accuracy import accuracy_reward
from ..agents.parser import parse_solver_output

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    n_examples: int
    format_rate: float   # fraction whose output contained valid <think> + tool_call_answer
    accuracy: float      # mean accuracy_reward (Jaccard name+key+value, range −3..3)


def run_eval(
    solver: SolverAgent,
    examples: list[GeneratedTask],
    config: Config,
    n: int = 200,
) -> EvalResult:
    """
    Evaluate the solver greedily (temperature=0, single sample) over `examples[:n]`.

    NOTE on ToolAlpaca: gold calls carry the tool name but empty parameters,
    because ToolAlpaca's Output field is free-text, not structured JSON.
    The accuracy score reflects tool-name selection only. A high score
    confirms the model learned *which* tool to call; argument quality
    requires a benchmark with structured gold calls (e.g. BFCL).
    """
    subset = examples[:n]
    format_hits = 0
    accuracy_total = 0.0

    logger.info("Evaluating %d examples", len(subset))
    for task, text in zip(subset, solver.generate_greedy_batch(subset)):
        predicted = parse_solver_output(text)

        if predicted is not None:
            format_hits += 1
            accuracy_total += accuracy_reward(predicted, task.gold_calls, config)

    n_eval = len(subset)
    return EvalResult(
        n_examples=n_eval,
        format_rate=format_hits / n_eval if n_eval else 0.0,
        accuracy=accuracy_total / n_eval if n_eval else 0.0,
    )

