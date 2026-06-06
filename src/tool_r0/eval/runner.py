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

    for i, task in enumerate(subset):
        if i % 20 == 0:
            logger.info("Eval progress: %d / %d", i, len(subset))

        text = _greedy_generate(solver, task)
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


def _greedy_generate(solver: SolverAgent, task: GeneratedTask) -> str:
    """Single greedy (temperature=0) forward pass through the solver."""
    prompt = solver.build_prompt(task)
    inputs = solver.tokenizer(prompt, return_tensors="pt").to(solver.model.device)
    outputs = solver.model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False,
        pad_token_id=solver.tokenizer.eos_token_id,
    )
    prompt_len = inputs["input_ids"].shape[1]
    return solver.tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
