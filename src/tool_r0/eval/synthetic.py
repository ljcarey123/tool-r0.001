from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from ..agents.generator import GeneratorAgent
from ..data.models import GeneratedTask, TaskSpec
from ..rewards.validity import validity_reward
from ..tools.schema import Tool, ToolCall

logger = logging.getLogger(__name__)


def create_eval_set(
    generator: GeneratorAgent,
    n: int,
    seed: int = 42,
) -> list[GeneratedTask]:
    """
    Generate n tasks from the current (untrained) generator as a held-out eval set.
    Uses a fixed seed so the set is reproducible.  Only fully valid tasks are kept
    (gold tool in the menu, required arguments present, every value grounded in the
    question): otherwise the gold values are invented and no solver can match them.
    May return fewer than n tasks if the generator rarely produces valid output.
    """
    rng = random.Random(seed)
    config = generator.config
    tasks: list[GeneratedTask] = []
    attempts = 0
    max_attempts = n * 10  # bail out rather than looping forever

    while len(tasks) < n and attempts < max_attempts:
        specs = [config.sample_spec(rng) for _ in range(config.generation_batch_size)]
        attempts += len(specs)
        tasks += [
            t for t in generator.generate_batch(specs)
            if t is not None and validity_reward(t) == 1.0
        ]
    tasks = tasks[:n]

    if len(tasks) < n:
        logger.warning(
            "Eval set: generated %d / %d tasks (fully valid rate: %.0f%%)",
            len(tasks), n, 100 * len(tasks) / max(attempts, 1),
        )
    return tasks


def save_eval_set(tasks: list[GeneratedTask], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [_task_to_dict(t) for t in tasks]
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info("Saved %d eval tasks to %s", len(tasks), path)


def load_eval_set(path: Path) -> list[GeneratedTask] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return [_task_from_dict(d) for d in data]
    except Exception as exc:
        logger.warning("Could not load eval set from %s: %s", path, exc)
        return None


def eval_pass_at_1(solver, tasks: list[GeneratedTask]) -> float:
    """
    Greedy pass@1 accuracy: fraction of tasks where the solver's single greedy
    sample exactly matches the gold calls (same names and arguments).
    """
    from ..agents.parser import parse_solver_output
    from ..rewards.curriculum import exact_match

    hits = 0
    for task, text in zip(tasks, solver.generate_greedy_batch(tasks)):
        predicted = parse_solver_output(text)
        if predicted is not None and exact_match(predicted, task.gold_calls):
            hits += 1
    return hits / len(tasks) if tasks else 0.0


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _task_to_dict(task: GeneratedTask) -> dict:
    return {
        "spec": {
            "domain": task.spec.domain,
            "context": task.spec.context,
            "n_tools": task.spec.n_tools,
            "n_calls": task.spec.n_calls,
        },
        "question": task.question,
        "available_tools": [t.to_dict() for t in task.available_tools],
        "gold_calls": [{"name": c.name, "parameters": c.parameters} for c in task.gold_calls],
    }


def _task_from_dict(d: dict) -> GeneratedTask:
    return GeneratedTask(
        spec=TaskSpec(**d["spec"]),
        question=d["question"],
        available_tools=[Tool(**t) for t in d["available_tools"]],
        gold_calls=[ToolCall(**c) for c in d["gold_calls"]],
    )
