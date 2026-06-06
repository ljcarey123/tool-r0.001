from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from ..agents.generator import GeneratorAgent
from ..data.models import GeneratedTask, TaskSpec
from ..tools.registry import ToolRegistry
from ..tools.schema import Tool, ToolCall

logger = logging.getLogger(__name__)


def create_eval_set(
    generator: GeneratorAgent,
    registry: ToolRegistry,
    n: int,
    seed: int = 42,
) -> list[GeneratedTask]:
    """
    Generate n tasks from the current (untrained) generator as a held-out eval set.
    Uses a fixed seed so the set is reproducible.  May return fewer than n tasks
    if the generator fails to produce valid output for some specs.
    """
    rng = random.Random(seed)
    config = generator.config
    tasks: list[GeneratedTask] = []
    attempts = 0
    max_attempts = n * 10  # bail out rather than looping forever

    while len(tasks) < n and attempts < max_attempts:
        attempts += 1
        n_tools = rng.randint(*config.n_tools_range)
        n_tools = min(n_tools, len(registry))
        n_calls = min(rng.randint(*config.n_calls_range), n_tools)
        spec = TaskSpec(
            domain=rng.choice(config.domains),
            context="single-turn",
            n_tools=n_tools,
            n_calls=n_calls,
        )
        task = generator.generate(spec)
        if task is not None:
            tasks.append(task)

    if len(tasks) < n:
        logger.warning(
            "Eval set: generated %d / %d tasks (generator valid-parse rate: %.0f%%)",
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
    from ..tools.schema import ToolCall as TC

    def _greedy(task: GeneratedTask) -> str:
        prompt = solver.build_prompt(task)
        inputs = solver.tokenizer(prompt, return_tensors="pt").to(solver.model.device)
        outputs = solver.model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=solver.tokenizer.eos_token_id,
        )
        plen = inputs["input_ids"].shape[1]
        return solver.tokenizer.decode(outputs[0][plen:], skip_special_tokens=True)

    def _exact(pred: list[TC], gold: list[TC]) -> bool:
        if len(pred) != len(gold):
            return False
        return {c.canonical() for c in pred} == {c.canonical() for c in gold}

    hits = 0
    for task in tasks:
        text = _greedy(task)
        predicted = parse_solver_output(text)
        if predicted is not None and _exact(predicted, task.gold_calls):
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
    from ..data.models import TaskSpec
    return GeneratedTask(
        spec=TaskSpec(**d["spec"]),
        question=d["question"],
        available_tools=[Tool(**t) for t in d["available_tools"]],
        gold_calls=[ToolCall(**c) for c in d["gold_calls"]],
    )
