from __future__ import annotations

import hashlib
import json
import logging

from ..config import Config
from .models import GeneratedTask, SolverExample
from ..tools.schema import ToolCall

logger = logging.getLogger(__name__)


class TaskPool:
    """
    Stateless pipeline: raw generated tasks → deduplicated → cross-verified
    → curriculum-ordered SolverExamples.
    """

    def __init__(self, config: Config) -> None:
        self.config = config

    def deduplicate(self, tasks: list[GeneratedTask]) -> list[GeneratedTask]:
        """Drop near-duplicates keyed on (question, tool names, gold call names)."""
        seen: set[str] = set()
        unique: list[GeneratedTask] = []
        for task in tasks:
            sig = self._signature(task)
            if sig not in seen:
                seen.add(sig)
                unique.append(task)
        logger.debug("Dedup: %d → %d tasks", len(tasks), len(unique))
        return unique

    def cross_verify(
        self,
        tasks: list[GeneratedTask],
        solver,              # SolverAgent — Any to avoid circular import
        k: int | None = None,
    ) -> list[SolverExample]:
        """
        For each task, sample the Solver k times.
        Keep tasks where the Solver gets it right at least once (difficulty > 0),
        which filters out tasks with broken or unanswerable gold answers.
        Attach difficulty = pass@k rate.
        """
        k = k or self.config.mc_samples
        examples: list[SolverExample] = []
        for task in tasks:
            samples = solver.solve_mc(task, k=k)
            valid = [s for s in samples if s is not None]
            if not valid:
                continue
            successes = sum(1 for s in valid if _exact_match(s, task.gold_calls))
            difficulty = successes / k
            if difficulty > 0.0:  # discard tasks the solver never gets right
                examples.append(SolverExample(task=task, difficulty=difficulty))
        logger.debug("Cross-verify: %d → %d examples", len(tasks), len(examples))
        return examples

    def build_curriculum(
        self, examples: list[SolverExample], n: int
    ) -> list[SolverExample]:
        """
        Bucket into easy / medium / hard, select a balanced n,
        sort easy→hard for curriculum ordering.
        """
        easy   = [e for e in examples if e.difficulty > self.config.p_high]
        medium = [e for e in examples if self.config.p_low <= e.difficulty <= self.config.p_high]
        hard   = [e for e in examples if e.difficulty < self.config.p_low]

        per_bucket = n // 3
        selected = (
            easy[:per_bucket]
            + medium[:per_bucket]
            + hard[: n - 2 * per_bucket]
        )
        # Sort descending difficulty → easiest first for curriculum
        selected.sort(key=lambda e: e.difficulty, reverse=True)
        logger.info(
            "Curriculum: %d easy, %d medium, %d hard (total %d)",
            min(len(easy), per_bucket),
            min(len(medium), per_bucket),
            min(len(hard), n - 2 * per_bucket),
            len(selected),
        )
        return selected

    @staticmethod
    def _signature(task: GeneratedTask) -> str:
        key = json.dumps(
            {
                "q": task.question.lower().strip(),
                "tools": sorted(t.name for t in task.available_tools),
                "calls": sorted(c.name for c in task.gold_calls),
            },
            sort_keys=True,
        )
        return hashlib.md5(key.encode()).hexdigest()  # noqa: S324 — not cryptographic


def _exact_match(predicted: list[ToolCall], gold: list[ToolCall]) -> bool:
    if len(predicted) != len(gold):
        return False
    return {c.canonical() for c in predicted} == {c.canonical() for c in gold}
