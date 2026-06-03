import pytest
from unittest.mock import MagicMock

from tool_r0.config import Config
from tool_r0.data.models import GeneratedTask, SolverExample, TaskSpec
from tool_r0.data.pool import TaskPool
from tool_r0.tools.schema import Tool, ToolCall


def _tool(name: str) -> Tool:
    return Tool(
        name=name,
        description=f"Tool {name}",
        parameters={"type": "object", "properties": {}, "required": []},
    )


def _task(question: str, tool_names: list[str], call_names: list[str]) -> GeneratedTask:
    return GeneratedTask(
        spec=TaskSpec("test", "single-turn", len(tool_names), len(call_names)),
        question=question,
        available_tools=[_tool(n) for n in tool_names],
        gold_calls=[ToolCall(name=n, parameters={}) for n in call_names],
    )


_CFG = Config()


class TestDeduplicate:
    def test_removes_exact_duplicate(self):
        pool = TaskPool(_CFG)
        task = _task("Do X", ["calc"], ["calc"])
        result = pool.deduplicate([task, task])
        assert len(result) == 1

    def test_keeps_different_questions(self):
        pool = TaskPool(_CFG)
        t1 = _task("Do X", ["calc"], ["calc"])
        t2 = _task("Do Y", ["calc"], ["calc"])
        result = pool.deduplicate([t1, t2])
        assert len(result) == 2

    def test_empty_input(self):
        assert TaskPool(_CFG).deduplicate([]) == []


class TestBuildCurriculum:
    def _example(self, difficulty: float) -> SolverExample:
        return SolverExample(task=_task("q", ["t"], ["t"]), difficulty=difficulty)

    def test_sorted_easy_first(self):
        pool = TaskPool(_CFG)
        examples = [self._example(d) for d in [0.1, 0.9, 0.5, 0.3, 0.8, 0.6]]
        curriculum = pool.build_curriculum(examples, n=6)
        difficulties = [e.difficulty for e in curriculum]
        assert difficulties == sorted(difficulties, reverse=True)

    def test_respects_n_limit(self):
        pool = TaskPool(_CFG)
        examples = [self._example(d) for d in [0.1, 0.2, 0.5, 0.6, 0.8, 0.9]]
        curriculum = pool.build_curriculum(examples, n=3)
        assert len(curriculum) <= 3

    def test_empty_returns_empty(self):
        assert TaskPool(_CFG).build_curriculum([], n=10) == []


class TestCrossVerify:
    def test_discards_unsolvable_tasks(self):
        pool = TaskPool(_CFG)
        task = _task("unsolvable", ["calc"], ["calc"])

        solver = MagicMock()
        # Solver always returns None (parse failure) → difficulty = 0 → discarded
        solver.solve_mc.return_value = [None] * _CFG.mc_samples

        result = pool.cross_verify([task], solver)
        assert result == []

    def test_keeps_partially_solvable(self):
        pool = TaskPool(_CFG)
        gold_call = ToolCall(name="calc", parameters={})
        task = _task("solvable", ["calc"], ["calc"])

        solver = MagicMock()
        # Half the samples match gold, half don't
        half = _CFG.mc_samples // 2
        solver.solve_mc.return_value = (
            [[gold_call]] * half + [None] * half
        )

        result = pool.cross_verify([task], solver)
        assert len(result) == 1
        assert result[0].difficulty == pytest.approx(half / _CFG.mc_samples)

    def test_difficulty_is_pass_rate(self):
        pool = TaskPool(_CFG)
        gold_call = ToolCall(name="calc", parameters={})
        task = _task("q", ["calc"], ["calc"])

        solver = MagicMock()
        # 3 out of mc_samples succeed
        solver.solve_mc.return_value = (
            [[gold_call]] * 3 + [None] * (_CFG.mc_samples - 3)
        )

        result = pool.cross_verify([task], solver)
        assert result[0].difficulty == pytest.approx(3 / _CFG.mc_samples)
