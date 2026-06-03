from __future__ import annotations

from dataclasses import dataclass

from ..tools.schema import Tool, ToolCall


@dataclass
class TaskSpec:
    domain: str
    context: str    # "single-turn" | "multi-turn"
    n_tools: int    # size of tool menu
    n_calls: int    # number of gold tool calls required


@dataclass
class GeneratedTask:
    spec: TaskSpec
    question: str
    available_tools: list[Tool]
    gold_calls: list[ToolCall]


@dataclass
class SolverExample:
    task: GeneratedTask
    difficulty: float   # pass@K rate — 0.0 (impossible) .. 1.0 (trivial)
