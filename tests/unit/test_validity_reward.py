import pytest
from tool_r0.data.models import GeneratedTask, TaskSpec
from tool_r0.tools.schema import Tool, ToolCall
from tool_r0.rewards.validity import validity_reward


def _make_tool(name: str, required: list[str], props: dict | None = None) -> Tool:
    props = props or {p: {"type": "string"} for p in required}
    return Tool(
        name=name,
        description=f"A tool called {name}",
        parameters={"type": "object", "properties": props, "required": required},
    )


def _make_task(
    question: str,
    tools: list[Tool],
    calls: list[ToolCall],
) -> GeneratedTask:
    return GeneratedTask(
        spec=TaskSpec("test", "single-turn", len(tools), len(calls)),
        question=question,
        available_tools=tools,
        gold_calls=calls,
    )


class TestValidityReward:
    def test_perfect_task_scores_one(self):
        tool = _make_tool("calculator", ["expression"])
        call = ToolCall(name="calculator", parameters={"expression": "2 + 2"})
        task = _make_task("What is 2 + 2?", [tool], [call])
        assert validity_reward(task) == pytest.approx(1.0)

    def test_tool_not_in_menu_scores_zero(self):
        tool = _make_tool("calculator", ["expression"])
        call = ToolCall(name="unknown_tool", parameters={"expression": "2 + 2"})
        task = _make_task("What is 2 + 2?", [tool], [call])
        assert validity_reward(task) == pytest.approx(0.0)

    def test_missing_required_param_loses_0_4(self):
        # Weights: name=0.4, required-params=0.4, values=0.2
        tool = _make_tool("calculator", ["expression"])
        call = ToolCall(name="calculator", parameters={})  # missing expression
        task = _make_task("What is 2 + 2?", [tool], [call])
        score = validity_reward(task)
        # name ok (0.4) + req missing (0.0) + no non-trivial values (0.2) = 0.6
        assert score == pytest.approx(0.6, abs=1e-6)

    def test_value_not_in_question_loses_0_2(self):
        # Weights: name=0.4, required-params=0.4, values=0.2
        tool = _make_tool("calculator", ["expression"])
        call = ToolCall(name="calculator", parameters={"expression": "999 + 888"})
        # Question doesn't contain "999" or "888"
        task = _make_task("Do some maths for me.", [tool], [call])
        score = validity_reward(task)
        # name ok (0.4) + req ok (0.4) + value missing (0.0) = 0.8
        assert score == pytest.approx(0.8, abs=1e-6)

    def test_no_gold_calls_returns_zero(self):
        tool = _make_tool("calculator", ["expression"])
        task = _make_task("What?", [tool], [])
        assert validity_reward(task) == pytest.approx(0.0)

    def test_integer_params_skip_word_check(self):
        props = {"task_id": {"type": "integer"}}
        tool = Tool(
            name="mark_done",
            description="Mark done",
            parameters={"type": "object", "properties": props, "required": ["task_id"]},
        )
        call = ToolCall(name="mark_done", parameters={"task_id": 99})
        task = _make_task("Mark task 99 as done.", [tool], [call])
        assert validity_reward(task) == pytest.approx(1.0)

    def test_multiple_calls_averaged(self):
        tool = _make_tool("calculator", ["expression"])
        good_call = ToolCall(name="calculator", parameters={"expression": "3 + 4"})
        bad_call = ToolCall(name="unknown", parameters={})
        task = _make_task("What is 3 + 4?", [tool], [good_call, bad_call])
        score = validity_reward(task)
        # good call = 1.0, bad call = 0.0 → average = 0.5
        assert score == pytest.approx(0.5, abs=1e-6)
