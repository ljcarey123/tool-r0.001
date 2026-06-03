"""
Integration tests for the Generator → reward → Solver pipeline.
All LLM model calls are mocked so no GPU is required.
"""
import json
import pytest
from unittest.mock import MagicMock, patch

from tool_r0.config import Config
from tool_r0.data.models import GeneratedTask, TaskSpec
from tool_r0.data.pool import TaskPool
from tool_r0.tools.builtins import build_default_registry
from tool_r0.tools.schema import Tool, ToolCall
from tool_r0.agents.generator import GeneratorAgent
from tool_r0.agents.solver import SolverAgent
from tool_r0.rewards.format import generator_format_reward, solver_format_reward
from tool_r0.rewards.validity import validity_reward
from tool_r0.rewards.accuracy import accuracy_reward


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg() -> Config:
    return Config(mc_samples=4, grpo_rollouts=2, task_pool_size=10, curriculum_size=4)


@pytest.fixture
def registry():
    return build_default_registry()


def _make_fake_tokenizer(decode_output: str = ""):
    """
    Fake tokenizer where:
    - tok(prompt) returns a batch_encoding with .to() and input_ids.shape = (1, 0)
    - tok.decode(ids, ...) returns decode_output (set per-test via .decode.return_value)
    Setting shape[1]=0 means output[0:] == output (full string), simplifying mocks.
    """
    tok = MagicMock()
    tok.eos_token_id = 0
    tok.pad_token = "<pad>"
    tok.apply_chat_template.side_effect = lambda msgs, **kw: "PROMPT:" + msgs[-1]["content"]

    input_ids = MagicMock()
    input_ids.shape = (1, 0)  # prompt_len = shape[1] = 0

    batch_encoding = MagicMock()
    batch_encoding.to.return_value = batch_encoding
    batch_encoding.__getitem__ = MagicMock(return_value=input_ids)
    tok.return_value = batch_encoding

    tok.decode.return_value = decode_output
    return tok


def _make_fake_model(outputs: list[str]):
    """
    Model whose generate() returns successive strings from `outputs`.
    Strings survive output[0:] slicing (since prompt_len=0) and are
    passed to tokenizer.decode() which returns decode_output unchanged.
    """
    model = MagicMock()
    model.device = "cpu"
    call_count = [0]

    def fake_generate(**kwargs):
        n = kwargs.get("num_return_sequences", 1)
        results = [outputs[call_count[0] % len(outputs)] for _ in range(n)]
        call_count[0] += 1
        return results

    model.generate.side_effect = fake_generate
    return model


# ---------------------------------------------------------------------------
# Generator agent integration
# ---------------------------------------------------------------------------

VALID_GEN_OUTPUT = """
<think>I'll create a calculator task.</think>
<question>What is 6 multiplied by 7?</question>
<available_tools>[{"name": "calculator", "description": "Evaluates math expressions", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}]</available_tools>
<tool_call_answer>[{"name": "calculator", "parameters": {"expression": "6 * 7"}}]</tool_call_answer>
"""


class TestGeneratorAgentIntegration:
    def test_generate_returns_task(self, cfg, registry):
        tok = _make_fake_tokenizer(decode_output=VALID_GEN_OUTPUT)
        model = _make_fake_model([VALID_GEN_OUTPUT])

        agent = GeneratorAgent(model, tok, registry, cfg)
        spec = TaskSpec("math", "single-turn", 1, 1)
        task = agent.generate(spec)

        assert task is not None
        assert task.gold_calls[0].name == "calculator"

    def test_generate_returns_none_on_bad_output(self, cfg, registry):
        tok = _make_fake_tokenizer(decode_output="garbage output with no tags")
        model = _make_fake_model(["garbage"])

        agent = GeneratorAgent(model, tok, registry, cfg)
        task = agent.generate(TaskSpec("math", "single-turn", 1, 1))
        assert task is None

    def test_build_prompt_contains_spec_info(self, cfg, registry):
        tok = MagicMock()
        tok.apply_chat_template.side_effect = lambda msgs, **kw: json.dumps(msgs)
        agent = GeneratorAgent(MagicMock(), tok, registry, cfg)
        prompt = agent.build_prompt(TaskSpec("math", "single-turn", 2, 1))
        assert "math" in prompt
        assert "single-turn" in prompt


# ---------------------------------------------------------------------------
# Solver agent integration
# ---------------------------------------------------------------------------

VALID_SOL_OUTPUT = """
<think>I'll call the calculator with 6*7.</think>
<tool_call_answer>[{"name": "calculator", "parameters": {"expression": "6 * 7"}}]</tool_call_answer>
"""


class TestSolverAgentIntegration:
    def _make_task(self) -> GeneratedTask:
        tool = Tool(
            name="calculator",
            description="math",
            parameters={"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
        )
        return GeneratedTask(
            spec=TaskSpec("math", "single-turn", 1, 1),
            question="What is 6 * 7?",
            available_tools=[tool],
            gold_calls=[ToolCall(name="calculator", parameters={"expression": "6 * 7"})],
        )

    def test_solve_returns_tool_calls(self, cfg):
        tok = _make_fake_tokenizer(decode_output=VALID_SOL_OUTPUT)
        model = _make_fake_model([VALID_SOL_OUTPUT])

        agent = SolverAgent(model, tok, cfg)
        calls = agent.solve(self._make_task())
        assert calls is not None
        assert calls[0].name == "calculator"

    def test_solve_returns_none_on_bad_output(self, cfg):
        tok = _make_fake_tokenizer(decode_output="no tags here")
        model = _make_fake_model(["no tags here"])

        agent = SolverAgent(model, tok, cfg)
        assert agent.solve(self._make_task()) is None


# ---------------------------------------------------------------------------
# Reward pipeline integration
# ---------------------------------------------------------------------------

class TestRewardPipeline:
    def _make_task(self) -> GeneratedTask:
        tool = Tool(
            name="calculator",
            description="Evaluates math expressions",
            parameters={
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        )
        return GeneratedTask(
            spec=TaskSpec("math", "single-turn", 1, 1),
            question="What is 6 multiplied by 7?",
            available_tools=[tool],
            gold_calls=[ToolCall(name="calculator", parameters={"expression": "6 * 7"})],
        )

    def test_full_generator_reward_pipeline(self):
        r_fmt = generator_format_reward(VALID_GEN_OUTPUT)
        task = self._make_task()
        r_val = validity_reward(task)
        total = r_fmt + r_val
        assert total > 1.5  # format(1.0) + validity(~0.67+) should be well above 1.5

    def test_full_solver_reward_pipeline(self):
        cfg = Config()
        gold = [ToolCall(name="calculator", parameters={"expression": "6 * 7"})]
        pred_correct = [ToolCall(name="calculator", parameters={"expression": "6 * 7"})]
        pred_wrong   = [ToolCall(name="calculator", parameters={"expression": "1 + 1"})]

        r_fmt = solver_format_reward(VALID_SOL_OUTPUT)
        r_correct = accuracy_reward(pred_correct, gold, cfg)
        r_wrong = accuracy_reward(pred_wrong, gold, cfg)

        assert r_fmt > 0
        assert r_correct > r_wrong


# ---------------------------------------------------------------------------
# TaskPool integration with mocked solver
# ---------------------------------------------------------------------------

class TestTaskPoolIntegration:
    def test_full_curriculum_pipeline(self, cfg):
        pool = TaskPool(cfg)
        tool = Tool(
            name="calculator",
            description="math",
            parameters={"type": "object", "properties": {}, "required": []},
        )
        gold = ToolCall(name="calculator", parameters={})

        tasks = [
            GeneratedTask(
                spec=TaskSpec("math", "single-turn", 1, 1),
                question=f"Question number {i}",
                available_tools=[tool],
                gold_calls=[gold],
            )
            for i in range(20)
        ]

        # Mock solver: returns correct answer for even-indexed tasks
        solver = MagicMock()
        def mock_solve_mc(task, k):
            if any(str(i) in task.question and i % 2 == 0 for i in range(20)):
                return [[gold]] * k
            return [[gold]] * (k // 2) + [None] * (k - k // 2)
        solver.solve_mc.side_effect = mock_solve_mc

        deduped = pool.deduplicate(tasks)
        assert len(deduped) == 20  # all unique

        examples = pool.cross_verify(deduped, solver, k=cfg.mc_samples)
        assert len(examples) > 0

        curriculum = pool.build_curriculum(examples, n=cfg.curriculum_size)
        # Should be sorted easy → hard (descending difficulty)
        difficulties = [e.difficulty for e in curriculum]
        assert difficulties == sorted(difficulties, reverse=True)
