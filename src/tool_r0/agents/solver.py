from __future__ import annotations

import json
import logging

from ..config import Config
from ..data.models import GeneratedTask
from ..tools.schema import ToolCall
from ..training.adapters import use_adapter
from . import parser

logger = logging.getLogger(__name__)

# Matches the paper's solver system prompt style: a single paragraph that describes
# the task and shows the exact output format inline as a literal example.
_SYSTEM_PROMPT = (
    "A conversation between a user and a tool-calling assistant. "
    "The user asks a question, and the assistant uses tools to solve it. "
    "The assistant first thinks about the reasoning process in the mind and then provides the answer. "
    "The reasoning process and answer are enclosed within <think></think> and "
    "<tool_call_answer></tool_call_answer> tags, i.e., "
    "<think>\nThis is my reasoning.\n</think>\n"
    '<tool_call_answer>[{"name": "tool_name", "arguments": {"arg1": "value1", "arg2": "value2"}}]'
    "</tool_call_answer>."
)

_USER_TEMPLATE = """\
Available tools:
{tools}

User request: {question}
"""

# Semantic-coherence judge (paper Fig. 11): the Solver rates a generated task 1–5.
_JUDGE_SYSTEM_PROMPT = (
    "You are a strict reviewer of tool-calling tasks. Rate how good the task is on a scale "
    "from 1 to 5:\n"
    "5 = a natural, specific user request, and the gold tool call(s) fully solve it.\n"
    "3 = understandable but vague, or the tool call only partly solves it.\n"
    "1-2 = template-like or instruction-like text rather than a real user request, or the "
    "tool call does not match the request.\n"
    "Reply with a single digit only."
)

_JUDGE_USER_TEMPLATE = """\
Available tools:
{tools}

User request: {question}

Gold tool call(s):
{calls}

Score (1-5):"""


class SolverAgent:
    def __init__(self, model, tokenizer, config: Config, adapter: str | None = None) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.adapter = adapter  # LoRA adapter to activate on a shared model

    def solve(self, task: GeneratedTask) -> list[ToolCall] | None:
        return self._sample(task, n=1)[0]

    def solve_mc(self, task: GeneratedTask, k: int) -> list[list[ToolCall] | None]:
        """k independent stochastic samples — used for difficulty estimation."""
        return self._sample(task, n=k)

    def generate_greedy(self, task: GeneratedTask, max_new_tokens: int = 512) -> str:
        """Single greedy decode, returning raw text — used for evaluation."""
        return self._greedy(self.build_prompt(task), max_new_tokens)

    def judge_coherence(self, task: GeneratedTask) -> int | None:
        """Rate a generated task's coherence 1–5, or None if no score is found."""
        user_msg = _JUDGE_USER_TEMPLATE.format(
            tools=json.dumps([t.to_dict() for t in task.available_tools]),
            question=task.question,
            calls=json.dumps([c.model_dump() for c in task.gold_calls]),
        )
        prompt = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        return parser.parse_score(self._greedy(prompt, max_new_tokens=8))

    def _greedy(self, prompt: str, max_new_tokens: int) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with use_adapter(self.model, self.adapter):
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=self.config.repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        return self.tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)

    def build_prompt(self, task: GeneratedTask) -> str:
        tools_json = json.dumps([t.to_dict() for t in task.available_tools], indent=2)
        user_msg = _USER_TEMPLATE.format(tools=tools_json, question=task.question)
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _sample(self, task: GeneratedTask, n: int) -> list[list[ToolCall] | None]:
        prompt = self.build_prompt(task)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with use_adapter(self.model, self.adapter):
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.solver_max_tokens,
                do_sample=True,
                temperature=self.config.generation_temperature,
                repetition_penalty=self.config.repetition_penalty,
                num_return_sequences=n,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        results: list[list[ToolCall] | None] = []
        prompt_len = inputs["input_ids"].shape[1]
        for output in outputs:
            text = self.tokenizer.decode(output[prompt_len:], skip_special_tokens=True)
            results.append(parser.parse_solver_output(text))
        return results
