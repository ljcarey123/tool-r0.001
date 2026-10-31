from __future__ import annotations

import json
import logging

from ..config import Config
from ..data.models import GeneratedTask
from ..tools.schema import ToolCall
from ..training.adapters import generate_texts
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

# The closing format reminder is not in the paper's prompt. Without it the 0.5B base
# model mostly answers in prose (2/16 parsed on valid eval tasks vs 13/16 with it).
_USER_TEMPLATE = """\
Available tools:
{tools}

User request: {question}

Answer in exactly this format:
<think>which tool fits and which values from the request to use</think>
<tool_call_answer>[{{"name": "<tool name>", "arguments": {{"<parameter>": <value copied from the request>}}}}]</tool_call_answer>
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
        return self.generate_greedy_batch([task], max_new_tokens)[0]

    def generate_greedy_batch(
        self, tasks: list[GeneratedTask], max_new_tokens: int = 512
    ) -> list[str]:
        return self._greedy([self.build_prompt(t) for t in tasks], max_new_tokens)

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
        return parser.parse_score(self._greedy([prompt], max_new_tokens=8)[0])

    def _greedy(self, prompts: list[str], max_new_tokens: int) -> list[str]:
        return generate_texts(
            self.model,
            self.tokenizer,
            prompts,
            self.adapter,
            self.config.generation_batch_size,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=self.config.repetition_penalty,
        )

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
        texts = generate_texts(
            self.model,
            self.tokenizer,
            [self.build_prompt(task)],
            self.adapter,
            batch_size=1,
            max_new_tokens=self.config.solver_max_tokens,
            do_sample=True,
            temperature=self.config.generation_temperature,
            repetition_penalty=self.config.repetition_penalty,
            num_return_sequences=n,
        )
        return [parser.parse_solver_output(text) for text in texts]
