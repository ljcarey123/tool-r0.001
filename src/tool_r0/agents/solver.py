from __future__ import annotations

import json
import logging

from ..config import Config
from ..data.models import GeneratedTask
from ..tools.schema import ToolCall
from . import parser

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a tool-calling AI assistant.
Given a user request and a list of available tools, select and call the correct tool(s).

Output exactly these two tagged blocks — no other text:

<think>
Reason about which tool(s) to call, with what parameters, and why.
</think>
<tool_call_answer>
A JSON array of tool call objects. Each must have "name" and "parameters" fields.
</tool_call_answer>
"""

_USER_TEMPLATE = """\
Available tools:
{tools}

User request: {question}
"""


class SolverAgent:
    def __init__(self, model, tokenizer, config: Config) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    def solve(self, task: GeneratedTask) -> list[ToolCall] | None:
        return self._sample(task, n=1)[0]

    def solve_mc(self, task: GeneratedTask, k: int) -> list[list[ToolCall] | None]:
        """k independent stochastic samples — used for difficulty estimation."""
        return self._sample(task, n=k)

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
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=self.config.grpo_temperature,
            num_return_sequences=n,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        results: list[list[ToolCall] | None] = []
        prompt_len = inputs["input_ids"].shape[1]
        for output in outputs:
            text = self.tokenizer.decode(output[prompt_len:], skip_special_tokens=True)
            results.append(parser.parse_solver_output(text))
        return results
