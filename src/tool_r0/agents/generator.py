from __future__ import annotations

import json
import logging

from ..config import Config
from ..data.models import GeneratedTask, TaskSpec
from ..tools.registry import ToolRegistry
from . import parser

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a task generator for tool-calling AI training data.
Given a specification, produce a realistic user task together with a matching tool menu and correct tool call(s).

Output exactly these four tagged blocks in order — no other text:

<think>
Reason about what task fits the domain, which tools are needed, and what realistic parameter values to use.
</think>
<question>
A natural-language user request that requires exactly the specified number of tool calls to answer.
</question>
<available_tools>
A JSON array of tool objects. Each object must have "name", "description", and "parameters" fields.
</available_tools>
<tool_call_answer>
A JSON array of tool call objects. Each must have "name" and "parameters" fields matching a tool in the menu.
</tool_call_answer>
"""

_USER_TEMPLATE = """\
Task specification:
- Domain: {domain}
- Context: {context}
- Tools in menu: {n_tools}
- Required tool calls: {n_calls}

Choose {n_tools} tools from this pool and build your task around them:
{tool_pool}
"""


class GeneratorAgent:
    def __init__(
        self,
        model,
        tokenizer,
        registry: ToolRegistry,
        config: Config,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.registry = registry
        self.config = config

    def generate(self, spec: TaskSpec) -> GeneratedTask | None:
        prompt = self.build_prompt(spec)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=self.config.grpo_temperature,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        parsed = parser.parse_generator_output(text)
        if parsed is None:
            logger.debug("Generator output failed to parse")
            return None
        question, tools, calls = parsed
        return GeneratedTask(spec=spec, question=question, available_tools=tools, gold_calls=calls)

    def build_prompt(self, spec: TaskSpec) -> str:
        pool_json = json.dumps([t.to_dict() for t in self.registry.schemas()], indent=2)
        user_msg = _USER_TEMPLATE.format(
            domain=spec.domain,
            context=spec.context,
            n_tools=spec.n_tools,
            n_calls=spec.n_calls,
            tool_pool=pool_json,
        )
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
