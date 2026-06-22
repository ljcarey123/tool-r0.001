from __future__ import annotations

import json
import logging

from ..config import Config
from ..data.models import GeneratedTask, TaskSpec
from ..tools.registry import ToolRegistry
from . import parser

logger = logging.getLogger(__name__)

# System prompt is a template — spec details are filled in per call so the model
# treats them as instructions rather than as data to echo.
_SYSTEM_PROMPT_TEMPLATE = """\
You are an expert task generator for tool-calling agents.

FIRST, in your private scratch-pad, reason step-by-step to design a realistic, \
non-trivial task that cannot be solved without correctly calling one or sometimes multiple tools.

CONTROL SPEC (MUST FOLLOW EXACTLY):
- Domain: {domain}
- Context type: {context}
- Number of available tools: {n_tools} (<available_tools>)
- Number of gold tool calls: {n_calls} (<tool_call_answer>)

RULES:
1) You MUST output exactly {n_tools} tools in <available_tools>.
2) You MUST output exactly {n_calls} tool calls in <tool_call_answer>.
3) Domain must be {domain}. Do not drift into other domains.
4) Tool arguments must be flat primitives only (no lists, no nested objects).
5) Every argument value in <tool_call_answer> MUST appear verbatim in <question>.

Output ONLY these four XML-tagged blocks in order — no other text, no markdown fences:
<think>...</think>
<question>...</question>
<available_tools>[...]</available_tools>
<tool_call_answer>[{{"name": "...", "arguments": {{...}}}}, ...]</tool_call_answer>\
"""

_USER_TEMPLATE = """\
Choose {n_tools} tools from this pool and build your task around them:
{tool_pool}\
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
            temperature=self.config.generation_temperature,
            repetition_penalty=self.config.repetition_penalty,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        logger.debug("Generator raw output:\n%s", text[:1500])
        parsed = parser.parse_generator_output(text)
        if parsed is None:
            tags = ["think", "question", "available_tools", "tool_call_answer"]
            missing = [t for t in tags if f"<{t}>" not in text]
            logger.debug("Parse failed — missing tags: %s", missing or "all tags present, JSON invalid")
            return None
        question, tools, calls = parsed
        return GeneratedTask(spec=spec, question=question, available_tools=tools, gold_calls=calls)

    def build_prompt(self, spec: TaskSpec) -> str:
        system_msg = _SYSTEM_PROMPT_TEMPLATE.format(
            domain=spec.domain,
            context=spec.context,
            n_tools=spec.n_tools,
            n_calls=spec.n_calls,
        )
        pool_json = json.dumps([t.to_dict() for t in self.registry.schemas()], indent=2)
        user_msg = _USER_TEMPLATE.format(n_tools=spec.n_tools, tool_pool=pool_json)
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
