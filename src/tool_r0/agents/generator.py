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
You are an expert task generator for tool-calling agents. You invent a realistic \
user request, a menu of tools (as JSON schemas), and the correct tool call(s) that \
answer the request.

Output ONLY four XML-tagged blocks, in the same order and format as the EXAMPLE below: \
<think>, <question>, <available_tools>, <tool_call_answer>. No other text, no markdown fences.

RULES:
1) Invent new tools that fit the requested domain. Each tool is a JSON object with \
"name", "description" and "parameters" (a JSON schema with "type", "properties", "required").
2) <tool_call_answer> is a JSON list of {{"name": ..., "arguments": {{...}}}} objects. \
Every called tool must appear in <available_tools>.
3) Tool arguments must be flat primitives only (no lists, no nested objects).
4) Every argument value in <tool_call_answer> MUST appear verbatim in <question>.

EXAMPLE (domain: library, 2 tools, 1 call):
<think>A user wants to reserve a specific book at a named branch. I need a search tool as a \
distractor and a reserve tool that takes the title and branch.</think>
<question>Please reserve "The Hobbit" for me at the Riverside branch.</question>
<available_tools>[{{"name": "search_catalog", "description": "Search the library catalog by keyword.", \
"parameters": {{"type": "object", "properties": {{"query": {{"type": "string", "description": "Search keywords"}}}}, \
"required": ["query"]}}}}, {{"name": "reserve_book", "description": "Reserve a book at a library branch.", \
"parameters": {{"type": "object", "properties": {{"title": {{"type": "string", "description": "Book title"}}, \
"branch": {{"type": "string", "description": "Branch name"}}}}, "required": ["title", "branch"]}}}}]</available_tools>
<tool_call_answer>[{{"name": "reserve_book", "arguments": {{"title": "The Hobbit", "branch": "Riverside"}}}}]</tool_call_answer>\
"""

_USER_TEMPLATE = """\
Generate one new task with this spec:
- Domain: {domain}
- Context type: {context}
- Number of tools in <available_tools>: exactly {n_tools}
- Number of calls in <tool_call_answer>: exactly {n_calls}\
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
        system_msg = _SYSTEM_PROMPT_TEMPLATE.format()
        user_msg = _USER_TEMPLATE.format(
            domain=spec.domain,
            context=spec.context,
            n_tools=spec.n_tools,
            n_calls=spec.n_calls,
        )
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
