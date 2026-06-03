from __future__ import annotations

from .registry import ToolRegistry
from .schema import Tool

_SCHEMAS: list[Tool] = [
    Tool(
        name="calculator",
        description="Evaluate a mathematical expression and return the numeric result.",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A safe arithmetic expression, e.g. '2 + 3 * 4'",
                }
            },
            "required": ["expression"],
        },
    ),
    Tool(
        name="get_date",
        description="Return today's date in ISO 8601 format (YYYY-MM-DD).",
        parameters={"type": "object", "properties": {}, "required": []},
    ),
    Tool(
        name="celsius_to_fahrenheit",
        description="Convert a temperature from Celsius to Fahrenheit.",
        parameters={
            "type": "object",
            "properties": {
                "celsius": {"type": "number", "description": "Temperature in Celsius"}
            },
            "required": ["celsius"],
        },
    ),
    Tool(
        name="count_words",
        description="Count the number of words in a piece of text.",
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The text to count words in"}
            },
            "required": ["text"],
        },
    ),
]


def build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    for tool in _SCHEMAS:
        registry.register(tool)
    return registry
