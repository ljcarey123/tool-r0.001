from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, model_validator


class Tool(BaseModel):
    name: str
    description: str
    parameters: dict  # JSON Schema format

    def to_dict(self) -> dict:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


class ToolCall(BaseModel):
    name: str
    parameters: dict[str, Any] = {}

    @model_validator(mode="before")
    @classmethod
    def _accept_arguments_alias(cls, data: Any) -> Any:
        # The paper's repo (and Qwen's pre-training) uses "arguments" as the key.
        # Accept either spelling and normalise to "parameters".
        if isinstance(data, dict) and "arguments" in data and "parameters" not in data:
            data = {**data, "parameters": data["arguments"]}
        return data

    def canonical(self) -> str:
        """Stable string representation for deduplication and matching."""
        return f"{self.name}({json.dumps(self.parameters, sort_keys=True)})"

    def __hash__(self) -> int:
        return hash(self.canonical())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ToolCall):
            return NotImplemented
        return self.canonical() == other.canonical()
