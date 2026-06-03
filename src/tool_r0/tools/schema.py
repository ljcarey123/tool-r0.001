from __future__ import annotations

import json

from pydantic import BaseModel


class Tool(BaseModel):
    name: str
    description: str
    parameters: dict  # JSON Schema format

    def to_dict(self) -> dict:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


class ToolCall(BaseModel):
    name: str
    parameters: dict = {}

    def canonical(self) -> str:
        """Stable string representation for deduplication and matching."""
        return f"{self.name}({json.dumps(self.parameters, sort_keys=True)})"

    def __hash__(self) -> int:
        return hash(self.canonical())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ToolCall):
            return NotImplemented
        return self.canonical() == other.canonical()
