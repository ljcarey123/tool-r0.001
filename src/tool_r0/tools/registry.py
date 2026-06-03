from __future__ import annotations

import random

from .schema import Tool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def sample_menu(self, n: int) -> list[Tool]:
        available = list(self._tools.values())
        return random.sample(available, min(n, len(available)))

    def schemas(self) -> list[Tool]:
        return list(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
