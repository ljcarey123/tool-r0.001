from tool_r0.tools.registry import ToolRegistry
from tool_r0.tools.schema import Tool
from tool_r0.tools.builtins import build_default_registry


def _simple_tool(name: str) -> Tool:
    return Tool(
        name=name,
        description=f"Tool {name}",
        parameters={"type": "object", "properties": {}, "required": []},
    )


class TestToolRegistry:
    def test_register_and_contains(self):
        reg = ToolRegistry()
        reg.register(_simple_tool("foo"))
        assert "foo" in reg

    def test_unknown_name_not_in_registry(self):
        reg = ToolRegistry()
        assert "missing" not in reg

    def test_sample_menu_returns_requested_size(self):
        reg = build_default_registry()
        menu = reg.sample_menu(3)
        assert len(menu) == 3

    def test_sample_menu_capped_at_registry_size(self):
        reg = ToolRegistry()
        reg.register(_simple_tool("only_one"))
        assert len(reg.sample_menu(100)) == 1

    def test_schemas_returns_all_tools(self):
        reg = build_default_registry()
        assert len(reg.schemas()) == len(reg)

    def test_len_increments_on_register(self):
        reg = ToolRegistry()
        assert len(reg) == 0
        reg.register(_simple_tool("a"))
        assert len(reg) == 1

    def test_default_registry_has_four_tools(self):
        assert len(build_default_registry()) == 4

    def test_sample_menu_tools_are_from_registry(self):
        reg = build_default_registry()
        menu = reg.sample_menu(2)
        registered_names = {t.name for t in reg.schemas()}
        for tool in menu:
            assert tool.name in registered_names
