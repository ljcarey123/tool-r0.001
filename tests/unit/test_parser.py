import pytest
from tool_r0.agents.parser import (
    extract_tag,
    load_lenient,
    parse_generator_output,
    parse_solver_output,
    parse_tools,
)


VALID_GENERATOR_OUTPUT = """
<think>I need to make a calculator task.</think>
<question>What is 12 multiplied by 7?</question>
<available_tools>[{"name": "calculator", "description": "Evaluates math", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}]</available_tools>
<tool_call_answer>[{"name": "calculator", "parameters": {"expression": "12 * 7"}}]</tool_call_answer>
"""

VALID_SOLVER_OUTPUT = """
<think>The user wants 12 * 7 so I call calculator.</think>
<tool_call_answer>[{"name": "calculator", "parameters": {"expression": "12 * 7"}}]</tool_call_answer>
"""


class TestExtractTag:
    def test_extracts_simple_tag(self):
        assert extract_tag("<think>hello</think>", "think") == "hello"

    def test_extracts_multiline(self):
        text = "<think>\nline one\nline two\n</think>"
        assert extract_tag(text, "think") == "line one\nline two"

    def test_returns_none_when_absent(self):
        assert extract_tag("no tags here", "think") is None

    def test_strips_whitespace(self):
        assert extract_tag("<think>  hi  </think>", "think") == "hi"

    def test_returns_last_match(self):
        # Models often re-reason and emit a corrected final block; take the last.
        text = "<think>first</think><think>second</think>"
        assert extract_tag(text, "think") == "second"

    def test_unclosed_final_block_runs_to_end(self):
        # Small models often stop right after the last block without closing it
        assert extract_tag("<think>a</think><tool_call_answer>[1]\n", "tool_call_answer") == "[1]"

    def test_unclosed_block_followed_by_another_tag_is_rejected(self):
        text = "<question>what?<available_tools>[]</available_tools>"
        assert extract_tag(text, "question") is None


class TestLenientRepairs:
    def test_unclosed_answer_parses(self):
        text = VALID_GENERATOR_OUTPUT.strip().removesuffix("</tool_call_answer>")
        assert parse_generator_output(text) is not None

    @pytest.mark.parametrize("split", [
        '[{"name": "a", "description": "x", "parameters": {}}], {"name": "b", "description": "y", "parameters": {}}]',
        '[{"name": "a", "description": "x", "parameters": {}}], [{"name": "b", "description": "y", "parameters": {}}]',
    ])
    def test_split_tool_list_is_merged(self, split):
        tools = parse_tools(split)
        assert tools is not None and [t.name for t in tools] == ["a", "b"]

    def test_set_of_dicts_is_unparseable_not_a_crash(self):
        # literal_eval raises TypeError on `{{...}}` (a set containing a dict)
        with pytest.raises(ValueError):
            load_lenient('{{"name": "a", "arguments": {}}}')

    def test_valid_nested_lists_are_untouched(self):
        assert load_lenient('[{"a": [1]}, {"b": [{"c": 2}]}]') == [{"a": [1]}, {"b": [{"c": 2}]}]


class TestParseGeneratorOutput:
    def test_valid_output(self):
        result = parse_generator_output(VALID_GENERATOR_OUTPUT)
        assert result is not None
        question, tools, calls = result
        assert "12" in question
        assert len(tools) == 1
        assert tools[0].name == "calculator"
        assert len(calls) == 1
        assert calls[0].name == "calculator"
        assert calls[0].parameters["expression"] == "12 * 7"

    def test_missing_question_returns_none(self):
        text = VALID_GENERATOR_OUTPUT.replace("<question>", "").replace("</question>", "")
        assert parse_generator_output(text) is None

    def test_missing_tools_returns_none(self):
        text = VALID_GENERATOR_OUTPUT.replace("<available_tools>", "").replace("</available_tools>", "")
        assert parse_generator_output(text) is None

    def test_malformed_json_returns_none(self):
        text = VALID_GENERATOR_OUTPUT.replace(
            "[{\"name\": \"calculator\"", "[{bad json"
        )
        assert parse_generator_output(text) is None

    def test_single_dict_call_coerced_to_list(self):
        text = VALID_GENERATOR_OUTPUT.replace(
            '[{"name": "calculator", "parameters": {"expression": "12 * 7"}}]',
            '{"name": "calculator", "parameters": {"expression": "12 * 7"}}',
        )
        result = parse_generator_output(text)
        assert result is not None
        _, _, calls = result
        assert len(calls) == 1

    def test_empty_string_returns_none(self):
        assert parse_generator_output("") is None

    def test_placeholder_question_returns_none(self):
        text = VALID_GENERATOR_OUTPUT.replace("What is 12 multiplied by 7?", "...")
        assert parse_generator_output(text) is None


class TestParseSolverOutput:
    def test_valid_output(self):
        result = parse_solver_output(VALID_SOLVER_OUTPUT)
        assert result is not None
        assert len(result) == 1
        assert result[0].name == "calculator"

    def test_missing_tag_returns_none(self):
        assert parse_solver_output("<think>only think</think>") is None

    def test_malformed_json_returns_none(self):
        assert parse_solver_output("<tool_call_answer>not json</tool_call_answer>") is None

    def test_single_dict_coerced(self):
        text = '<tool_call_answer>{"name": "get_date", "parameters": {}}</tool_call_answer>'
        result = parse_solver_output(text)
        assert result is not None
        assert result[0].name == "get_date"

    def test_empty_returns_none(self):
        assert parse_solver_output("") is None


class TestParseScore:
    @pytest.mark.parametrize("text,expected", [
        ("4", 4), ("Score: 5", 5), (" 2.", 2), ("I'd say 3/5", 3), ("none", None), ("10", None),
    ])
    def test_parse_score(self, text, expected):
        from tool_r0.agents.parser import parse_score
        assert parse_score(text) == expected
