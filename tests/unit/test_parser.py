import pytest
from tool_r0.agents.parser import extract_tag, parse_generator_output, parse_solver_output


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
