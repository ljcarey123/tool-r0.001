import pytest
from tool_r0.rewards.format import generator_format_reward, solver_format_reward


PERFECT_GENERATOR = """
<think>reasoning</think>
<question>What is 2+2?</question>
<available_tools>[{"name": "calculator", "description": "math", "parameters": {"type": "object", "properties": {}, "required": []}}]</available_tools>
<tool_call_answer>[{"name": "calculator", "parameters": {"expression": "2+2"}}]</tool_call_answer>
"""

PERFECT_SOLVER = """
<think>I'll use the calculator.</think>
<tool_call_answer>[{"name": "calculator", "parameters": {"expression": "2+2"}}]</tool_call_answer>
"""


class TestGeneratorFormatReward:
    """
    Weights are from the Tool-R0 paper: I_tag=0.3, I_parse=0.3, I_norm=0.4.
    Each criterion is binary (0 or 1); possible totals: 0, 0.3, 0.6, 0.7, 1.0.
    """

    def test_perfect_output_scores_one(self):
        assert generator_format_reward(PERFECT_GENERATOR) == pytest.approx(1.0)

    def test_empty_scores_zero(self):
        assert generator_format_reward("") == pytest.approx(0.0)

    def test_missing_tag_loses_0_3(self):
        text = PERFECT_GENERATOR.replace("<think>reasoning</think>", "")
        assert generator_format_reward(text) == pytest.approx(0.7, abs=1e-6)

    def test_tools_not_a_list_loses_0_3(self):
        # Valid JSON but a dict, not a list — fails I_parse
        text = PERFECT_GENERATOR.replace(
            '[{"name": "calculator", "description": "math", "parameters": {"type": "object", "properties": {}, "required": []}}]',
            '{"name": "calculator", "description": "math", "parameters": {}}',
        )
        assert generator_format_reward(text) == pytest.approx(0.7, abs=1e-6)

    def test_tools_missing_name_field_loses_0_3(self):
        # List but items lack "name" — fails Tool(**t) hydration
        text = PERFECT_GENERATOR.replace(
            '[{"name": "calculator", "description": "math", "parameters": {"type": "object", "properties": {}, "required": []}}]',
            '[{"description": "no name here", "parameters": {}}]',
        )
        assert generator_format_reward(text) == pytest.approx(0.7, abs=1e-6)

    def test_tool_without_parameters_parses_but_loses_0_3(self):
        # The parser fills in an empty schema, but omitting it earns no format credit
        text = PERFECT_GENERATOR.replace(
            ', "parameters": {"type": "object", "properties": {}, "required": []}}]', "}]", 1
        )
        assert '"parameters": {"type"' not in text
        assert generator_format_reward(text) == pytest.approx(0.7, abs=1e-6)

    def test_invalid_calls_json_loses_0_4(self):
        text = PERFECT_GENERATOR.replace(
            '[{"name": "calculator", "parameters": {"expression": "2+2"}}]',
            "not json at all",
        )
        assert generator_format_reward(text) == pytest.approx(0.6, abs=1e-6)

    def test_calls_missing_name_loses_0_4(self):
        # Parses as JSON but ToolCall(**c) fails without "name"
        text = PERFECT_GENERATOR.replace(
            '[{"name": "calculator", "parameters": {"expression": "2+2"}}]',
            '[{"parameters": {"expression": "2+2"}}]',
        )
        assert generator_format_reward(text) == pytest.approx(0.6, abs=1e-6)


class TestSolverFormatReward:
    """
    Partial credit per the Tool-R0 paper: I_tag=0.3 (<think> + <tool_call_answer>),
    I_parse=0.3 (answer block parses), I_norm=0.4 (normalises to >= 1 ToolCall).
    """

    def test_perfect_output_scores_one(self):
        assert solver_format_reward(PERFECT_SOLVER) == pytest.approx(1.0)

    def test_empty_scores_zero(self):
        assert solver_format_reward("") == pytest.approx(0.0)

    def test_missing_think_loses_tag_credit(self):
        text = PERFECT_SOLVER.replace("<think>I'll use the calculator.</think>", "")
        assert solver_format_reward(text) == pytest.approx(0.7)

    def test_missing_calls_scores_zero(self):
        text = PERFECT_SOLVER.replace(
            '<tool_call_answer>[{"name": "calculator", "parameters": {"expression": "2+2"}}]</tool_call_answer>',
            "",
        )
        assert solver_format_reward(text) == pytest.approx(0.0)

    def test_malformed_calls_json_gets_tag_credit_only(self):
        text = "<think>ok</think><tool_call_answer>not valid json</tool_call_answer>"
        assert solver_format_reward(text) == pytest.approx(0.3)

    def test_calls_without_name_parse_but_do_not_normalise(self):
        text = '<think>ok</think><tool_call_answer>[{"parameters": {}}]</tool_call_answer>'
        assert solver_format_reward(text) == pytest.approx(0.6)

    def test_unclosed_answer_parses_but_loses_tag_credit(self):
        text = PERFECT_SOLVER.strip().removesuffix("</tool_call_answer>")
        assert solver_format_reward(text) == pytest.approx(0.7)

    def test_placeholder_answer_scores_zero(self):
        text = "<think>ok</think><tool_call_answer>[...]</tool_call_answer>"
        assert solver_format_reward(text) == pytest.approx(0.0)

    @pytest.mark.parametrize("answer", [
        '```json\n[{"name": "get_date", "arguments": {}}]\n```',          # code fence
        "[{'name': 'get_date', 'arguments': {}}]",                       # Python literal
        '[{"type": "function", "function": {"name": "get_date", "arguments": "{}"}}]',  # OpenAI
    ])
    def test_lenient_formats_score_one(self, answer):
        text = f"<think>thinking</think><tool_call_answer>{answer}</tool_call_answer>"
        assert solver_format_reward(text) == pytest.approx(1.0)


class TestQwenSolverFormatReward:
    """Same partial credit as the paper format, for Qwen's native <tool_call> blocks."""

    def test_closed_parseable_block_scores_one(self):
        from tool_r0.rewards.format import qwen_solver_format_reward
        assert qwen_solver_format_reward('<tool_call>\n{"name": "a", "arguments": {}}\n</tool_call>') == pytest.approx(1.0)

    def test_unclosed_block_loses_tag_credit(self):
        from tool_r0.rewards.format import qwen_solver_format_reward
        assert qwen_solver_format_reward('<tool_call>\n{"name": "a", "arguments": {}}') == pytest.approx(0.7)

    def test_malformed_block_gets_tag_credit_only(self):
        from tool_r0.rewards.format import qwen_solver_format_reward
        assert qwen_solver_format_reward("<tool_call>\nnot json\n</tool_call>") == pytest.approx(0.3)

    def test_prose_scores_zero(self):
        from tool_r0.rewards.format import qwen_solver_format_reward
        assert qwen_solver_format_reward("I need more details.") == pytest.approx(0.0)
