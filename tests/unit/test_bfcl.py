import pytest
from tool_r0.eval.bfcl import _parse_row, bfcl_match
from tool_r0.tools.schema import ToolCall

# Real row shape from BFCL_v3_live_simple.json and its possible_answer file
ROW = {
    "id": "live_simple_0-0-0",
    "question": [[{"role": "user", "content": "Retrieve the details for user 7890, special request black."}]],
    "function": [{
        "name": "get_user_info",
        "description": "Retrieve details for a specific user.",
        "parameters": {"type": "dict", "required": ["user_id"], "properties": {
            "user_id": {"type": "integer"}, "special": {"type": "string", "default": "none"}}},
    }],
}
ANSWER = {"id": "live_simple_0-0-0", "ground_truth": [{"get_user_info": {"user_id": [7890], "special": ["black"]}}]}
GT_OPTIONAL = [{"get_weather": {"city": ["Paris", "Paris, France"], "unit": ["", "celsius"]}}]


def call(name, **params):
    return ToolCall(name=name, parameters=params)


class TestParseRow:
    def test_parses_question_tools_and_gold(self):
        task = _parse_row(ROW, ANSWER)
        assert task is not None
        assert task.question.startswith("Retrieve the details")
        assert [t.name for t in task.available_tools] == ["get_user_info"]
        assert task.gold_calls == [call("get_user_info", user_id=7890, special="black")]

    def test_missing_answer_is_skipped(self):
        assert _parse_row(ROW, None) is None

    def test_gold_call_omits_optional_parameters(self):
        task = _parse_row(ROW, {"ground_truth": GT_OPTIONAL})
        assert task is not None
        assert task.gold_calls == [call("get_weather", city="Paris")]


class TestBfclMatch:
    def test_exact(self):
        assert bfcl_match([call("get_user_info", user_id=7890, special="black")], ANSWER["ground_truth"])

    @pytest.mark.parametrize("city", ["Paris", "paris", "Paris, France", "paris france"])
    def test_any_allowed_value_after_string_standardisation(self, city):
        assert bfcl_match([call("get_weather", city=city)], GT_OPTIONAL)

    def test_optional_parameter_may_be_given(self):
        assert bfcl_match([call("get_weather", city="Paris", unit="celsius")], GT_OPTIONAL)

    @pytest.mark.parametrize("predicted", [
        [call("get_weather", unit="celsius")],                       # required parameter missing
        [call("get_weather", city="London")],                         # value not allowed
        [call("get_weather", city="Paris", country="FR")],           # unexpected parameter
        [call("get_forecast", city="Paris")],                         # wrong function
        [call("get_weather", city="Paris"), call("get_weather", city="Paris")],  # extra call
    ])
    def test_mismatches(self, predicted):
        assert not bfcl_match(predicted, GT_OPTIONAL)

    def test_numeric_string_matches_number(self):
        assert bfcl_match([call("get_user_info", user_id="7890", special="black")], ANSWER["ground_truth"])
