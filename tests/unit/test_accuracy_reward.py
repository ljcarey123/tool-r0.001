import pytest
from tool_r0.config import Config
from tool_r0.tools.schema import ToolCall
from tool_r0.rewards.accuracy import accuracy_reward, pair_score, values_match

_CFG = Config()


def _call(name: str, **params) -> ToolCall:
    return ToolCall(name=name, parameters=params)


class TestPairScore:
    def test_perfect_pair_scores_one(self):
        assert pair_score(_call("calc", x="1"), _call("calc", x="1")) == pytest.approx(1.0)

    def test_wrong_name_loses_name_weight_only(self):
        assert pair_score(_call("other", x="1"), _call("calc", x="1")) == pytest.approx(0.8)

    def test_wrong_value_loses_value_weight(self):
        assert pair_score(_call("calc", x="1"), _call("calc", x="2")) == pytest.approx(0.5)

    def test_key_f1_partial(self):
        # pred keys {a, b}, gold keys {a}: P=0.5, R=1 → F1=2/3; value of a matches
        s = pair_score(_call("calc", a="1", b="2"), _call("calc", a="1"))
        assert s == pytest.approx(0.2 + 0.3 * (2 / 3) + 0.5)

    def test_no_shared_keys(self):
        assert pair_score(_call("calc", a="1"), _call("calc", b="1")) == pytest.approx(0.2)

    def test_zero_argument_tool(self):
        assert pair_score(_call("get_date"), _call("get_date")) == pytest.approx(1.0)


class TestValuesMatch:
    @pytest.mark.parametrize("a,b", [
        (5, "5"), ("5.0", 5), (2.5, "2.50"),       # numeric coercion
        ("2 + 2", "2+2"), ("New  York", "New York"),  # whitespace-insensitive
        ({"b": 1, "a": 2}, {"a": 2, "b": 1}),     # canonical JSON
    ])
    def test_matches(self, a, b):
        assert values_match(a, b)

    @pytest.mark.parametrize("a,b", [
        ("5", "6"), ("paris", "Paris"), (True, 1),
        # Long numeric strings compare as strings, not floats
        ("12345678901234567", "12345678901234568"),
    ])
    def test_mismatches(self, a, b):
        assert not values_match(a, b)


class TestAccuracyReward:
    def test_perfect_prediction_scores_one(self):
        gold = [_call("calculator", expression="2+2")]
        assert accuracy_reward(gold, gold, _CFG) == pytest.approx(1.0)

    def test_empty_prediction_scores_zero(self):
        assert accuracy_reward([], [_call("calculator", expression="2+2")], _CFG) == 0.0

    def test_empty_gold_returns_zero(self):
        assert accuracy_reward([_call("calculator")], [], _CFG) == 0.0

    def test_extra_call_penalty(self):
        gold = [_call("calculator", expression="2+2")]
        pred = [_call("calculator", expression="2+2"), _call("get_date")]
        assert accuracy_reward(pred, gold, _CFG) == pytest.approx(1.0 / 1.25)

    def test_missing_call_scores_zero_for_that_call(self):
        gold = [_call("a", x="1"), _call("b", y="2")]
        assert accuracy_reward([_call("a", x="1")], gold, _CFG) == pytest.approx(0.5)

    def test_greedy_picks_best_prediction_not_first(self):
        gold = [_call("calc", x="1")]
        pred = [_call("calc", x="9"), _call("calc", x="1")]
        # Best match scores 1.0; one extra call → 1 / (1 + 0.25)
        assert accuracy_reward(pred, gold, _CFG) == pytest.approx(0.8)

    def test_prediction_not_reused(self):
        gold = [_call("calc", x="1"), _call("calc", x="1")]
        assert accuracy_reward([_call("calc", x="1")], gold, _CFG) == pytest.approx(0.5)

    def test_order_independent(self):
        gold = [_call("a", x="1"), _call("b", y="2")]
        pred = [_call("b", y="2"), _call("a", x="1")]
        assert accuracy_reward(pred, gold, _CFG) == pytest.approx(1.0)
