import pytest
from tool_r0.config import Config
from tool_r0.tools.schema import ToolCall
from tool_r0.rewards.accuracy import (
    accuracy_reward,
    _greedy_match,
    _name_score,
    _key_score,
    _value_score,
)

_CFG = Config()


def _call(name: str, **params) -> ToolCall:
    return ToolCall(name=name, parameters=params)


class TestGreedyMatch:
    def test_exact_name_match(self):
        pred = [_call("calc", expression="1+1")]
        gold = [_call("calc", expression="1+1")]
        pairs = _greedy_match(pred, gold)
        assert len(pairs) == 1

    def test_no_name_match(self):
        pairs = _greedy_match([_call("foo")], [_call("bar")])
        assert pairs == []

    def test_does_not_reuse_prediction(self):
        pred = [_call("calc"), _call("calc")]
        gold = [_call("calc"), _call("calc")]
        pairs = _greedy_match(pred, gold)
        assert len(pairs) == 2


class TestNameScore:
    def test_perfect_match(self):
        assert _name_score([_call("a")], [_call("a")]) == pytest.approx(1.0)

    def test_no_overlap(self):
        assert _name_score([_call("a")], [_call("b")]) == pytest.approx(0.0)

    def test_partial_overlap(self):
        # predicted={a,b}, gold={a,c} → intersection={a}, union={a,b,c}
        assert _name_score([_call("a"), _call("b")], [_call("a"), _call("c")]) == pytest.approx(1 / 3)


class TestAccuracyReward:
    def test_perfect_prediction_above_zero(self):
        gold = [_call("calculator", expression="2+2")]
        pred = [_call("calculator", expression="2+2")]
        r = accuracy_reward(pred, gold, _CFG)
        assert r > 0

    def test_perfect_prediction_maximum(self):
        gold = [_call("calculator", expression="2+2")]
        pred = [_call("calculator", expression="2+2")]
        r = accuracy_reward(pred, gold, _CFG)
        assert r == pytest.approx(3.0)  # max normalised score with no extra calls

    def test_empty_prediction_below_zero(self):
        gold = [_call("calculator", expression="2+2")]
        r = accuracy_reward([], gold, _CFG)
        assert r < 0

    def test_extra_calls_penalised(self):
        gold = [_call("calculator", expression="2+2")]
        pred_exact = [_call("calculator", expression="2+2")]
        pred_extra = [_call("calculator", expression="2+2"), _call("get_date")]
        r_exact = accuracy_reward(pred_exact, gold, _CFG)
        r_extra = accuracy_reward(pred_extra, gold, _CFG)
        assert r_exact > r_extra

    def test_wrong_name_penalised(self):
        gold = [_call("calculator", expression="2+2")]
        pred = [_call("wrong_tool", expression="2+2")]
        r = accuracy_reward(pred, gold, _CFG)
        assert r < accuracy_reward([_call("calculator", expression="2+2")], gold, _CFG)

    def test_wrong_value_penalised(self):
        gold = [_call("calculator", expression="2+2")]
        right = [_call("calculator", expression="2+2")]
        wrong = [_call("calculator", expression="9+9")]
        assert accuracy_reward(right, gold, _CFG) > accuracy_reward(wrong, gold, _CFG)

    def test_empty_gold_returns_zero(self):
        assert accuracy_reward([_call("calculator")], [], _CFG) == pytest.approx(0.0)
