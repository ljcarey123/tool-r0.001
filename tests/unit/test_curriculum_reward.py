import math
import pytest
from tool_r0.rewards.curriculum import _bandpass, _exact_match
from tool_r0.tools.schema import ToolCall


class TestBandpass:
    def test_centre_of_band_returns_one(self):
        assert _bandpass(0.5, 0.25, 0.75, 0.12) == pytest.approx(1.0)

    def test_low_edge_returns_one(self):
        assert _bandpass(0.25, 0.25, 0.75, 0.12) == pytest.approx(1.0)

    def test_high_edge_returns_one(self):
        assert _bandpass(0.75, 0.25, 0.75, 0.12) == pytest.approx(1.0)

    def test_zero_below_band_decays(self):
        r = _bandpass(0.0, 0.25, 0.75, 0.12)
        assert 0.0 < r < 1.0
        expected = math.exp(-(0.25**2) / (2 * 0.12**2))
        assert r == pytest.approx(expected)

    def test_one_above_band_decays(self):
        r = _bandpass(1.0, 0.25, 0.75, 0.12)
        assert 0.0 < r < 1.0
        expected = math.exp(-(0.25**2) / (2 * 0.12**2))
        assert r == pytest.approx(expected)

    def test_extreme_values_near_zero(self):
        # Very far outside the band
        assert _bandpass(-1.0, 0.25, 0.75, 0.12) < 0.01
        assert _bandpass(2.0, 0.25, 0.75, 0.12) < 0.01


class TestExactMatch:
    def test_matching_calls(self):
        a = [ToolCall(name="calc", parameters={"expression": "1+1"})]
        b = [ToolCall(name="calc", parameters={"expression": "1+1"})]
        assert _exact_match(a, b)

    def test_different_length(self):
        a = [ToolCall(name="calc", parameters={})]
        b = [ToolCall(name="calc", parameters={}), ToolCall(name="get_date", parameters={})]
        assert not _exact_match(a, b)

    def test_different_name(self):
        a = [ToolCall(name="foo", parameters={})]
        b = [ToolCall(name="bar", parameters={})]
        assert not _exact_match(a, b)

    def test_different_params(self):
        a = [ToolCall(name="calc", parameters={"expression": "1+1"})]
        b = [ToolCall(name="calc", parameters={"expression": "2+2"})]
        assert not _exact_match(a, b)
