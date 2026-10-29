import math
import pytest
from tool_r0.rewards.curriculum import _bandpass, exact_match
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
        assert exact_match(a, b)

    def test_different_length(self):
        a = [ToolCall(name="calc", parameters={})]
        b = [ToolCall(name="calc", parameters={}), ToolCall(name="get_date", parameters={})]
        assert not exact_match(a, b)

    def test_different_name(self):
        a = [ToolCall(name="foo", parameters={})]
        b = [ToolCall(name="bar", parameters={})]
        assert not exact_match(a, b)

    def test_different_params(self):
        a = [ToolCall(name="calc", parameters={"expression": "1+1"})]
        b = [ToolCall(name="calc", parameters={"expression": "2+2"})]
        assert not exact_match(a, b)


class TestCurriculumReward:
    def _setup(self, n_correct: int, judge_score, k: int = 8):
        from unittest.mock import MagicMock
        from tool_r0.config import Config
        from tool_r0.data.models import GeneratedTask, TaskSpec
        from tool_r0.rewards.curriculum import curriculum_reward

        gold = [ToolCall(name="calc", parameters={"x": "1"})]
        wrong = [ToolCall(name="calc", parameters={"x": "2"})]
        task = GeneratedTask(TaskSpec("math", "single-turn", 1, 1), "calc 1", [], gold)
        solver = MagicMock()
        solver.solve_mc.return_value = [gold] * n_correct + [wrong] * (k - n_correct)
        solver.judge_coherence.return_value = judge_score
        return curriculum_reward(task, solver, Config(mc_samples=k))

    def test_no_successes_gives_zero_difficulty_reward(self):
        # Paper: r_diff = 0 when p̂ < 1/K, not the Gaussian tail value.
        assert self._setup(0, judge_score=1) == pytest.approx(0.0)

    def test_in_band_with_perfect_coherence(self):
        assert self._setup(4, judge_score=5) == pytest.approx(2.0)

    def test_semantic_score_mapped_to_unit_interval(self):
        assert self._setup(4, judge_score=3) == pytest.approx(1.5)

    def test_unparseable_judge_reply_gives_zero_semantic(self):
        assert self._setup(4, judge_score=None) == pytest.approx(1.0)

    def test_always_solved_decays(self):
        r = self._setup(8, judge_score=1)
        assert r == pytest.approx(math.exp(-(0.25**2) / (2 * 0.12**2)))
