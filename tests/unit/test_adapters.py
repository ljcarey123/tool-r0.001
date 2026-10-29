import pytest
from tool_r0.config import Config
from tool_r0.training.adapters import use_adapter


class _FakePeftModel:
    def __init__(self, active: str) -> None:
        self.active_adapter = active
        self.switches: list[str] = []

    def set_adapter(self, name: str) -> None:
        self.switches.append(name)
        self.active_adapter = name


class TestUseAdapter:
    def test_switches_and_restores(self):
        model = _FakePeftModel("generator")
        with use_adapter(model, "solver"):
            assert model.active_adapter == "solver"
        assert model.active_adapter == "generator"

    def test_restores_on_exception(self):
        model = _FakePeftModel("generator")
        with pytest.raises(RuntimeError):
            with use_adapter(model, "solver"):
                raise RuntimeError
        assert model.active_adapter == "generator"

    def test_already_active_does_not_switch(self):
        # set_adapter also resets requires_grad, so avoid redundant calls.
        model = _FakePeftModel("solver")
        with use_adapter(model, "solver"):
            pass
        assert model.switches == []

    def test_none_is_noop_for_plain_models(self):
        model = object()  # no adapter API at all
        with use_adapter(model, None):
            pass


class TestPresets:
    def test_dev_preset_overrides_defaults(self):
        cfg = Config.from_preset("dev")
        assert cfg.n_iterations == 1
        assert cfg.mc_samples == 4

    def test_full_preset_matches_defaults(self):
        assert Config.from_preset("full") == Config()

    def test_explicit_override_wins_and_none_ignored(self):
        cfg = Config.from_preset("dev", generator_steps=3, solver_steps=None)
        assert cfg.generator_steps == 3
        assert cfg.solver_steps == Config.from_preset("dev").solver_steps

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError):
            Config.from_preset("huge")


class TestSampleSpec:
    def test_distribution_matches_paper(self):
        import random
        rng = random.Random(0)
        cfg = Config()
        specs = [cfg.sample_spec(rng) for _ in range(4000)]
        two_call = [s for s in specs if s.n_calls == 2]
        assert 0.17 < len(two_call) / len(specs) < 0.23
        assert all(3 <= s.n_tools <= 5 for s in two_call)
        assert all(2 <= s.n_tools <= 8 for s in specs if s.n_calls == 1)
        assert max(s.n_tools for s in specs) == 8  # no cap from a fixed tool pool
        assert {s.context for s in specs} == {"single-turn"}
