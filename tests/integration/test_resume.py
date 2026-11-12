"""
Resume after a crash mid-iteration: generator training is checkpointed separately, so a
crash while building the task pool must not repeat it. Uses a tiny random Qwen2 model on
CPU with the training phases stubbed out; skipped if the model cannot be downloaded.
"""
import json
from pathlib import Path

import pytest

import tool_r0.loop as loop_mod
from tool_r0.config import Config

TINY_MODEL = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"


@pytest.fixture
def cfg(tmp_path):
    from transformers import AutoConfig

    try:
        AutoConfig.from_pretrained(TINY_MODEL)
    except Exception as exc:  # offline
        pytest.skip(f"tiny model unavailable: {exc}")
    return Config(model_name=TINY_MODEL, n_iterations=2, eval_set_size=0, checkpoint_dir=str(tmp_path))


def test_crash_after_generator_phase_resumes_at_task_pool(cfg, monkeypatch):
    calls: list[str] = []
    crash = {"pool": True}

    def train_generator(self):
        calls.append("generator")

    def build_curriculum(self):
        calls.append("pool")
        if crash["pool"]:
            raise RuntimeError("CUBLAS_STATUS_EXECUTION_FAILED")
        return []  # empty curriculum: solver training is skipped

    monkeypatch.setattr(loop_mod.SelfPlayLoop, "_train_generator", train_generator)
    monkeypatch.setattr(loop_mod.SelfPlayLoop, "_build_curriculum", build_curriculum)

    with pytest.raises(RuntimeError):
        loop_mod.SelfPlayLoop(cfg).run()
    assert calls == ["generator", "pool"]
    state_path = Path(cfg.checkpoint_dir) / "state.json"
    state = json.loads(state_path.read_text())
    assert state["completed_iterations"] == 0 and state["generator_trained"] is True

    crash["pool"] = False
    calls.clear()
    loop_mod.SelfPlayLoop(cfg).run()
    # Iteration 1 resumes at the pool; iteration 2 runs in full
    assert calls == ["pool", "generator", "pool"]
    state = json.loads(state_path.read_text())
    assert state["completed_iterations"] == 2 and state["generator_trained"] is False
