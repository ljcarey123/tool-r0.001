"""
Regression test: generation must not depend on the model state TRL leaves behind.

GRPO training leaves the shared model in train mode with gradient checkpointing on and
config.use_cache=False. Generating in that state silently degenerates (the KV cache is
dropped per layer), which once zeroed every curriculum reward and post-training eval.
Uses a tiny random Qwen2 model on CPU; skipped if it cannot be downloaded.
"""
import pytest
import torch

from tool_r0.training.adapters import generate_texts

TINY_MODEL = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"


@pytest.fixture(scope="module")
def tiny():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(TINY_MODEL)
        model = AutoModelForCausalLM.from_pretrained(TINY_MODEL, dtype=torch.float32)
    except Exception as exc:  # offline
        pytest.skip(f"tiny model unavailable: {exc}")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok, model.eval()


PROMPTS = ["hi", "a much longer prompt with many more tokens in it", "medium prompt"]
GREEDY = dict(max_new_tokens=12, do_sample=False)


def test_batched_matches_unbatched(tiny):
    tok, model = tiny
    batched = generate_texts(model, tok, PROMPTS, None, batch_size=8, **GREEDY)
    single = [generate_texts(model, tok, [p], None, batch_size=1, **GREEDY)[0] for p in PROMPTS]
    assert batched == single
    assert tok.padding_side == "right"  # restored after left-padded generation


def test_trl_training_state_does_not_change_outputs(tiny):
    tok, model = tiny
    fresh = generate_texts(model, tok, PROMPTS, None, batch_size=8, **GREEDY)

    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.train()
    try:
        after = generate_texts(model, tok, PROMPTS, None, batch_size=8, **GREEDY)
        assert model.training  # train mode restored for the trainer
    finally:
        model.gradient_checkpointing_disable()
        model.config.use_cache = True
        model.eval()
    assert after == fresh
