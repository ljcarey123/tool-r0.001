from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from ..config import Config

logger = logging.getLogger(__name__)

# Generator and Solver are two LoRA adapters on one frozen base model. Each role
# keeps its own trainable weights (the paper's ablation shows fully shared weights
# cost ~17pp), but the base weights are only held in memory once.
GENERATOR_ADAPTER = "generator"
SOLVER_ADAPTER = "solver"


@contextmanager
def use_adapter(model, name: str | None) -> Iterator[None]:
    """Temporarily activate a LoRA adapter, restoring the previous one on exit.

    No-op when `name` is None, so agents also work with plain (non-PEFT) models.
    PEFT's set_adapter also flips requires_grad to the active adapter, so restoring
    the previous adapter matters when this is used inside a GRPO reward function.
    """
    if name is None:
        yield
        return
    previous = model.active_adapter
    if previous == name:
        yield
        return
    model.set_adapter(name)
    try:
        yield
    finally:
        model.set_adapter(previous)


@contextmanager
def inference_mode(model) -> Iterator[None]:
    """Put the model in eval mode for generation, restoring train mode afterwards.

    TRL leaves the shared model in train mode with gradient checkpointing on (also while
    reward functions run). In that state transformers drops the KV cache inside each layer
    while generate() still feeds only the newest token, so outputs degenerate into
    repetition. Eval mode disables checkpointing (and LoRA dropout); callers should also
    pass use_cache=True, since TRL sets config.use_cache=False.
    """
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        if was_training:
            model.train()


def generate_texts(
    model, tokenizer, prompts: list[str], adapter: str | None, batch_size: int, **gen_kwargs
) -> list[str]:
    """Decode the completion for each prompt, generating `batch_size` prompts at a time.

    Uses left padding so every completion starts right after its prompt.
    """
    texts: list[str] = []
    previous_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        for i in range(0, len(prompts), batch_size):
            inputs = tokenizer(
                prompts[i : i + batch_size], return_tensors="pt", padding=True
            ).to(model.device)
            with use_adapter(model, adapter), inference_mode(model):
                outputs = model.generate(
                    **inputs, pad_token_id=tokenizer.pad_token_id, use_cache=True, **gen_kwargs
                )
            prompt_len = inputs["input_ids"].shape[1]
            texts += tokenizer.batch_decode(outputs[:, prompt_len:], skip_special_tokens=True)
    finally:
        tokenizer.padding_side = previous_side
    return texts


def _dtype():
    import torch

    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def load_base_model(model_name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=_dtype(),
        device_map="auto" if torch.cuda.is_available() else None,
    )
    return tokenizer, model


def load_shared_model(config: Config, checkpoint_dir: str | Path | None = None):
    """Load the base model with `generator` and `solver` LoRA adapters attached.

    Fresh adapters are created when `checkpoint_dir` is None; otherwise both
    adapters are restored from `checkpoint_dir/generator` and `checkpoint_dir/solver`.
    """
    from peft import LoraConfig, PeftModel, get_peft_model

    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)

    tokenizer, base = load_base_model(config.model_name)

    if checkpoint_dir is None:
        lora = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base, lora, adapter_name=GENERATOR_ADAPTER)
        model.add_adapter(SOLVER_ADAPTER, lora)
    else:
        for name in (GENERATOR_ADAPTER, SOLVER_ADAPTER):
            if not (checkpoint_dir / name / "adapter_config.json").exists():
                raise FileNotFoundError(
                    f"No LoRA adapter at {checkpoint_dir / name}. Checkpoints from before the "
                    "LoRA switch hold full model weights — delete the checkpoint dir to start fresh."
                )
        model = PeftModel.from_pretrained(
            base, checkpoint_dir / GENERATOR_ADAPTER,
            adapter_name=GENERATOR_ADAPTER, is_trainable=True,
        )
        model.load_adapter(checkpoint_dir / SOLVER_ADAPTER, adapter_name=SOLVER_ADAPTER, is_trainable=True)

    model.set_adapter(GENERATOR_ADAPTER)
    return tokenizer, model


def save_adapters(model, checkpoint_dir: Path) -> None:
    """Save both adapters to `checkpoint_dir/generator` and `checkpoint_dir/solver`."""
    for name in (GENERATOR_ADAPTER, SOLVER_ADAPTER):
        model.save_pretrained(checkpoint_dir, selected_adapters=[name])


def load_solver_for_eval(path: str):
    """Load a model for evaluation: a solver LoRA adapter dir, or a plain HF model.

    Adapters are merged into the base weights so greedy decoding runs at full speed.
    Returns (tokenizer, model, base_model_name_or_None).
    """
    adapter_cfg = Path(path) / "adapter_config.json"
    if not adapter_cfg.exists():
        tokenizer, model = load_base_model(path)
        return tokenizer, model, None

    from peft import PeftConfig, PeftModel

    base_name = PeftConfig.from_pretrained(path).base_model_name_or_path
    if base_name is None:
        raise ValueError(f"Adapter at {path} does not record its base model")
    tokenizer, base = load_base_model(base_name)
    model = PeftModel.from_pretrained(base, path).merge_and_unload()
    return tokenizer, model, base_name
