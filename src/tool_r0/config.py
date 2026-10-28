from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class Config:
    # Model
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"

    # LoRA — Generator and Solver are separate adapters on one shared base model
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Self-play loop (paper: 3 iterations x 50 GRPO steps)
    n_iterations: int = 3
    generator_steps: int = 50
    solver_steps: int = 50

    # Max new tokens per completion. Generator outputs carry full tool schemas and
    # need more room; solver outputs are a short <think> plus one or two calls.
    generator_max_tokens: int = 768
    solver_max_tokens: int = 384

    # Dataset sizes
    task_pool_size: int = 500
    curriculum_size: int = 200

    # Difficulty estimation (curriculum reward)
    mc_samples: int = 8
    p_low: float = 0.25
    p_high: float = 0.75
    sigma: float = 0.12

    # GRPO
    grpo_rollouts: int = 4
    grpo_temperature: float = 1.0   # temperature used during GRPO rollouts
    generation_temperature: float = 0.7  # temperature for inference outside of GRPO (curriculum, eval)
    repetition_penalty: float = 1.0  # >1 penalises the repeated braces/quotes/keys that JSON needs
    learning_rate: float = 5e-6

    # Accuracy reward
    extra_call_penalty: float = 0.25

    # Evaluation
    eval_set_size: int = 100   # held-out synthetic tasks generated before training starts

    # Output / checkpointing
    checkpoint_dir: str = "./checkpoints"

    # Task spec sampling distribution — mirrors the 32 domains from the paper
    domains: list[str] = field(
        default_factory=lambda: [
            "finance", "travel", "math", "sports", "weather", "system", "database",
            "vehicle_control", "communication", "entertainment", "retail_ecommerce",
            "scheduling", "cloud_infrastructure", "geolocation", "web_search",
            "memory_management", "programming", "iot", "social_media", "logistics",
            "real_estate", "food_ordering", "healthcare", "education", "productivity",
            "insurance", "cybersecurity", "legal", "government", "news", "translation",
            "utilities",
        ]
    )
    context_types: list[str] = field(
        default_factory=lambda: ["single-turn"]
    )
    n_tools_range: tuple[int, int] = (2, 4)
    n_calls_range: tuple[int, int] = (1, 2)

    @classmethod
    def from_preset(cls, preset: str, **overrides) -> "Config":
        """Build a Config from a named preset; `overrides` that are None are ignored."""
        if preset not in PRESETS:
            raise ValueError(f"Unknown preset {preset!r}; choose from {sorted(PRESETS)}")
        values = {**PRESETS[preset], **{k: v for k, v in overrides.items() if v is not None}}
        return cls(**values)

    def sample_spec_kwargs(self) -> dict:
        return {
            "domain": random.choice(self.domains),
            "context": random.choice(self.context_types),
            "n_tools": random.randint(*self.n_tools_range),
            "n_calls": random.randint(*self.n_calls_range),
        }


# "dev" is a fast end-to-end check of the full loop at reduced scale;
# "full" is the paper's schedule at a pool size that fits a single small GPU.
PRESETS: dict[str, dict] = {
    "dev": {
        "n_iterations": 1,
        "generator_steps": 10,
        "solver_steps": 10,
        "task_pool_size": 48,
        "curriculum_size": 24,
        "mc_samples": 4,
        "eval_set_size": 20,
        "generator_max_tokens": 640,
        "solver_max_tokens": 256,
    },
    "full": {},
}
