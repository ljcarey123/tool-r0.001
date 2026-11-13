from __future__ import annotations

import random
from dataclasses import dataclass, field

from .data.models import TaskSpec


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
    generator_max_tokens: int = 1024
    solver_max_tokens: int = 384

    # Dataset sizes
    task_pool_size: int = 500
    curriculum_size: int = 200

    # Difficulty estimation (curriculum reward)
    mc_samples: int = 8
    p_low: float = 0.25
    p_high: float = 0.75
    sigma: float = 0.12
    semantic_reward: bool = True  # add r_sem: Solver judges task coherence (paper Fig. 11)
    # Solver answer format: "paper" (<think> + <tool_call_answer>[...]) or "qwen" (the model's
    # native chat-template tools prompt and <tool_call> blocks, as BFCL evaluates Qwen).
    solver_format: str = "paper"

    # GRPO
    grpo_rollouts: int = 4
    # Completions per forward/backward pass; the rest is gradient accumulation, so each
    # step still trains on one group of `grpo_rollouts`. On a 4 GB GPU, 2 ran out of memory
    # on long generator completions; raise it on bigger GPUs for speed.
    grpo_micro_batch: int = 1
    grpo_temperature: float = 1.0   # temperature used during GRPO rollouts
    generation_temperature: float = 0.7  # temperature for inference outside of GRPO (curriculum, eval)
    repetition_penalty: float = 1.0  # >1 penalises the repeated braces/quotes/keys that JSON needs
    generation_batch_size: int = 16  # prompts per generate() call when building pools and evaluating
    # Paper uses 1e-6 for full fine-tuning; LoRA typically needs ~10x higher.
    learning_rate: float = 2e-5

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
    # Spec distribution — paper Appendix B. Multi-turn context (10% in the paper) is
    # out of scope here, so every task is single-turn.
    p_two_calls: float = 0.2                    # n=2 gold calls w.p. 0.2, else n=1
    multi_call_menu: tuple[int, int] = (3, 5)   # menu size m when n > 1
    small_menu: tuple[int, int] = (2, 4)        # menu size m when n = 1 ...
    large_menu: tuple[int, int] = (5, 8)        # ... drawn from one of two buckets
    p_large_menu: float = 0.5                   # bucket probability (not given in the paper)

    @classmethod
    def from_preset(cls, preset: str, **overrides) -> "Config":
        """Build a Config from a named preset; `overrides` that are None are ignored."""
        if preset not in PRESETS:
            raise ValueError(f"Unknown preset {preset!r}; choose from {sorted(PRESETS)}")
        values = {**PRESETS[preset], **{k: v for k, v in overrides.items() if v is not None}}
        return cls(**values)

    def sample_spec(self, rng: random.Random | None = None) -> TaskSpec:
        rng = rng or random.Random()
        n_calls = 2 if rng.random() < self.p_two_calls else 1
        if n_calls > 1:
            menu = self.multi_call_menu
        else:
            menu = self.large_menu if rng.random() < self.p_large_menu else self.small_menu
        return TaskSpec(
            domain=rng.choice(self.domains),
            context="single-turn",
            n_tools=rng.randint(*menu),
            n_calls=n_calls,
        )


# "dev" is a fast end-to-end check of the full loop at reduced scale;
# "full" is the paper's schedule at a pool size that fits a single small GPU.
PRESETS: dict[str, dict] = {
    # A few minutes: checks every stage produces a healthy signal, not that learning happens
    "smoke": {
        "n_iterations": 1,
        "generator_steps": 3,
        "solver_steps": 3,
        "task_pool_size": 16,
        "curriculum_size": 8,
        "mc_samples": 4,
        "eval_set_size": 16,
        "generator_max_tokens": 768,
        "solver_max_tokens": 256,
    },
    "dev": {
        "n_iterations": 1,
        "generator_steps": 10,
        "solver_steps": 10,
        "task_pool_size": 48,
        "curriculum_size": 24,
        "mc_samples": 4,
        "eval_set_size": 20,
        "generator_max_tokens": 768,
        "solver_max_tokens": 256,
    },
    "full": {},
}
