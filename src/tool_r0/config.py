from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class Config:
    # Model
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"

    # Self-play loop
    n_iterations: int = 2
    generator_steps: int = 50
    solver_steps: int = 50

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
    grpo_temperature: float = 1.0
    learning_rate: float = 5e-6

    # Accuracy reward
    extra_call_penalty: float = 0.25

    # Task spec sampling distribution
    domains: list[str] = field(
        default_factory=lambda: ["math", "calendar", "temperature", "text"]
    )
    context_types: list[str] = field(
        default_factory=lambda: ["single-turn"]
    )
    n_tools_range: tuple[int, int] = (2, 4)
    n_calls_range: tuple[int, int] = (1, 2)

    def sample_spec_kwargs(self) -> dict:
        return {
            "domain": random.choice(self.domains),
            "context": random.choice(self.context_types),
            "n_tools": random.randint(*self.n_tools_range),
            "n_calls": random.randint(*self.n_calls_range),
        }
