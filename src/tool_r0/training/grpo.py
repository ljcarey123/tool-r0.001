from __future__ import annotations

import logging
from typing import Callable

from ..config import Config

logger = logging.getLogger(__name__)


class GRPOTrainer:
    """
    Thin wrapper around trl.GRPOTrainer.

    Accepts a reward_fn with signature:
        reward_fn(prompts: list[str], completions: list[str], **kwargs) -> list[float]

    and a list of prompt strings. Handles Dataset construction and TRL API wiring.
    """

    def __init__(
        self,
        model,
        tokenizer,
        reward_fn: Callable[..., list[float]],
        config: Config,
        extra_columns: dict[str, list] | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.config = config
        self.extra_columns = extra_columns or {}

    def train(self, prompts: list[str], n_steps: int) -> None:
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer as TRLGRPOTrainer  # type: ignore[attr-defined]

        data = {"prompt": prompts, **self.extra_columns}
        dataset = Dataset.from_dict(data)

        # Wrap our reward_fn into TRL's expected signature:
        #   fn(completions: list[str], **kwargs) -> list[float]
        # TRL passes dataset columns (including "prompts") via kwargs.
        reward_fn = self.reward_fn

        def trl_reward_fn(completions: list[str], **kwargs) -> list[float | None]:
            prompts_batch = kwargs.get("prompt", [""] * len(completions))
            extra = {k: kwargs[k] for k in self.extra_columns if k in kwargs}
            return reward_fn(prompts_batch, completions, **extra)  # type: ignore[return-value]

        grpo_config = GRPOConfig(
            output_dir="./grpo_output",
            max_steps=n_steps,
            per_device_train_batch_size=min(4, len(prompts)),
            num_generations=self.config.grpo_rollouts,
            temperature=self.config.grpo_temperature,
            max_completion_length=1024,
            learning_rate=self.config.learning_rate,
            beta=0.0,           # remove KL penalty — per ToolRL finding
            logging_steps=5,
            report_to="none",
            save_strategy="no",
        )

        trainer = TRLGRPOTrainer(
            model=self.model,
            args=grpo_config,
            reward_funcs=trl_reward_fn,  # type: ignore[arg-type]
            train_dataset=dataset,
        )
        trainer.train()
        logger.info("GRPO training complete (%d steps)", n_steps)
