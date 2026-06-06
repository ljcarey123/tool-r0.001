from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path

from .config import Config
from .data.models import GeneratedTask, SolverExample, TaskSpec
from .data.pool import TaskPool
from .agents.generator import GeneratorAgent
from .agents.solver import SolverAgent
from .agents.parser import parse_generator_output
from .rewards.format import generator_format_reward, solver_format_reward
from .rewards.validity import validity_reward
from .rewards.curriculum import curriculum_reward
from .rewards.accuracy import accuracy_reward
from .training.grpo import GRPOTrainer
from .tools.registry import ToolRegistry
from .tools.schema import ToolCall
from .eval.synthetic import create_eval_set, save_eval_set, load_eval_set, eval_pass_at_1

logger = logging.getLogger(__name__)

_STATE_FILE = "state.json"
_GEN_DIR = "generator"
_SOL_DIR = "solver"


class SelfPlayLoop:
    """
    Outer K-iteration self-play loop.

    Each iteration:
      1. Train Generator (GRPO) to produce tasks at the Solver's frontier.
      2. Freeze Generator; sample task pool; deduplicate + cross-verify + order.
      3. Train Solver (GRPO) on the curated curriculum.
      4. Save a checkpoint (both models + state).

    Checkpoints are written to `config.checkpoint_dir` after every completed
    iteration, so training can be resumed after a crash by re-running the same
    command — the loop detects the state file and skips completed iterations.
    """

    def __init__(self, config: Config, registry: ToolRegistry) -> None:
        self.config = config
        self.registry = registry
        self.pool = TaskPool(config)
        # Initialised in run() before any private methods are called.
        self.generator: GeneratorAgent
        self.solver: SolverAgent

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self) -> SolverAgent:
        start_iter, tokenizer, gen_model, sol_model = self._init()
        self.generator = GeneratorAgent(gen_model, tokenizer, self.registry, self.config)
        self.solver = SolverAgent(sol_model, tokenizer, self.config)

        eval_set = self._get_or_create_eval_set()

        if start_iter == 0 and eval_set:
            baseline = eval_pass_at_1(self.solver, eval_set)
            logger.info("Baseline pass@1 (untrained): %.3f over %d tasks", baseline, len(eval_set))

        for i in range(start_iter, self.config.n_iterations):
            logger.info("=== Self-play iteration %d / %d ===", i + 1, self.config.n_iterations)
            self._train_generator()
            curriculum = self._build_curriculum()
            if not curriculum:
                logger.warning("Empty curriculum — skipping Solver training this iteration")
            else:
                self._train_solver(curriculum)
            self._save_checkpoint(i)

            if eval_set:
                score = eval_pass_at_1(self.solver, eval_set)
                logger.info("Synthetic eval — pass@1 after iteration %d: %.3f", i + 1, score)

        return self.solver

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _init(self):
        """Load from checkpoint if one exists, otherwise start fresh."""
        ckpt = Path(self.config.checkpoint_dir)
        state_path = ckpt / _STATE_FILE

        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            completed = state.get("completed_iterations", 0)
            if completed >= self.config.n_iterations:
                logger.info("All %d iterations already completed.", self.config.n_iterations)
                completed = self.config.n_iterations
            else:
                logger.info(
                    "Resuming training from checkpoint (completed %d / %d iterations).",
                    completed, self.config.n_iterations,
                )
            tokenizer, gen_model, sol_model = self._load_models(
                gen_path=str(ckpt / _GEN_DIR),
                sol_path=str(ckpt / _SOL_DIR),
            )
            return completed, tokenizer, gen_model, sol_model

        logger.info("No checkpoint found — starting fresh training.")
        tokenizer, gen_model, sol_model = self._load_models()
        return 0, tokenizer, gen_model, sol_model

    def _get_or_create_eval_set(self) -> list[GeneratedTask]:
        ckpt = Path(self.config.checkpoint_dir)
        eval_path = ckpt / "eval_set.json"
        existing = load_eval_set(eval_path)
        if existing is not None:
            logger.info("Loaded %d tasks from existing eval set.", len(existing))
            return existing
        logger.info("Generating held-out eval set (%d tasks) …", self.config.eval_set_size)
        tasks = create_eval_set(self.generator, self.registry, n=self.config.eval_set_size)
        if tasks:
            save_eval_set(tasks, eval_path)
        return tasks

    def _save_checkpoint(self, iteration: int) -> None:
        ckpt = Path(self.config.checkpoint_dir)
        ckpt.mkdir(parents=True, exist_ok=True)

        logger.info("Saving checkpoint after iteration %d …", iteration + 1)
        self.generator.model.save_pretrained(ckpt / _GEN_DIR)
        self.generator.tokenizer.save_pretrained(ckpt / _GEN_DIR)
        self.solver.model.save_pretrained(ckpt / _SOL_DIR)
        self.solver.tokenizer.save_pretrained(ckpt / _SOL_DIR)

        state = {
            "completed_iterations": iteration + 1,
            "model_name": self.config.model_name,
            "n_iterations": self.config.n_iterations,
            "config": dataclasses.asdict(self.config),
        }
        (ckpt / _STATE_FILE).write_text(json.dumps(state, indent=2), encoding="utf-8")
        logger.info("Checkpoint saved to %s", ckpt)

    # ------------------------------------------------------------------
    # Private: Generator training
    # ------------------------------------------------------------------

    def _train_generator(self) -> None:
        logger.info("Training Generator for %d steps…", self.config.generator_steps)
        n_prompts = self.config.grpo_rollouts * self.config.generator_steps
        specs = [self._sample_spec() for _ in range(n_prompts)]
        prompts = [self.generator.build_prompt(s) for s in specs]

        solver_ref = self.solver  # capture current solver in closure

        def reward_fn(
            prompts_batch: list[str],
            completions: list[str],
            **_,
        ) -> list[float]:
            rewards = []
            for completion in completions:
                r_fmt = generator_format_reward(completion)
                parsed = parse_generator_output(completion)
                if parsed is None:
                    rewards.append(r_fmt)
                    continue
                question, tools, calls = parsed
                task = GeneratedTask(
                    spec=TaskSpec("unknown", "single-turn", len(tools), len(calls)),
                    question=question,
                    available_tools=tools,
                    gold_calls=calls,
                )
                r_val = validity_reward(task)
                r_cur = curriculum_reward(task, solver_ref, self.config)
                rewards.append(r_fmt + r_val + r_cur)
            return rewards

        GRPOTrainer(
            self.generator.model,
            self.generator.tokenizer,
            reward_fn,
            self.config,
        ).train(prompts, n_steps=self.config.generator_steps)

    # ------------------------------------------------------------------
    # Private: Curriculum construction
    # ------------------------------------------------------------------

    def _build_curriculum(self) -> list[SolverExample]:
        logger.info("Sampling %d tasks from frozen Generator…", self.config.task_pool_size)
        raw: list[GeneratedTask] = []
        for _ in range(self.config.task_pool_size):
            spec = self._sample_spec()
            task = self.generator.generate(spec)
            if task is not None:
                raw.append(task)
        logger.info("Valid tasks: %d / %d", len(raw), self.config.task_pool_size)

        deduped = self.pool.deduplicate(raw)
        examples = self.pool.cross_verify(deduped, self.solver)
        return self.pool.build_curriculum(examples, self.config.curriculum_size)

    # ------------------------------------------------------------------
    # Private: Solver training
    # ------------------------------------------------------------------

    def _train_solver(self, curriculum: list[SolverExample]) -> None:
        logger.info("Training Solver on %d examples for %d steps…",
                    len(curriculum), self.config.solver_steps)
        prompts = [self.solver.build_prompt(ex.task) for ex in curriculum]
        # Serialise gold calls so TRL can pass them to the reward function
        gold_json = [
            json.dumps([c.model_dump() for c in ex.task.gold_calls])
            for ex in curriculum
        ]

        config_ref = self.config

        def reward_fn(
            prompts_batch: list[str],
            completions: list[str],
            gold_calls: list[str] | None = None,
            **_,
        ) -> list[float]:
            rewards = []
            for i, completion in enumerate(completions):
                r_fmt = solver_format_reward(completion)
                from .agents.parser import parse_solver_output
                predicted = parse_solver_output(completion)
                if predicted is None or gold_calls is None or i >= len(gold_calls):
                    rewards.append(r_fmt)
                    continue
                gold = [ToolCall(**c) for c in json.loads(gold_calls[i])]
                r_acc = accuracy_reward(predicted, gold, config_ref)
                rewards.append(r_fmt + r_acc)
            return rewards

        GRPOTrainer(
            self.solver.model,
            self.solver.tokenizer,
            reward_fn,
            self.config,
            extra_columns={"gold_calls": gold_json},
        ).train(prompts, n_steps=self.config.solver_steps)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sample_spec(self) -> TaskSpec:
        kwargs = self.config.sample_spec_kwargs()
        kwargs["n_tools"] = min(kwargs["n_tools"], len(self.registry))
        kwargs["n_calls"] = min(kwargs["n_calls"], kwargs["n_tools"])
        return TaskSpec(**kwargs)

    def _load_models(self, gen_path: str | None = None, sol_path: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        gen_src = gen_path or self.config.model_name
        sol_src = sol_path or self.config.model_name
        logger.info("Loading generator from: %s", gen_src)
        logger.info("Loading solver from:    %s", sol_src)

        # Tokenizer is shared (both models use the same vocab)
        tokenizer = AutoTokenizer.from_pretrained(gen_src)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        device_map = "auto" if torch.cuda.is_available() else None

        gen_model = AutoModelForCausalLM.from_pretrained(
            gen_src, torch_dtype=dtype, device_map=device_map
        )
        sol_model = AutoModelForCausalLM.from_pretrained(
            sol_src, torch_dtype=dtype, device_map=device_map
        )
        logger.info("Both models loaded.")
        return tokenizer, gen_model, sol_model
