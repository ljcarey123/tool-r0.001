from __future__ import annotations

import argparse
import logging

from .config import Config
from .loop import SelfPlayLoop
from .tools.builtins import build_default_registry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    p = argparse.ArgumentParser(description="Tool-R0: Self-Evolving Tool-Calling Agent")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--iterations", type=int, default=2)
    p.add_argument("--pool-size", type=int, default=500)
    p.add_argument("--curriculum-size", type=int, default=200)
    p.add_argument("--generator-steps", type=int, default=50)
    p.add_argument("--solver-steps", type=int, default=50)
    args = p.parse_args()

    config = Config(
        model_name=args.model,
        n_iterations=args.iterations,
        task_pool_size=args.pool_size,
        curriculum_size=args.curriculum_size,
        generator_steps=args.generator_steps,
        solver_steps=args.solver_steps,
    )

    registry = build_default_registry()
    loop = SelfPlayLoop(config, registry)
    loop.run()
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
