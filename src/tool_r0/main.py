from __future__ import annotations

import argparse
import logging
import sys

# TRL reads chat-template files without an explicit encoding; on Windows the
# default is cp1252, which cannot decode the DeepSeek V3 jinja template.
# Patch pathlib before TRL's lazy import fires so all read_text() calls
# default to UTF-8 on Windows.
if sys.platform == "win32":
    import pathlib as _pathlib
    _orig_read_text = _pathlib.Path.read_text
    def _utf8_read_text(self, encoding=None, **kw):  # type: ignore[override]
        return _orig_read_text(self, encoding=encoding or "utf-8", **kw)
    _pathlib.Path.read_text = _utf8_read_text  # type: ignore[method-assign]

from .config import Config


def _configure_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def cmd_train(args: argparse.Namespace) -> None:
    from .loop import SelfPlayLoop
    from .tools.builtins import build_default_registry

    config = Config(
        model_name=args.model,
        n_iterations=args.iterations,
        task_pool_size=args.pool_size,
        curriculum_size=args.curriculum_size,
        generator_steps=args.generator_steps,
        solver_steps=args.solver_steps,
        checkpoint_dir=args.checkpoint_dir,
    )
    registry = build_default_registry()
    loop = SelfPlayLoop(config, registry)
    loop.run()
    logging.getLogger(__name__).info(
        "Training complete. Checkpoints in %s", args.checkpoint_dir
    )


def cmd_eval(args: argparse.Namespace) -> None:
    import json
    from pathlib import Path

    from .eval.runner import run_eval, EvalResult
    from .agents.solver import SolverAgent

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger = logging.getLogger(__name__)

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device_map = "auto" if torch.cuda.is_available() else None

    # Resolve base model name from checkpoint state if not supplied
    checkpoint = args.checkpoint
    state_path = Path(checkpoint) / "state.json"
    base_model = args.model
    if base_model is None and state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        base_model = state.get("model_name")
        logger.info("Base model detected from checkpoint: %s", base_model)
    if base_model is None:
        logger.error("Cannot determine base model name. Pass --model explicitly.")
        sys.exit(1)

    # Load benchmark examples
    if args.benchmark == "synthetic":
        from .eval.synthetic import load_eval_set
        eval_path = Path(checkpoint).parent / "eval_set.json"
        examples = load_eval_set(eval_path)
        if not examples:
            logger.error(
                "No synthetic eval set found at %s.\n"
                "Run training first — the eval set is created automatically at training start.",
                eval_path,
            )
            sys.exit(1)
        logger.info("Loaded %d synthetic eval tasks.", len(examples))
    elif args.benchmark == "bfcl":
        from .eval.bfcl import load_bfcl
        examples = load_bfcl(split=args.bfcl_split, n=args.n)
        if not examples:
            sys.exit(1)
    else:
        logger.error("Unknown benchmark: %s", args.benchmark)
        sys.exit(1)

    def _load_solver(model_path: str, cfg: Config) -> SolverAgent:
        tok = AutoTokenizer.from_pretrained(model_path)
        mdl = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, device_map=device_map
        )
        return SolverAgent(mdl, tok, cfg)

    trained_config = Config(model_name=checkpoint)
    base_config = Config(model_name=base_model)

    logger.info("Evaluating trained solver …")
    solver_trained = _load_solver(checkpoint, trained_config)
    result_trained = run_eval(solver_trained, examples, trained_config, n=args.n)

    logger.info("Evaluating baseline model …")
    solver_base = _load_solver(base_model, base_config)
    result_base = run_eval(solver_base, examples, base_config, n=args.n)

    benchmark_label = {
        "synthetic": "Synthetic held-out",
        "bfcl": f"BFCL ({args.bfcl_split})",
    }[args.benchmark]

    print(f"\n{'='*58}")
    print(f"  {benchmark_label} eval  ({result_trained.n_examples} examples)")
    print(f"{'='*58}")
    print(f"  {'Metric':<22} {'Baseline':>10} {'Trained':>10}  {'Δ':>8}")
    print(f"  {'-'*50}")
    fmt_d = result_trained.format_rate - result_base.format_rate
    acc_d = result_trained.accuracy - result_base.accuracy
    print(f"  {'Format rate':<22} {result_base.format_rate:>9.1%} {result_trained.format_rate:>9.1%}  {fmt_d:>+8.1%}")
    if args.benchmark == "synthetic":
        print(f"  {'Pass@1 (exact match)':<22} {result_base.accuracy:>10.3f} {result_trained.accuracy:>10.3f}  {acc_d:>+8.3f}")
    else:
        print(f"  {'Accuracy (−3..3)':<22} {result_base.accuracy:>10.3f} {result_trained.accuracy:>10.3f}  {acc_d:>+8.3f}")
    print(f"{'='*58}\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Tool-R0: Self-Evolving Tool-Calling Agent")
    sub = p.add_subparsers(dest="command", required=True)

    # --- train ---
    tr = sub.add_parser("train", help="Run the self-play training loop")
    tr.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    tr.add_argument("--iterations", type=int, default=2)
    tr.add_argument("--pool-size", type=int, default=500)
    tr.add_argument("--curriculum-size", type=int, default=200)
    tr.add_argument("--generator-steps", type=int, default=50)
    tr.add_argument("--solver-steps", type=int, default=50)
    tr.add_argument("--checkpoint-dir", default="./checkpoints")
    tr.add_argument("--debug", action="store_true", help="Enable DEBUG logging")

    # --- eval ---
    ev = sub.add_parser("eval", help="Evaluate a trained checkpoint vs the base model")
    ev.add_argument("--checkpoint", required=True,
                    help="Path to the solver checkpoint (checkpoints/solver)")
    ev.add_argument("--model", default=None,
                    help="Base model name for comparison (auto-detected from checkpoint state)")
    ev.add_argument("--benchmark", choices=["synthetic", "bfcl"], default="synthetic",
                    help="synthetic: held-out tasks generated at training start; "
                         "bfcl: Berkeley Function Calling Leaderboard (requires internet)")
    ev.add_argument("--bfcl-split", default="live_simple",
                    help="BFCL dataset split (default: live_simple)")
    ev.add_argument("--n", type=int, default=200,
                    help="Number of examples to evaluate (default: 200)")
    ev.add_argument("--debug", action="store_true", help="Enable DEBUG logging")

    args = p.parse_args()
    _configure_logging(getattr(args, "debug", False))

    if args.command == "train":
        cmd_train(args)
    elif args.command == "eval":
        cmd_eval(args)


if __name__ == "__main__":
    main()
