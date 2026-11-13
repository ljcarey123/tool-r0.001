"""Compare base vs trained Solver on BFCL in both answer formats, with paired statistics.

Tests whether a training gain is a capability gain rather than format learning:
  - "qwen" format is the model's native tool-calling template, as BFCL evaluates Qwen, so
    base/qwen is the fair baseline;
  - paired comparisons count, on the same examples, which model alone gets each one right.

    python scripts/eval_compare.py --checkpoint runs/full/solver --splits live_simple,simple

Variants are "<model>/<format>[+option...]": model base|trained, format paper|qwen, options
nothink (pre-fill an empty <think> block), tagged / labels (paper-format prompt template).
Both models default to the prompt settings the checkpoint's run was trained with.
"""
import argparse
import dataclasses
import gc
import json
import logging
from math import comb
from pathlib import Path

import torch

from tool_r0.agents.solver import SolverAgent
from tool_r0.config import Config
from tool_r0.eval.bfcl import bfcl_match, load_bfcl
from tool_r0.eval.runner import EvalResult, run_eval
from tool_r0.training.adapters import load_base_model, load_solver_for_eval

DEFAULT_VARIANTS = "base/paper,base/qwen,trained/paper,trained/qwen"
DEFAULT_PAIRS = [  # (A, B): is A better than B on the same examples?
    ("trained/paper", "base/qwen"),   # headline vs the fair baseline
    ("trained/qwen", "base/qwen"),    # does the trained adapter help in the native format too?
    ("trained/paper", "base/paper"),  # the comparison the README originally reported
]
OPTIONS = {
    "nothink": {"solver_answer_prefix": "<think>\n</think>\n"},
    "tagged": {"solver_prompt": "tagged"},
    "labels": {"solver_prompt": "labels"},
}


def match(task, predicted):
    return bfcl_match(predicted, task.ground_truth)


def variant_config(run_cfg: Config, variant: str) -> Config:
    fmt, *options = variant.split("/", 1)[1].split("+")
    overrides: dict = {"solver_format": fmt}
    for opt in options:
        overrides.update(OPTIONS[opt])
    return dataclasses.replace(run_cfg, **overrides)


def evaluate(model, tok, examples, run_cfg: Config, variants: list[str]) -> dict[str, EvalResult]:
    results = {}
    for v in variants:
        cfg = variant_config(run_cfg, v)
        results[v] = run_eval(SolverAgent(model, tok, cfg), examples, cfg, n=len(examples), match=match)
    return results


def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value for b vs c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def report(split: str, results: dict[str, EvalResult], pairs_wanted: list[tuple[str, str]]) -> dict:
    n = len(next(iter(results.values())).exact)
    print(f"\n=== BFCL {split} ({n} examples) ===")
    print(f"{'variant':<16}{'parses':>8}{'exact':>8}{'acc':>8}")
    for name, r in results.items():
        print(f"{name:<16}{r.format_rate:>8.3f}{r.exact_rate:>8.3f}{r.accuracy:>8.3f}")

    pairs = {}
    print(f"\n{'A vs B':<30}{'A only':>8}{'B only':>8}{'p':>8}   exact when both parse (n)")
    for a, b in pairs_wanted:
        ra, rb = results[a], results[b]
        a_only = sum(x and not y for x, y in zip(ra.exact, rb.exact))
        b_only = sum(y and not x for x, y in zip(ra.exact, rb.exact))
        both = [i for i in range(n) if ra.parsed[i] and rb.parsed[i]]
        ea = sum(ra.exact[i] for i in both) / len(both) if both else 0.0
        eb = sum(rb.exact[i] for i in both) / len(both) if both else 0.0
        p = mcnemar_p(a_only, b_only)
        print(f"{a + ' vs ' + b:<30}{a_only:>8}{b_only:>8}{p:>8.3f}   {ea:.3f} vs {eb:.3f} ({len(both)})")
        pairs[f"{a} vs {b}"] = {"a_only": a_only, "b_only": b_only, "p": p,
                                "both_parse_n": len(both), "both_parse_exact": [ea, eb]}
    return {
        "n": n,
        "variants": {k: {"parses": r.format_rate, "exact": r.exact_rate, "accuracy": r.accuracy}
                     for k, r in results.items()},
        "pairs": pairs,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Trained solver adapter directory")
    ap.add_argument("--splits", default="live_simple,simple")
    ap.add_argument("--n", type=int, default=None, help="Examples per split (default: all)")
    ap.add_argument("--variants", default=DEFAULT_VARIANTS)
    ap.add_argument("--pairs", default=None,
                    help='Comparisons as "A:B,A:B" (default: trained/paper vs each base variant)')
    ap.add_argument("--out", default=None, help="Write results as JSON")
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    variants = a.variants.split(",")
    pairs = ([tuple(p.split(":")) for p in a.pairs.split(",")] if a.pairs
             else [p for p in DEFAULT_PAIRS if p[0] in variants and p[1] in variants])
    run_dir = Path(a.checkpoint).parent
    run_cfg = Config.from_run(run_dir) if (run_dir / "state.json").exists() else Config()
    print(f"prompt settings from the run: solver_prompt={run_cfg.solver_prompt}")

    data = {s: load_bfcl(split=s, n=a.n) for s in a.splits.split(",")}
    results: dict[str, dict[str, EvalResult]] = {s: {} for s in data}

    tok, trained, base_name = load_solver_for_eval(a.checkpoint)
    if base_name is None:
        raise SystemExit("Cannot determine the base model from the adapter config")
    trained_variants = [v for v in variants if v.startswith("trained/")]
    for split, examples in data.items():
        results[split].update(evaluate(trained, tok, examples, run_cfg, trained_variants))
    del trained
    gc.collect()
    torch.cuda.empty_cache()

    base_variants = [v for v in variants if v.startswith("base/")]
    if base_variants:
        tok, base = load_base_model(base_name)
        for split, examples in data.items():
            results[split].update(evaluate(base, tok, examples, run_cfg, base_variants))

    summary = {split: report(split, res, pairs) for split, res in results.items()}
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
