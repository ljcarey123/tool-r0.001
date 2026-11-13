"""Compare base vs trained Solver on BFCL in both answer formats, with paired statistics.

Tests whether a training gain is a capability gain rather than format learning:
  - "qwen" format is the model's native tool-calling template, as BFCL evaluates Qwen, so
    base/qwen is the fair baseline;
  - paired comparisons count, on the same examples, which model alone gets each one right.

    python scripts/eval_compare.py --checkpoint runs/full/solver --splits live_simple,simple
"""
import argparse
import gc
import json
import logging
from math import comb

import torch

from tool_r0.agents.solver import SolverAgent
from tool_r0.config import Config
from tool_r0.eval.bfcl import bfcl_match, load_bfcl
from tool_r0.eval.runner import EvalResult, run_eval
from tool_r0.training.adapters import load_base_model, load_solver_for_eval

FORMATS = ("paper", "qwen")
PAIRS = [  # (A, B): is A better than B on the same examples?
    ("trained/paper", "base/qwen"),   # headline vs the fair baseline
    ("trained/qwen", "base/qwen"),    # does the trained adapter help in the native format too?
    ("trained/paper", "base/paper"),  # the comparison the README originally reported
]


def match(task, predicted):
    return bfcl_match(predicted, task.ground_truth)


def evaluate(model, tok, examples) -> dict[str, EvalResult]:
    results = {}
    for fmt in FORMATS:
        cfg = Config(solver_format=fmt)
        results[fmt] = run_eval(SolverAgent(model, tok, cfg), examples, cfg, n=len(examples), match=match)
    return results


def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value for b vs c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def report(split: str, results: dict[str, EvalResult]) -> dict:
    n = len(next(iter(results.values())).exact)
    print(f"\n=== BFCL {split} ({n} examples) ===")
    print(f"{'variant':<16}{'parses':>8}{'exact':>8}{'acc':>8}")
    for name, r in results.items():
        print(f"{name:<16}{r.format_rate:>8.3f}{r.exact_rate:>8.3f}{r.accuracy:>8.3f}")

    pairs = {}
    print(f"\n{'A vs B':<30}{'A only':>8}{'B only':>8}{'p':>8}   exact when both parse (n)")
    for a, b in PAIRS:
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
    ap.add_argument("--out", default=None, help="Write results as JSON")
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    data = {s: load_bfcl(split=s, n=a.n) for s in a.splits.split(",")}
    results: dict[str, dict[str, EvalResult]] = {s: {} for s in data}

    tok, trained, base_name = load_solver_for_eval(a.checkpoint)
    if base_name is None:
        raise SystemExit("Cannot determine the base model from the adapter config")
    for split, examples in data.items():
        for fmt, r in evaluate(trained, tok, examples).items():
            results[split][f"trained/{fmt}"] = r
    del trained
    gc.collect()
    torch.cuda.empty_cache()

    tok, base = load_base_model(base_name)
    for split, examples in data.items():
        for fmt, r in evaluate(base, tok, examples).items():
            results[split][f"base/{fmt}"] = r

    summary = {split: report(split, res) for split, res in results.items()}
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
