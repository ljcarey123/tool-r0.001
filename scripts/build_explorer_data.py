"""Build the data files behind the results explorer in docs/.

Reads training logs and comparison JSONs from runs/ and writes compact JSON to docs/data/:

    runs.json      per run: baseline, and per iteration the pool funnel, pass@1 and
                   the reward components logged for every GRPO step
    results.json   the BFCL comparisons (run 1, run 2, the run 1 ablation, the 1.5B check)
    examples.json  every BFCL example with each variant's raw output (from
                   `eval_compare.py --save-outputs`)

    python scripts/build_explorer_data.py
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RUNS = {  # run id -> (label, training log)
    "run1": ("Run 1", "runs/full.log"),
    "run2": ("Run 2", "runs/full2.log"),
}
RESULTS = {  # result id -> comparison JSON
    "run1": "runs/compare-full.json",
    "run1_ablation": "runs/ablation-full.json",
    "run2": "runs/compare-full2.json",
    "base15": "runs/base15-compare.json",
}
EXAMPLES = "runs/outputs-full2.json"

_ITERATION = re.compile(r"Self-play iteration (\d+) / \d+")
_BASELINE = re.compile(r"Baseline pass@1 \(untrained\): ([\d.]+)")
_PASS = re.compile(r"pass@1 after iteration (\d+): ([\d.]+)")
_VALID = re.compile(r"Valid tasks: (\d+) / (\d+)")
_CROSS = re.compile(r"Cross-verify: (\d+) \S+ (\d+) tasks solved at least once \(pass rates: \[(.*)\]\)")
_DROPPED = re.compile(r"Curriculum: dropped (\d+) tasks")
_CURRICULUM = re.compile(r"Curriculum: (\d+) easy, (\d+) medium, (\d+) hard")
_REWARDS = re.compile(r"(Generator|Solver) rewards: parsed=(\d+)/(\d+)\s+(.*)$")
_COMPONENT = re.compile(r"(r_\w+)=([\d.]+)")


def parse_training_log(path: Path) -> dict:
    """Extract the per-iteration story from a training log.

    A crash or resume restarts an iteration and logs it again, so whenever an iteration
    starts, anything recorded for it (or later iterations) is discarded. A resume that
    skips an already-trained Generator logs no rewards, so its earlier rewards are kept.
    """
    baseline = None
    iterations: dict[int, dict] = {}
    current = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if m := _BASELINE.search(line):
            baseline = float(m[1])
        elif m := _ITERATION.search(line):
            current = int(m[1])
            kept = iterations.get(current, {}).get("generator", [])
            for k in [k for k in iterations if k >= current]:
                del iterations[k]
            iterations[current] = _empty_iteration(current)
            iterations[current]["_previous_generator"] = kept
        elif "Generator already trained" in line and current:
            it = iterations[current]
            it["generator"] = it.pop("_previous_generator", [])
        elif not current:
            continue
        elif m := _VALID.search(line):
            iterations[current]["valid"] = int(m[1])
            iterations[current]["sampled"] = int(m[2])
        elif m := _CROSS.search(line):
            # Pass rates are successes / mc_samples (8), so k = round(rate * 8) solves of 8.
            rates = [float(r) for r in m[3].split(",") if r.strip()]
            hist = [0] * 9
            for r in rates:
                hist[round(r * 8)] += 1
            hist[0] = int(m[1]) - int(m[2])  # unique tasks never solved
            iterations[current].update(unique=int(m[1]), solved_once=int(m[2]), solves_of_8=hist)
        elif m := _DROPPED.search(line):
            iterations[current]["dropped"] = int(m[1])
        elif m := _CURRICULUM.search(line):
            iterations[current]["curriculum"] = {"easy": int(m[1]), "medium": int(m[2]), "hard": int(m[3])}
        elif m := _PASS.search(line):
            iterations[int(m[1])]["pass_at_1"] = float(m[2])
        elif m := _REWARDS.search(line):
            role = m[1].lower()
            step = {"parsed": int(m[2]) / int(m[3])}
            step.update({k: float(v) for k, v in _COMPONENT.findall(m[4])})
            iterations[current][role].append(step)
    for it in iterations.values():
        it.pop("_previous_generator", None)
    return {"baseline_pass_at_1": baseline, "iterations": [iterations[k] for k in sorted(iterations)]}


def _empty_iteration(n: int) -> dict:
    return {"iteration": n, "sampled": None, "valid": None, "unique": None,
            "solved_once": None, "solves_of_8": None, "dropped": 0, "curriculum": None,
            "pass_at_1": None, "generator": [], "solver": []}


def build_examples(path: Path) -> dict:
    """Keep what the example browser shows: question, tools, ground truth, outputs."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        split: [
            {
                "q": ex["question"],
                "tools": [{"name": t["name"], "description": t.get("description", ""),
                           "parameters": t.get("parameters", {})} for t in ex["tools"]],
                "truth": ex["ground_truth"],
                "out": {v: [o["text"], o["parsed"], o["exact"]] for v, o in ex["outputs"].items()},
            }
            for ex in examples
        ]
        for split, examples in data.items()
    }


def write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size / 1024:.0f} KB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="docs/data")
    out = ROOT / ap.parse_args().out

    runs = {rid: {"label": label, **parse_training_log(ROOT / log)} for rid, (label, log) in RUNS.items()}
    write(out / "runs.json", runs)

    results = {rid: json.loads((ROOT / p).read_text(encoding="utf-8"))
               for rid, p in RESULTS.items() if (ROOT / p).exists()}
    write(out / "results.json", results)

    if (ROOT / EXAMPLES).exists():
        write(out / "examples.json", build_examples(ROOT / EXAMPLES))
    else:
        print(f"skipped examples.json: {EXAMPLES} not found (run eval_compare.py --save-outputs)")


if __name__ == "__main__":
    main()
