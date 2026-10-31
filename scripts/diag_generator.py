"""Generator health check: parse and validity rates at several sampling temperatures.

GRPO samples the Generator at `grpo_temperature`; if few of those samples parse, it only
ever receives the format reward. Run this before spending GPU time on training.

    python scripts/diag_generator.py --temps 0.7,0.85,1.0 --n 32 [--checkpoint DIR]
"""
import argparse
import collections
import json
import random
import time

import torch

from tool_r0.agents.generator import GeneratorAgent
from tool_r0.config import Config
from tool_r0.rewards.format import generator_format_reward
from tool_r0.rewards.validity import validity_reward
from tool_r0.training.adapters import GENERATOR_ADAPTER, generate_texts, load_shared_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=Config().model_name)
    ap.add_argument("--checkpoint", default=None, help="Run dir with trained adapters (default: base)")
    ap.add_argument("--temps", default="0.7,0.85,1.0")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", choices=["auto", "float32", "bfloat16", "float16"], default="auto")
    ap.add_argument("--dump", type=int, default=0, help="Print this many unparsed outputs per temperature")
    ap.add_argument("--save", default=None, help="Write raw outputs to this JSON file")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    cfg = Config(model_name=a.model)
    tok, model = load_shared_model(cfg, checkpoint_dir=a.checkpoint)
    if a.dtype != "auto":
        model.to(getattr(torch, a.dtype))
    print("dtype:", next(model.parameters()).dtype, "| device:", model.device)
    gen = GeneratorAgent(model, tok, cfg, adapter=GENERATOR_ADAPTER)
    specs = [cfg.sample_spec(random.Random(a.seed + i)) for i in range(a.n)]
    prompts = [gen.build_prompt(s) for s in specs]

    saved = {}
    for temp in (float(t) for t in a.temps.split(",")):
        t0 = time.time()
        texts = generate_texts(
            model, tok, prompts, GENERATOR_ADAPTER, cfg.generation_batch_size,
            max_new_tokens=cfg.generator_max_tokens, do_sample=True, temperature=temp,
        )
        saved[str(temp)] = texts
        tasks = [gen._parse(s, t) for s, t in zip(specs, texts)]
        parsed = [t for t in tasks if t is not None]
        valid = [validity_reward(t) for t in parsed]
        fmt = [generator_format_reward(t) for t in texts]
        tokens = [len(tok(t)["input_ids"]) for t in texts]
        print(
            f"temp={temp:<4}  parsed={len(parsed)}/{a.n}  fully_valid={sum(v == 1.0 for v in valid)}/{a.n}"
            f"  mean_fmt={sum(fmt) / len(fmt):.2f}  mean_tokens={sum(tokens) / len(tokens):.0f}"
            f"  hit_max={sum(n >= cfg.generator_max_tokens for n in tokens)}  {time.time() - t0:.0f}s"
        )
        print("   format scores:", dict(sorted(collections.Counter(round(f, 1) for f in fmt).items())))
        failures = [(s, t) for s, t, task in zip(specs, texts, tasks) if task is None]
        for spec, text in failures[: a.dump]:
            print(f"   --- unparsed ({spec.n_tools} tools, {spec.n_calls} calls):\n{text[:1200]}\n")

    if a.save:
        with open(a.save, "w", encoding="utf-8") as f:
            json.dump(saved, f, indent=1)


if __name__ == "__main__":
    main()
