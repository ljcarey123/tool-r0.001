# Tool-R0

A reimplementation of [Tool-R0: Self-Evolving LLM Agents for Tool-Learning from Zero Data](https://arxiv.org/pdf/2602.21320), incorporating reward design findings from the companion [ToolRL paper](https://arxiv.org/html/2504.13958v1).

## Quick start

```bash
# 1. Install
python -m venv .venv && .venv\Scripts\activate
pip install -e ".[dev]"

# 2. Dev run: the full loop at reduced scale (1 iteration, 10+10 GRPO steps, small pool)
tool-r0 train --preset dev

# 3. Full run: paper schedule (3 iterations x 50 steps). Resume-safe: re-run the same command if interrupted
tool-r0 train --preset full

# 4a. Eval — synthetic held-out tasks (zero dependencies, created automatically at training start)
tool-r0 eval --benchmark synthetic --checkpoint ./checkpoints/solver

# 4b. Eval — Berkeley Function Calling Leaderboard (external, structured gold calls)
tool-r0 eval --benchmark bfcl --checkpoint ./checkpoints/solver --n 100
```

Both eval commands run the same benchmark against the untrained base model and print a side-by-side delta table.

## What this is

Tool-R0 trains a tool-calling agent from **zero annotated data** using self-play reinforcement learning. A Generator model proposes realistic tasks; a Solver model learns to answer them by selecting the right tools and parameters. Both models co-evolve via GRPO — no human labels, no SFT warm-up.

The paper roughly doubles Qwen2.5-0.5B-Instruct's tool-calling score (15.47 → 30.57 averaged over five benchmarks). This repo reproduces that size of gain at 0.5B on a 4 GB laptop GPU.

## Results

One full run (`--preset full`: 3 iterations × 50 Generator + 50 Solver GRPO steps) of
Qwen2.5-0.5B-Instruct on an RTX 3050 Ti laptop GPU (4 GB). Training took about 10.5 hours.
Each table compares the trained Solver with the untrained base model, using the same prompt.

**BFCL `live_simple`** (first 100 examples, external benchmark, BFCL-style AST matching):

| Metric | Base | Trained | Δ |
|--------|------|---------|---|
| Exact match (pass@1) | 0.29 | **0.58** | +0.29 (2.0×) |
| Output parses to a tool call | 0.45 | 0.99 | +0.54 |
| Shaped accuracy reward (0..1) | 0.38 | 0.78 | +0.41 |

**Synthetic held-out set** (100 fully valid tasks from the untrained Generator, pass@1 exact match):

| Base | Iteration 1 | Iteration 2 | Iteration 3 |
|------|-------------|-------------|-------------|
| 0.23 | 0.35 | 0.41 | **0.43** |

The Generator improves alongside the Solver. Of 500 sampled tasks, the share that parse rose
from about 17% for the base model to 55%, 90% and 95% over the three iterations. The
curriculum grew from 77 to 124 tasks once cross-verification had filtered the pool.

Caveats: this is a single run, so there are no error bars. The BFCL numbers are for the first
100 `live_simple` examples, where every task has exactly one tool and one call. The paper
instead averages five benchmarks, so the gains match in size but are not directly comparable.
Most of the BFCL gain comes from the model reliably producing a parseable call. Among answers
that parse, exact match goes from 64% (29/45) to 59% (58/99).

## Architecture

```
task spec (domain, n_tools, n_calls)
    │
    ▼
GeneratorAgent  ──GRPO──►  evolving generator   (invents a question, a tool menu, gold calls)
    │
    ▼
TaskPool  (deduplicate → cross-verify difficulty → sort easy→hard)
    │
    ▼
SolverAgent  ──GRPO──►  evolving solver
```

Generator and Solver are separate LoRA adapters on one frozen base model.

### Reward functions

**Generator**: `r_fmt + r_valid + r_curr`

| Reward | Definition |
|--------|------------|
| `r_fmt` | 0.3 · all four tags present + 0.3 · `<available_tools>` parses + 0.4 · `<tool_call_answer>` normalises |
| `r_valid` | 0.4 · gold tool in menu + 0.4 · required params present + 0.2 · argument values appear in the question |
| `r_curr` | `r_diff + r_sem`. `r_diff`: band-pass on the Solver's pass rate over K=8 samples (1 inside [0.25, 0.75], Gaussian decay σ=0.12 outside, 0 if never solved). `r_sem`: the Solver rates task coherence 1–5, scaled to [0, 1] |

**Solver**: `r_fmt + r_acc`

| Reward | Definition |
|--------|------------|
| `r_fmt` | 0.3 · `<think>` + answer tags + 0.3 · answer parses + 0.4 · normalises to ≥1 call (placeholders score 0) |
| `r_acc` | Each gold call is greedily matched to its best prediction, scored `0.2·name + 0.3·key-F1 + 0.5·value-match`, averaged, then × `1/(1 + 0.25·extra calls)`. Range [0, 1] |

The parser accepts strict JSON, Python-literal syntax, code-fenced blocks and OpenAI-style call wrappers.

### Key hyperparameters and deviations from the paper

| Parameter | Here | Paper | Why different |
|-----------|------|-------|---------------|
| Base model | Qwen2.5-0.5B-Instruct | 0.5B–3B (1.5B main) | Reproducing the 0.5B result |
| Fine-tuning | LoRA r=16, α=32 | Full | Fits a 4 GB GPU |
| Learning rate | 2e-5 | 1e-6 | LoRA needs a higher rate |
| KL β | 0 | 0.01 | Named adapters have no reference copy in TRL; ToolRL also found KL hurts |
| Batch | 4 completions/step, back-propagated 1 at a time (`grpo_micro_batch`) | 24–32 | Memory: 2 at a time ran out of 4 GB on long generator completions |
| Solver prompt | paper prompt + a closing format reminder | paper prompt | The 0.5B base model mostly answered in prose (2/16 valid tasks parsed vs 11/16) |
| Task pool | 500 sampled, ≤200 kept | 10k → 2k | Compute |
| Coherence judge (`r_sem`) | kept | same | At 0.5B the Solver rates almost every task 5/5, so in practice this adds a near-constant bonus to any parseable task |
| Context | single-turn | 10% multi-turn | Scope |
| Rollouts / temperature / iterations × steps | 4 / 1.0 / 3 × 50 | same | |

## Project layout

```
src/tool_r0/
├── config.py          # Config dataclass, presets, task-spec distribution
├── main.py            # CLI entry point
├── loop.py            # SelfPlayLoop — outer training loop
├── agents/
│   ├── generator.py   # GeneratorAgent
│   ├── solver.py      # SolverAgent
│   └── parser.py      # XML tag extraction + output parsing
├── rewards/
│   ├── format.py      # Format rewards (generator + solver)
│   ├── validity.py    # Generator validity reward
│   ├── curriculum.py  # Curriculum reward (difficulty band-pass + coherence judge)
│   └── accuracy.py    # Solver accuracy reward
├── tools/
│   └── schema.py      # Tool + ToolCall pydantic models
├── training/
│   ├── adapters.py    # Shared base model + generator/solver LoRA adapters
│   └── grpo.py        # Thin wrapper around TRL's GRPOTrainer
├── eval/
│   ├── synthetic.py   # Held-out generated tasks
│   ├── bfcl.py        # BFCL loader
│   └── runner.py      # Trained vs base comparison
└── data/
    ├── models.py      # TaskSpec, GeneratedTask, SolverExample
    └── pool.py        # TaskPool — dedup, cross-verify, curriculum build
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

pip install -e ".[dev]"
```

## Run

```bash
tool-r0 train --preset dev     # fast end-to-end check of the full loop
tool-r0 train --preset full    # paper schedule
tool-r0 train --preset dev --generator-steps 20   # explicit flags override the preset
```

The default model is `Qwen/Qwen2.5-0.5B-Instruct`. Generator and Solver are separate LoRA
adapters on one shared, frozen base model, so training fits on a 4 GB GPU and checkpoints are a
few MB. Adapters are saved to `./checkpoints/generator` and `./checkpoints/solver`
(override with `--checkpoint-dir`).

### On a cloud GPU (Modal)

[modal_app.py](modal_app.py) runs the same commands on a Modal GPU (L4, falling back to A10G or T4 when capacity is short). Roughly $0.60–1.10/hour, billed per second.

```bash
pip install modal && modal token new                   # one-time setup
modal run --detach modal_app.py::train --preset dev    # train; keeps running if you disconnect
modal run modal_app.py::evaluate --run dev             # trained vs base
modal volume get tool-r0-runs dev ./runs               # download adapters
```

## Evaluate

Two benchmarks are supported. Both compare the trained solver against the untrained base model.

```bash
# Synthetic held-out eval (generated automatically at training start, zero dependencies)
tool-r0 eval --benchmark synthetic --checkpoint ./checkpoints/solver

# Berkeley Function Calling Leaderboard (external, structured JSON gold calls)
tool-r0 eval --benchmark bfcl --checkpoint ./checkpoints/solver --n 100
```

The synthetic eval uses 100 tasks generated from the base generator before training begins
and saved to `checkpoints/eval_set.json`. Pass@1 exact match is reported before training
starts and after each iteration in the training log, so you can track improvement in real time.

BFCL (`live_simple` split) provides out-of-distribution validation with fully structured
gold calls (tool name + arguments), giving a meaningful accuracy score across all dimensions.

## Tests

```bash
pytest                    # all tests
pytest tests/unit/        # unit only (no GPU needed)
pytest tests/integration/ # integration (mocked models)
```

## Scope and future work

This repo covers the paper's single-turn self-play loop at 0.5B. Training is resumable: a
checkpoint is written after each Generator phase and after each iteration. Not yet covered:

- The paper's other benchmarks and model sizes, and repeated runs for error bars
- Multi-turn tasks (10% of the paper's task distribution) and multi-step tool use

See [Phases.md](Phases.md) for earlier plans.

## References

- [Tool-R0 paper](https://arxiv.org/pdf/2602.21320): Self-Evolving LLM Agents for Tool-Learning from Zero Data
- [ToolRL paper](https://arxiv.org/html/2504.13958v1): When Tool Use Gets Better with Reinforcement Learning
