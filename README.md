# Tool-R0

A reimplementation of [Tool-R0: Self-Evolving LLM Agents for Tool-Learning from Zero Data](https://arxiv.org/pdf/2602.21320), incorporating reward design findings from the companion [ToolRL paper](https://arxiv.org/html/2504.13958v1).

## Quick start

```bash
# 1. Install
python -m venv .venv && .venv\Scripts\activate
pip install -e ".[dev]"

# 2. Smoke-test: verify the pipeline runs end-to-end (plumbing check only — 0.5B is too
#    small to follow the structured format, so expect 0 valid tasks and an empty curriculum)
tool-r0 train --model Qwen/Qwen2.5-0.5B-Instruct --iterations 1 --pool-size 20 --curriculum-size 10 --generator-steps 5 --solver-steps 5 --debug

# 3. Full training — paper settings (resume-safe: re-run the same command if interrupted)
tool-r0 train --model Qwen/Qwen2.5-1.5B-Instruct --iterations 2 --pool-size 500 --curriculum-size 200 --generator-steps 50 --solver-steps 50

# 4a. Eval — synthetic held-out tasks (zero dependencies, created automatically at training start)
tool-r0 eval --benchmark synthetic --checkpoint ./checkpoints/solver

# 4b. Eval — Berkeley Function Calling Leaderboard (external, structured gold calls)
tool-r0 eval --benchmark bfcl --checkpoint ./checkpoints/solver --n 200
```

Both eval commands run the same benchmark against the untrained base model and print a side-by-side delta table.

## What this is

Tool-R0 trains a tool-calling agent from **zero annotated data** using self-play reinforcement learning. A Generator model proposes realistic tasks; a Solver model learns to answer them by selecting the right tools and parameters. Both models co-evolve via GRPO — no human labels, no SFT warm-up.

The paper reports a **92.5% relative improvement** over the Qwen2.5-1.5B base model on standard tool-calling benchmarks.

## Architecture

```
registry (tool schemas)
    │
    ▼
GeneratorAgent  ──GRPO──►  evolving generator
    │ TaskSpec → GeneratedTask (question + available_tools + gold_calls)
    │
    ▼
TaskPool  (deduplicate → cross-verify difficulty → sort easy→hard)
    │
    ▼
SolverAgent  ──GRPO──►  evolving solver
```

### Reward functions

**Generator rewards** (3 binary criteria, weighted per paper Table 1):

| Criterion | Weight | Passes when |
|-----------|--------|-------------|
| `I_tag`   | 0.3    | All four required XML tags are present |
| `I_parse` | 0.3    | `<available_tools>` hydrates into valid `Tool` objects |
| `I_norm`  | 0.4    | `<tool_call_answer>` hydrates into valid `ToolCall` objects |

Plus a **validity reward** (tool-in-menu + required params + value plausibility) and a **curriculum reward** (band-pass [0.25, 0.75] so tasks are neither trivial nor impossible).

**Solver rewards** (per ToolRL paper):

- **Format**: binary 0/1 — `<think>` present + parseable tool calls.
- **Accuracy**: greedy name match → Jaccard on name + key + value exact, normalised to [−3, 3], with a penalty for extra calls (`α = 0.25`).

### Key hyperparameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` | Tool-R0 paper |
| GRPO β (KL penalty) | 0.0 | ToolRL finding: KL hurts |
| MC samples per task | 8 | Tool-R0 curriculum |
| Curriculum band | [0.25, 0.75] | Tool-R0 Section 3.3 |
| Extra-call penalty α | 0.25 | ToolRL accuracy reward |

## Project layout

```
src/tool_r0/
├── config.py          # Config dataclass — all hyperparameters
├── main.py            # CLI entry point
├── loop.py            # SelfPlayLoop — outer training loop
├── agents/
│   ├── generator.py   # GeneratorAgent
│   ├── solver.py      # SolverAgent
│   └── parser.py      # XML tag extraction + output parsing
├── rewards/
│   ├── format.py      # Format rewards (generator + solver)
│   ├── validity.py    # Generator validity reward
│   ├── curriculum.py  # Curriculum band-pass reward
│   └── accuracy.py    # Solver accuracy reward
├── tools/
│   ├── schema.py      # Tool + ToolCall pydantic models
│   ├── registry.py    # ToolRegistry (schema-only, no execution)
│   └── builtins.py    # Default 4-tool registry
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
# Smoke-test (fast, small model, enable --debug to see raw generator outputs)
tool-r0 train --model Qwen/Qwen2.5-0.5B-Instruct --iterations 1 --pool-size 20 --curriculum-size 10 --generator-steps 5 --solver-steps 5 --debug

# Full training run (paper settings)
tool-r0 train --model Qwen/Qwen2.5-1.5B-Instruct --iterations 2 --pool-size 500 --curriculum-size 200 --generator-steps 50 --solver-steps 50
```

The trained Solver is saved to `./solver_checkpoint` (override with `--output`).

## Evaluate

Two benchmarks are supported. Both compare the trained solver against the untrained base model.

```bash
# Synthetic held-out eval (generated automatically at training start, zero dependencies)
tool-r0 eval --benchmark synthetic --checkpoint ./checkpoints/solver

# Berkeley Function Calling Leaderboard (external, structured JSON gold calls)
tool-r0 eval --benchmark bfcl --checkpoint ./checkpoints/solver --n 200
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

## Phases

- **Phase 1** *(current)*: Core self-play loop — Generator + Solver co-evolving via GRPO. No tool execution; reward is structural JSON matching only. Verified against synthetic held-out eval and BFCL.
- **Phase 2**: Benchmark evaluation on ToolBench / APIBench / BFCL.
- **Phase 3**: Multi-step tool use (chained calls, dependency graphs).

See [Phases.md](Phases.md) for detailed plans.

## References

- [Tool-R0 paper](https://arxiv.org/pdf/2602.21320): Self-Evolving LLM Agents for Tool-Learning from Zero Data
- [ToolRL paper](https://arxiv.org/html/2504.13958v1): When Tool Use Gets Better with Reinforcement Learning
