# Tool-R0

A reimplementation of [Tool-R0: Self-Evolving LLM Agents for Tool-Learning from Zero Data](https://arxiv.org/pdf/2602.21320), incorporating reward design findings from the companion [ToolRL paper](https://arxiv.org/html/2504.13958v1).

**[Explore the architecture and results →](https://ljcarey123.github.io/tool-r0.001/)** (source in [docs/](docs/), served by GitHub Pages)

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

The paper reports roughly doubling Qwen2.5-0.5B-Instruct's tool-calling score (15.47 → 30.57,
averaged over five benchmarks). This repo reproduces the paper's training loop on a 4 GB laptop
GPU and checks where that gain comes from.

## Results

**Summary.** At 0.5B, the self-play loop reproduces the paper's reported size of gain, but on a
fair comparison the gain is entirely the model learning the output format. Accuracy did not
improve. A two-token prompt prefix gives the untrained model the same score, and the trained
model's reasoning step is empty boilerplate. A second run that gave the Solver 4× more training signal gave the
same result. At 1.5B, the base model already handles the paper's format and its reasoning step
helps, so 1.5B is where a real gain could show up.

Setup: two full runs (`--preset full`, 3 iterations × 50 Generator + 50 Solver GRPO steps) of
Qwen2.5-0.5B-Instruct on an RTX 3050 Ti laptop GPU (4 GB), about 10.5 hours of compute each.
Sections 1–3 describe run 1 and section 4 run 2. Evaluation uses all
of BFCL `live_simple` (258 examples) and `simple` (400), scored with BFCL-style AST matching.
Decoding is greedy. Comparisons are paired, on the same examples, with McNemar's test.

### 1. Against its own training format, the trained model looks twice as good

| Exact match | Base (paper format) | Trained (paper format) |
|---|---|---|
| BFCL `live_simple` | 0.264 | **0.512** |
| BFCL `simple` | 0.502 | **0.765** |
| Synthetic held-out set (100 tasks) | 0.23 | **0.43** (0.35 → 0.41 → 0.43 by iteration) |

This is the comparison the paper reports. The base model is given the paper's
`<think>` / `<tool_call_answer>` format, which it was never trained on, and it often fails to
produce a parseable answer: only 50% (`live_simple`) and 63% (`simple`) of its answers parse.

### 2. Against a fair baseline, there is no gain

| Exact match | Base, Qwen's own tool format | Base + empty `<think>` pre-filled | Trained |
|---|---|---|---|
| BFCL `live_simple` | 0.508 | 0.516 | 0.512 |
| BFCL `simple` | **0.830** | 0.757 | 0.765 |

- **Base model in its own built-in tool-calling format** (how BFCL evaluates Qwen): the trained
  model ties on `live_simple` (31 vs 30 examples only one model gets right, p = 1.0) and is
  worse on `simple` (8 vs 34, p < 0.001).
- **Only answers that parse:** the trained and base models are equally accurate (0.535 vs 0.535
  on `live_simple`; 0.787 vs 0.803 on `simple`).
- **Base model with an empty `<think></think>` block pre-filled at the start of its answer:** it parses 97–98%
  of the time and matches the trained model. The roughly 10.5 hours of training bought what two
  tokens of prompt do.

### 3. The trained model does not reason

The trained Solver always writes a `<think>` block, but there is no reasoning in it:

- **Run 1** writes only the name of the function it is about to call (e.g. `get_current_weather`).
- **Run 2** writes `This is my reasoning.` in all 658 BFCL answers. That is the placeholder text
  from the format example at the end of the prompt.

Self-play rewarded the format and the answer, not the reasoning, so the reasoning collapsed to
the cheapest text that satisfies the format. Pre-filling an empty `<think>` block, so that run 1's
model answers directly, scores *higher* on `live_simple` (0.581 vs 0.512; 25 vs 7, p = 0.002) and
the same on `simple` (0.780 vs 0.765, p = 0.18).

### 4. More Solver training signal: the same result

Run 1's Solver saw one task per GRPO step, and about half its curriculum was tasks it already
solved every time, which give GRPO nothing to learn from. Run 2 trains on 4 tasks per step, drops
always-solved tasks, and uses the paper's tagged prompt (Fig. 12).

| Exact match | Base, Qwen format | Base + empty `<think>` | Trained, run 1 | Trained, run 2 |
|---|---|---|---|---|
| BFCL `live_simple` | 0.508 | 0.504 | 0.512 | **0.527** |
| BFCL `simple` | **0.830** | 0.748 | 0.765 | 0.733 |

Run 2's baseline columns use the tagged prompt, so they differ slightly from section 2.

- Run 2 vs the base model with an empty `<think>` block: p = 0.36 (`live_simple`) and 0.42
  (`simple`). On answers that parse for both, accuracy is identical (0.603 vs 0.603) or slightly
  lower (0.747 vs 0.762).
- The synthetic held-out score went 0.30 → 0.41 → 0.40 → 0.40. All the gain came in iteration 1,
  even though the Solver went on to solve more and more of its own tasks every time (19, 102 and
  162 tasks dropped from the curriculum as always solved).
- Pre-filling an empty `<think>` block no longer changes run 2's score (0.761 vs 0.757,
  p = 1.0), as expected when its `<think>` block is always the same placeholder sentence.

### 5. At 1.5B, there is no format gap, and reasoning helps

An evaluation of the untrained Qwen2.5-1.5B-Instruct (the paper's main model) on the first 200
examples of each split, using the run-1 prompt:

| Exact match | Paper format, with reasoning | Paper format, empty `<think>` | Qwen's own format |
|---|---|---|---|
| BFCL `live_simple` | **0.680** | 0.570 | 0.640 |
| BFCL `simple` | 0.905 | 0.850 | **0.925** |

- **The 1.5B model already produces the paper's format.** 92.5% (`live_simple`) and 99.5%
  (`simple`) of its answers parse, so a gain from training could not come from format learning.
- **Reasoning helps.** Removing the reasoning step costs 11 points on `live_simple` (33 vs 11
  examples, p = 0.001) and 5.5 on `simple` (p = 0.02). At 0.5B it made no difference or hurt.
- **The paper's format with reasoning matches Qwen's own format** (0.680 vs 0.640 and 0.905 vs
  0.925).

1.5B is therefore the size where self-play could show a real gain. Training it does not fit in
4 GB, so it is left as future work.

### What this means for the paper's 0.5B result

The paper does not say how base models were prompted at evaluation, nor how unparseable answers
were scored. Its 0.5B base scores are very low on some benchmarks (1.57 on SNIPS, 4.71 on
NexusRaven), which is what format failures would look like, and the paper reports that format
rewards rise faster than accuracy. If its base models were evaluated in the custom format, its
0.5B gains may largely be format learning, as ours are. This is a possibility the paper's text
leaves open, not a demonstrated flaw: our setup differs (LoRA, a smaller batch and task pool, one
run), and the effect should be weaker for the paper's main 1.5B and 3B models.

### Other observations

- The Generator does learn: of 500 sampled tasks, the share that parse rose from about 17% for
  the base model to 55%, 90% and 95% over the three iterations.
- In run 2, 470 and 487 of 500 sampled tasks were valid in iterations 2 and 3.
- The paper's tagged prompt template (Fig. 12) makes the base model parse more often than the
  plain-label prompt used in the first run (0.50 → 0.76 on `live_simple`). It is now the default.

Reproduce the comparisons with `scripts/eval_compare.py` (see its docstring).

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
| Batch | Generator 4 completions/step (1 task); Solver 16 (4 tasks × 4). Back-propagated 1 at a time (`grpo_micro_batch`) | 24–32 | Memory: 2 at a time ran out of 4 GB on long generator completions. Run 1's Solver used 1 task/step |
| Solver prompt | paper's tagged prompt (Fig. 12) + a closing format reminder | paper's tagged prompt | The 0.5B base model mostly answered in prose (2/16 valid tasks parsed vs 11/16). Run 1 used plain labels instead of tags |
| Curriculum | tasks the Solver always solves are dropped | not stated | They give GRPO zero learning signal. Run 1 kept them |
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

BFCL (`live_simple` by default, `--bfcl-split simple` also works) provides out-of-distribution
validation with structured gold answers, scored by BFCL-style AST matching.

Both commands use the same prompt for the base model as for the trained one, which favours the
trained model on format (see Results). For a fair comparison, including the base model in Qwen's
own tool-calling format, use:

```bash
python scripts/eval_compare.py --checkpoint ./checkpoints/solver --splits live_simple,simple
```

## Tests

```bash
pytest                    # all tests
pytest tests/unit/        # unit only (no GPU needed)
pytest tests/integration/ # integration (mocked models)
```

## Scope and future work

This repo covers the paper's single-turn self-play loop at 0.5B. Training is resumable: a
checkpoint is written after each Generator phase and after each iteration. Not yet covered:

- Training at 1.5B, where section 5 suggests a real gain is possible (needs more than 4 GB;
  `modal_app.py` runs training on a cloud GPU)
- The paper's other benchmarks, and repeated runs for error bars
- Multi-turn tasks (10% of the paper's task distribution) and multi-step tool use

See [Phases.md](Phases.md) for earlier plans.

## References

- [Tool-R0 paper](https://arxiv.org/pdf/2602.21320): Self-Evolving LLM Agents for Tool-Learning from Zero Data
- [ToolRL paper](https://arxiv.org/html/2504.13958v1): When Tool Use Gets Better with Reinforcement Learning
