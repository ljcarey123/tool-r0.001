# Tool-R0: Paper Summary & Implementation Goals

**Paper:** Tool-R0: Self-Evolving LLM Agents for Tool-Learning from Zero Data
**arXiv:** https://arxiv.org/abs/2602.21320 | HTML: https://arxiv.org/html/2602.21320v1
**Authors:** Emre Can Acikgoz, Cheng Qian, Jonas Hübotter, Heng Ji, Dilek Hakkani-Tür, Gokhan Tur

---

## Core Idea

Train a tool-calling LLM agent **from zero annotated data** using self-play RL between two co-evolving agents:

- **Generator** — proposes tool-calling tasks at the Solver's competence frontier
- **Solver** — learns to solve those tasks with real tool calls

The key insight: the Generator is rewarded for producing tasks that are *neither too easy nor too hard* for the current Solver (band-pass difficulty filter). This naturally induces curriculum learning without any human-curated dataset.

---

## Full Method

### High-Level Loop (K iterations)

```
Initialise Generator and Solver from the same base LLM
For each iteration:
  1. Train Generator (GRPO) to produce tasks targeting Solver's frontier
  2. Freeze Generator; sample 10,000 candidate tasks
  3. Filter, deduplicate, cross-verify → 2,000 curated tasks (curriculum ordered easy→hard)
  4. Train Solver (GRPO) on curated dataset
  5. Carry Solver forward to next iteration
```

Primary model: **Qwen2.5-1.5B-Instruct**. Three iterations, 50 GRPO steps each.

---

### Grounded Task Specification

To prevent mode collapse, every generated task is grounded by a spec tuple:

```
s = (d, c, m, n)
  d = task domain          (e.g. math, finance, weather)
  c = interaction context  (e.g. single-turn, multi-turn)
  m = number of available tools in the menu
  n = number of gold tool-calls required
```

Specs are sampled from a user-defined weighted distribution and injected into the Generator's prompt at each training step.

---

### Generator Training

**Output format** — four required tagged blocks:

```xml
<think>...</think>
<question>...</question>
<available_tools>[{...}, ...]</available_tools>
<tool_call_answer>[{...}]</tool_call_answer>
```

**Reward signals** (three complementary):

#### 1. Format Reward
```
r_fmt = λ_tag · I_tag + λ_parse · I_parse + λ_norm · I_norm
weights: (0.3, 0.3, 0.4)
```
Rewards tag completeness and well-formed JSON. All blocks must be extractable, `<available_tools>` must parse as a JSON list, `<tool_call_answer>` must parse into a canonical tool-call representation.

#### 2. Validity Reward
Enforces internal consistency:
- Gold tool must exist in the tool menu
- All schema-required parameters must be present
- Every non-trivial argument value must appear as a word-boundary match in the question

#### 3. Curriculum (Difficulty) Reward
The key reward. Query the current Solver K=8 times (Monte Carlo stochastic decoding) on the generated task; estimate success rate p̄. Then:

```
Maximal reward when p̄ ∈ [P_low=0.25, P_high=0.75]
Gaussian decay (σ=0.12) outside the band
```
This means: rewarded for tasks the Solver gets ~25–75% of the time — hard enough to teach, easy enough to verify.

---

### Solver Dataset Construction

After Generator is frozen:

1. Sample **10,000 candidate tasks**
2. **Deduplicate** via canonicalized question–tool–call signatures
3. **Cross-verify** — sample multiple Solver predictions per task; keep only tasks with consistent solutions (filters ambiguous/broken tasks)
4. **Estimate difficulty** via pass@K success rates → bucket into easy / medium / hard
5. Select **2,000 final samples**, preserving domain diversity, ordered easy→hard as a curriculum

---

### Solver Training

Also trained with **GRPO**.

**Format Reward** — same structure as Generator's format reward (well-formed tagged output).

**Accuracy Reward:**
```
r_acc = s̄ · 1/(1 + α·max(0, |Ĉ| - |C*|))
```
Where s̄ is computed by greedy matching of predicted vs gold tool-calls across:
- **Name match** — binary (weight 0.2)
- **Key overlap** — F1 on parameter keys (weight 0.3)
- **Value match** — fraction of matching values (weight 0.5)

The `α=0.25` penalty term discourages spurious extra tool calls.

---

## Results

### Main Result (Qwen2.5-1.5B, zero training data)

| Benchmark   | Baseline | Tool-R0 | Δ      | Relative |
|-------------|----------|---------|--------|----------|
| ToolAlpaca  | 35.96    | 47.36   | +11.40 | +31.7%   |
| SealTool    | 47.27    | 83.00   | +35.73 | +75.6%   |
| NexusRaven  | 17.61    | 34.59   | +16.98 | +86.4%   |
| API-Bank    | 19.13    | 50.62   | +31.49 | +164.6%  |
| SNIPS       | 4.29     | 20.86   | +16.57 | +386.3%  |
| **Average** | 24.85    | **47.84** | +22.99 | **+92.5%** |

### Tool-R0 Beats Supervised Baselines (Table 2)

| Method       | Training Data | Avg Score |
|--------------|---------------|-----------|
| xLAM         | 60k samples   | 43.60     |
| Hammer       | 210k samples  | 43.74     |
| ToolAce      | 12k samples   | 44.71     |
| ToolRL       | 4k samples    | 46.06     |
| **Tool-R0**  | **0 samples** | **47.84** |

### Ablations (Table 3)

| Variant                  | Avg Acc | Drop    |
|--------------------------|---------|---------|
| Full Tool-R0             | 47.84   | —       |
| Shared Generator+Solver  | 30.42   | -17.42  |
| Frozen Generator         | 41.65   | -6.19   |
| No difficulty reward     | 43.54   | -4.30   |
| No Gaussian falloff      | 44.10   | -3.74   |

Key takeaway: **separate Generator and Solver weights are critical** (-17pp if shared). The difficulty reward provides +4.3pp.

### Other Base Models

| Model              | Baseline | Tool-R0 | Relative |
|--------------------|----------|---------|----------|
| Qwen2.5-0.5B       | 15.47    | 30.57   | +101%    |
| Qwen2.5-1.5B       | 24.85    | 47.84   | +92.5%   |
| Qwen2.5-3B         | 43.97    | 48.50   | +10.3%   |
| Llama-3.2-3B       | 36.12    | 40.47   | +12.0%   |

Smaller models benefit more — the 0.5B model doubles its score.

---

## Evaluation Benchmarks

Five tool-use benchmarks:
1. **ToolAlpaca** — simulated API call scenarios
2. **SealTool** — detailed function-calling benchmark
3. **NexusRaven** — function calling tasks
4. **API-Bank** — comprehensive API usage benchmark
5. **SNIPS** — intent/slot classification for tool routing

---

## Simplified Implementation Plan

My goal is to build a minimal version capturing the core loop, then extend it with file I/O tools for a simple agentic todo-list.

### Phase 1 — Minimal Tool-R0 Core

**What to simplify:**
- Use a single small model (e.g. Qwen2.5-1.5B or 0.5B via Ollama/vLLM) for both roles initially
- Use a small fixed tool set (3–5 tools: calculator, string_search, get_date, read_file, write_file)
- Skip GRPO; start with supervised fine-tuning on self-generated data (simpler to iterate)
- 1–2 self-play iterations instead of 3

**Minimum viable Generator:**
- Prompt the model with a task spec `(domain, n_tools, n_calls)`
- Parse its `<question>`, `<available_tools>`, `<tool_call_answer>` output
- Score with format + validity rewards (no curriculum yet)

**Minimum viable Solver:**
- Given question + tool menu, predict `<tool_call>` in JSON
- Execute the tool call against real implementations
- Score accuracy against Generator's gold answer

**Curriculum pool (simplified):**
- Generate 200–500 tasks, filter valid ones, sort by Solver pass rate
- Train Solver on sorted curriculum

### Phase 2 — Agentic Loop with File Tools

Extend the Solver's tool set with:
- `read_file(path)` — read a file's contents
- `write_file(path, content)` — write/append to a file
- `list_todos()` — read a todo list file
- `add_todo(task)` — append a task to the todo list
- `mark_done(task_id)` — update a task's status

The Generator creates tasks like:
- "Add 'buy milk' to the todo list"
- "Mark task 2 as done"
- "List all incomplete tasks"

This gives a grounded agentic loop where the Solver's tool calls have real, verifiable effects — and the gold tool-call from the Generator provides the RL reward signal.

### Key Design Decisions for Implementation

| Decision | Paper | Simplified Version |
|----------|-------|--------------------|
| RL algorithm | GRPO | SFT first, GRPO later |
| Iterations | 3 × 50 steps | 1–2 iterations |
| Task pool | 10k → 2k | 500 → 200 |
| Base model | Qwen2.5-1.5B | Qwen2.5-0.5B or 1.5B |
| Tool set | Domain-agnostic JSON | Fixed 5-tool set including file I/O |
| Curriculum | 3-bucket easy/med/hard | Simple pass-rate sort |
| Difficulty reward | Band-pass [0.25, 0.75] | Optional — add in phase 2 |

---

## Key Equations Reference

**Curriculum reward (band-pass):**
```
p̄ = (1/K) Σ success(solver_k(q, T))   # K=8 Monte Carlo samples
r_curr = gaussian_bandpass(p̄, low=0.25, high=0.75, σ=0.12)
```

**Accuracy reward (Solver):**
```
s̄ = λ_name·name_match + λ_key·key_F1 + λ_val·val_match
   weights: (0.2, 0.3, 0.5)
r_acc = s̄ · 1/(1 + 0.25·max(0, |predicted_calls| - |gold_calls|))
```

**Format reward (both agents):**
```
r_fmt = 0.3·I_tag + 0.3·I_parse + 0.4·I_norm
```
