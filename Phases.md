# Tool-R0 Project Phases

---

## Phase 1 — Core Tool-R0 Reimplementation (current)

**Goal:** Faithfully reimplement the main result of the Tool-R0 paper — a self-evolving tool-calling agent trained from zero data via Generator/Solver self-play — and verify it reproduces the paper's improvement over the base model.

### What the training loop does

```
Initialise Generator and Solver from the same base LLM (separate weight copies)
For K iterations:
  1. Train Generator (GRPO) — produces tool-calling tasks at the Solver's competence frontier
  2. Freeze Generator → sample task pool → deduplicate → cross-verify → order by difficulty
  3. Train Solver (GRPO) — predicts tool calls against Generator's gold answers
```

Every reward is **structural** (JSON matching). We never execute a tool during training. The tool schemas exist only so the Generator can construct realistic menus and the Solver has something concrete to reason about.

### On tool implementations

We do not need runnable tool implementations for Phase 1. The registry only needs to serve **schemas** (Tool objects with name/description/parameters) for:
- Sampling menus into Generator prompts
- Passing menus into Solver prompts
- Validity reward: checking gold call is consistent with the schema

`ToolRegistry.execute()` is vestigial in Phase 1. It will become load-bearing in Phase 3.

### What's left to do

- [ ] Smoke-test the GRPO training loop end-to-end on a tiny model/tiny dataset to verify TRL wiring works
- [ ] Run full training on `Qwen2.5-1.5B-Instruct` (paper's primary model, 2 iterations, 50 steps each)
- [ ] Evaluate trained Solver against the five paper benchmarks:
  - ToolAlpaca, SealTool, NexusRaven, API-Bank, SNIPS
- [ ] Compare to paper's reported numbers (+92.5% relative improvement over base)

### Success criterion

The trained Solver scores meaningfully higher than the uninstructed base model on at least three of the five benchmarks, directionally consistent with the paper's results.

---

## Phase 2 — Multi-Step Reasoning Benchmark

**Goal:** Build a benchmark that tests whether a tool-calling model can carry out *multi-step* reasoning tasks — specifically, tasks that require reading and writing to files or managing a structured todo list across multiple dependent tool calls in a single trajectory.

This phase is **evaluation only** — no new training. We run both the baseline model and the Phase 1 R0-trained model through the benchmark and compare.

### Why this matters

Tool-R0 trains single-turn tool calling: one question → one (or a small fixed number of) tool calls. The benchmark gap we want to expose is whether that capability generalises to **chained, stateful** tool use, where:
- The output of one tool call is needed to construct the next
- The task cannot be solved with a fixed plan — it requires reading intermediate state
- The agent must self-correct if a step fails

### Benchmark design

Existing benchmarks to draw from or adapt:
- **τ-bench** (tau-bench) — multi-step tool use with state, closest to our intent
- **BFCL v3** — has a multi-turn / multi-step category
- **API-Bank Level 2/3** — multi-turn dialogue with dependent API calls
- **ToolBench** — complex real-world APIs with chained calls

Our custom extension adds tasks in two categories:

#### Category A — File-based tasks
The agent must read one or more files and write results to another.

Example tasks:
- "Read `shopping.txt`, count the number of items, write the count to `count.txt`"
- "Read `notes.txt`, find all lines containing the word 'urgent', write them to `urgent.txt`"
- "Read `data.csv`, compute the sum of the second column, write the result to `summary.txt`"

Requires: `read_file` → compute → `write_file`

#### Category B — Todo list management
The agent must perform a sequence of modifications to a structured todo list, reading intermediate state between steps.

Example tasks:
- "Add three tasks to the todo list, then mark the second one as done"
- "Read the current todo list, count incomplete items, add a summary task at the end"
- "Check if task 'buy milk' exists; if not, add it, then mark it done"

Requires: `list_todos` → conditional logic → `add_todo` / `mark_done` → `list_todos` (verify)

### Evaluation metric

For each task, score 0/1 on **final state correctness** — does the file/todo list end up in the expected state? No partial credit for "almost right" multi-step trajectories.

Secondary metric: step efficiency — did the model use the minimum number of tool calls, or did it waste steps?

### What we need to build

- [ ] Tool implementations: `read_file`, `write_file`, `list_todos`, `add_todo`, `mark_done` (already in codebase, removed for Phase 1, restore here)
- [ ] A sandboxed eval harness that resets file/todo state between tasks
- [ ] Task dataset: ~50–100 tasks across both categories, with expected final states
- [ ] Adapt existing benchmark tasks from τ-bench or BFCL multi-step where applicable
- [ ] Evaluation runner: feed each task to the model in an agentic loop (model sees tool result, then decides next call)

### Expected findings

The hypothesis is that the Phase 1 R0 model will be **no better than baseline** on multi-step tasks, because it was never trained to chain tool calls or use intermediate results. This gap motivates Phase 3.

---

## Phase 3 — Training Multi-Step Tool-Calling Capabilities

**Goal:** Starting from the Phase 1 R0 model, extend the self-play training to evolve multi-step, stateful tool-calling capabilities — and measure whether this closes the gap identified in Phase 2.

### Key design decisions (to be determined by Phase 2 findings)

**Option A — Extend Tool-R0's Generator to produce multi-step tasks**

Modify the Generator to produce trajectories instead of single-turn tasks:
```
<think>...</think>
<question>Multi-step user goal</question>
<available_tools>[...]</available_tools>
<trajectory>
  [{"tool_call": {...}, "result": "..."}, {"tool_call": {...}, "result": "..."}, ...]
</trajectory>
<final_answer>...</final_answer>
```

The Solver is trained to predict the full trajectory (auto-regressively), using the result of each step to construct the next call.

Challenges:
- Generator reward needs to verify the trajectory is internally consistent (each call uses the result of the prior one)
- Curriculum reward becomes more expensive (sampling Solver on multi-step tasks requires executing tools)
- Tool execution is now required in training — `ToolRegistry.execute()` becomes load-bearing

**Option B — SFT on multi-step demonstrations, then GRPO fine-tune**

Generate a small set of high-quality multi-step demonstrations (either by hand or by prompting a larger model), SFT the Phase 1 model on them, then GRPO fine-tune with a task-completion reward.

Simpler to implement; less aligned with Tool-R0's zero-data philosophy but more practical if Phase 2 shows a large gap.

**Option C — Extend the reward to credit intermediate steps**

Keep single-turn training but change the reward: instead of matching a single gold tool call, reward partial completion of a multi-step plan. This is closer to process reward modelling.

### Training infrastructure needed

- Sandboxed tool execution during training (each Solver rollout executes real tools and reads results)
- State reset between rollouts (file system / todo state must be clean for each episode)
- Trajectory-level reward: did the final state match the target? (binary) + intermediate step credit (optional)
- Significantly more compute: multi-step rollouts are much slower than single-turn

### Success criterion

The Phase 3 model scores measurably higher than the Phase 1 R0 model on the Phase 2 benchmark, demonstrating that the self-play training approach can be extended to evolve multi-step capabilities.

---

## Summary

| Phase | What it is | Model input | Model output | Reward basis | Tool execution needed? |
|-------|-----------|-------------|--------------|-------------|----------------------|
| 1 | Core R0 training | Question + tool menu | Single tool call | JSON matching against gold | No |
| 2 | Benchmark eval | Multi-step task + tools | Tool call sequence | Final state correctness | Yes (eval harness) |
| 3 | Multi-step training | Multi-step task + tools | Tool call trajectory | Final state + trajectory consistency | Yes (training) |
