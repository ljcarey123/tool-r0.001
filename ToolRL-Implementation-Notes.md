# ToolRL: Key Implementation Notes for Tool-R0 Reimplementation

**Paper:** ToolRL: Reward is All Tool Learning Needs (https://arxiv.org/html/2504.13958v1)
**Same research group as Tool-R0** — overlapping authors, complementary work. ToolRL is the Solver-side reward study; Tool-R0 adds the Generator/self-play loop on top.

---

## Why This Paper Matters

ToolRL is a systematic ablation of reward design for tool-calling RL training (GRPO). It directly informs how to design the **Solver's reward function** in your Tool-R0 reimplementation, with numbers to justify each choice.

---

## Output Format (use this for your Solver)

```xml
<think> reasoning and analysis </think>
<tool_call>
[{"name": "tool_name", "parameters": {"key": "value"}}, ...]
</tool_call>
<response> optional text response </response>
```

Rules:
- `<think>` is always required
- At least one of `<tool_call>` or `<response>` must be present
- Multiple simultaneous tool calls go in the same JSON array

---

## Reward Design — The Core Contribution

### Total Reward

```
R_final = R_format + R_correct  ∈ [-3, 4]
```

### Format Reward (binary)
```
R_format = 1 if all required tags present in correct order, else 0
```

### Correctness Reward (decomposed, fine-grained)

Three components computed by greedy matching predicted calls against gold calls:

**a) Tool name match (Jaccard)**
```
r_name = |N_gold ∩ N_pred| / |N_gold ∪ N_pred|   ∈ [0, 1]
```

**b) Parameter key match (summed Jaccard per tool)**
```
r_param = Σ_j  |keys(gold_j) ∩ keys(pred_j)| / |keys(gold_j) ∪ keys(pred_j)|   ∈ [0, |G|]
```

**c) Parameter value match (exact match per key)**
```
r_value = Σ_j Σ_k  𝟙[gold_j[k] == pred_j[k]]   ∈ [0, Σ|keys(gold_j)|]
```

**d) Normalize to [-3, 3]**
```
S_max = 1 + |G| + Σ|keys(gold_j)|     # max possible raw score
R_correct = 6 · (r_name + r_param + r_value) / S_max  - 3
```

This is nearly identical to Tool-R0's accuracy reward — the weights differ slightly but the structure is the same.

---

## Key Findings (with numbers)

### 1. Fine-grained reward beats coarse — always

| Granularity | Qwen1.5B | Qwen3B | Llama3B |
|-------------|----------|--------|---------|
| Fine-grained (above) | **46.20%** | **52.98%** | **44.10%** |
| Intermediate (combined param) | 37.65% | 51.36% | 38.62% |
| Coarse (binary exact match) | 36.72% | 51.40% | 35.95% |

**Implication:** Always decompose into name / param-keys / param-values. Binary exact match loses 8–9pp.

### 2. Do NOT add a length reward

| Variant | Qwen1.5B | Qwen3B |
|---------|----------|--------|
| No length reward (baseline) | **46.20%** | **52.98%** |
| With length reward | 33.23% (-13pp) | 48.89% (-4pp) |
| Dynamic length reward | 28.51% (-18pp) | 48.24% |

**Implication:** Don't penalise or incentivise response length. Just reward correctness.

### 3. Cold-start GRPO beats SFT → RL

| Training approach | Qwen1.5B | Qwen3B | Qwen7B |
|-------------------|----------|--------|--------|
| SFT (4k examples) | 40.67% | 41.97% | 36.53% |
| SFT → GRPO | 40.93% | 46.42% | 39.25% |
| **GRPO cold start (no SFT)** | **46.20%** | **52.98%** | **58.38%** |

**Implication:** Skip the SFT warmup. Train GRPO directly from the instruction-tuned base model. SFT overfits and limits RL exploration.

### 4. Remove KL penalty

Standard GRPO includes a KL divergence penalty to keep the policy close to a reference. Removing it enables broader exploration and consistently improves results here. Start without KL regularisation.

### 5. Dynamic reward scaling — marginal benefit, worth noting

Linearly shift reward scale over training (format reward decreases, correctness reward increases as training progresses):
```
scale_format  = [-2+p, 2-p]    # shrinks as training progresses
scale_correct = [-2-p, 2+p]    # grows as training progresses
p = S_current / S_total         # normalised training step
```

Effect: +0–2pp vs fixed scale, model-dependent. Not essential for a first implementation but easy to add.

---

## GRPO Hyperparameters

| Parameter | Value |
|-----------|-------|
| Batch size | 512 samples |
| Responses per query (rollouts) | 4 |
| Training epochs | 15 |
| Generation temperature | 1.0 (rollout) |
| KL penalty | None |
| Training data size | 4,000 examples |

For a simplified reimplementation, scale these down proportionally (e.g. 4 rollouts, 200–500 training examples, 5–10 epochs).

---

## Training Data (for ToolRL's supervised baseline — not needed for Tool-R0's self-play approach)

ToolRL uses 4k human-annotated samples from ToolACE + Hammer + xLAM. **Tool-R0 avoids this entirely** via self-generated data from the Generator. The ToolRL reward design is what you want to borrow, not its dataset.

---

## Summary: What to Take Into Your Implementation

| Decision | ToolRL Recommendation |
|----------|-----------------------|
| Solver reward | Fine-grained: name Jaccard + param-key Jaccard + value exact-match, normalised to [-3,3] |
| Format reward | Binary tag-presence check, contributes +1 |
| Length reward | Do not add one |
| KL penalty | Remove it |
| Training init | GRPO cold start directly from instruction-tuned base model |
| Rollouts per query | 4 |
| Temperature during rollout | 1.0 |
| Reward granularity | Always decompose — never use binary exact match |
