# Tool-R0: Self-Evolving LLM Agents for Tool-Learning from Zero Data

https://arxiv.org/html/2602.21320v1

## Outline

This project aims to reimplement the Tool-R0 paper.

It is a framework for training general purpose tool-calling agents from scratch with self-play RL.

We evolve a Generator and a Solver with complementary awards.
- The Generator proposes tasks, and is rewarded for generating challenging tasks aligned with the Solver's evolving capabilities
- The Solver learns to solve them with real-world tool calls

For effective co-evolution, the reward is difficulty-guided and based on the Solver's answer uncertainty: hard enough to teach and easy enough to verify.

Generated tasks are filtered and ranked easy-to-hard into a cirriculum pool, and the Solver trains on this curated data to predict tool calls.

## Method

### Overview

K self-play iterations.
Both components initialised from the same base LLM.
At each iteration, we first train Generator to produce tool-calling tasks that adaptively target the Solver's componentence frontier.
We then freeze the Generator and use it to construct a high-quality dataset with deduplication, cross-verification, and difficulty-based curriculum ordering.
Then the Solver is trained on the curated dataset, and carried forward to the next iteration.

### Grounded Task Specification

To avoid mode-collapsed generation, we should ground the task generation using task spec s=(d,c,m,n)
- d = task domain
- c = interaction context type
- m = the number of available tools
- n = the number of gold tool-calls as answer

At each training step, specs are dynamically sampled from a user-defined weighted distribution, and injected into the Generator's prompt.

### Training the Generator

Generator policy trained with GRPO. No external data, only the main prompt and task specifications.

Each generated sample consists of:
- a user request
- an explicit tool menu
- a gold tool-call

The Generator outputs exactly four tagged blocks:
- <think>, <question>, <available_tools>. <tool_call_answer>

Generator trained with a set of rewards that serve complementary purposes:
- enforce a strict code-verifiable output interface
- guarantee internal consistency between the tool menu and the gold tool-call
- induce an adaptive cirriculum by targeting tasks that aren't trivial or unsolvable for the current Solver

Format Reward:
- Reward tag completeness and well-formed JSON artifacts
- All required blocks should be extractable
- <available_tools> should parse as a JSON list of tool specs
- <tool_call_answer> should parse and normalise into a canonical tool-call representation

Validity Reward:
- Enforce internal consistency between the tool menu, gold tool answer, and the question itself
- Gold tool should exist in the menu, all schema-required parameters should be present, and every non-trivial argument value appears as a word-boundary match

Cirriculum Reward:
- Given a generated question and tool menu, we query the current Solver K times under Monte Carlo stochastic decoding to obtain predicted tool-calls, and estimate success against the gold tool-call
- Band-pass signal filter assigns maximal difficulty reward when the average predicted tool call lies in a target interval with smooth decay outside

### Solver Dataset Construction

After training Generator, it is frozen, and we sample a large pool of candidate tasks.
- Remove near-duplicates
- Cross-verify each candidate by sampling multiple predictions from Solver and measuring agreement with the generated gold tool-call by retaining only tasks with consistent solutions
- Estimate difficulty via pass@K success rates and bucket tasks

### Solver Training

Solver is trained to predict correct tool calls given a user query and tool menu, using a propmt template that elicits explicit reasoning.

Generates reasoning within <think> tags followed by predicted tool calls in <tool_call_answer> tags.

Format Reward:
- Assign partial credit on parseability for special tokens

Accuracy Reward:
- Decompose tool-call correctness into three components
- Match each gold tool-call to the best unused prediction
- For a matched pair, compute dense sub-scores of name match, key overlap, and value match


