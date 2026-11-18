// Text for the architecture section: one entry per stage of a self-play iteration.
// Kept as data so the section module only does layout.

export const STAGES = [
  {
    id: "generator",
    title: "Train the Generator",
    short: "Invent tasks at the edge of the Solver's ability",
    adapter: "generator",
    body: [
      "The Generator writes a whole task: a user question, a menu of tools as JSON schemas, and the gold tool calls that answer it.",
      "It is trained with GRPO for 50 steps. Each step samples 4 tasks from one prompt and rewards them relative to each other.",
      "Part of its reward comes from the Solver: the Solver attempts each new task 8 times, and tasks it solves 25–75% of the time score highest. That link is what makes this self-play.",
    ],
    reward: [
      ["r_fmt", "0–1", "All four tags present, tools parse as JSON, gold calls normalise"],
      ["r_valid", "0–1", "Gold tool is in the menu, required parameters present, argument values appear in the question"],
      ["r_curr", "0–2", "Difficulty band-pass on the Solver's pass rate, plus a 1–5 coherence rating from the Solver"],
    ],
    code: ["src/tool_r0/loop.py", "src/tool_r0/rewards/curriculum.py", "src/tool_r0/rewards/validity.py"],
  },
  {
    id: "pool",
    title: "Sample a task pool",
    short: "500 tasks from the frozen Generator",
    adapter: "generator",
    body: [
      "The trained Generator, now frozen, samples 500 tasks across domains and tool counts.",
      "Tasks that do not parse into a question, tools and calls are discarded, and duplicates are removed.",
    ],
    code: ["src/tool_r0/agents/generator.py", "src/tool_r0/agents/parser.py"],
  },
  {
    id: "verify",
    title: "Cross-verify difficulty",
    short: "The Solver attempts each task 8 times",
    adapter: "solver",
    body: [
      "Each task's pass rate is how many of 8 sampled Solver answers exactly match the gold calls.",
      "Tasks never solved are dropped: they are usually broken or unanswerable. Tasks always solved are dropped too, because every GRPO sample would score the same and teach nothing.",
    ],
    code: ["src/tool_r0/data/pool.py"],
  },
  {
    id: "curriculum",
    title: "Build a curriculum",
    short: "Balanced easy / medium / hard, sorted easy → hard",
    adapter: null,
    body: [
      "Tasks are bucketed by pass rate (easy above 0.75, hard below 0.25), a balanced set of up to 200 is selected, and the set is ordered from easy to hard.",
    ],
    code: ["src/tool_r0/data/pool.py"],
  },
  {
    id: "solver",
    title: "Train the Solver",
    short: "Learn to answer with reasoning, then calls",
    adapter: "solver",
    body: [
      "The Solver reads the question and tool menu, writes its reasoning in <think>, then the calls in <tool_call_answer>.",
      "It is trained with GRPO for 50 steps, 4 curriculum tasks × 4 samples per step.",
    ],
    reward: [
      ["r_fmt", "0–1", "Tags present, answer parses, at least one real call"],
      ["r_acc", "0–1", "Per gold call: 0.2 name + 0.3 parameter-key F1 + 0.5 value match, with a penalty for extra calls"],
    ],
    code: ["src/tool_r0/loop.py", "src/tool_r0/rewards/accuracy.py", "src/tool_r0/agents/solver.py"],
  },
];

/** How the two roles share one model. */
export const ADAPTERS = {
  title: "One frozen model, two LoRA adapters",
  body: [
    "Qwen2.5-0.5B-Instruct is loaded once and frozen. PEFT replaces every linear layer with a wrapper that holds the original weights plus a small A·B pair per adapter (rank 16).",
    "Switching role is a dictionary lookup, not a reload. Only the active adapter's 8.8M parameters (1.7% of the model) receive gradients.",
  ],
  formula: "y = W·x + 2 · B_role · (A_role · x)",
  code: ["src/tool_r0/training/adapters.py", "src/tool_r0/training/grpo.py"],
};
