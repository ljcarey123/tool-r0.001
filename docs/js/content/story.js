// The results narrative: four steps, each a set of BFCL bars and the paired tests behind them.
// `source` is a key of data/results.json, `variant` a variant inside it.

export const STORY = [
  {
    title: "The paper's comparison",
    callout: "<strong>The trained model looks twice as good.</strong> Both models answer in the paper's <code>&lt;think&gt;</code> / <code>&lt;tool_call_answer&gt;</code> format. The base model was never trained on it, so many of its answers don't parse at all.",
    rows: [
      { entity: "paper", source: "run1", variant: "base/paper" },
      { entity: "trained1", source: "run1", variant: "trained/paper" },
    ],
    pairs: [{ source: "run1", key: "trained/paper vs base/paper", a: "trained1", b: "paper" }],
  },
  {
    title: "Fair baselines",
    callout: "<strong>The gain disappears.</strong> In Qwen's own tool-calling format, or with an empty <code>&lt;think&gt;</code> block pre-filled (two tokens of prompt), the base model parses reliably and matches the trained model.",
    rows: [
      { entity: "paper", source: "run1", variant: "base/paper" },
      { entity: "qwen", source: "run1", variant: "base/qwen" },
      { entity: "nothink", source: "run1_ablation", variant: "base/paper+nothink" },
      { entity: "trained1", source: "run1", variant: "trained/paper" },
    ],
    pairs: [
      { source: "run1", key: "trained/paper vs base/qwen", a: "trained1", b: "qwen" },
      { source: "run1_ablation", key: "base/paper vs base/paper+nothink", a: "paper", b: "nothink" },
    ],
  },
  {
    title: "Run 2: more signal",
    callout: "<strong>Same result.</strong> Run 2 trained the Solver on 4 tasks per step instead of 1, dropped tasks it always solved, and used the paper's tagged prompt. It still ties the fair baselines, and is worse than Qwen's own format on <code>simple</code>.",
    rows: [
      { entity: "paper", source: "run2", variant: "base/paper" },
      { entity: "qwen", source: "run2", variant: "base/qwen" },
      { entity: "nothink", source: "run2", variant: "base/paper+nothink" },
      { entity: "trained1", source: "run1", variant: "trained/paper", detail: "labels prompt" },
      { entity: "trained2", source: "run2", variant: "trained/paper" },
    ],
    pairs: [
      { source: "run2", key: "trained/paper vs base/qwen", a: "trained2", b: "qwen" },
      { source: "run2", key: "trained/paper vs base/paper+nothink", a: "trained2", b: "nothink" },
      { source: "run2", key: "trained/paper vs base/paper", a: "trained2", b: "paper" },
    ],
  },
  {
    title: "At 1.5B",
    model: "1.5B",
    callout: "<strong>No format gap, and reasoning helps.</strong> The untrained 1.5B model already produces the paper's format, and removing its reasoning costs 11 points on <code>live_simple</code>. A gain from training here could only be real skill. (First 200 examples per split.)",
    rows: [
      { entity: "paper", source: "base15", variant: "base/paper+labels", detail: "with reasoning" },
      { entity: "nothink", source: "base15", variant: "base/paper+labels+nothink" },
      { entity: "qwen", source: "base15", variant: "base/qwen" },
    ],
    pairs: [
      { source: "base15", key: "base/paper+labels+nothink vs base/paper+labels", a: "nothink", b: "paper" },
      { source: "base15", key: "base/qwen vs base/paper+labels+nothink", a: "qwen", b: "nothink" },
    ],
  },
];
