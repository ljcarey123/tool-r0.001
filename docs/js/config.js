// Site-wide constants: where code links point, and the fixed colour of each model variant.
// A variant keeps its colour in every chart (colour follows the entity, never its rank).

export const REPO = "https://github.com/ljcarey123/tool-r0.001";
export const codeUrl = (path) => `${REPO}/blob/main/${path}`;

/** Model variants compared on BFCL, in fixed palette order. */
export const ENTITIES = {
  trained2: { name: "Trained, run 2", color: "var(--series-1)" },
  trained1: { name: "Trained, run 1", color: "var(--series-2)" },
  qwen: { name: "Base, Qwen's own format", color: "var(--series-3)" },
  nothink: { name: "Base, empty <think> pre-filled", color: "var(--series-4)" },
  paper: { name: "Base, paper format", color: "var(--series-5)" },
};

export const RUNS = {
  run1: { name: "Run 1", color: "var(--series-2)" },
  run2: { name: "Run 2", color: "var(--series-1)" },
};

export const SPLITS = {
  live_simple: "BFCL live_simple",
  simple: "BFCL simple",
};
