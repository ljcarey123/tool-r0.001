// Hero: what the project is, the headline numbers, and the findings in one list.

import { h } from "../lib/dom.js";
import { load } from "../lib/data.js";
import { dec, pValue } from "../lib/format.js";

export async function render() {
  const results = await load("results");
  const r1 = results.run1.live_simple;
  const r2 = results.run2.live_simple;
  const b15 = results.base15.live_simple.variants;
  const fair = r2.pairs["trained/paper vs base/qwen"];

  return h("header", { class: "hero", id: "top" },
    h("div", { class: "section__kicker" }, "Tool-R0 reproduction · Qwen2.5-0.5B · 4 GB laptop GPU"),
    h("h1", {}, "Self-play taught the format, not the skill"),
    h("p", { class: "hero__lede" },
      "Tool-R0 trains a tool-calling model with no labelled data: one model invents tasks, another learns to solve them. ",
      "This reproduction gets the paper's doubling at 0.5B, then shows that a fairly prompted base model scores the same.",
    ),
    h("div", { class: "stats" },
      stat(`${dec(r1.variants["base/paper"].exact, 2)} → ${dec(r1.variants["trained/paper"].exact, 2)}`,
        "BFCL live_simple, base vs trained, both in the paper's answer format"),
      stat(`${dec(fair ? r2.variants["base/qwen"].exact : null, 2)} vs ${dec(r2.variants["trained/paper"].exact, 2)}`,
        `The base model in its own tool format vs the trained model (${pValue(fair?.p)})`),
      stat(`+${Math.round(100 * (b15["base/paper+labels"].exact - b15["base/paper+labels+nothink"].exact))} pts`,
        "What the reasoning step is worth to the untrained 1.5B model; at 0.5B it is worth nothing"),
    ),
    h("ol", { class: "findings" },
      finding("The paper-style comparison reproduces the gain.",
        "Base and trained models both answer in the paper's <think> / <tool_call_answer> format. The base model was never trained on it and half its answers fail to parse."),
      finding("Against a fair baseline the gain disappears.",
        "Given its own tool-calling format, or two pre-filled tokens, the base model matches the trained one. On answers both models parse, accuracy is the same."),
      finding("More training signal did not change it.",
        "Run 2 trained the Solver on 4× more tasks per step and dropped tasks it always solved. Same result."),
      finding("The trained model stopped reasoning.",
        "In all 658 BFCL answers, run 2's <think> block reads \"This is my reasoning.\", the placeholder from the prompt's format example. Run 1's holds just a function name. The Generator stalled too: its tasks became either unsolvable or trivially solved."),
      finding("1.5B is where a real gain could appear.",
        "The untrained 1.5B model already produces the format, and its reasoning step helps."),
    ),
  );
}

const stat = (value, label) =>
  h("div", { class: "card stat" }, h("div", { class: "stat__value" }, value), h("div", { class: "stat__label" }, label));

const finding = (title, body) =>
  h("li", { class: "finding" },
    h("div", {}, h("div", { class: "finding__title" }, title), h("div", { class: "finding__body" }, body)));
