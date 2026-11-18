// Browse BFCL examples: the question, the tools, the ground truth, and what each model
// variant actually wrote. Filters pick out the cases that explain the headline result.

import { h, mount, segmented } from "../lib/dom.js";
import { load } from "../lib/data.js";
import { section, card, errorBox } from "../lib/layout.js";
import { int, pct } from "../lib/format.js";
import { ENTITIES, SPLITS } from "../config.js";

/** Variant key in examples.json -> entity in config.js, in display order. */
const VARIANTS = [
  ["base/paper", "paper"],
  ["base/paper+nothink", "nothink"],
  ["base/qwen", "qwen"],
  ["trained/paper", "trained2"],
];

const [TEXT, PARSED, EXACT] = [0, 1, 2];
const out = (ex, v) => ex.out[v] ?? ["", false, false];

const FILTERS = {
  all: { label: "All examples", test: () => true },
  formatOnly: {
    label: "Base failed only on format",
    test: (ex) => !out(ex, "base/paper")[PARSED] && out(ex, "base/paper+nothink")[EXACT] && out(ex, "trained/paper")[EXACT],
  },
  trainedWins: {
    label: "Trained right, Qwen format wrong",
    test: (ex) => out(ex, "trained/paper")[EXACT] && !out(ex, "base/qwen")[EXACT],
  },
  qwenWins: {
    label: "Qwen format right, trained wrong",
    test: (ex) => out(ex, "base/qwen")[EXACT] && !out(ex, "trained/paper")[EXACT],
  },
  allWrong: {
    label: "Every variant wrong",
    test: (ex) => VARIANTS.every(([v]) => !out(ex, v)[EXACT]),
  },
};

export async function render() {
  let data;
  try {
    data = await load("examples");
  } catch {
    return section({ id: "examples", kicker: "Examples", title: "What the models actually wrote" },
      errorBox("Example outputs not generated yet. Run scripts/eval_compare.py with --save-outputs, then scripts/build_explorer_data.py."));
  }

  const state = { split: "live_simple", filter: "formatOnly", index: 0 };
  const thinkSlot = h("div");
  const view = h("div", { class: "stack" });
  const counter = h("span", { class: "small muted" });
  const filterSelect = h("select", {
    class: "select",
    "aria-label": "Filter examples",
    onchange: (e) => { state.filter = e.target.value; state.index = 0; draw(); },
  }, Object.entries(FILTERS).map(([value, f]) => h("option", { value, selected: value === state.filter }, f.label)));

  const matching = () => data[state.split].filter(FILTERS[state.filter].test);

  function draw() {
    const list = matching();
    state.index = list.length ? (state.index + list.length) % list.length : 0;
    counter.textContent = list.length ? `${state.index + 1} of ${int(list.length)}` : "No matching examples";
    mount(view, list.length ? example(list[state.index]) : null);
    mount(thinkSlot, thinkSummary(data[state.split]));
  }
  const move = (delta) => { state.index += delta; draw(); };
  draw();

  return section(
    {
      id: "examples",
      kicker: "Examples",
      title: "What the models actually wrote",
      lede: "Every BFCL example with the raw output of each variant (run 2's trained Solver). Start with the cases where the base model only failed on format.",
    },
    card({ title: "What the trained model writes in <think>", note: "The most common reasoning texts across all of its answers on this split." }, thinkSlot),
    h("div", { class: "controls" },
      segmented(Object.entries(SPLITS).map(([value, label]) => ({ value, label })), state.split,
        (v) => { state.split = v; state.index = 0; draw(); }, "Split"),
      filterSelect,
      h("div", { class: "controls pager" },
        h("button", { type: "button", class: "icon-button", "aria-label": "Previous example", onclick: () => move(-1) }, "‹"),
        counter,
        h("button", { type: "button", class: "icon-button", "aria-label": "Next example", onclick: () => move(1) }, "›"),
        h("button", { type: "button", class: "icon-button", "aria-label": "Random example", title: "Random",
          onclick: () => { state.index = Math.floor(Math.random() * matching().length); draw(); } }, "⤨"),
      ),
    ),
    view,
  );
}

function example(ex) {
  return [
    card({ title: "Question" },
      h("p", {}, ex.q),
      h("details", { class: "disclosure", style: { marginTop: "12px" } },
        h("summary", {}, `${ex.tools.length} tool${ex.tools.length === 1 ? "" : "s"} available: ${ex.tools.map((t) => t.name).join(", ")}`),
        h("pre", { class: "code" }, JSON.stringify(ex.tools, null, 2)),
      ),
      h("details", { class: "disclosure", style: { marginTop: "8px" } },
        h("summary", {}, "Ground truth (allowed values per parameter)"),
        h("pre", { class: "code" }, JSON.stringify(ex.truth, null, 2)),
      ),
    ),
    h("div", { class: "grid-2" },
      VARIANTS.map(([variant, entity]) => {
        const [text, parsed, exact] = out(ex, variant);
        return h("div", { class: "card output" },
          h("div", { class: "card__title" },
            h("h3", { class: "output__name" }, h("span", { class: "swatch", style: { background: ENTITIES[entity].color } }), ENTITIES[entity].name),
            h("span", { class: "controls" },
              h("span", { class: `badge ${parsed ? "badge--neutral" : "badge--bad"}` }, parsed ? "parsed" : "✕ no parse"),
              h("span", { class: `badge ${exact ? "badge--good" : "badge--neutral"}` }, exact ? "✓ correct" : "wrong"),
            ),
          ),
          h("pre", { class: "code" }, highlightThink(text || "(empty)")),
        );
      }),
    ),
  ];
}

/** Wrap the contents of <think>…</think> in <mark> so the reasoning stands out. */
function highlightThink(text) {
  const m = text.match(/<think>([\s\S]*?)<\/think>/);
  if (!m) return text;
  const start = m.index + "<think>".length;
  return [text.slice(0, start), h("mark", {}, m[1]), text.slice(start + m[1].length)];
}

function thinkSummary(examples) {
  const counts = new Map();
  for (const ex of examples) {
    const m = out(ex, "trained/paper")[TEXT].match(/<think>([\s\S]*?)<\/think>/);
    const key = m ? m[1].trim() || "(empty)" : "(no <think> block)";
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  const top = [...counts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 3);
  return h("div", { class: "table-wrap" },
    h("table", { class: "table" },
      h("thead", {}, h("tr", {}, h("th", {}, "Reasoning text"), h("th", {}, "Answers"), h("th", {}, "Share"))),
      h("tbody", {},
        top.map(([text, n]) => h("tr", {},
          h("td", { style: { whiteSpace: "normal" } }, h("code", {}, text.length > 140 ? `${text.slice(0, 140)}…` : text)),
          h("td", {}, int(n)),
          h("td", {}, pct(n / examples.length, 0)),
        )),
        h("tr", {}, h("td", { class: "muted" }, `${int(counts.size)} distinct texts in total`), h("td", {}), h("td", {})),
      ),
    ),
  );
}
