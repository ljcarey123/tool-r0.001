// BFCL results as a four-step story. Each step shows the variants it is about on both
// splits, plus the paired McNemar tests that say whether a difference is real.

import { h, mount, segmented } from "../lib/dom.js";
import { load } from "../lib/data.js";
import { section, card } from "../lib/layout.js";
import { barList } from "../lib/charts.js";
import { tooltipBody } from "../lib/tooltip.js";
import { dec, pct, pValue, significant } from "../lib/format.js";
import { ENTITIES, SPLITS } from "../config.js";
import { STORY } from "../content/story.js";

const METRICS = {
  exact: { label: "Exact match", note: "Share of examples whose calls match BFCL's ground truth (AST matching)." },
  parses: { label: "Parse rate", note: "Share of answers that parse into at least one tool call, right or wrong." },
};

export async function render() {
  const results = await load("results");
  const state = { step: 0, metric: "exact" };

  const callout = h("div", { class: "callout" });
  const panels = h("div", { class: "grid-2" });
  const tests = h("div");

  const stepButtons = STORY.map((step, i) =>
    h("button", { type: "button", class: "story__step", "aria-pressed": "false", onclick: () => { state.step = i; draw(); } },
      h("span", { class: "story__num" }, `Step ${i + 1}`),
      h("span", { class: "story__title" }, step.title),
    ),
  );

  function draw() {
    const step = STORY[state.step];
    stepButtons.forEach((b, i) => b.setAttribute("aria-pressed", String(i === state.step)));
    callout.innerHTML = step.callout; // trusted, authored in content/story.js
    mount(panels, Object.entries(SPLITS).map(([split, title]) => splitPanel(results, step, split, title, state.metric)));
    mount(tests, pairTable(results, step));
  }
  draw();

  return section(
    {
      id: "results",
      kicker: "Results",
      title: "Where the doubling comes from",
      lede: "BFCL live_simple (258 examples) and simple (400), greedy decoding. Step through the comparisons in the order they were run.",
    },
    h("div", { class: "story", role: "group", "aria-label": "Comparison" }, stepButtons),
    callout,
    h("div", { class: "controls" },
      h("span", { class: "controls__label" }, "Metric"),
      segmented(Object.entries(METRICS).map(([value, m]) => ({ value, label: m.label })), state.metric,
        (v) => { state.metric = v; draw(); }, "Metric"),
    ),
    panels,
    card({ title: "Is the difference real?", note: "Paired on the same examples: how many only A gets right vs only B, with McNemar's exact test. The last column compares accuracy on examples where both answers parse, which removes the format effect." }, tests),
  );
}

function splitPanel(results, step, split, title, metric) {
  const rows = step.rows.map((row) => {
    const v = results[row.source]?.[split]?.variants[row.variant];
    const entity = ENTITIES[row.entity];
    return {
      label: step.model ? entity.name.replace("Base", `Base ${step.model}`) : entity.name,
      detail: row.detail,
      value: v?.[metric] ?? 0,
      color: entity.color,
      tooltip: v ? tooltipBody(entity.name, [
        ["Exact match", dec(v.exact)],
        ["Parse rate", pct(v.parses)],
        ["Shaped accuracy", dec(v.accuracy)],
      ]) : null,
    };
  });
  const n = results[step.rows[0].source]?.[split]?.n;
  return card({ title, aside: h("span", { class: "small muted" }, `${n} examples`), note: METRICS[metric].note },
    barList({ rows, max: 1, format: (x) => dec(x, 3) }));
}

function pairTable(results, step) {
  const body = [];
  for (const pair of step.pairs) {
    for (const split of Object.keys(SPLITS)) {
      const t = results[pair.source]?.[split]?.pairs[pair.key];
      if (!t) continue;
      const [ea, eb] = t.both_parse_exact;
      body.push(h("tr", {},
        h("td", {}, `${ENTITIES[pair.a].name} vs ${ENTITIES[pair.b].name}`),
        h("td", {}, split),
        h("td", {}, t.a_only),
        h("td", {}, t.b_only),
        h("td", {}, h("span", { class: `badge ${significant(t.p) ? "badge--accent" : "badge--neutral"}` },
          pValue(t.p), significant(t.p) ? " · significant" : "")),
        h("td", {}, `${dec(ea)} vs ${dec(eb)} (n=${t.both_parse_n})`),
      ));
    }
  }
  return h("div", { class: "table-wrap" },
    h("table", { class: "table" },
      h("thead", {}, h("tr", {},
        ["A vs B", "Split", "Only A right", "Only B right", "McNemar", "Exact when both parse"].map((c) => h("th", {}, c)))),
      h("tbody", {}, body),
    ),
  );
}
