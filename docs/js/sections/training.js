// What happened during training: held-out pass@1, the task-pool funnel, how hard the
// Generator's tasks were for the Solver, and the logged GRPO rewards.

import { h, mount, segmented } from "../lib/dom.js";
import { load } from "../lib/data.js";
import { section, card } from "../lib/layout.js";
import { lineChart, columnChart, legend } from "../lib/charts.js";
import { tooltipBody } from "../lib/tooltip.js";
import { dec, int, pct } from "../lib/format.js";
import { RUNS } from "../config.js";

const STEPS_PER_PHASE = 50;

const REWARD_SERIES = {
  generator: [
    { key: "r_fmt", name: "Format (r_fmt)", color: "var(--series-1)" },
    { key: "r_valid", name: "Validity (r_valid)", color: "var(--series-2)" },
    { key: "r_curr", name: "Curriculum (r_curr)", color: "var(--series-3)" },
  ],
  solver: [
    { key: "r_fmt", name: "Format (r_fmt)", color: "var(--series-1)" },
    { key: "r_acc", name: "Accuracy (r_acc)", color: "var(--series-2)" },
  ],
};

export async function render() {
  const runs = await load("runs");
  const state = { run: "run2", role: "generator" };

  const funnelSlot = h("div");
  const histSlot = h("div");
  const rewardSlot = h("div");
  const redraw = () => {
    mount(funnelSlot, funnel(runs[state.run]));
    mount(histSlot, solveHistograms(runs[state.run]));
    mount(rewardSlot, rewardCurves(runs[state.run], state.role));
  };
  redraw();

  const runPicker = segmented(
    Object.entries(RUNS).map(([value, r]) => ({ value, label: r.name })),
    state.run,
    (v) => { state.run = v; redraw(); },
    "Training run",
  );
  const rolePicker = segmented(
    [{ value: "generator", label: "Generator" }, { value: "solver", label: "Solver" }],
    state.role,
    (v) => { state.role = v; mount(rewardSlot, rewardCurves(runs[state.run], state.role)); },
    "Role",
  );

  return section(
    {
      id: "training",
      kicker: "Training",
      title: "All the gain came in iteration 1",
      lede: "Each run is 3 iterations of 50 Generator and 50 Solver GRPO steps, about 10.5 hours on a laptop GPU. Run 2 gave the Solver 4× more tasks per step and dropped tasks it always solved.",
    },
    card({ title: "Held-out synthetic tasks", note: "Share of 100 generated tasks the Solver answers exactly right with greedy decoding, after each iteration." },
      passAtOne(runs),
    ),
    h("div", { class: "controls" }, h("span", { class: "controls__label" }, "Run"), runPicker),
    h("div", { class: "grid-2" },
      card({ title: "Task pool funnel", note: "From 500 sampled tasks to the Solver's curriculum, per iteration." }, funnelSlot),
      card({
        title: "How often the Solver solves each task",
        note: "Cross-verification: 8 Solver attempts per unique task. Only tasks in between give GRPO a learning signal.",
      }, histSlot),
    ),
    card({
      title: "GRPO rewards per step",
      aside: rolePicker,
      note: "Mean reward components per logged batch, smoothed over 5 steps. Validity and curriculum are averaged over the completions that parsed.",
    }, rewardSlot),
  );
}

function passAtOne(runs) {
  const series = Object.entries(runs).map(([id, run]) => ({
    name: RUNS[id].name,
    color: RUNS[id].color,
    markers: true,
    points: [{ x: 0, y: run.baseline_pass_at_1 }, ...run.iterations.map((it) => ({ x: it.iteration, y: it.pass_at_1 }))],
  }));
  return h("div", { class: "stack" },
    legend(series),
    lineChart({
      series,
      x: { domain: [0, 3], ticks: [0, 1, 2, 3], format: (t) => (t ? `Iteration ${t}` : "Untrained") },
      y: { domain: [0, 0.5], ticks: [0, 0.1, 0.2, 0.3, 0.4, 0.5], format: (t) => dec(t, 1) },
      height: 220,
      tooltipTitle: (x) => (x ? `After iteration ${x}` : "Untrained"),
      valueFormat: (v) => dec(v, 2),
    }),
    h("p", { class: "small muted" },
      "Run 2 starts higher because it uses the paper's tagged prompt, which the untrained model follows more often."),
  );
}

function funnel(run) {
  const stages = [
    ["Sampled", (it) => it.sampled],
    ["Parsed", (it) => it.valid],
    ["Unique", (it) => it.unique],
    ["Solved at least once", (it) => it.solved_once],
    ["Always solved (dropped)", (it) => it.dropped || null],
    ["Curriculum", (it) => (it.curriculum ? it.curriculum.easy + it.curriculum.medium + it.curriculum.hard : null)],
  ];
  return h("div", { class: "table-wrap" },
    h("table", { class: "table" },
      h("thead", {}, h("tr", {}, h("th", {}, "Stage"), run.iterations.map((it) => h("th", {}, `Iter ${it.iteration}`)))),
      h("tbody", {},
        stages.map(([label, get]) =>
          h("tr", {},
            h("td", {}, label),
            run.iterations.map((it) => {
              const v = get(it);
              return h("td", {}, v == null ? h("span", { class: "muted" }, label.startsWith("Always") ? "kept" : "–") : int(v));
            }),
          ),
        ),
      ),
    ),
  );
}

function solveHistograms(run) {
  // Tasks never solved (0 of 8) are dropped and far outnumber the rest, so they are
  // reported as a number rather than a bar that would flatten the others.
  const max = Math.max(...run.iterations.flatMap((it) => (it.solves_of_8 ?? [0]).slice(1)));
  return h("div", { class: "stack" },
    legend([
      { name: "Sometimes solved (useful signal)", color: "var(--series-1)" },
      { name: "Always solved (no signal)", color: "var(--series-2)" },
    ]),
    h("div", { class: "hist-grid" },
      run.iterations.map((it) => {
        const counts = it.solves_of_8 ?? [];
        const solved = counts.slice(1);
        const total = solved.reduce((a, b) => a + b, 0) || 1;
        return h("div", {},
          h("div", { class: "small" }, `Iteration ${it.iteration}`),
          h("div", { class: "small muted" }, `${int(counts[0])} never solved`),
          columnChart({
            bins: solved.map((v, i) => {
              const k = i + 1;
              return {
                label: k === 1 || k === 4 || k === 8 ? String(k) : "",
                value: v,
                color: k === 8 ? "var(--series-2)" : "var(--series-1)",
                showValue: k === 8,
                tooltip: tooltipBody(`Solved ${k} of 8`, [["Tasks", int(v)], ["Share of solved", pct(v / total, 0)]]),
              };
            }),
            max,
            height: 150,
            xTitle: "solves of 8",
          }),
        );
      }),
    ),
  );
}

function rewardCurves(run, role) {
  const defs = REWARD_SERIES[role];
  // Smooth within each iteration only, so one iteration's rewards never leak into the next.
  const series = defs.map((d) => ({
    name: d.name,
    color: d.color,
    points: run.iterations.flatMap((it, i) =>
      smooth(it[role].map((step, j) => ({ x: i * STEPS_PER_PHASE + j + 1, y: step[d.key] ?? null })), 5)),
  }));
  const yMax = role === "generator" ? 2 : 1;
  return h("div", { class: "stack" },
    legend(series),
    lineChart({
      series,
      x: {
        domain: [1, run.iterations.length * STEPS_PER_PHASE],
        ticks: run.iterations.map((_, i) => i * STEPS_PER_PHASE + 25),
        format: (t) => `Iteration ${Math.ceil(t / STEPS_PER_PHASE)}`,
      },
      y: { domain: [0, yMax], ticks: yMax === 2 ? [0, 0.5, 1, 1.5, 2] : [0, 0.25, 0.5, 0.75, 1], format: (t) => dec(t, t % 1 ? 2 : 0) },
      marks: run.iterations.slice(1).map((_, i) => ({ x: (i + 1) * STEPS_PER_PHASE + 0.5 })),
      height: 260,
      tooltipTitle: (x) => `Iteration ${Math.ceil(x / STEPS_PER_PHASE)}, step ${((x - 1) % STEPS_PER_PHASE) + 1}`,
      valueFormat: (v) => dec(v, 2),
    }),
  );
}

/** Trailing moving average over the last `n` non-missing values. */
function smooth(points, n) {
  const window = [];
  return points.map((p) => {
    if (p.y != null) window.push(p.y);
    if (window.length > n) window.shift();
    return { x: p.x, y: window.length ? window.reduce((a, b) => a + b, 0) / window.length : null };
  });
}
