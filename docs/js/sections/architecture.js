// The self-play loop as a clickable flow: pick a stage, read what it does and where the code is.

import { h, mount } from "../lib/dom.js";
import { section, card, codeLinks } from "../lib/layout.js";
import { STAGES, ADAPTERS } from "../content/stages.js";

export function render() {
  const detail = h("div", { class: "loop__detail card", "aria-live": "polite" });
  const buttons = STAGES.map((stage, i) =>
    h("button", {
      type: "button",
      class: "loop__stage",
      "aria-pressed": "false",
      onclick: () => select(i),
    },
      h("span", { class: "loop__index" }, i + 1),
      h("span", { class: "loop__text" },
        h("span", { class: "loop__title" }, stage.title),
        h("span", { class: "loop__short" }, stage.short),
      ),
      stage.adapter ? h("span", { class: "badge badge--neutral" }, stage.adapter) : null,
    ),
  );

  function select(i) {
    buttons.forEach((b, j) => b.setAttribute("aria-pressed", String(i === j)));
    const stage = STAGES[i];
    mount(detail,
      h("div", { class: "card__title" }, h("h3", {}, stage.title),
        stage.adapter ? h("span", { class: "badge badge--neutral" }, `${stage.adapter} adapter active`) : null),
      h("div", { class: "stack loop__body" },
        stage.body.map((p) => h("p", {}, p)),
        stage.reward ? rewardTable(stage.reward) : null,
        codeLinks(stage.code),
      ),
    );
  }
  select(0);

  return section(
    {
      id: "architecture",
      kicker: "Architecture",
      title: "The self-play loop",
      lede: "One iteration runs these five stages. The full run repeats it three times, each time with both adapters carried forward.",
    },
    h("div", { class: "loop" },
      h("div", { class: "loop__flow" },
        buttons,
        h("div", { class: "loop__repeat" }, "↻ repeat × 3 iterations"),
      ),
      detail,
    ),
    card({ title: ADAPTERS.title },
      h("div", { class: "stack" },
        ADAPTERS.body.map((p) => h("p", { class: "small" }, p)),
        h("pre", { class: "code" }, ADAPTERS.formula),
        codeLinks(ADAPTERS.code),
      ),
    ),
  );
}

function rewardTable(rows) {
  return h("div", { class: "table-wrap" },
    h("table", { class: "table" },
      h("thead", {}, h("tr", {}, h("th", {}, "Reward"), h("th", {}, "Range"), h("th", { style: { textAlign: "left" } }, "What it checks"))),
      h("tbody", {},
        rows.map(([name, range, what]) =>
          h("tr", {},
            h("td", {}, h("code", {}, name)),
            h("td", {}, range),
            h("td", { style: { textAlign: "left", whiteSpace: "normal" } }, what),
          ),
        ),
      ),
    ),
  );
}
