// One shared tooltip, positioned next to the pointer and kept inside the viewport.

import { h, mount } from "./dom.js";

let el;

function element() {
  if (!el) {
    el = h("div", { class: "tooltip", role: "tooltip", hidden: true });
    document.body.append(el);
  }
  return el;
}

/** Show content (node, string or array) near a pointer event. */
export function showTooltip(event, content) {
  const tip = element();
  mount(tip, content);
  tip.hidden = false;
  const pad = 12;
  const { innerWidth: vw, innerHeight: vh } = window;
  const rect = tip.getBoundingClientRect();
  let x = event.clientX + pad;
  let y = event.clientY + pad;
  if (x + rect.width > vw - pad) x = event.clientX - rect.width - pad;
  if (y + rect.height > vh - pad) y = event.clientY - rect.height - pad;
  tip.style.transform = `translate(${Math.max(pad, x)}px, ${Math.max(pad, y)}px)`;
}

export function hideTooltip() {
  if (el) el.hidden = true;
}

/** A tooltip body: a title line plus [label, value, color?] rows. */
export function tooltipBody(title, rows = []) {
  return [
    h("div", { class: "tooltip__title" }, title),
    ...rows.map(([label, value, color]) =>
      h("div", { class: "tooltip__row" },
        color ? h("span", { class: "swatch", style: { background: color } }) : null,
        h("span", { class: "tooltip__label" }, label),
        h("span", { class: "tooltip__value" }, value),
      ),
    ),
  ];
}
