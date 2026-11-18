// Tiny element builders. h() for HTML, s() for SVG; both take (tag, props, ...children).
//
//   h("button", { class: "x", onclick: fn, "aria-pressed": true }, "Label")
//
// Props: `class`, `style` (object), `on<event>` (listener), `dataset` (object), anything
// else becomes an attribute (false/null/undefined are skipped). Children may be nested
// arrays, strings, numbers, nodes, or null/false (skipped).

const SVG_NS = "http://www.w3.org/2000/svg";

export function h(tag, props = {}, ...children) {
  return build(document.createElement(tag), props, children);
}

export function s(tag, props = {}, ...children) {
  return build(document.createElementNS(SVG_NS, tag), props, children);
}

function build(el, props, children) {
  for (const [key, value] of Object.entries(props ?? {})) {
    if (value === false || value == null) continue;
    if (key === "style" && typeof value === "object") Object.assign(el.style, value);
    else if (key === "dataset") Object.assign(el.dataset, value);
    else if (key.startsWith("on") && typeof value === "function") el.addEventListener(key.slice(2), value);
    else el.setAttribute(key, value === true ? "" : String(value));
  }
  append(el, children);
  return el;
}

function append(el, children) {
  for (const child of children) {
    if (child == null || child === false) continue;
    if (Array.isArray(child)) append(el, child);
    else el.append(child instanceof Node ? child : String(child));
  }
}

/** Replace an element's children. */
export function mount(el, ...children) {
  el.replaceChildren();
  append(el, children);
  return el;
}

/**
 * A segmented control. Calls onChange(value) on selection; returns the element.
 * options: [{ value, label }]
 */
export function segmented(options, selected, onChange, label) {
  const root = h("div", { class: "segmented", role: "group", "aria-label": label });
  const buttons = options.map((opt) =>
    h("button", {
      type: "button",
      "aria-pressed": String(opt.value === selected),
      onclick: () => {
        buttons.forEach((b, i) => b.setAttribute("aria-pressed", String(options[i].value === opt.value)));
        onChange(opt.value);
      },
    }, opt.label),
  );
  root.append(...buttons);
  return root;
}

/** Re-run draw(width) whenever the element's width changes. */
export function onResize(el, draw) {
  let last = 0;
  const observer = new ResizeObserver(([entry]) => {
    const width = Math.round(entry.contentRect.width);
    if (width && width !== last) {
      last = width;
      draw(width);
    }
  });
  observer.observe(el);
  return observer;
}
