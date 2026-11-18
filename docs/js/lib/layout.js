// Page-level building blocks shared by every section.

import { h } from "./dom.js";
import { codeUrl } from "../config.js";

/** A page section with the standard heading block. */
export function section({ id, kicker, title, lede }, ...children) {
  return h("section", { class: "section", id, "aria-labelledby": `${id}-title` },
    h("header", { class: "section__head" },
      kicker ? h("div", { class: "section__kicker" }, kicker) : null,
      h("h2", { id: `${id}-title` }, title),
      lede ? h("p", { class: "section__lede" }, lede) : null,
    ),
    h("div", { class: "stack" }, children),
  );
}

export function card({ title, aside, note }, ...children) {
  return h("div", { class: "card" },
    title ? h("div", { class: "card__title" }, h("h3", {}, title), aside ?? null) : null,
    note ? h("p", { class: "card__note" }, note) : null,
    children,
  );
}

export function codeLinks(paths) {
  return h("div", { class: "controls" },
    paths.map((p) => h("a", { class: "file-link", href: codeUrl(p), target: "_blank", rel: "noopener" }, p)),
  );
}

export function errorBox(message) {
  return h("div", { class: "error", role: "alert" }, message);
}
