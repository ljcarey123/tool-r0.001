// Entry point: render each section into the page in order. A section that fails to load
// shows an error in its place instead of breaking the rest of the page.

import { h, mount } from "./lib/dom.js";
import { errorBox } from "./lib/layout.js";
import * as overview from "./sections/overview.js";
import * as architecture from "./sections/architecture.js";
import * as training from "./sections/training.js";
import * as results from "./sections/results.js";
import * as examples from "./sections/examples.js";
import { initTheme } from "./theme.js";

const SECTIONS = [overview, architecture, results, training, examples];

async function main() {
  initTheme(document.querySelector("[data-theme-toggle]"));
  const root = document.getElementById("app");
  const slots = SECTIONS.map(() => h("div"));
  mount(root, slots);
  await Promise.all(SECTIONS.map(async (mod, i) => {
    try {
      mount(slots[i], await mod.render());
    } catch (err) {
      console.error(err);
      mount(slots[i], errorBox(`This section failed to load: ${err.message}`));
    }
  }));
  trackActiveNav();
}

/** Highlight the nav link of the section in view. */
function trackActiveNav() {
  const links = new Map([...document.querySelectorAll(".topbar__nav a")].map((a) => [a.hash.slice(1), a]));
  const observer = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      links.forEach((a, id) => a.classList.toggle("is-active", id === entry.target.id));
    }
  }, { rootMargin: "-40% 0px -55% 0px" });
  links.forEach((_, id) => {
    const el = document.getElementById(id);
    if (el) observer.observe(el);
  });
}

main();
