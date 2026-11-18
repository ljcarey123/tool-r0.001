// Light/dark toggle. Follows the OS until the viewer picks; the choice is remembered.

const KEY = "tool-r0-theme";

export function initTheme(button) {
  const saved = read();
  if (saved) document.documentElement.dataset.theme = saved;
  button?.addEventListener("click", () => {
    const dark = current() === "dark";
    const next = dark ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    write(next);
  });
}

function current() {
  const set = document.documentElement.dataset.theme;
  if (set) return set;
  return matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function read() {
  try { return localStorage.getItem(KEY); } catch { return null; }
}

function write(value) {
  try { localStorage.setItem(KEY, value); } catch { /* storage unavailable: theme lasts this visit */ }
}
