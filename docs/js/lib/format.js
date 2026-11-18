// Number formatting used across sections.

export const pct = (x, digits = 1) => (x == null ? "–" : `${(x * 100).toFixed(digits)}%`);
export const dec = (x, digits = 3) => (x == null ? "–" : x.toFixed(digits));
export const int = (x) => (x == null ? "–" : Math.round(x).toLocaleString("en"));

/** A p-value, reported the way the README does. */
export function pValue(p) {
  if (p == null) return "–";
  if (p < 0.001) return "p < 0.001";
  return `p = ${p.toFixed(p < 0.01 ? 3 : 2)}`;
}

export const significant = (p) => p != null && p < 0.05;
