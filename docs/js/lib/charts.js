// Chart builders. Each returns an element; SVG charts redraw to fit their container width.
// Colours arrive as CSS custom-property references (e.g. "var(--series-1)") so charts
// follow the light/dark tokens without knowing about them.

import { h, s, mount, onResize } from "./dom.js";
import { showTooltip, hideTooltip, tooltipBody } from "./tooltip.js";

const linear = ([d0, d1], [r0, r1]) => (v) => r0 + ((v - d0) / (d1 - d0 || 1)) * (r1 - r0);

/**
 * Horizontal labelled bars: the label, bar and value sit on one row, so every bar is
 * directly labelled. rows: [{ label, detail?, value, color, tooltip? }]
 */
export function barList({ rows, max = 1, format = String }) {
  return h("div", { class: "bars" },
    rows.map((row) =>
      h("div", {
        class: "bars__row",
        onpointermove: row.tooltip ? (e) => showTooltip(e, row.tooltip) : null,
        onpointerleave: hideTooltip,
      },
        h("div", { class: "bars__label" },
          h("span", { class: "swatch", style: { background: row.color } }),
          h("span", {}, row.label),
          row.detail ? h("span", { class: "bars__detail" }, row.detail) : null,
        ),
        h("div", { class: "bars__track" },
          h("div", { class: "bars__fill", style: { width: `${(100 * row.value) / max}%`, background: row.color } }),
        ),
        h("div", { class: "bars__value" }, format(row.value)),
      ),
    ),
  );
}

/**
 * Line chart with a crosshair tooltip.
 *   series: [{ name, color, points: [{ x, y }], dashed?, markers? }]
 *   x: { domain, ticks, format, title? }   y: { domain, ticks, format }
 *   marks: [{ x, label }] vertical reference lines (e.g. iteration boundaries)
 */
export function lineChart({ series, x, y, marks = [], height = 240, tooltipTitle = String, valueFormat }) {
  const root = h("div", { class: "chart" });
  const fmt = valueFormat ?? y.format ?? String;
  const xs = [...new Set(series.flatMap((sr) => sr.points.map((p) => p.x)))].sort((a, b) => a - b);

  onResize(root, (width) => {
    const m = { top: 12, right: 16, bottom: x.title ? 40 : 28, left: 44 };
    const X = linear(x.domain, [m.left, width - m.right]);
    const Y = linear(y.domain, [height - m.bottom, m.top]);
    const path = (pts) => pts.filter((p) => p.y != null)
      .map((p, i) => `${i ? "L" : "M"}${X(p.x).toFixed(1)},${Y(p.y).toFixed(1)}`).join("");

    const crosshair = s("line", { class: "chart__crosshair", y1: m.top, y2: height - m.bottom, visibility: "hidden" });
    const dots = s("g");

    const svg = s("svg", { width, height, viewBox: `0 0 ${width} ${height}`, role: "img" },
      // grid + y axis
      y.ticks.map((t) => s("g", {},
        s("line", { class: "chart__grid", x1: m.left, x2: width - m.right, y1: Y(t), y2: Y(t) }),
        s("text", { class: "chart__tick", x: m.left - 8, y: Y(t), "text-anchor": "end", "dominant-baseline": "middle" }, y.format(t)),
      )),
      s("line", { class: "chart__axis", x1: m.left, x2: width - m.right, y1: Y(y.domain[0]), y2: Y(y.domain[0]) }),
      // x ticks
      x.ticks.map((t) => s("text", { class: "chart__tick", x: X(t), y: height - m.bottom + 18, "text-anchor": "middle" }, x.format(t))),
      x.title ? s("text", { class: "chart__tick", x: (m.left + width - m.right) / 2, y: height - 4, "text-anchor": "middle" }, x.title) : null,
      // reference marks
      marks.map((mk) => s("g", {},
        s("line", { class: "chart__mark", x1: X(mk.x), x2: X(mk.x), y1: m.top, y2: height - m.bottom }),
        mk.label ? s("text", { class: "chart__mark-label", x: X(mk.x) + 4, y: m.top + 10 }, mk.label) : null,
      )),
      // series
      series.map((sr) => s("g", {},
        s("path", { class: "chart__line", d: path(sr.points), stroke: sr.color, "stroke-dasharray": sr.dashed ? "4 3" : null }),
        sr.markers ? sr.points.filter((p) => p.y != null).map((p) =>
          s("circle", { class: "chart__marker", cx: X(p.x), cy: Y(p.y), r: 4, fill: sr.color })) : null,
      )),
      crosshair,
      dots,
    );

    // Hover layer: snap to the nearest x that has data.
    const hit = s("rect", { x: m.left, y: m.top, width: width - m.left - m.right, height: height - m.top - m.bottom, fill: "transparent" });
    hit.addEventListener("pointermove", (event) => {
      const px = event.offsetX;
      const nearest = xs.reduce((best, v) => (Math.abs(X(v) - px) < Math.abs(X(best) - px) ? v : best), xs[0]);
      crosshair.setAttribute("x1", X(nearest));
      crosshair.setAttribute("x2", X(nearest));
      crosshair.setAttribute("visibility", "visible");
      const rows = [];
      mount(dots);
      for (const sr of series) {
        const p = sr.points.find((q) => q.x === nearest);
        if (!p || p.y == null) continue;
        rows.push([sr.name, fmt(p.y), sr.color]);
        dots.append(s("circle", { class: "chart__marker", cx: X(nearest), cy: Y(p.y), r: 4, fill: sr.color }));
      }
      showTooltip(event, tooltipBody(tooltipTitle(nearest), rows));
    });
    hit.addEventListener("pointerleave", () => {
      crosshair.setAttribute("visibility", "hidden");
      mount(dots);
      hideTooltip();
    });
    svg.append(hit);
    mount(root, svg);
  });
  return root;
}

/**
 * Column chart for a small number of categories (a histogram).
 *   bins: [{ label, value, color, tooltip }]
 */
export function columnChart({ bins, max, height = 160, xTitle }) {
  const root = h("div", { class: "chart" });
  const top = max ?? Math.max(...bins.map((b) => b.value), 1);
  onResize(root, (width) => {
    const m = { top: 18, right: 4, bottom: xTitle ? 36 : 22, left: 4 };
    const band = (width - m.left - m.right) / bins.length;
    const Y = linear([0, top], [height - m.bottom, m.top]);
    const barW = Math.max(4, band - 4); // 2px surface gap each side
    const svg = s("svg", { width, height, viewBox: `0 0 ${width} ${height}`, role: "img" },
      s("line", { class: "chart__axis", x1: m.left, x2: width - m.right, y1: Y(0), y2: Y(0) }),
      bins.map((b, i) => {
        const x = m.left + i * band + (band - barW) / 2;
        const y0 = Y(b.value);
        const hgt = Math.max(0, Y(0) - y0);
        const r = Math.min(4, barW / 2, hgt);
        const d = hgt <= 0 ? "" :
          `M${x},${Y(0)}V${y0 + r}Q${x},${y0} ${x + r},${y0}H${x + barW - r}Q${x + barW},${y0} ${x + barW},${y0 + r}V${Y(0)}Z`;
        const g = s("g", { class: "chart__col" },
          s("rect", { x: m.left + i * band, y: m.top, width: band, height: height - m.top - m.bottom, fill: "transparent" }),
          s("path", { d, fill: b.color }),
          s("text", { class: "chart__tick", x: x + barW / 2, y: height - m.bottom + 14, "text-anchor": "middle" }, b.label),
          b.showValue ? s("text", { class: "chart__value", x: x + barW / 2, y: y0 - 5, "text-anchor": "middle" }, b.value) : null,
        );
        if (b.tooltip) {
          g.addEventListener("pointermove", (e) => showTooltip(e, b.tooltip));
          g.addEventListener("pointerleave", hideTooltip);
        }
        return g;
      }),
      xTitle ? s("text", { class: "chart__tick", x: width / 2, y: height - 4, "text-anchor": "middle" }, xTitle) : null,
    );
    mount(root, svg);
  });
  return root;
}

/** Legend row. items: [{ name, color, dashed? }] */
export function legend(items) {
  return h("div", { class: "legend" },
    items.map((it) => h("span", { class: "legend__item" },
      h("span", { class: "swatch", style: { background: it.color } }), it.name)),
  );
}
