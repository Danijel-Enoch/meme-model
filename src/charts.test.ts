import { expect, test, describe } from "bun:test";
import {
  sparkline, lineChart, dualLineChart, candleChart, axisLabels,
  regimeStrip, positionStrip, meter,
} from "./charts";
import type { Candle } from "./features";

const mkCandle = (t: number, o: number, h: number, l: number, c: number, v = 100): Candle =>
  ({ time: t, open: o, high: h, low: l, close: c, volume: v });

describe("sparkline", () => {
  test("8 evenly spaced values on an 8-wide chart hit every ramp step in order", () => {
    // n === width, so this is a 1:1 mapping with no aggregation — the
    // clearest possible golden case: it should walk the ramp start to end.
    expect(sparkline([0, 1, 2, 3, 4, 5, 6, 7], 8)).toBe("▁▂▃▄▅▆▇█");
  });

  test("flat series sits mid-ramp, not at an edge", () => {
    expect(sparkline([5, 5, 5, 5, 5], 5)).toBe("▅▅▅▅▅");
  });

  test("empty input is all gaps, still exactly `width` wide", () => {
    expect(sparkline([], 5)).toBe("     ");
    expect(sparkline([], 5).length).toBe(5);
  });

  test("width of 1 does not throw and returns exactly one glyph", () => {
    expect(sparkline([1, 2, 3], 1).length).toBe(1);
  });

  test("a single point (n=1) does not throw and fills the requested width", () => {
    const s = sparkline([42], 6);
    expect(s.length).toBe(6);
    expect(s).not.toMatch(/\s/); // a lone finite point is never a "gap"
  });

  test("NaN/Infinity entries render as gaps (spaces), not a broken glyph", () => {
    const s = sparkline([1, NaN, 2, Infinity, 3], 5);
    expect(s.length).toBe(5);
    expect(s).not.toMatch(/[^\x20-\x7E▁▂▃▄▅▆▇█]/); // only ASCII space or a ramp glyph
  });

  test("all-NaN input is all gaps and never emits a glyph", () => {
    expect(sparkline([NaN, NaN, NaN], 4)).toBe("    ");
  });

  test("a spike shorter than one column survives downsampling", () => {
    // 20 zeros with a single 100 at index 9, squeezed into 5 columns (bucket
    // size 4). A naive "last value per bucket" or "mean per bucket" would
    // wash the spike out (mean of [0,100,0,0] is 25, still low-ramp); picking
    // the bucket's own outlier finds it. Bucket 2 (indices 8..11) is the one
    // that contains it, so only the middle column should light up brightly.
    const values = Array(20).fill(0);
    values[9] = 100;
    const s = sparkline(values, 5);
    expect(s[2]).toBe("█");
    expect(s[0]).toBe("▁");
    expect(s[4]).toBe("▁");
  });

  test("more columns than data upsamples without gaps", () => {
    const s = sparkline([1, 2], 6);
    expect(s.length).toBe(6);
    expect(s).not.toMatch(/\s/);
  });
});

describe("lineChart", () => {
  test("20-point monotonic ramp on a 10x3 chart: exact golden", () => {
    // Verified by hand: with height=3 the vertical scale is 12 sub-levels
    // (4 per row). Row 0 (top) only lights up in the rightmost columns,
    // where the ramp reaches its highest values; row 2 (bottom) only in the
    // leftmost columns, where it's lowest. That's the "monotonic in, stairs
    // out" shape a line chart should produce.
    const rows = lineChart(Array.from({ length: 20 }, (_, i) => i), { width: 10, height: 3 });
    expect(rows).toEqual([
      "       ⣀⠤⠊",
      "   ⣀⠤⠒⠉   ",
      "⡠⠒⠉       ",
    ]);
  });

  test("every row is exactly `width` and there are exactly `height` rows", () => {
    for (const [w, h] of [[1, 1], [1, 5], [5, 1], [7, 4], [60, 12]] as const) {
      const rows = lineChart([1, 5, 2, 8, 3], { width: w, height: h });
      expect(rows.length).toBe(h);
      for (const r of rows) expect(r.length).toBe(w);
    }
  });

  test("empty input renders `height` blank rows of `width` spaces, no throw", () => {
    const rows = lineChart([], { width: 5, height: 2 });
    expect(rows).toEqual(["     ", "     "]);
  });

  test("a single point does not throw and centers on a flat padded range", () => {
    const rows = lineChart([42], { width: 5, height: 2 });
    expect(rows.length).toBe(2);
    for (const r of rows) expect(r.length).toBe(5);
    expect(rows.join("")).not.toMatch(/NaN/);
  });

  test("all-equal values (flat series) do not throw or divide by zero", () => {
    const rows = lineChart([7, 7, 7, 7, 7], { width: 6, height: 3 });
    expect(rows.length).toBe(3);
    // Exactly one row should carry the flat line; the padding keeps it off
    // both edges.
    const nonBlank = rows.filter((r) => r.trim().length > 0);
    expect(nonBlank.length).toBeGreaterThanOrEqual(1);
  });

  test("NaN/Infinity entries leave a blank gap column instead of a bogus glyph", () => {
    // Column 2 (NaN) and column 5 (Infinity) are single-source columns
    // (n === width), so they must render fully blank in both rows.
    const rows = lineChart([0, 1, NaN, 3, 4, Infinity, 6, 7], { width: 8, height: 2 });
    for (const r of rows) {
      expect(r.length).toBe(8);
      expect(r[2]).toBe(" ");
      expect(r[5]).toBe(" ");
    }
  });

  test("width=1 and height=1 do not throw", () => {
    const rows = lineChart([1, 2, 3], { width: 1, height: 1 });
    expect(rows.length).toBe(1);
    expect(rows[0].length).toBe(1);
  });

  test("more columns than data points (upsample) still fills exactly `width`", () => {
    const rows = lineChart([1, 2], { width: 9, height: 2 });
    expect(rows.length).toBe(2);
    for (const r of rows) expect(r.length).toBe(9);
  });

  test("axis:true reserves exactly one leading column and keeps total width exact", () => {
    const rows = lineChart(Array.from({ length: 10 }, (_, i) => i), { width: 10, height: 2, axis: true });
    for (const r of rows) {
      expect(r.length).toBe(10);
      expect(r[0]).toBe("│");
    }
  });

  test("axis:true at width=1 is dropped rather than overflowing to width 2", () => {
    const rows = lineChart([1, 2, 3], { width: 1, height: 2, axis: true });
    for (const r of rows) expect(r.length).toBe(1);
  });

  test("monotonic series: the highest ink is in the last column", () => {
    // Scan the top row (the row that can only be reached by the maximum
    // value) and confirm the rightmost data column is the one that reaches
    // it — a stand-in for "visually monotonic" that doesn't depend on
    // reading individual braille dot bits.
    const width = 12, height = 4;
    const rows = lineChart(Array.from({ length: 40 }, (_, i) => i), { width, height });
    const topRow = rows[0];
    const inkCols = [...topRow].map((ch, i) => (ch !== " " ? i : -1)).filter((i) => i >= 0);
    expect(inkCols.length).toBeGreaterThan(0);
    expect(Math.max(...inkCols)).toBe(width - 1);
    // and the top row must have no ink in the first half of the chart —
    // the series hasn't gotten anywhere near its max yet there.
    expect(inkCols.every((i) => i >= width / 2)).toBe(true);
  });
});

describe("dualLineChart", () => {
  test("returns exactly height rows of width chars", () => {
    const a = Array.from({ length: 12 }, (_, i) => i);
    const b = Array.from({ length: 12 }, (_, i) => i * 0.5);
    const rows = dualLineChart(a, b, { width: 10, height: 3 });
    expect(rows.length).toBe(3);
    for (const r of rows) expect(r.length).toBe(10);
  });

  test("both series share one scale: b never exceeds a's own range assumption", () => {
    // If b used its own scale, a flat b at the same absolute level as a's
    // midpoint would land at a different visual row than a plotted alone at
    // that level. Cross-check: plotting b alone (as `a`) over the *combined*
    // range should land its ink in the same rows as it does inside the dual
    // chart, proving one shared axis is in effect.
    const a = [0, 100];
    const b = [50, 50];
    const dual = dualLineChart(a, b, { width: 4, height: 4 });
    const soloBOnCombinedScale = lineChart(b, { width: 4, height: 4, min: 0, max: 100 });
    // Every row lit in the solo (combined-scale) b render must also be lit
    // somewhere in the dual render (b's ink is a subset, since a adds more).
    for (let r = 0; r < 4; r++) {
      const soloHasInk = soloBOnCombinedScale[r].trim().length > 0;
      if (soloHasInk) expect(dual[r].trim().length).toBeGreaterThan(0);
    }
  });

  test("empty inputs do not throw", () => {
    const rows = dualLineChart([], [], { width: 5, height: 2 });
    expect(rows).toEqual(["     ", "     "]);
  });

  test("NaN/Infinity in either series do not produce broken glyphs", () => {
    const rows = dualLineChart([1, NaN, 3], [Infinity, 2, 3], { width: 6, height: 2 });
    expect(rows.length).toBe(2);
    for (const r of rows) expect(r.length).toBe(6);
  });

  test("width=1, height=1 do not throw", () => {
    const rows = dualLineChart([1, 2], [3, 4], { width: 1, height: 1 });
    expect(rows).toEqual([rows[0]]);
    expect(rows[0].length).toBe(1);
  });
});

describe("candleChart", () => {
  // Five hand-built candles, strictly rising, alternating up/down bars so
  // both body glyphs appear: (open,high,low,close)
  //   0: 10,12,9,11   (up,   body 10-11)
  //   1: 11,14,10,10.5 (down, body 10.5-11)
  //   2: 12,16,11,15   (up,   body 12-15)
  //   3: 15,17,13,13.5 (down, body 13.5-15)
  //   4: 14,20,14,19   (up,   body 14-19)
  const candles: Candle[] = [
    mkCandle(0, 10, 12, 9, 11),
    mkCandle(1, 11, 14, 10, 10.5),
    mkCandle(2, 12, 16, 11, 15),
    mkCandle(3, 15, 17, 13, 13.5),
    mkCandle(4, 14, 20, 14, 19),
  ];

  test("one column per candle (n === width) is exactly height x width", () => {
    const rows = candleChart(candles, { width: 5, height: 8 });
    expect(rows.length).toBe(8);
    for (const r of rows) expect(r.length).toBe(5);
  });

  test("up bars use the solid body, down bars use the hollow body", () => {
    const rows = candleChart(candles, { width: 5, height: 20 });
    const colHasGlyph = (c: number, glyph: string) => rows.some((r) => r[c] === glyph);
    // Candle 0 and 2 and 4 are up -> solid body somewhere in their column.
    expect(colHasGlyph(0, "█")).toBe(true);
    expect(colHasGlyph(2, "█")).toBe(true);
    expect(colHasGlyph(4, "█")).toBe(true);
    // Candle 1 and 3 are down -> hollow body somewhere in their column.
    expect(colHasGlyph(1, "░")).toBe(true);
    expect(colHasGlyph(3, "░")).toBe(true);
  });

  test("wick extends above/below the body using │", () => {
    // Candle 4: high=20, body top=19 — there must be a │ above the body.
    const rows = candleChart(candles, { width: 5, height: 40 });
    const col = 4;
    const glyphsInCol = rows.map((r) => r[col]);
    expect(glyphsInCol.includes("│")).toBe(true);
    expect(glyphsInCol.includes("█")).toBe(true);
  });

  test("no candles: blank grid, no throw", () => {
    const rows = candleChart([], { width: 6, height: 3 });
    expect(rows).toEqual(["      ", "      ", "      "]);
  });

  test("a single flat candle (open=high=low=close) does not throw or divide by zero", () => {
    const rows = candleChart([mkCandle(0, 10, 10, 10, 10)], { width: 3, height: 5 });
    expect(rows.length).toBe(5);
    for (const r of rows) expect(r.length).toBe(3);
    expect(rows.join("")).not.toMatch(/NaN/);
    // exactly one row should show the flat body, the rest blank.
    const bodyRows = rows.filter((r) => r.includes("█"));
    expect(bodyRows.length).toBe(1);
  });

  test("width=1, height=1 do not throw", () => {
    const rows = candleChart(candles, { width: 1, height: 1 });
    expect(rows).toEqual([rows[0]]);
    expect(rows[0].length).toBe(1);
  });

  test("more candles than columns aggregates (min/max) rather than dropping a spike", () => {
    // 20 flat candles at 10, with one huge-range candle stuck in the middle
    // (high=100). Downsampled to 5 columns, the bucket containing index 10
    // must still show the spike reaching the top row.
    const flat = Array.from({ length: 20 }, (_, i) => mkCandle(i, 10, 10.5, 9.5, 10));
    flat[10] = mkCandle(10, 10, 100, 9.5, 10);
    const rows = candleChart(flat, { width: 5, height: 10 });
    // bucket 2 (indices 8..11) owns index 10; its column should reach row 0.
    expect(rows[0][2]).not.toBe(" ");
  });

  test("fewer candles than columns (upsample) still fills exactly `width`", () => {
    const rows = candleChart(candles.slice(0, 2), { width: 9, height: 4 });
    for (const r of rows) expect(r.length).toBe(9);
  });

  test("axis:true reserves exactly one leading column", () => {
    const rows = candleChart(candles, { width: 6, height: 4, axis: true });
    for (const r of rows) {
      expect(r.length).toBe(6);
      expect(r[0]).toBe("│");
    }
  });
});

describe("axisLabels", () => {
  test("returns exactly `height` labels", () => {
    expect(axisLabels(0, 10, 5).length).toBe(5);
    expect(axisLabels(0, 10, 1).length).toBe(1);
    expect(axisLabels(0, 10, 0).length).toBe(0);
  });

  test("all labels share one padded width, right-aligned", () => {
    const labels = axisLabels(0, 12345, 4);
    const w = labels[0].length;
    for (const l of labels) {
      expect(l.length).toBe(w);
      expect(l).toBe(l.trimStart().padStart(w)); // right-aligned: no gaps except leading
    }
  });

  test("micro-scale range (1e-6ish) reads as a real decimal, not 0.00 or exponential", () => {
    // Exact spec example: 0.00000381 at the low end of a near-zero-width
    // range must round-trip legibly.
    const labels = axisLabels(0.00000381, 0.00000381, 1);
    expect(labels[0].trim()).toBe("0.00000381");
    expect(labels[0]).not.toMatch(/e[-+]/i);
  });

  test("macro-scale range (1e5ish) reads as an integer-ish price, not drowned in decimals", () => {
    const labels = axisLabels(77132, 77132, 1);
    expect(labels[0].trim()).toBe("77132");
  });

  test("micro- and macro-scale ranges produce genuinely different formatting", () => {
    const micro = axisLabels(0.0000038, 0.0000042, 3).map((s) => s.trim());
    const macro = axisLabels(70000, 77132, 3).map((s) => s.trim());
    // distinguishable: every micro label has a decimal point with several
    // digits, every macro label is a short integer.
    for (const l of micro) expect(l).toMatch(/^0\.\d{6,}$/);
    for (const l of macro) expect(l).toMatch(/^\d{5}$/);
    // and within each, rows must actually differ (not all collapsed to one
    // value by rounding).
    expect(new Set(micro).size).toBeGreaterThan(1);
    expect(new Set(macro).size).toBeGreaterThan(1);
  });

  test("degenerate range (min === max) does not throw or divide by zero", () => {
    const labels = axisLabels(5, 5, 3);
    expect(labels.length).toBe(3);
    for (const l of labels) expect(l).not.toMatch(/NaN/);
  });

  test("NaN/Infinity bounds do not throw", () => {
    expect(() => axisLabels(NaN, NaN, 3)).not.toThrow();
    expect(() => axisLabels(-Infinity, Infinity, 3)).not.toThrow();
    for (const l of axisLabels(NaN, NaN, 3)) expect(l).not.toMatch(/NaN/);
  });

  test("row 0 is the max, the last row is the min", () => {
    const labels = axisLabels(0, 100, 5).map((s) => Number(s));
    expect(labels[0]).toBe(100);
    expect(labels[4]).toBe(0);
    // strictly decreasing top to bottom
    for (let i = 1; i < labels.length; i++) expect(labels[i]).toBeLessThan(labels[i - 1]);
  });
});

describe("regimeStrip", () => {
  test("one glyph per bar, ordered bearish-to-bullish, exact golden", () => {
    // K=3: state 0 -> ramp[0]='▁', state 1 -> ramp mid ('▅' at round(0.5*7)=4),
    // state 2 -> ramp[7]='█'. Downsampling with bucket>1 takes the bucket's
    // most recent state.
    expect(regimeStrip([0, 0, 1, 1, 2, 2], 3, 6)).toBe("▁▁▅▅██");
  });

  test("categorical downsample (n > width) takes the last state per bucket", () => {
    // 6 states into 3 columns: buckets are [0,1],[1,1... wait bucket sizes
    // are 2 each: [0,0]->0, [1,1]->1, [2,2]->2.
    expect(regimeStrip([0, 0, 1, 1, 2, 2], 3, 3)).toBe("▁▅█");
  });

  test("out-of-range or NaN states render as a blank gap, not a wrong glyph", () => {
    const s = regimeStrip([0, NaN, -1, 5], 3, 4);
    expect(s[0]).toBe("▁");
    expect(s[1]).toBe(" ");
    expect(s[2]).toBe(" "); // negative state, no data
    expect(s[3]).toBe(" "); // state 5 >= K=3, out of range
  });

  test("K=1 does not throw (single state maps to the top of the ramp)", () => {
    expect(regimeStrip([0, 0, 0], 1, 3)).toBe("███");
  });

  test("empty input is all gaps at exactly `width`", () => {
    expect(regimeStrip([], 3, 5)).toBe("     ");
  });

  test("width=1 does not throw", () => {
    expect(regimeStrip([0, 1, 2], 3, 1).length).toBe(1);
  });
});

describe("positionStrip", () => {
  test("long/flat/short glyphs, exact golden", () => {
    expect(positionStrip([1, -1, 0, 2, -0.5], 5)).toBe("▲▼·▲▼");
  });

  test("NaN renders as a blank gap", () => {
    expect(positionStrip([1, NaN, -1], 3)).toBe("▲ ▼");
  });

  test("empty input is all gaps at exactly `width`", () => {
    expect(positionStrip([], 4)).toBe("    ");
  });

  test("downsampling takes the bucket's most recent position", () => {
    expect(positionStrip([1, 1, -1, -1], 2)).toBe("▲▼");
  });

  test("width=1 does not throw", () => {
    expect(positionStrip([1, -1, 0], 1).length).toBe(1);
  });
});

describe("meter", () => {
  test("exact golden for the documented example", () => {
    expect(meter(0.62, 6)).toBe("████░░ 62%");
  });

  test("label is prefixed when given", () => {
    expect(meter(0.62, 6, "confidence")).toBe("confidence ████░░ 62%");
  });

  test("0 and 1 are the empty and full bar", () => {
    expect(meter(0, 4)).toBe("░░░░ 0%");
    expect(meter(1, 4)).toBe("████ 100%");
  });

  test("out-of-range fractions clamp instead of throwing or overflowing the bar", () => {
    expect(meter(1.5, 4)).toBe("████ 100%");
    expect(meter(-0.5, 4)).toBe("░░░░ 0%");
  });

  test("NaN fraction does not throw and renders as empty", () => {
    expect(meter(NaN, 4)).toBe("░░░░ 0%");
  });

  test("width=0 does not throw", () => {
    expect(meter(0.5, 0)).toBe(" 50%");
  });
});
