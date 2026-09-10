/**
 * Pure text chart renderers for the terminal UI.
 *
 * Hard rule: no I/O, no ANSI colour, no dependencies. Every function takes
 * plain numbers and returns plain strings — the UI layer decides how (or
 * whether) to colour them. That split is what makes these golden-testable:
 * a string is either exactly right or it isn't, with no terminal state to
 * fake.
 *
 * Shared machinery: most of these charts need the same two primitives —
 * "map N source samples onto W output columns" (bucketOf) and "map a value
 * onto [min,max] with headroom" (effectiveRange) — so both live once at the
 * top rather than being re-derived per function.
 */

import type { Candle } from "./features";

export interface ChartOpts {
  width: number; height: number;
  min?: number; max?: number;      // default: data range with a small pad
  /** Rows to reserve on the left for the y-axis gutter, default 0 (no axis). */
  axis?: boolean;
}

/**
 * Bucket boundaries for mapping `n` source samples onto `cols` output
 * columns — the one downsampling/upsampling rule every chart in this file
 * shares. When n > cols this aggregates (many samples -> one column, so a
 * spike inside the bucket can still be picked up by the caller instead of
 * being dropped by naive stride-sampling). When n <= cols it's a
 * nearest-neighbour upsample: consecutive columns can map to the same
 * source index, which is exactly what you want for "10 candles stretched
 * across 60 columns" rather than leaving 50 columns blank.
 */
function bucketOf(n: number, cols: number, i: number): [number, number] {
  if (n <= 0 || cols <= 0) return [0, 0];
  const start = Math.floor((i * n) / cols);
  const end = Math.max(start + 1, Math.floor(((i + 1) * n) / cols));
  return [start, Math.min(end, n)];
}

/**
 * Effective [min,max] for a chart. Explicit opts win. Otherwise it's the
 * finite data range plus a 5% pad so extremes don't sit flush against the
 * top/bottom row. A flat series (or a single point, or explicit min===max)
 * would otherwise divide by zero downstream, so that case gets a pad
 * relative to its own magnitude instead (an absolute epsilon would either
 * do nothing at 1e5 or dwarf the value at 1e-8).
 */
function effectiveRange(values: number[], optMin?: number, optMax?: number): [number, number] {
  let lo = Infinity, hi = -Infinity;
  for (const v of values) if (Number.isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; }
  if (!Number.isFinite(lo)) { lo = 0; hi = 1; } // no finite data anywhere
  let min = optMin ?? lo;
  let max = optMax ?? hi;
  if (!(max > min)) {
    const pad = Math.max(Math.abs(min) * 0.05, 1e-9);
    return [min - pad, max + pad];
  }
  if (optMin === undefined && optMax === undefined) {
    const pad = (max - min) * 0.05;
    min -= pad; max += pad;
  }
  return [min, max];
}

/** Value -> discrete level in [0, steps-1], 0 at the top (high value). NaN/Inf -> -1 (no dot). */
function levelOf(v: number, min: number, max: number, steps: number): number {
  if (!Number.isFinite(v)) return -1;
  const frac = Math.min(1, Math.max(0, (v - min) / (max - min)));
  return Math.round((1 - frac) * (steps - 1));
}

const BRAILLE_BASE = 0x2800;
// Dot bit for (subcol 0|1, subrow 0..3) in a 2x4 unicode braille cell. Dot
// numbering per the standard is 1,2,3,7 down the left column and 4,5,6,8
// down the right; the bit values below follow that layout.
const DOT_BIT: readonly [number, number, number, number][] = [
  [0x01, 0x02, 0x04, 0x40], // left column,  subrows 0..3
  [0x08, 0x10, 0x20, 0x80], // right column, subrows 0..3
];

/**
 * A width*height grid of braille cells addressed by subpixel coordinates
 * (2*width horizontal x 4*height vertical), which is what gives braille
 * charts their resolution advantage over one-glyph-per-value rendering.
 * Internal to this module — callers only ever see the rendered rows.
 */
class BrailleCanvas {
  private cells: Uint8Array;
  constructor(private width: number, private height: number) {
    this.cells = new Uint8Array(Math.max(0, width * height));
  }
  private idx(col: number, row: number): number { return row * this.width + col; }
  set(subcol: number, level: number): void {
    if (subcol < 0 || level < 0 || subcol >= this.width * 2 || level >= this.height * 4) return;
    const col = subcol >> 1, sc = subcol & 1;
    const row = level >> 2, sr = level & 3;
    this.cells[this.idx(col, row)] |= DOT_BIT[sc][sr];
  }
  /** Vertical run of dots in one subpixel column — the "wick", so a bucket's
   *  min/max survives downsampling even when it's neither first nor last. */
  vline(subcol: number, levelA: number, levelB: number): void {
    if (levelA < 0 || levelB < 0) return;
    const lo = Math.min(levelA, levelB), hi = Math.max(levelA, levelB);
    for (let l = lo; l <= hi; l++) this.set(subcol, l);
  }
  /** Bresenham between two subpixel points, so consecutive columns read as
   *  a connected line rather than a scatter of unrelated dots. */
  line(x0: number, y0: number, x1: number, y1: number): void {
    if (y0 < 0 || y1 < 0) { this.set(x1, y1); return; }
    const dx = Math.abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    const dy = -Math.abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    let err = dx + dy, x = x0, y = y0;
    for (;;) {
      this.set(x, y);
      if (x === x1 && y === y1) break;
      const e2 = 2 * err;
      if (e2 >= dy) { err += dy; x += sx; }
      if (e2 <= dx) { err += dx; y += sy; }
    }
  }
  rows(): string[] {
    const out: string[] = [];
    for (let r = 0; r < this.height; r++) {
      let s = "";
      for (let c = 0; c < this.width; c++) {
        const b = this.cells[this.idx(c, r)];
        s += b === 0 ? " " : String.fromCodePoint(BRAILLE_BASE + b);
      }
      out.push(s);
    }
    return out;
  }
}

/** How many columns of `width` to reserve for the axis rule. Only ever 1 —
 *  ChartOpts.axis is a boolean, not a gutter width — and never taken from a
 *  width of 1, since reserving a column there would leave zero for the
 *  chart itself and break the "returns exactly `width` chars" contract. */
function axisGutter(opts: ChartOpts): number {
  return opts.axis && opts.width > 1 ? 1 : 0;
}

const blankRows = (width: number, height: number): string[] =>
  Array.from({ length: Math.max(0, height) }, () => " ".repeat(Math.max(0, width)));

const SPARK_RAMP = "▁▂▃▄▅▆▇█";

/** One-line sparkline using ▁▂▃▄▅▆▇█. Handles flat series and NaN gaps. */
export function sparkline(values: number[], width: number): string {
  if (width <= 0) return "";
  const n = values.length;
  const [min, max] = effectiveRange(values);
  let out = "";
  for (let c = 0; c < width; c++) {
    const [a, b] = bucketOf(n, width, c);
    // Pick the bucket's own outlier (max deviation from the bucket's own
    // mean), not the mean itself or "last" — a one-bar spike surrounded by
    // flat bars would get averaged away or missed entirely by "last". Using
    // the bucket's own mean (rather than the chart's global midline) avoids
    // a tie when the spike and the baseline happen to be equidistant from
    // the global center (e.g. a lone 100 among a run of 0s, centered on 50).
    let sum = 0, cnt = 0;
    for (let i = a; i < b; i++) { const v = values[i]; if (Number.isFinite(v)) { sum += v; cnt++; } }
    if (cnt === 0) { out += " "; continue; }
    const localMean = sum / cnt;
    let best = NaN, bestDev = -1;
    for (let i = a; i < b; i++) {
      const v = values[i];
      if (!Number.isFinite(v)) continue;
      const dev = Math.abs(v - localMean);
      if (dev > bestDev) { bestDev = dev; best = v; }
    }
    if (!Number.isFinite(best)) { out += " "; continue; }
    const frac = Math.min(1, Math.max(0, (best - min) / (max - min)));
    out += SPARK_RAMP[Math.min(SPARK_RAMP.length - 1, Math.floor(frac * SPARK_RAMP.length))];
  }
  return out;
}

/**
 * Plot one series into a fresh (or shared) braille canvas: per output
 * subpixel column, aggregate the bucket to a mean (the representative
 * point, connected across columns) plus a min/max wick (so an in-bucket
 * spike still shows even though the mean would dilute it).
 */
function plotSolid(canvas: BrailleCanvas, values: number[], min: number, max: number, subCols: number, vSteps: number): void {
  let prevX = -1, prevY = -1;
  for (let s = 0; s < subCols; s++) {
    const [a, b] = bucketOf(values.length, subCols, s);
    let lo = Infinity, hi = -Infinity, sum = 0, cnt = 0;
    for (let i = a; i < b; i++) {
      const v = values[i];
      if (!Number.isFinite(v)) continue;
      if (v < lo) lo = v; if (v > hi) hi = v; sum += v; cnt++;
    }
    if (cnt === 0) { prevX = -1; continue; } // gap: don't draw a line across missing data
    const yMean = levelOf(sum / cnt, min, max, vSteps);
    canvas.vline(s, levelOf(hi, min, max, vSteps), levelOf(lo, min, max, vSteps));
    if (prevX >= 0) canvas.line(prevX, prevY, s, yMean); else canvas.set(s, yMean);
    prevX = s; prevY = yMean;
  }
}

/** Braille line chart (⠁⠂⠄⡀…), 2x4 subpixels per cell, so `height` rows give
 *  4*height vertical resolution. Returns exactly `height` rows of `width` chars. */
export function lineChart(values: number[], opts: ChartOpts): string[] {
  const { width, height } = opts;
  if (width <= 0 || height <= 0) return blankRows(width, height);
  const [min, max] = effectiveRange(values, opts.min, opts.max);
  const gutter = axisGutter(opts);
  const plotW = width - gutter;
  const canvas = new BrailleCanvas(plotW, height);
  plotSolid(canvas, values, min, max, plotW * 2, height * 4);
  const body = canvas.rows();
  return gutter ? body.map((r) => "│" + r) : body;
}

/** Two series on one chart: `a` in braille, `b` as a dotted/lighter overlay.
 *  Used for strategy equity vs buy & hold. Both share one scale. */
export function dualLineChart(a: number[], b: number[], opts: ChartOpts): string[] {
  const { width, height } = opts;
  if (width <= 0 || height <= 0) return blankRows(width, height);
  // One shared scale: comparing "strategy vs buy & hold" only means
  // something if both are read off the same axis.
  const [min, max] = effectiveRange(a.concat(b), opts.min, opts.max);
  const gutter = axisGutter(opts);
  const plotW = width - gutter;
  const subCols = plotW * 2, vSteps = height * 4;
  const canvas = new BrailleCanvas(plotW, height);

  // `b` first, sparse and unconnected (every other subpixel column, no wick,
  // no line) so it reads as a lighter dotted texture; `a` drawn solid over
  // it. Braille dots OR together, so a crossing point still shows both.
  for (let s = 0; s < subCols; s += 2) {
    const [lo, hiI] = bucketOf(b.length, subCols, s);
    let sum = 0, cnt = 0;
    for (let i = lo; i < hiI; i++) { const v = b[i]; if (Number.isFinite(v)) { sum += v; cnt++; } }
    if (cnt > 0) canvas.set(s, levelOf(sum / cnt, min, max, vSteps));
  }
  plotSolid(canvas, a, min, max, subCols, vSteps);

  const body = canvas.rows();
  return gutter ? body.map((r) => "│" + r) : body;
}

/** OHLC candles, one column per candle, wick │ and body █ (hollow ▯/░ for down
 *  bars so direction survives a monochrome terminal). Newest candle on the
 *  right; if there are more candles than columns, show the most recent `width`. */
export function candleChart(candles: Candle[], opts: ChartOpts): string[] {
  const { width, height } = opts;
  if (width <= 0 || height <= 0) return blankRows(width, height);
  const gutter = axisGutter(opts);
  const plotW = width - gutter;
  const n = candles.length;

  // Resample the whole series into exactly plotW columns (standard OHLC
  // aggregation: open=first, close=last, high=max, low=min of the bucket).
  // A width-sized window of raw candles collapses to one candle per bucket
  // (no aggregation, newest on the right); a longer history is aggregated
  // rather than having older bars silently dropped, so a spike anywhere in
  // the window still shows up in the wick.
  interface Agg { open: number; high: number; low: number; close: number; has: boolean; }
  const cols: Agg[] = [];
  for (let c = 0; c < plotW; c++) {
    const [a, b] = bucketOf(n, plotW, c);
    let open = NaN, close = NaN, hi = -Infinity, lo = Infinity, has = false;
    if (b > a) {
      open = candles[a].open;
      close = candles[b - 1].close;
      for (let i = a; i < b; i++) {
        const k = candles[i];
        for (const v of [k.open, k.high, k.low, k.close]) {
          if (Number.isFinite(v)) { if (v > hi) hi = v; if (v < lo) lo = v; has = true; }
        }
      }
    }
    has = has && Number.isFinite(open) && Number.isFinite(close);
    cols.push(has ? { open, high: hi, low: lo, close, has } : { open: NaN, high: NaN, low: NaN, close: NaN, has: false });
  }

  const spanVals: number[] = [];
  for (const c of cols) if (c.has) { spanVals.push(c.high, c.low); }
  const [min, max] = effectiveRange(spanVals, opts.min, opts.max);

  const grid: string[][] = Array.from({ length: height }, () => new Array<string>(plotW).fill(" "));
  for (let c = 0; c < plotW; c++) {
    const k = cols[c];
    if (!k.has) continue;
    const bodyLo = Math.min(k.open, k.close), bodyHi = Math.max(k.open, k.close);
    const up = k.close >= k.open;
    for (let r = 0; r < height; r++) {
      const rowTop = max - (r / height) * (max - min);
      const rowBot = max - ((r + 1) / height) * (max - min);
      if (!(rowBot <= k.high && rowTop >= k.low)) continue; // row misses the whole candle
      const inBody = rowBot <= bodyHi && rowTop >= bodyLo;
      grid[r][c] = inBody ? (up ? "█" : "░") : "│";
    }
  }
  const rows = grid.map((r) => r.join(""));
  return gutter ? rows.map((r) => "│" + r) : rows;
}

/** Decimal places that suit a value of this magnitude. Crypto prices here
 *  range from ~1e-8 (a fresh meme coin) to ~1e5, and a single fixed decimal
 *  count would either print "0.00" for the small end or drown the large end
 *  in digits. Deliberately never switches to exponential notation —
 *  `3.81e-6` is harder to eyeball in a price column than `0.00000381`. */
function decimalsForMagnitude(a: number): number {
  return a >= 1000 ? 0 :
    a >= 1 ? 2 :
    a >= 0.01 ? 4 :
    a >= 0.0001 ? 6 :
    a >= 0.000001 ? 8 : 10;
}

/** Y-axis labels for a chart of `height` rows over [min, max]: right-aligned,
 *  padded to a common width, formatted with significant digits that suit the
 *  range (crypto prices span 1e-8 to 1e5 here, so 0.00000381 and 77132 must both
 *  read correctly). Returns `height` strings. */
export function axisLabels(min: number, max: number, height: number, width?: number): string[] {
  if (height <= 0) return [];
  const lo0 = Number.isFinite(min) ? min : 0;
  let lo = lo0, hi = Number.isFinite(max) ? max : lo0 + 1;
  if (!(hi > lo)) {
    // Degenerate range (flat series, or min===max): pad relative to the
    // value's own magnitude, same reasoning as effectiveRange — a fixed
    // +1 pad would be invisible at 1e5 and absurd at 1e-8.
    const pad = Math.max(Math.abs(lo) * 0.05, 1e-9);
    lo -= pad; hi += pad;
  }
  // Row 0 (top of the chart) gets `hi`, the last row gets `lo`, evenly
  // spaced in between — matches how lineChart/candleChart bucket their rows.
  const values = height === 1
    ? [(lo + hi) / 2]
    : Array.from({ length: height }, (_, r) => hi - (r * (hi - lo)) / (height - 1));
  // One shared decimal count for the whole axis (driven by the larger-
  // magnitude end of the range), not per-row — a column of labels with a
  // different number of decimals per row reads as broken, not adaptive.
  const decimals = decimalsForMagnitude(Math.max(Math.abs(lo), Math.abs(hi)));
  const strs = values.map((v) => (Number.isFinite(v) ? v.toFixed(decimals) : "?"));
  const w = Math.max(width ?? 0, ...strs.map((s) => s.length));
  return strs.map((s) => s.padStart(w));
}

// Same 8-glyph ink ramp as sparkline, for one consistent "more ink = more
// bullish" visual language across the module.
const REGIME_RAMP = "▁▂▃▄▅▆▇█";

/** One glyph per bar showing which regime the model was in — the strip that
 *  goes directly under a price chart. Distinct, visually ordered glyphs per
 *  state (e.g. ░ for the most bearish through █ for the most bullish), and a
 *  space for "no data". `K` is the number of states. */
export function regimeStrip(states: number[], K: number, width: number): string {
  if (width <= 0) return "";
  const n = states.length;
  const kk = Math.max(1, Math.floor(K));
  let out = "";
  for (let c = 0; c < width; c++) {
    const [a, b] = bucketOf(n, width, c);
    // Categorical, not numeric: the bucket's most recent state wins rather
    // than an average, which would land on a meaningless "state 1.5".
    let s = NaN;
    for (let i = a; i < b; i++) { const v = states[i]; if (Number.isFinite(v)) s = v; }
    if (!Number.isFinite(s) || s < 0 || s >= kk) { out += " "; continue; }
    const idx = kk === 1 ? REGIME_RAMP.length - 1 : Math.round((s / (kk - 1)) * (REGIME_RAMP.length - 1));
    out += REGIME_RAMP[Math.min(REGIME_RAMP.length - 1, Math.max(0, idx))];
  }
  return out;
}

/** Positions over time as a strip: long/flat/short. */
export function positionStrip(positions: number[], width: number): string {
  if (width <= 0) return "";
  const n = positions.length;
  let out = "";
  for (let c = 0; c < width; c++) {
    const [a, b] = bucketOf(n, width, c);
    let p = NaN;
    for (let i = a; i < b; i++) { const v = positions[i]; if (Number.isFinite(v)) p = v; }
    out += !Number.isFinite(p) ? " " : p > 0 ? "▲" : p < 0 ? "▼" : "·";
  }
  return out;
}

/** A horizontal bar for a 0..1 quantity (progress, confidence), e.g. "███░░░ 62%". */
export function meter(fraction: number, width: number, label?: string): string {
  const w = Math.max(0, Math.floor(width));
  const f = Number.isFinite(fraction) ? Math.min(1, Math.max(0, fraction)) : 0;
  const filled = Math.min(w, Math.round(f * w));
  const bar = "█".repeat(filled) + "░".repeat(w - filled);
  const pct = `${Math.round(f * 100)}%`;
  return label ? `${label} ${bar} ${pct}` : `${bar} ${pct}`;
}
