/**
 * The screen.
 *
 * app.ts owns the renderable tree and nothing else: it turns an AppState into
 * text, and turns keys into actions. Every side effect — fetching, fitting,
 * trading — is a hook supplied by the caller, so the whole UI can be driven
 * headlessly in a test with `createTestRenderer` and stub hooks.
 *
 * Layout is flexbox (Yoga, under the hood):
 *
 *   header                       1 row
 *   body        universe list | chart over tabs      flexGrow 1
 *   footer                       1 row
 */

import {
  BoxRenderable, TextRenderable, type CliRenderer,
} from "@opentui/core";
import { candleChart, axisLabels, regimeStrip, positionStrip, dualLineChart, meter } from "../charts";
import {
  TABS, currentRow, keyToAction, reduce,
  type Action, type AppState, type CoinRow, type Tab,
} from "./model";

const C = {
  bg: "#0b0d12",
  panel: "#11141c",
  border: "#2a3040",
  borderFocus: "#4d7cff",
  text: "#c8d0e0",
  dim: "#5c6680",
  head: "#8fa3c8",
  up: "#3fb950",
  down: "#f85149",
  warn: "#d29922",
  accent: "#4d7cff",
};

export interface AppHooks {
  fit(): void;
  backtest(): void;
  replay(): void;
  paperToggle(): void;
  sweep(): void;
  quit(): void;
}

const NO_HOOKS: AppHooks = {
  fit() {}, backtest() {}, replay() {}, paperToggle() {}, sweep() {}, quit() {},
};

/**
 * A pool of single-line TextRenderables inside a box.
 *
 * One renderable per line rather than one multi-line Text, because colour in
 * this UI is per line — a red drawdown next to a green return — and a single
 * Text would force one foreground for the lot.
 */
class Lines {
  private pool: TextRenderable[] = [];
  constructor(private renderer: CliRenderer, private parent: BoxRenderable) {}

  /**
   * `capacity` is the number of rows the parent box can actually show. Lines
   * beyond it are hidden rather than dropped, because a Yoga box does not clip
   * its children: an extra renderable does not disappear, it paints over
   * whatever is below the panel — which showed up as the universe list
   * spilling past the footer.
   */
  set(lines: { text: string; fg?: string; bg?: string }[], capacity = lines.length) {
    lines = lines.slice(0, Math.max(0, capacity));
    while (this.pool.length < lines.length) {
      const t = new TextRenderable(this.renderer, { content: "", fg: C.text, wrapMode: "none" });
      this.pool.push(t);
      this.parent.add(t);
    }
    for (let i = 0; i < this.pool.length; i++) {
      const line = lines[i];
      const r = this.pool[i];
      if (!line) { r.visible = false; continue; }
      r.visible = true;
      r.content = line.text;
      r.fg = line.fg ?? C.text;
      if (line.bg) r.bg = line.bg;
      else r.bg = undefined as unknown as string;
    }
  }
}

const pct = (x: number | null | undefined, dp = 1) =>
  x === null || x === undefined || !Number.isFinite(x) ? "  —  " : `${(x * 100).toFixed(dp)}%`;
const signColor = (x: number | null | undefined) =>
  x === null || x === undefined || !Number.isFinite(x) ? C.dim : x > 0 ? C.up : x < 0 ? C.down : C.text;
const pad = (s: string, n: number) => (s.length >= n ? s.slice(0, n) : s + " ".repeat(n - s.length));
const rpad = (s: string, n: number) => (s.length >= n ? s.slice(0, n) : " ".repeat(n - s.length) + s);

function money(x: number): string {
  const a = Math.abs(x);
  const s = a >= 1000 ? a.toFixed(0) : a >= 1 ? a.toFixed(2) : a.toFixed(4);
  return `${x < 0 ? "-" : ""}$${s}`;
}

function price(x: number): string {
  if (!Number.isFinite(x)) return "—";
  if (x >= 1000) return x.toFixed(1);
  if (x >= 1) return x.toFixed(3);
  if (x >= 0.001) return x.toFixed(5);
  return x.toExponential(2);
}

export interface App {
  state(): AppState;
  dispatch(a: Action): void;
  render(): void;
  handleKey(name: string, shift?: boolean): void;
  destroy(): void;
}

export function createApp(renderer: CliRenderer, initial: AppState, hooks: AppHooks = NO_HOOKS): App {
  let state = initial;

  const root = new BoxRenderable(renderer, {
    width: "100%", height: "100%", flexDirection: "column", backgroundColor: C.bg,
  });
  const header = new TextRenderable(renderer, { content: "", fg: C.head, height: 1, wrapMode: "none" });
  const body = new BoxRenderable(renderer, { flexGrow: 1, flexDirection: "row" });
  const footer = new TextRenderable(renderer, { content: "", fg: C.dim, height: 1, wrapMode: "none" });

  const listBox = new BoxRenderable(renderer, {
    width: 34, flexDirection: "column", border: true, borderColor: C.border,
    borderStyle: "rounded", title: " universe ", titleColor: C.head, backgroundColor: C.panel,
    overflow: "hidden",
  });
  const right = new BoxRenderable(renderer, { flexGrow: 1, flexDirection: "column" });
  const chartBox = new BoxRenderable(renderer, {
    flexGrow: 1, flexDirection: "column", border: true, borderColor: C.border,
    borderStyle: "rounded", title: " chart ", titleColor: C.head, backgroundColor: C.panel,
    overflow: "hidden",
  });
  const tabBox = new BoxRenderable(renderer, {
    height: 14, flexDirection: "column", border: true, borderColor: C.border,
    borderStyle: "rounded", title: " fit ", titleColor: C.head, backgroundColor: C.panel,
    overflow: "hidden",
  });

  root.add(header);
  root.add(body);
  body.add(listBox);
  body.add(right);
  right.add(chartBox);
  right.add(tabBox);
  root.add(footer);
  renderer.root.add(root);

  const listLines = new Lines(renderer, listBox);
  const chartLines = new Lines(renderer, chartBox);
  const tabLines = new Lines(renderer, tabBox);

  // Inner drawing area of a bordered box. The border eats one cell a side, and
  // asking the renderable for its size before the first layout pass returns 0 —
  // hence the floors, which keep the chart functions from being handed a
  // negative width during the first frame.
  const inner = (b: BoxRenderable) => ({
    w: Math.max(10, (b.width || 0) - 2),
    h: Math.max(3, (b.height || 0) - 2),
  });

  function renderHeader() {
    const row = currentRow(state);
    const p = state.paper;
    const left = row
      ? `${row.coin}-PERP ${row.timeframe} ${row.modelType.toUpperCase()}  ${price(row.markPrice)}`
      : "no markets loaded";
    const equity = p ? `paper ${money(p.equity)} ${pct((p.equity - p.startingEquity) / p.startingEquity)}` : "paper —";
    const status = state.busy
      ? `⋯ ${state.busy}${state.progress ? ` ${state.progress.done}/${state.progress.total}` : ""}`
      : state.message ?? "";
    header.content = ` ${pad(left, 34)}${pad(equity, 30)}${status}`;
    header.fg = state.error ? C.down : state.busy ? C.warn : C.head;
  }

  function renderList() {
    const { w } = inner(listBox);
    const rows = state.rows;
    const height = Math.max(1, inner(listBox).h - 1);
    // Keep the cursor in view without a scrollbar: a simple window that only
    // moves when the cursor would leave it.
    const start = Math.max(0, Math.min(state.cursor - Math.floor(height / 2), rows.length - height));
    const head = { text: pad(" coin      tf    excess    p", w), fg: C.dim };
    const lines = [head];
    for (let i = start; i < Math.min(rows.length, start + height); i++) {
      const r = rows[i];
      const sel = i === state.cursor;
      const excess = r.excessRoi === null ? "   —  " : rpad(pct(r.excessRoi, 0), 6);
      const pv = r.pValue === null ? "  — " : rpad(r.pValue.toFixed(2), 4);
      const text = pad(` ${pad(r.coin, 8)} ${pad(r.timeframe, 4)} ${excess} ${pv}`, w);
      lines.push({
        text,
        fg: sel ? "#ffffff" : signColor(r.excessRoi),
        bg: sel ? C.accent : undefined,
      } as { text: string; fg: string; bg?: string });
    }
    listLines.set(lines, inner(listBox).h);
  }

  function renderChart() {
    const { w, h } = inner(chartBox);
    const c = state.chart;
    if (!c || c.candles.length === 0) {
      chartLines.set([{ text: "", fg: C.dim }, { text: "  no data — press f to fit, b to backtest", fg: C.dim }], h);
      return;
    }
    // Two strips (regime, position) plus a legend line live under the candles.
    const chartH = Math.max(3, h - 3);
    const gutter = 9;
    const plotW = Math.max(10, w - gutter);
    const visible = c.candles.slice(-plotW);
    let lo = Infinity, hi = -Infinity;
    for (const k of visible) { if (k.low < lo) lo = k.low; if (k.high > hi) hi = k.high; }
    const rows = candleChart(visible, { width: plotW, height: chartH, min: lo, max: hi });
    const labels = axisLabels(lo, hi, chartH, gutter - 1);

    const lines = rows.map((r, i) => ({ text: `${rpad(labels[i] ?? "", gutter - 1)} ${r}`, fg: C.text }));
    const tailStates = c.states.slice(-plotW);
    const tailPos = c.positions.slice(-plotW);
    lines.push({ text: `${rpad("regime", gutter - 1)} ${regimeStrip(tailStates, c.K, plotW)}`, fg: C.accent });
    lines.push({ text: `${rpad("position", gutter - 1)} ${positionStrip(tailPos, plotW)}`, fg: C.warn });
    const span = `${visible.length} bars  ${price(lo)} … ${price(hi)}`;
    lines.push({ text: `${rpad("", gutter - 1)} ${span}`, fg: C.dim });
    chartLines.set(lines, h);
    chartBox.title = ` ${c.coin} ${c.timeframe} `;
  }

  function tabHeader(): { text: string; fg: string } {
    const names = TABS.map((t, i) => (t === state.tab ? `[${i + 1} ${t}]` : ` ${i + 1} ${t} `)).join(" ");
    return { text: ` ${names}`, fg: C.head };
  }

  function renderFitTab(w: number): { text: string; fg: string }[] {
    const f = state.fit;
    if (!f) return [{ text: "  press f to fit this coin on the current window", fg: C.dim }];
    const out = [
      { text: `  ${f.bars} bars   log-lik/bar ${f.logLikPerBar.toFixed(4)}   ${f.converged ? "converged" : "hit iteration cap"}`, fg: C.dim },
      { text: "  #  label       mean ret    bar vol     freq    dwell", fg: C.dim },
    ];
    f.states.forEach((s, k) => {
      out.push({
        text: `  ${k}  ${pad(s.label, 10)} ${rpad(s.meanRetBps.toFixed(1) + "bps", 9)} ` +
          `${rpad(s.volPct.toFixed(2) + "%", 9)} ${rpad(pct(s.freq, 1), 7)} ${rpad(s.durationBars.toFixed(1) + "b", 7)}`,
        fg: signColor(s.meanRetBps),
      });
    });
    if (f.durationModes) {
      out.push({ text: "  duration pmf (top modes)", fg: C.dim });
      f.durationModes.forEach((modes, k) => {
        out.push({
          text: `  ${k}  ` + modes.map((m) => `${m.d}b=${(m.p * 100).toFixed(0)}%`).join("  "),
          fg: C.text,
        });
      });
    }
    out.push({ text: "  in-sample labels — never trade a smoothed state", fg: C.dim });
    return out;
  }

  function renderBacktestTab(w: number): { text: string; fg: string }[] {
    const b = state.backtest;
    if (!b) return [{ text: "  press b to walk this coin forward out of sample", fg: C.dim }];
    const verdict = b.pValue === null ? "no permutation test"
      : b.pValue >= 0.05 ? "not distinguishable from luck"
        : b.roi > 0 ? "REAL TIMING SKILL" : "beats random timing, still loses money";
    const out = [
      { text: `  ${b.barsTraded} bars traded, ${b.refits} refits, ${b.trades} trades`, fg: C.dim },
      { text: `  return   ${rpad(pct(b.roi), 8)}     buy & hold ${rpad(pct(b.buyHold), 8)}     vs null ${rpad(pct(b.medianRandom), 8)}`, fg: signColor(b.roi - b.buyHold) },
      { text: `  sharpe   ${rpad(b.sharpe.toFixed(2), 8)}     maxDD      ${rpad(pct(b.maxDD), 8)}     exposure ${rpad(pct(b.exposure), 8)}`, fg: C.text },
      { text: `  p-value  ${rpad(b.pValue === null ? "—" : b.pValue.toFixed(3), 8)}     ${verdict}`, fg: b.pValue !== null && b.pValue < 0.05 ? C.up : C.warn },
    ];
    const h = 5;
    if (b.equity.length > 2) {
      const rows = dualLineChart(b.equity, b.equityBuyHold, { width: Math.max(20, w - 4), height: h });
      rows.forEach((r) => out.push({ text: `  ${r}`, fg: C.accent }));
      out.push({ text: "  ─ strategy   · buy & hold", fg: C.dim });
    }
    return out;
  }

  function renderPaperTab(w: number): { text: string; fg: string }[] {
    const p = state.paper;
    if (!p) return [{ text: "  press r to replay history, p to start the live loop", fg: C.dim }];
    const pnl = p.equity - p.startingEquity;
    const out = [
      { text: `  equity ${rpad(money(p.equity), 10)}  pnl ${rpad(money(pnl), 10)}  ${rpad(pct(pnl / p.startingEquity), 8)}` +
          `  fees ${money(p.feesUsd)}${p.fundingUsd ? `  funding ${money(p.fundingUsd)}` : ""}`, fg: signColor(pnl) },
      { text: `  ${p.live ? "LIVE" : "idle"}${p.liquidated ? "   LIQUIDATED" : ""}` +
          `${p.nextBarAt ? `   next bar in ${Math.max(0, Math.round((p.nextBarAt - Date.now()) / 1000))}s` : ""}`,
        fg: p.liquidated ? C.down : p.live ? C.up : C.dim },
      { text: "  coin      pos    entry      mark      notional   unreal   bars", fg: C.dim },
    ];
    const positions = p.positions.filter((x) => Math.abs(x.position) > 1e-9);
    if (positions.length === 0) out.push({ text: "  (flat)", fg: C.dim });
    for (const q of positions) {
      out.push({
        text: `  ${pad(q.coin, 8)} ${rpad(q.position.toFixed(2), 5)} ${rpad(price(q.entryPrice), 10)} ` +
          `${rpad(price(q.markPrice), 9)} ${rpad(money(q.notionalUsd), 10)} ${rpad(money(q.unrealizedUsd), 8)} ${rpad(String(q.barsHeld), 6)}`,
        fg: signColor(q.unrealizedUsd),
      });
    }
    if (p.equityCurve.length > 2) {
      const rows = dualLineChart(p.equityCurve, p.buyHoldCurve, { width: Math.max(20, w - 4), height: 4 });
      rows.forEach((r) => out.push({ text: `  ${r}`, fg: C.accent }));
    }
    return out;
  }

  function renderBlotterTab(w: number): { text: string; fg: string }[] {
    const p = state.paper;
    if (!p || p.fills.length === 0) return [{ text: "  no fills yet", fg: C.dim }];
    const out = [{ text: "  time              coin     side   price       size      fee    why", fg: C.dim }];
    for (const f of p.fills.slice(-10)) {
      const ts = new Date(f.time * 1000).toISOString().slice(5, 16).replace("T", " ");
      out.push({
        text: `  ${pad(ts, 17)} ${pad(f.coin, 8)} ${pad(f.side, 6)} ${rpad(price(f.price), 10)} ` +
          `${rpad(money(f.sizeUsd), 9)} ${rpad(money(f.feeUsd), 7)}  ${f.reason}`,
        fg: f.side === "buy" ? C.up : C.down,
      });
    }
    return out;
  }

  function renderTabs() {
    const { w, h } = inner(tabBox);
    tabBox.title = ` ${state.tab} `;
    const lines = [tabHeader()];
    if (state.help) {
      lines.push(
        { text: "  j/k ↑↓ coin    h/l ←→ timeframe    m model    1-4 tab    tab cycles", fg: C.text },
        { text: "  f fit    b backtest (walk-forward + permutation)    r replay paper", fg: C.text },
        { text: "  p paper live on/off    s sweep the universe    ? help    q quit", fg: C.text },
        { text: "", fg: C.dim },
        { text: "  the p-value is the only number here that answers 'is this skill'", fg: C.warn },
      );
    } else {
      const body = state.tab === "fit" ? renderFitTab(w)
        : state.tab === "backtest" ? renderBacktestTab(w)
          : state.tab === "paper" ? renderPaperTab(w)
            : renderBlotterTab(w);
      lines.push(...body);
    }
    tabLines.set(lines, h);
  }

  function renderFooter() {
    const bar = state.progress
      ? ` ${meter(state.progress.total ? state.progress.done / state.progress.total : 0, 20)}`
      : "";
    footer.content = ` f fit  b backtest  r replay  p paper  s sweep  ? help  q quit${bar}`;
  }

  function render() {
    // 80 columns is the floor (a non-TTY defaults there, and so does an
    // unresized SSH window). At that width a 34-column list leaves the chart
    // too narrow to read, so it gives some back.
    listBox.width = renderer.width < 100 ? 24 : 34;
    // The tab pane shrinks before the chart does, since the chart is the part
    // that cannot be scrolled.
    tabBox.height = renderer.height < 30 ? 9 : 14;
    renderHeader();
    renderList();
    renderChart();
    renderTabs();
    renderFooter();
  }

  function dispatch(a: Action) {
    state = reduce(state, a);
    render();
  }

  function handleKey(name: string, shift = false) {
    const a = keyToAction(name, shift);
    if (a === null) return;
    if (typeof a === "string") {
      switch (a) {
        case "quit": hooks.quit(); return;
        case "fit": hooks.fit(); return;
        case "backtest": hooks.backtest(); return;
        case "replay": hooks.replay(); return;
        case "paper": hooks.paperToggle(); return;
        case "sweep": hooks.sweep(); return;
      }
      return;
    }
    dispatch(a);
  }

  // A resized terminal changes how many candles fit, so the chart has to be
  // recomputed rather than merely reflowed.
  // The first render runs before Yoga has laid anything out, so every pane is
  // drawn again once its real size is known.
  listBox.onSizeChange = () => render();
  chartBox.onSizeChange = () => render();
  tabBox.onSizeChange = () => render();

  render();

  return {
    state: () => state,
    dispatch,
    render,
    handleKey,
    destroy: () => root.destroyRecursively(),
  };
}
