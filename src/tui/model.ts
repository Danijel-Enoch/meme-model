/**
 * The TUI's view model, and the reducer that moves it.
 *
 * Everything the screen shows lives in one plain object, and every keystroke is
 * a pure function from (state, action) to state. Nothing in this file imports
 * the renderer, the model code, or the network — which is the point. The
 * layout in app.ts renders this; the worker in worker.ts fills it; and the
 * tests exercise the whole interaction model without a terminal or an API.
 *
 * The alternative — reading state off renderables and mutating them from key
 * handlers — is how TUIs become untestable.
 */

import type { Candle } from "../features";

export type ModelType = "hmm" | "hsmm";
export type Tab = "fit" | "backtest" | "paper" | "blotter";

export const TABS: Tab[] = ["fit", "backtest", "paper", "blotter"];

/** Timeframes offered in the UI. 5m is deliberately absent: Hyperliquid keeps
 *  ~5000 candles per interval, so 5m cannot reach the 3-week floor this whole
 *  exercise is scoped to. */
export const TIMEFRAMES = ["15m", "30m", "1h", "2h", "4h"];

/** One row of the universe list: a coin, and whatever the sweep learned about it. */
export interface CoinRow {
  coin: string;
  timeframe: string;
  modelType: ModelType;
  volume24h: number;
  markPrice: number;
  /** Sweep results. Null until the sweep has covered this coin. */
  excessRoi: number | null;
  roi: number | null;
  buyHold: number | null;
  exposure: number | null;
  trades: number | null;
  pValue: number | null;
  /** Set when this coin could not be evaluated, with the reason. */
  note?: string;
}

export interface FitView {
  coin: string;
  timeframe: string;
  modelType: ModelType;
  bars: number;
  logLikPerBar: number;
  converged: boolean;
  states: {
    label: string;
    meanRetBps: number;
    volPct: number;
    freq: number;
    durationBars: number;
  }[];
  /** K*K, row-major. The HSMM's diagonal is zero by construction. */
  transitions: number[];
  /** Per state, the top few (duration, probability) pairs. HSMM only. */
  durationModes: { state: number; d: number; p: number }[][] | null;
}

export interface BacktestView {
  coin: string;
  timeframe: string;
  modelType: ModelType;
  barsTraded: number;
  refits: number;
  roi: number;
  buyHold: number;
  sharpe: number;
  maxDD: number;
  exposure: number;
  trades: number;
  costDrag: number;
  /** Permutation test, filled in separately because it costs another pass. */
  pValue: number | null;
  medianRandom: number | null;
  equity: number[];
  equityBuyHold: number[];
}

export interface PaperView {
  equity: number;
  startingEquity: number;
  cash: number;
  realizedUsd: number;
  feesUsd: number;
  fundingUsd: number;
  liquidated: boolean;
  positions: {
    coin: string;
    position: number;
    entryPrice: number;
    markPrice: number;
    notionalUsd: number;
    unrealizedUsd: number;
    barsHeld: number;
  }[];
  equityCurve: number[];
  buyHoldCurve: number[];
  fills: {
    time: number;
    coin: string;
    side: "buy" | "sell";
    price: number;
    sizeUsd: number;
    feeUsd: number;
    reason: string;
  }[];
  /** Live loop state: when the next bar closes, and whether we are polling. */
  live: boolean;
  nextBarAt: number | null;
}

/** What the chart pane draws. Bars, and three things aligned to them. */
export interface ChartView {
  coin: string;
  timeframe: string;
  candles: Candle[];
  /** Most likely state per bar, -1 where the model has no opinion (warmup). */
  states: number[];
  /** Target position per bar in [-1, 1]. */
  positions: number[];
  K: number;
}

export interface AppState {
  rows: CoinRow[];
  cursor: number;
  tab: Tab;
  /** Label of the job currently running, or null. The UI must stay responsive
   *  while this is set — that is what the worker is for. */
  busy: string | null;
  progress: { done: number; total: number } | null;
  /** Transient status line. Errors land here rather than crashing the frame. */
  message: string | null;
  error: boolean;
  help: boolean;
  chart: ChartView | null;
  fit: FitView | null;
  backtest: BacktestView | null;
  paper: PaperView | null;
  /** Wall clock of the last data update, for the header. */
  updatedAt: number;
}

export type Action =
  | { type: "cursor"; delta: number }
  | { type: "cursorTo"; index: number }
  | { type: "tab"; delta: number }
  | { type: "tabTo"; tab: Tab }
  | { type: "timeframe"; delta: number }
  | { type: "modelType" }
  | { type: "busy"; label: string | null }
  | { type: "progress"; done: number; total: number }
  | { type: "message"; text: string | null; error?: boolean }
  | { type: "help" }
  | { type: "rows"; rows: CoinRow[] }
  | { type: "row"; row: CoinRow }
  | { type: "chart"; chart: ChartView | null }
  | { type: "fit"; fit: FitView | null }
  | { type: "backtest"; backtest: BacktestView | null }
  | { type: "paper"; paper: PaperView | null }
  | { type: "clock"; now: number };

export function initialState(rows: CoinRow[] = []): AppState {
  return {
    rows, cursor: 0, tab: "fit",
    busy: null, progress: null, message: null, error: false, help: false,
    chart: null, fit: null, backtest: null, paper: null,
    updatedAt: 0,
  };
}

export function currentRow(s: AppState): CoinRow | null {
  return s.rows[s.cursor] ?? null;
}

const wrap = (i: number, n: number) => (n <= 0 ? 0 : ((i % n) + n) % n);

export function reduce(s: AppState, a: Action): AppState {
  switch (a.type) {
    case "cursor": {
      if (s.rows.length === 0) return s;
      const cursor = wrap(s.cursor + a.delta, s.rows.length);
      // Moving off a coin invalidates everything derived from it. Showing the
      // previous coin's fit under a new coin's name is the kind of quiet lie a
      // trading UI must never tell.
      return cursor === s.cursor ? s
        : { ...s, cursor, chart: null, fit: null, backtest: null, message: null, error: false };
    }
    case "cursorTo": {
      if (s.rows.length === 0) return s;
      const cursor = Math.max(0, Math.min(s.rows.length - 1, a.index));
      return cursor === s.cursor ? s
        : { ...s, cursor, chart: null, fit: null, backtest: null, message: null, error: false };
    }
    case "tab":
      return { ...s, tab: TABS[wrap(TABS.indexOf(s.tab) + a.delta, TABS.length)] };
    case "tabTo":
      return { ...s, tab: a.tab };
    case "timeframe": {
      const row = currentRow(s);
      if (!row) return s;
      const i = TIMEFRAMES.indexOf(row.timeframe);
      const timeframe = TIMEFRAMES[wrap((i < 0 ? 0 : i) + a.delta, TIMEFRAMES.length)];
      return replaceRow(s, { ...row, timeframe }, true);
    }
    case "modelType": {
      const row = currentRow(s);
      if (!row) return s;
      return replaceRow(s, { ...row, modelType: row.modelType === "hmm" ? "hsmm" : "hmm" }, true);
    }
    case "busy":
      return { ...s, busy: a.label, progress: a.label ? s.progress : null };
    case "progress":
      return { ...s, progress: { done: a.done, total: a.total } };
    case "message":
      return { ...s, message: a.text, error: a.error ?? false };
    case "help":
      return { ...s, help: !s.help };
    case "rows": {
      const cursor = Math.min(s.cursor, Math.max(0, a.rows.length - 1));
      return { ...s, rows: a.rows, cursor, updatedAt: Date.now() };
    }
    case "row": {
      const rows = s.rows.map((r) => (r.coin === a.row.coin ? a.row : r));
      return { ...s, rows, updatedAt: Date.now() };
    }
    case "chart":
      return { ...s, chart: a.chart };
    case "fit":
      return { ...s, fit: a.fit };
    case "backtest":
      return { ...s, backtest: a.backtest };
    case "paper":
      return { ...s, paper: a.paper };
    case "clock":
      return { ...s, updatedAt: a.now };
  }
}

/** Swap the selected row, optionally dropping the results it invalidates. */
function replaceRow(s: AppState, row: CoinRow, invalidate: boolean): AppState {
  const rows = [...s.rows];
  rows[s.cursor] = row;
  return invalidate
    ? { ...s, rows, chart: null, fit: null, backtest: null, message: null, error: false }
    : { ...s, rows };
}

/**
 * The one place that maps keys to actions, so the key map is a data structure
 * the tests can walk rather than a pile of if-statements inside a handler.
 * Returns null for keys this app does not claim, which lets the caller decide
 * whether a focused widget should see them.
 */
export function keyToAction(name: string, shift = false): Action | "quit" | "fit" | "backtest" | "replay" | "paper" | "sweep" | null {
  switch (name) {
    case "up": case "k": return { type: "cursor", delta: -1 };
    case "down": case "j": return { type: "cursor", delta: 1 };
    case "pageup": return { type: "cursor", delta: -10 };
    case "pagedown": return { type: "cursor", delta: 10 };
    case "tab": return { type: "tab", delta: shift ? -1 : 1 };
    case "left": case "h": return { type: "timeframe", delta: -1 };
    case "right": case "l": return { type: "timeframe", delta: 1 };
    case "m": return { type: "modelType" };
    case "1": return { type: "tabTo", tab: "fit" };
    case "2": return { type: "tabTo", tab: "backtest" };
    case "3": return { type: "tabTo", tab: "paper" };
    case "4": return { type: "tabTo", tab: "blotter" };
    case "?": return { type: "help" };
    case "q": case "escape": return "quit";
    case "f": return "fit";
    case "b": return "backtest";
    case "r": return "replay";
    case "p": return "paper";
    case "s": return "sweep";
    default: return null;
  }
}
