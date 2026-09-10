#!/usr/bin/env bun
/**
 * meme-hmm TUI: fit, backtest and paper trade Hyperliquid perps from one screen.
 *
 *   bun run tui                       # top 30 perps, 45-day window
 *   bun run tui --limit 10 --days 30
 *   bun run tui --equity 5000 --leverage 3
 *
 * This file is the wiring only: it owns the renderer, the worker, the paper
 * account and the poll timer, and hands app.ts a view model. Anything that
 * computes lives in the worker; anything that decides lives in backtest.ts.
 *
 * The compute rule is worth stating because it shapes everything here: a fit is
 * seconds of blocking arithmetic, so it happens on a Worker and the UI keeps
 * repainting. The paper account is the one thing the UI thread owns outright —
 * applying a closed bar is microseconds, and it must never race the fit that
 * might replace the model underneath it.
 */

import { createCliRenderer } from "@opentui/core";
import { createApp } from "./app";
import { initialState, currentRow, type AppState, type CoinRow, type PaperView } from "./model";
import { JobClient, type BacktestResultMsg, type FitResult, type RuntimeModel, type SweepResultMsg } from "./jobs";
import { pollOnce } from "./live";
import {
  DEFAULT_MODEL_DIR, DEFAULT_PAPER_PATH, DEFAULT_SWEEP_PATH,
  loadRuntimeModel, loadSweep, rowsFrom,
} from "./store";
import { newPaperState, onBar, savePaper, loadPaper, type PaperState } from "../paper";
import { barBudget, DEFAULT_TIMEFRAMES, type SweepResult } from "../sweep";
import type { StrategyConfig } from "../backtest";
import * as hl from "../hyperliquid";

const argv = process.argv.slice(2);
const flag = (name: string, dflt: string) => {
  const i = argv.indexOf(`--${name}`);
  return i >= 0 && argv[i + 1] && !argv[i + 1].startsWith("--") ? argv[i + 1] : dflt;
};
const numFlag = (name: string, dflt: number) => Number(flag(name, String(dflt)));

const LIMIT = numFlag("limit", 30);
const DAYS = numFlag("days", 45);
const COST = numFlag("cost", hl.HL_TAKER_BPS);
const STATES = numFlag("states", 3);
const MAX_DURATION = numFlag("max-duration", 30);
const EQUITY = numFlag("equity", 1000);
const LEVERAGE = numFlag("leverage", 1);
const TRIALS = numFlag("trials", 200);
const POLL_MS = numFlag("poll", 15) * 1000;
const PAPER_PATH = flag("paper", DEFAULT_PAPER_PATH);
const SWEEP_PATH = flag("sweep", DEFAULT_SWEEP_PATH);
const MODEL_DIR = flag("models", DEFAULT_MODEL_DIR);

const STRATEGY: StrategyConfig = { costBps: COST, durationAware: true };

/** Cache of fitted models, keyed coin|timeframe|modelType. */
const models = new Map<string, RuntimeModel>();
const modelKey = (coin: string, tf: string, mt: string) => `${coin}|${tf}|${mt}`;

function paperView(p: PaperState, live: boolean, nextBarAt: number | null): PaperView {
  const positions = Object.values(p.positions).map((q) => ({
    coin: q.coin,
    position: q.position,
    entryPrice: q.entryPrice,
    markPrice: p.lastPrice[q.coin] ?? q.entryPrice,
    notionalUsd: q.notionalUsd,
    unrealizedUsd: q.unrealizedUsd,
    barsHeld: q.barsHeld,
  }));
  return {
    equity: p.equity,
    startingEquity: p.config.startingEquity ?? EQUITY,
    cash: p.cash,
    realizedUsd: p.realizedUsd,
    feesUsd: p.feesUsd,
    fundingUsd: p.fundingUsd,
    liquidated: p.liquidated,
    positions,
    equityCurve: p.equityCurve.map((e) => e.equity),
    buyHoldCurve: p.equityCurve.map((e) => e.buyHold),
    fills: p.fills.slice(-40),
    live,
    nextBarAt,
  };
}

async function main() {
  process.stderr.write("loading Hyperliquid markets…\n");
  const [markets, sweepTable] = await Promise.all([
    hl.topMarkets(LIMIT),
    loadSweep(SWEEP_PATH),
  ]);
  const rows: CoinRow[] = rowsFrom(markets, sweepTable, {
    timeframe: DEFAULT_TIMEFRAMES[1] ?? "30m",
    modelType: "hsmm",
  });

  // Resume the account rather than starting a fresh one every launch: a paper
  // record that resets on restart cannot answer the only question it exists for.
  let paper: PaperState;
  try {
    paper = await loadPaper(PAPER_PATH);
  } catch {
    paper = newPaperState({
      startingEquity: EQUITY, leverage: LEVERAGE, costBps: COST,
      maxCoins: 1, maxLeverage: markets[0]?.maxLeverage ?? 10,
    });
  }

  const renderer = await createCliRenderer({
    exitOnCtrlC: true,
    // Otherwise every console.log in the model code lands in an overlay nobody
    // asked for, on top of the chart.
    consoleMode: "disabled",
    targetFps: 30,
    backgroundColor: "#0b0d12",
  });

  const jobs = new JobClient();
  let live = false;
  let liveCoin: string | null = null;
  let nextBarAt: number | null = null;
  let pollTimer: ReturnType<typeof setInterval> | null = null;
  let clockTimer: ReturnType<typeof setInterval> | null = null;
  let dirty = false;

  const app = createApp(renderer, {
    ...initialState(rows),
    paper: paperView(paper, live, nextBarAt),
  }, {
    fit: () => void runFit(),
    backtest: () => void runBacktest(),
    replay: () => void runReplay(),
    paperToggle: () => void togglePaper(),
    sweep: () => void runSweep(),
    quit: () => void shutdown(),
  });

  const state = () => app.state();
  const busy = () => state().busy !== null;
  const say = (text: string, error = false) => app.dispatch({ type: "message", text, error });
  const refreshPaper = () => app.dispatch({ type: "paper", paper: paperView(paper, live, nextBarAt) });

  async function persist() {
    dirty = true;
    try { await savePaper(PAPER_PATH, paper); dirty = false; } catch { /* keep trading, retry next bar */ }
  }

  async function ensureModel(row: CoinRow): Promise<RuntimeModel | null> {
    const key = modelKey(row.coin, row.timeframe, row.modelType);
    const cached = models.get(key);
    if (cached) return cached;
    const onDisk = await loadRuntimeModel(row.coin, row.timeframe, row.modelType, MODEL_DIR);
    if (onDisk) models.set(key, onDisk);
    return onDisk;
  }

  async function runFit() {
    const row = currentRow(state());
    if (!row || busy()) return;
    const { bars } = barBudget(row.timeframe, DAYS);
    app.dispatch({ type: "busy", label: `fitting ${row.coin} ${row.timeframe} ${row.modelType}` });
    const started = Date.now();
    try {
      const res = await jobs.run<FitResult>({
        kind: "fit", coin: row.coin, timeframe: row.timeframe, bars,
        states: STATES, modelType: row.modelType, maxDuration: MAX_DURATION, seed: 42, restarts: 8,
      }).promise;
      models.set(modelKey(row.coin, row.timeframe, row.modelType), res.model);
      app.dispatch({ type: "busy", label: null });
      app.dispatch({ type: "fit", fit: res.fit });
      app.dispatch({ type: "chart", chart: res.chart });
      app.dispatch({ type: "tabTo", tab: "fit" });
      say(`fitted in ${((Date.now() - started) / 1000).toFixed(1)}s`);
    } catch (e) {
      app.dispatch({ type: "busy", label: null });
      say(String(e instanceof Error ? e.message : e), true);
    }
  }

  async function runBacktest() {
    const row = currentRow(state());
    if (!row || busy()) return;
    const { bars } = barBudget(row.timeframe, DAYS);
    app.dispatch({ type: "busy", label: `walking ${row.coin} ${row.timeframe} forward` });
    try {
      const res = await jobs.run<BacktestResultMsg>({
        kind: "backtest", coin: row.coin, timeframe: row.timeframe, bars,
        states: STATES, modelType: row.modelType, maxDuration: MAX_DURATION,
        costBps: COST, trials: TRIALS, seed: 42, restarts: 4,
      }).promise;
      if (res.model) models.set(modelKey(row.coin, row.timeframe, row.modelType), res.model);
      app.dispatch({ type: "busy", label: null });
      app.dispatch({ type: "backtest", backtest: res.backtest });
      app.dispatch({ type: "chart", chart: res.chart });
      app.dispatch({ type: "tabTo", tab: "backtest" });
      const b = res.backtest;
      say(b.pValue !== null && b.pValue >= 0.05
        ? `p=${b.pValue.toFixed(3)} — not distinguishable from luck`
        : `p=${b.pValue?.toFixed(3)}`);
    } catch (e) {
      app.dispatch({ type: "busy", label: null });
      say(String(e instanceof Error ? e.message : e), true);
    }
  }

  /**
   * Replay the walk-forward's own decisions through the paper account.
   *
   * The positions come from the backtest rather than being recomputed bar by
   * bar, which is both far faster and exactly equivalent: paper.test.ts pins
   * `replay` to `walkForward`'s equity curve at 1e-9, so feeding those same
   * positions through `onBar` reproduces it by construction. What it buys is
   * the account view — fills, fees, drawdown in dollars — of a run whose
   * headline numbers you already have.
   */
  async function runReplay() {
    const row = currentRow(state());
    const chart = state().chart;
    if (!row || busy()) return;
    if (!chart || chart.coin !== row.coin || chart.positions.every((p) => p === 0)) {
      say("press b first — replay walks the backtest's own positions", true);
      return;
    }
    app.dispatch({ type: "busy", label: `replaying ${row.coin}` });
    paper = newPaperState({
      startingEquity: EQUITY, leverage: LEVERAGE, costBps: COST,
      maxCoins: 1, maxLeverage: markets.find((m) => m.coin === row.coin)?.maxLeverage ?? 10,
    });
    for (let i = 0; i < chart.candles.length && i < chart.positions.length; i++) {
      onBar(paper, row.coin, chart.candles[i], chart.positions[i], "replay");
      if (paper.liquidated) break;
    }
    await persist();
    app.dispatch({ type: "busy", label: null });
    app.dispatch({ type: "tabTo", tab: "paper" });
    refreshPaper();
    say(`replayed ${chart.candles.length} bars — ${paper.fills.length} fills`);
  }

  async function togglePaper() {
    const row = currentRow(state());
    if (!row) return;
    if (live) {
      live = false;
      liveCoin = null;
      if (pollTimer) clearInterval(pollTimer);
      pollTimer = null;
      refreshPaper();
      say("paper loop stopped");
      return;
    }
    const model = await ensureModel(row);
    if (!model) {
      say(`no model for ${row.coin} ${row.timeframe} ${row.modelType} — press f to fit`, true);
      return;
    }
    live = true;
    liveCoin = row.coin;
    say(`paper loop live on ${row.coin} ${row.timeframe} — closed bars only`);
    await poll();
    pollTimer = setInterval(() => void poll(), POLL_MS);
  }

  async function poll() {
    if (!live || !liveCoin) return;
    const row = state().rows.find((r) => r.coin === liveCoin);
    if (!row) return;
    const model = models.get(modelKey(row.coin, row.timeframe, row.modelType));
    if (!model) return;
    try {
      const res = await pollOnce(paper, {
        coin: row.coin, timeframe: row.timeframe, model, strategy: STRATEGY,
      });
      nextBarAt = res.nextCloseMs;
      if (res.applied > 0) {
        await persist();
        say(`${row.coin}: ${res.applied} bar${res.applied > 1 ? "s" : ""} applied` +
          (res.signal ? `, target ${res.signal.target.toFixed(2)}` : ""));
      }
      refreshPaper();
    } catch (e) {
      say(`poll failed: ${e instanceof Error ? e.message : e}`, true);
    }
  }

  async function runSweep() {
    if (busy()) return;
    app.dispatch({ type: "busy", label: "sweeping the universe" });
    try {
      const res = await jobs.run<SweepResultMsg>(
        {
          kind: "sweep", limit: LIMIT, timeframes: DEFAULT_TIMEFRAMES, days: DAYS,
          costBps: COST, trials: TRIALS,
        },
        (done, total) => app.dispatch({ type: "progress", done, total }),
      ).promise;
      const table = { rows: res.rows, best: res.best } as unknown as SweepResult;
      app.dispatch({ type: "busy", label: null });
      app.dispatch({ type: "rows", rows: rowsFrom(markets, table, { timeframe: "30m", modelType: "hsmm" }) });
      say(`swept ${res.rows.length} combinations — ${Object.keys(res.best).length} coins qualified`);
    } catch (e) {
      app.dispatch({ type: "busy", label: null });
      say(String(e instanceof Error ? e.message : e), true);
    }
  }

  async function shutdown() {
    if (pollTimer) clearInterval(pollTimer);
    if (clockTimer) clearInterval(clockTimer);
    if (dirty) await persist();
    jobs.terminate();
    renderer.destroy();
    process.exit(0);
  }

  renderer.keyInput.on("keypress", (key) => {
    app.handleKey(key.name ?? key.sequence, key.shift);
  });

  // A slow tick so the countdown and any relative times stay honest even when
  // nothing else changes.
  clockTimer = setInterval(() => {
    if (live) refreshPaper();
    else app.dispatch({ type: "clock", now: Date.now() });
  }, 1000);

  process.on("SIGTERM", () => void shutdown());
  app.render();
}

main().catch((e) => {
  process.stderr.write(`${e instanceof Error ? e.stack : String(e)}\n`);
  process.exit(1);
});
