/**
 * The compute thread.
 *
 * Everything expensive — fetching candles, fitting, walking forward, permuting —
 * happens here, so the render thread never blocks. Results come back as view
 * models (see model.ts), already shaped for the screen, because doing that
 * shaping on the UI side would mean shipping raw Float64Arrays across the wire
 * and re-deriving labels on every repaint.
 */

import { buildFeatures, fitScaler, applyScaler, unscale, type FeatureConfig } from "../features";
import { fit, viterbi, expectedDuration, type HmmParams } from "../hmm";
import { fitHsmm, viterbiHsmm, expectedDurations, type HsmmParams } from "../hsmm";
import { walkForward } from "../backtest";
import { permutationTest, tradedSeries } from "../diagnostics";
import { barBudget, walkForwardSizes, sweep, SWEEP_WINDOW } from "../sweep";
import * as hl from "../hyperliquid";
import type { BacktestView, ChartView, FitView, ModelType } from "./model";
import type { JobRequest, JobResponse, RuntimeModel } from "./jobs";

/** Must match sweep.ts's featureConfig exactly, or the TUI would fit a
 *  different model than the sweep ranked. */
const FEATURES: FeatureConfig = { window: SWEEP_WINDOW, useVolatility: true, useVolume: true };

function labelStates(muRet: number[]): string[] {
  return muRet.map((r) => {
    const b = r * 10_000;
    if (b > 20) return "PUMP";
    if (b > 3) return "drift-up";
    if (b < -20) return "DUMP";
    if (b < -3) return "bleed";
    return "chop";
  });
}

/** Bun's worker global. Declared locally rather than pulling the whole
 *  webworker lib into a tsconfig the rest of the repo shares. */
declare const self: {
  postMessage(message: unknown): void;
  addEventListener(type: "message", listener: (e: { data: unknown }) => void): void;
};

const post = (msg: JobResponse) => self.postMessage(msg);

async function candlesFor(coin: string, timeframe: string, bars: number) {
  const res = await hl.fetchCandles(coin, timeframe, bars);
  if (res.candles.length < SWEEP_WINDOW + 32) {
    throw new Error(`${coin} ${timeframe}: only ${res.candles.length} candles available`);
  }
  return res.candles;
}

/** argmax of one row of a K-wide probability block, or -1 if the row is empty. */
function argmax(row: number[] | undefined): number {
  if (!row || row.length === 0) return -1;
  let best = 0;
  for (let k = 1; k < row.length; k++) if (row[k] > row[best]) best = k;
  return best;
}

async function doFit(req: Extract<JobRequest, { kind: "fit" }>) {
  const candles = await candlesFor(req.coin, req.timeframe, req.bars);
  const fs = buildFeatures(candles, FEATURES);
  const scaler = fitScaler(fs.X, fs.T, fs.D);
  const Z = applyScaler(fs.X, fs.T, fs.D, scaler);

  let params: HmmParams | HsmmParams;
  let logLik: number;
  let converged: boolean;
  let path: Int32Array;
  let durations: number[];

  if (req.modelType === "hsmm") {
    const res = fitHsmm(Z, fs.T, fs.D, {
      states: req.states, seed: req.seed, restarts: req.restarts, maxDuration: req.maxDuration,
    });
    params = res.params; logLik = res.logLik; converged = res.converged;
    path = viterbiHsmm(Z, fs.T, res.params);
    durations = expectedDurations(res.params);
  } else {
    const res = fit(Z, fs.T, fs.D, { states: req.states, seed: req.seed, restarts: req.restarts });
    params = res.params; logLik = res.logLik; converged = res.converged;
    path = viterbi(Z, fs.T, res.params);
    durations = Array.from({ length: res.params.K }, (_, k) => expectedDuration(res.params, k));
  }

  const K = params.K, D = params.D;
  const means = Array.from({ length: K }, (_, k) => unscale(params.mu[k * D], 0, scaler));
  const volIdx = fs.names.indexOf("realizedVol");
  const vols = volIdx < 0 ? null
    : Array.from({ length: K }, (_, k) => Math.exp(unscale(params.mu[k * D + volIdx], volIdx, scaler)));
  const counts = new Array(K).fill(0);
  for (let t = 0; t < fs.T; t++) counts[path[t]]++;
  const labels = labelStates(means);

  // The duration pmf is maxDuration long and mostly zeros; only the modes are
  // worth screen space.
  let durationModes: FitView["durationModes"] = null;
  if (req.modelType === "hsmm") {
    const p = params as HsmmParams;
    durationModes = Array.from({ length: K }, (_, k) => {
      const row = Array.from({ length: p.maxDuration }, (_, d) => ({ state: k, d: d + 1, p: p.dur[k * p.maxDuration + d] }));
      return row.sort((a, b) => b.p - a.p).slice(0, 5).filter((x) => x.p > 0.01);
    });
  }

  const fitView: FitView = {
    coin: req.coin, timeframe: req.timeframe, modelType: req.modelType,
    bars: fs.T,
    logLikPerBar: logLik / fs.T,
    converged,
    states: Array.from({ length: K }, (_, k) => ({
      label: labels[k],
      meanRetBps: means[k] * 10_000,
      volPct: vols ? vols[k] * 100 : 0,
      freq: counts[k] / fs.T,
      durationBars: durations[k],
    })),
    transitions: Array.from(params.A),
    durationModes,
  };

  const chart: ChartView = {
    coin: req.coin, timeframe: req.timeframe,
    candles: candles.slice(fs.index[0]),
    states: Array.from(path),
    positions: new Array(fs.T).fill(0),
    K,
  };

  const model: RuntimeModel = {
    coin: req.coin, timeframe: req.timeframe, modelType: req.modelType,
    params, scaler, names: fs.names, window: SWEEP_WINDOW,
  };
  return { fit: fitView, chart, model };
}

async function doBacktest(req: Extract<JobRequest, { kind: "backtest" }>) {
  const candles = await candlesFor(req.coin, req.timeframe, req.bars);
  const fs = buildFeatures(candles, FEATURES);
  const sizes = walkForwardSizes(fs.T);
  if (!sizes.fits) {
    throw new Error(
      `${req.coin} ${req.timeframe}: ${fs.T} rows, walk-forward needs ${sizes.needed}` +
      ` (train ${sizes.trainSize} / test ${sizes.testSize})`);
  }

  const res = walkForward(
    candles, FEATURES,
    { costBps: req.costBps, durationAware: true },
    {
      modelType: req.modelType, states: req.states, maxDuration: req.maxDuration,
      trainSize: sizes.trainSize, testSize: sizes.testSize,
      seed: req.seed, restarts: req.restarts,
      barsPerYear: hl.hlBarsPerYear(req.timeframe),
    },
  );

  // The permutation test is the expensive honest part: same positions, same
  // exposure, timing destroyed. Reported alongside the ROI so the two are never
  // read apart.
  let pValue: number | null = null;
  let medianRandom: number | null = null;
  if (req.trials > 0) {
    const ts = tradedSeries(candles, FEATURES, res.positions, sizes.trainSize);
    const perm = permutationTest(ts.positions, ts.returns,
      { trials: req.trials, seed: 99, costBps: req.costBps });
    pValue = perm.pValue;
    medianRandom = perm.medianRandom;
  }

  const backtest: BacktestView = {
    coin: req.coin, timeframe: req.timeframe, modelType: req.modelType,
    barsTraded: res.metrics.bars,
    refits: res.refits,
    roi: res.metrics.totalReturn,
    buyHold: res.metrics.buyHoldReturn,
    sharpe: res.metrics.sharpe,
    maxDD: res.metrics.maxDrawdown,
    exposure: res.metrics.exposure,
    trades: res.metrics.trades,
    costDrag: res.metrics.costDrag,
    pValue, medianRandom,
    equity: res.equity,
    equityBuyHold: res.buyHoldEquity,
  };

  const chart: ChartView = {
    coin: req.coin, timeframe: req.timeframe,
    candles: candles.slice(fs.index[0]),
    // Causal states here, unlike the fit tab's Viterbi path: these are the
    // beliefs the strategy actually traded on.
    states: Array.from({ length: fs.T }, (_, i) => argmax(res.stateProbs[i])),
    positions: res.positions,
    K: req.states,
  };

  const model: RuntimeModel | null = res.lastModel && {
    coin: req.coin, timeframe: req.timeframe, modelType: req.modelType,
    params: res.lastModel.params, scaler: res.lastModel.scaler,
    names: fs.names, window: SWEEP_WINDOW,
  };
  return { backtest, chart, model };
}

self.addEventListener("message", async (e) => {
  const req = e.data as JobRequest;
  try {
    let result: unknown;
    switch (req.kind) {
      case "candles":
        result = { candles: await candlesFor(req.coin, req.timeframe, req.bars) };
        break;
      case "fit":
        result = await doFit(req);
        break;
      case "backtest":
        result = await doBacktest(req);
        break;
      case "sweep": {
        const res = await sweep(
          {
            limit: req.limit, timeframes: req.timeframes, days: req.days,
            costBps: req.costBps, trials: req.trials, coins: req.coins,
          },
          (done, total, label) => post({ id: req.id, kind: "progress", done, total, label }),
        );
        result = { rows: res.rows, best: res.best };
        break;
      }
    }
    post({ id: req.id, kind: "ok", result });
  } catch (err) {
    post({ id: req.id, kind: "error", error: err instanceof Error ? err.message : String(err) });
  }
});

export { barBudget };
