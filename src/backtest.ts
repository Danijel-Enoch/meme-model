/**
 * Walk-forward backtest.
 *
 * The rule that makes or breaks an HMM trading study: the model must never see
 * the bar it is trading. Two guards enforce that here.
 *   1. Parameters are re-fit only on a trailing training block, then frozen for
 *      the test block that follows it.
 *   2. State beliefs come from `filter` (forward pass only), not from smoothed
 *      posteriors. Smoothed states look spectacular in-sample and are pure
 *      hindsight — they use future bars to decide what state you were in.
 * The position taken at the close of bar i earns the return of bar i+1.
 */

import { fit, filter, predictNext, expectedDuration, type HmmParams } from "./hmm";
import { fitHsmm, filterHsmm, expectedDurations, type HsmmParams } from "./hsmm";
import { buildFeatures, fitScaler, applyScaler, unscale, type Candle, type FeatureSet, type Scaler, type FeatureConfig } from "./features";

export type ModelType = "hmm" | "hsmm";

export interface StrategyConfig {
  /**
   * If set, position on state confidence instead of blended expected return:
   * go long only when P(next state = most bullish) exceeds this.
   * The blended expectation smears across states; the oracle is rare and sharp.
   * Needs a model that can actually become confident — with a plain HMM the
   * one-step forecast is capped near the transition diagonal, so this is
   * mostly useful with modelType "hsmm".
   */
  confidence?: number;
  /** Go long when expected next-bar return exceeds this, in basis points. */
  entryBps?: number;
  /** Exit only when it drops below this. Hysteresis cuts churn. */
  exitBps?: number;
  allowShort?: boolean;
  /** Round-trip-per-side cost applied to every unit of position change. */
  costBps?: number;
  /**
   * Compare expected edge over the whole expected HOLD to the round trip,
   * instead of demanding a single bar clear it.
   *
   * A regime model holds for the duration of the regime, so the profit of
   * entering is roughly E[return per bar] x E[bars held] - round trip. Testing
   * the per-bar figure against the full round trip is a category error, and it
   * is why a fixed `entryBps` of 2x cost silently produces zero trades: 5bps a
   * bar never clears 9bps, but 5bps across 10 bars clears it comfortably.
   */
  durationAware?: boolean;
  /** Scale position down when the model predicts high volatility. */
  volTarget?: number;
  maxPosition?: number;
}

export interface WalkForwardConfig {
  modelType?: ModelType;
  /** HSMM only: longest dwell time the duration pmf can represent. */
  maxDuration?: number;
  trainSize?: number;
  testSize?: number;
  states?: number;
  seed?: number;
  restarts?: number;
  barsPerYear?: number;
  verbose?: boolean;
}

export interface BacktestMetrics {
  totalReturn: number;
  buyHoldReturn: number;
  sharpe: number;
  maxDrawdown: number;
  hitRate: number;
  trades: number;
  turnover: number;
  exposure: number;
  bars: number;
  costDrag: number;
}

export interface BacktestResult {
  metrics: BacktestMetrics;
  equity: number[];
  buyHoldEquity: number[];
  positions: number[];
  expectedReturns: number[];
  stateProbs: number[][];
  refits: number;
  lastModel: { params: HmmParams | HsmmParams; scaler: Scaler } | null;
}

/**
 * One interface over both models, so the walk-forward harness does not care
 * which is underneath. Both expose a *causal* one-step-ahead state forecast:
 * row i of `nextProb` uses observations up to and including row i, never past it.
 */
interface FittedModel {
  K: number;
  logLik: number;
  meanReturns: number[];
  vols: number[] | null;
  durations: number[];
  /** n*K one-step-ahead state probabilities across the supplied span. */
  nextProb(Z: Float64Array, n: number): Float64Array;
  params: HmmParams | HsmmParams;
}

/** Per-state mean log return, converted back into real units. */
export function stateMeanReturns(p: HmmParams, s: Scaler): number[] {
  return Array.from({ length: p.K }, (_, k) => unscale(p.mu[k * p.D], 0, s));
}

/** Per-state expected volatility per bar, if the vol feature is present. */
function stateVols(p: HmmParams, s: Scaler, names: string[]): number[] | null {
  const d = names.indexOf("realizedVol");
  if (d < 0) return null;
  return Array.from({ length: p.K }, (_, k) => Math.exp(unscale(p.mu[k * p.D + d], d, s)));
}

function decidePosition(
  expR: number, expVol: number | null, prev: number,
  cfg: Required<StrategyConfig>, expectedHoldBars = 1,
): number {
  // Duration-aware: spread the round trip across the bars we expect to hold.
  const entry = cfg.durationAware
    ? (2 * cfg.costBps) / 10_000 / Math.max(expectedHoldBars, 1)
    : cfg.entryBps / 10_000;
  const exit = cfg.durationAware ? 0 : cfg.exitBps / 10_000;

  let target: number;
  if (prev > 0) target = expR > exit ? 1 : 0;            // stay long until conviction fades
  else if (prev < 0) target = expR < -exit ? -1 : 0;
  else target = expR > entry ? 1 : cfg.allowShort && expR < -entry ? -1 : 0;

  if (target !== 0 && cfg.volTarget > 0 && expVol && expVol > 0) {
    target *= Math.min(1, cfg.volTarget / expVol);
  }
  return Math.max(-cfg.maxPosition, Math.min(cfg.maxPosition, target));
}

function fitModel(
  trainZ: Float64Array, n: number, D: number,
  scaler: Scaler, names: string[], wf: WalkForwardConfig, refits: number,
): FittedModel {
  const seed = (wf.seed ?? 42) + refits;
  const states = wf.states ?? 3;

  if ((wf.modelType ?? "hmm") === "hsmm") {
    const res = fitHsmm(trainZ, n, D, {
      states, seed, restarts: wf.restarts ?? 4,
      maxDuration: wf.maxDuration ?? 60,
    });
    const p = res.params;
    return {
      K: p.K, logLik: res.logLik, params: p,
      meanReturns: Array.from({ length: p.K }, (_, k) => unscale(p.mu[k * p.D], 0, scaler)),
      vols: hsmmVols(p, scaler, names),
      durations: expectedDurations(p),
      nextProb: (Z, len) => filterHsmm(Z, len, p).nextProb,
    };
  }

  const res = fit(trainZ, n, D, { states, seed, restarts: wf.restarts ?? 4, verbose: false });
  const p = res.params;
  return {
    K: p.K, logLik: res.logLik, params: p,
    meanReturns: stateMeanReturns(p, scaler),
    vols: stateVols(p, scaler, names),
    durations: Array.from({ length: p.K }, (_, k) => expectedDuration(p, k)),
    nextProb: (Z, len) => {
      const { alpha } = filter(Z, len, p);
      const out = new Float64Array(len * p.K);
      for (let t = 0; t < len; t++) {
        const nx = predictNext(alpha, t * p.K, p);
        for (let k = 0; k < p.K; k++) out[t * p.K + k] = nx[k];
      }
      return out;
    },
  };
}

function hsmmVols(p: HsmmParams, s: Scaler, names: string[]): number[] | null {
  const d = names.indexOf("realizedVol");
  if (d < 0) return null;
  return Array.from({ length: p.K }, (_, k) => Math.exp(unscale(p.mu[k * p.D + d], d, s)));
}

export interface Trade {
  /** Feature-row indices; entry is decided at the close of the entry bar. */
  entryRow: number;
  exitRow: number;
  /** Unix seconds of the bar whose close the position was taken/left at. */
  entryTime: number;
  exitTime: number;
  direction: 1 | -1;
  barsHeld: number;
  entryPrice: number;
  exitPrice: number;
  /** Price move in the traded direction, before costs. */
  grossReturn: number;
  costPaid: number;
  netReturn: number;
  win: boolean;
}

/**
 * Turn a position series into a ledger of round trips.
 *
 * A trade opens when the position leaves zero and closes when it returns to
 * zero or flips sign. Costs are charged on both legs, so `netReturn` is what
 * the trade actually earned. Win rate over these is a fairer read than a
 * per-bar hit rate: a regime model is supposed to make few durable calls, and
 * per-bar accounting flatters a strategy that sits in one good position.
 */
export function extractTrades(
  candles: Candle[],
  featureConfig: FeatureConfig,
  positions: number[],
  costBps: number,
  fromRow = 0,
): Trade[] {
  const fs = buildFeatures(candles, featureConfig);
  const trades: Trade[] = [];
  const cost = costBps / 10_000;

  let open: { row: number; dir: 1 | -1; equity: number } | null = null;

  const close = (exitRow: number) => {
    if (!open) return;
    const gross = open.equity - 1;
    // Two legs: one to get in, one to get out.
    const costPaid = 2 * cost;
    const net = (1 + gross) * (1 - cost) * (1 - cost) - 1;
    trades.push({
      entryRow: open.row,
      exitRow,
      entryTime: candles[fs.index[open.row]].time,
      exitTime: candles[fs.index[Math.min(exitRow, fs.T - 1)]].time,
      direction: open.dir,
      barsHeld: exitRow - open.row,
      entryPrice: candles[fs.index[open.row]].close,
      exitPrice: candles[fs.index[Math.min(exitRow, fs.T - 1)]].close,
      grossReturn: gross,
      costPaid,
      netReturn: net,
      win: net > 0,
    });
    open = null;
  };

  for (let i = fromRow; i < fs.T - 1; i++) {
    const pos = positions[i] ?? 0;
    const dir: 1 | -1 | 0 = pos > 1e-9 ? 1 : pos < -1e-9 ? -1 : 0;

    if (open && dir !== open.dir) close(i);
    if (!open && dir !== 0) open = { row: i, dir, equity: 1 };

    if (open) {
      // The position held at row i earns the return of row i+1.
      const barRet = Math.exp(fs.rawReturn[i + 1]) - 1;
      open.equity *= 1 + open.dir * barRet;
    }
  }
  if (open) close(fs.T - 1);
  return trades;
}

export interface TradeStats {
  trades: number;
  wins: number;
  winRate: number;
  roi: number;
  avgWin: number;
  avgLoss: number;
  bestTrade: number;
  worstTrade: number;
  avgBarsHeld: number;
  profitFactor: number;
}

/** Compound a set of trades as if each was taken with the full stake in turn. */
export function summarizeTrades(trades: Trade[]): TradeStats {
  if (trades.length === 0) {
    return { trades: 0, wins: 0, winRate: 0, roi: 0, avgWin: 0, avgLoss: 0,
             bestTrade: 0, worstTrade: 0, avgBarsHeld: 0, profitFactor: 0 };
  }
  let eq = 1;
  for (const t of trades) eq *= 1 + t.netReturn;
  const wins = trades.filter((t) => t.win);
  const losses = trades.filter((t) => !t.win);
  const sum = (a: Trade[]) => a.reduce((x, t) => x + t.netReturn, 0);
  const grossWin = sum(wins);
  const grossLoss = Math.abs(sum(losses));
  return {
    trades: trades.length,
    wins: wins.length,
    winRate: wins.length / trades.length,
    roi: eq - 1,
    avgWin: wins.length ? grossWin / wins.length : 0,
    avgLoss: losses.length ? -grossLoss / losses.length : 0,
    bestTrade: Math.max(...trades.map((t) => t.netReturn)),
    worstTrade: Math.min(...trades.map((t) => t.netReturn)),
    avgBarsHeld: trades.reduce((a, t) => a + t.barsHeld, 0) / trades.length,
    profitFactor: grossLoss > 0 ? grossWin / grossLoss : Infinity,
  };
}

export function walkForward(
  candles: Candle[],
  featureConfig: FeatureConfig = {},
  strategy: StrategyConfig = {},
  wf: WalkForwardConfig = {},
): BacktestResult {
  const cfg: Required<StrategyConfig> = {
    // Default the entry bar to a round trip: taking a signal whose expected edge
    // is smaller than the cost of expressing it is the fastest way to bleed out.
    costBps: strategy.costBps ?? 30,
    entryBps: strategy.entryBps ?? 2 * (strategy.costBps ?? 30),
    exitBps: strategy.exitBps ?? 0,
    allowShort: strategy.allowShort ?? false,
    volTarget: strategy.volTarget ?? 0,
    maxPosition: strategy.maxPosition ?? 1,
    confidence: strategy.confidence ?? 0,
    durationAware: strategy.durationAware ?? false,
  };
  const trainSize = wf.trainSize ?? 1500;
  const testSize = wf.testSize ?? 500;
  const barsPerYear = wf.barsPerYear ?? 105_120; // 5-minute bars

  const fs: FeatureSet = buildFeatures(candles, featureConfig);
  const { X, T, D, names } = fs;
  if (T < trainSize + testSize + 2) {
    throw new Error(`need ${trainSize + testSize + 2} feature rows for this walk-forward config, have ${T}`);
  }

  const positions = new Array<number>(T).fill(0);
  const expectedReturns = new Array<number>(T).fill(0);
  const stateProbs: number[][] = Array.from({ length: T }, () => []);

  let refits = 0;
  let lastModel: { params: HmmParams | HsmmParams; scaler: Scaler } | null = null;

  for (let start = 0; start + trainSize < T - 1; start += testSize) {
    const trainEnd = start + trainSize;
    // Absorb a short trailing remainder into the final block.
    let testEnd = Math.min(trainEnd + testSize, T);
    if (T - testEnd < testSize / 2) testEnd = T;

    // Scaler and model both see training rows only.
    const trainX = X.slice(start * D, trainEnd * D);
    const scaler = fitScaler(trainX, trainEnd - start, D);
    const trainZ = applyScaler(trainX, trainEnd - start, D, scaler);

    const model = fitModel(trainZ, trainEnd - start, D, scaler, names, wf, refits);
    refits++;
    lastModel = { params: model.params, scaler };

    const muRet = model.meanReturns;
    const vols = model.vols;

    // Run the model across train+test so the test block inherits a warmed-up
    // belief, while row i still depends on rows <= i only.
    const spanZ = applyScaler(X.slice(start * D, testEnd * D), testEnd - start, D, scaler);
    const pred = model.nextProb(spanZ, testEnd - start);

    let prev = 0;
    for (let i = trainEnd; i < testEnd; i++) {
      const local = i - start;
      let expR = 0;
      let expVol = 0;
      const row: number[] = [];
      for (let k = 0; k < model.K; k++) {
        const pk = pred[local * model.K + k];
        row.push(pk);
        expR += pk * muRet[k];
        if (vols) expVol += pk * vols[k];
      }
      // Expected hold = how long the model thinks the most bullish state lasts.
      const holdBars = model.durations[model.K - 1] ?? 1;
      const pos = cfg.confidence > 0
        ? (row[model.K - 1] > cfg.confidence ? 1 : 0)
        : decidePosition(expR, vols ? expVol : null, prev, cfg, holdBars);
      positions[i] = pos;
      expectedReturns[i] = expR;
      stateProbs[i] = row;
      prev = pos;
    }

    if (wf.verbose) {
      console.error(
        `  block ${refits}: train[${start}:${trainEnd}] test[${trainEnd}:${testEnd}] ` +
        `logLik/T=${(model.logLik / (trainEnd - start)).toFixed(3)} ` +
        `stateRet=[${muRet.map((r) => (r * 10_000).toFixed(1)).join(", ")}]bps ` +
        `dur=[${model.durations.map((d) => d.toFixed(0)).join(", ")}]`,
      );
    }
    if (testEnd >= T) break;
  }

  // --- PnL accounting ---
  const firstTest = trainSize;
  const equity: number[] = [1];
  const buyHoldEquity: number[] = [1];
  let eq = 1, bh = 1, peak = 1, maxDd = 0;
  let wins = 0, decided = 0, trades = 0, turnover = 0, exposed = 0, costDrag = 0;
  let prevPos = 0;
  const rets: number[] = [];

  for (let i = firstTest; i < T - 1; i++) {
    const pos = positions[i];
    const barRet = Math.exp(fs.rawReturn[i + 1]) - 1; // simple return of the next bar

    const delta = Math.abs(pos - prevPos);
    const cost = delta * (cfg.costBps / 10_000);
    if (delta > 1e-9) trades++;
    turnover += delta;
    costDrag += cost;

    const net = pos * barRet - cost;
    eq *= 1 + net;
    bh *= 1 + barRet;
    // Log returns for the risk stats: arithmetic means can be positive while the
    // equity curve compounds downward (volatility drag), which reads as a great
    // Sharpe on a losing strategy. Log returns keep the two consistent.
    rets.push(Math.log(Math.max(1 + net, 1e-9)));

    if (Math.abs(pos) > 1e-9) {
      exposed++;
      decided++;
      if (pos * barRet > 0) wins++;
    }
    peak = Math.max(peak, eq);
    maxDd = Math.max(maxDd, 1 - eq / peak);

    equity.push(eq);
    buyHoldEquity.push(bh);
    prevPos = pos;
  }

  const n = rets.length || 1;
  const mean = rets.reduce((a, b) => a + b, 0) / n;
  const variance = rets.reduce((a, b) => a + (b - mean) * (b - mean), 0) / Math.max(n - 1, 1);
  const sd = Math.sqrt(variance);
  const sharpe = sd > 0 ? (mean / sd) * Math.sqrt(barsPerYear) : 0;

  return {
    metrics: {
      totalReturn: eq - 1,
      buyHoldReturn: bh - 1,
      sharpe,
      maxDrawdown: maxDd,
      hitRate: decided > 0 ? wins / decided : 0,
      trades,
      turnover,
      exposure: n > 0 ? exposed / n : 0,
      bars: n,
      costDrag,
    },
    equity,
    buyHoldEquity,
    positions,
    expectedReturns,
    stateProbs,
    refits,
    lastModel,
  };
}
