/**
 * Top-down multi-timeframe confluence.
 *
 * One Markov model per timeframe, combined into a scalping signal:
 *
 *   BIAS    (slow, e.g. 1h)  — which way is the regime leaning? Gate only.
 *   SETUP   (mid,  e.g. 15m) — does the intermediate structure agree?
 *   TRIGGER (fast, e.g. 5m)  — enter here, so the stop is close and the
 *                              holding period short.
 *
 * The reason to expect this to beat any single timeframe is not mysticism about
 * "confluence". It is that the per-bar signal grows with bar length (measured:
 * ~5bps at 1m, ~63bps at 1h) while the cost of a trade is fixed. Taking the
 * DIRECTION from a slow chart and the TIMING from a fast one is an attempt to
 * get the large signal and the cheap entry at once.
 *
 * ---------------------------------------------------------------------------
 * THE LOOKAHEAD TRAP
 *
 * At 10:05 the 10:00-11:00 hourly bar does not exist yet. Its close is an hour
 * in the future. Backtests that read the "current" hourly bar at every 5m step
 * are reading tomorrow's newspaper, and they produce spectacular equity curves.
 *
 * Everything here aligns through `lastCompletedIndex`, which maps a base bar to
 * the most recent higher-timeframe bar that had already CLOSED. A test asserts
 * that corrupting future base bars leaves every earlier signal untouched.
 * ---------------------------------------------------------------------------
 */

import { fit, filter, predictNext } from "./hmm";
import { fitHsmm, filterHsmm } from "./hsmm";
import {
  buildFeatures, fitScaler, applyScaler, unscale,
  type Candle, type FeatureConfig,
} from "./features";
import type { ModelType } from "./backtest";

export interface TimeframeSpec {
  label: string;
  /** How many base bars make one bar at this timeframe. 1 = the base itself. */
  factor: number;
  /** Role in the stack. Bias and setup gate; trigger fires the entry. */
  role: "bias" | "setup" | "trigger";
}

export const DEFAULT_STACK: TimeframeSpec[] = [
  { label: "1h", factor: 12, role: "bias" },
  { label: "15m", factor: 3, role: "setup" },
  { label: "5m", factor: 1, role: "trigger" },
];

/** Roll `factor` consecutive base candles into one. */
export function aggregate(candles: Candle[], factor: number): Candle[] {
  if (factor <= 1) return candles;
  const out: Candle[] = [];
  for (let i = 0; i + factor <= candles.length; i += factor) {
    let high = -Infinity, low = Infinity, volume = 0;
    for (let j = i; j < i + factor; j++) {
      if (candles[j].high > high) high = candles[j].high;
      if (candles[j].low < low) low = candles[j].low;
      volume += candles[j].volume;
    }
    out.push({
      time: candles[i].time,
      open: candles[i].open,
      high, low,
      close: candles[i + factor - 1].close,
      volume,
    });
  }
  return out;
}

/**
 * Index of the most recent higher-timeframe bar that has CLOSED by the end of
 * base bar `i`. Returns -1 when none has closed yet.
 *
 * HTF bar j spans base bars [j*factor, j*factor + factor - 1] and is complete
 * only at the last of them, so the count of finished bars is floor((i+1)/factor).
 */
export function lastCompletedIndex(i: number, factor: number): number {
  return Math.floor((i + 1) / factor) - 1;
}

export interface TimeframeSignal {
  label: string;
  role: TimeframeSpec["role"];
  /** Per base bar: expected next-bar return at this timeframe, in real units. */
  expectedReturn: Float64Array;
  /** Per base bar: probability mass on the most bullish state. */
  bullProb: Float64Array;
  /** Per base bar: probability mass on the most bearish state. */
  bearProb: Float64Array;
  /** Per base bar: true once this timeframe has a usable, completed reading. */
  ready: Uint8Array;
}

export interface ConfluenceOptions {
  featureConfig?: FeatureConfig;
  states?: number;
  modelType?: ModelType;
  maxDuration?: number;
  seed?: number;
  restarts?: number;
}

/**
 * Fit one model per timeframe on a training window, then emit causally aligned
 * per-base-bar readings across the whole span.
 *
 * `trainBaseRows` counts BASE candles; each timeframe trains on however many of
 * its own bars fit inside that stretch of history.
 */
export function buildSignals(
  candles: Candle[],
  stack: TimeframeSpec[],
  trainBaseBars: number,
  spanBaseBars: number,
  opts: ConfluenceOptions = {},
): TimeframeSignal[] {
  const featureConfig = opts.featureConfig ?? { window: 5 };
  const states = opts.states ?? 3;
  const modelType = opts.modelType ?? "hmm";

  return stack.map((tf) => {
    const agg = aggregate(candles, tf.factor);
    const fs = buildFeatures(agg, featureConfig);

    // Which aggregated bars belong to the training stretch, and which to the span.
    const trainAggBars = Math.floor(trainBaseBars / tf.factor);
    const spanAggBars = Math.floor(spanBaseBars / tf.factor);
    // Feature rows lag the candle array by the feature window.
    const rowOffset = agg.length - fs.T;
    const trainRows = Math.max(trainAggBars - rowOffset, 0);
    const spanRows = Math.min(Math.max(spanAggBars - rowOffset, 0), fs.T);

    const expectedReturn = new Float64Array(spanBaseBars);
    const bullProb = new Float64Array(spanBaseBars);
    const bearProb = new Float64Array(spanBaseBars);
    const ready = new Uint8Array(spanBaseBars);

    if (trainRows < states * 12 || spanRows <= trainRows) {
      return { label: tf.label, role: tf.role, expectedReturn, bullProb, bearProb, ready };
    }

    // Scaler and model see training rows only.
    const trainX = fs.X.slice(0, trainRows * fs.D);
    const scaler = fitScaler(trainX, trainRows, fs.D);
    const trainZ = applyScaler(trainX, trainRows, fs.D, scaler);
    const spanZ = applyScaler(fs.X.slice(0, spanRows * fs.D), spanRows, fs.D, scaler);

    let K: number;
    let muRet: number[];
    let nextP: Float64Array;

    if (modelType === "hsmm") {
      const res = fitHsmm(trainZ, trainRows, fs.D, {
        states, seed: opts.seed ?? 42, restarts: opts.restarts ?? 2,
        maxDuration: opts.maxDuration ?? 30,
      });
      K = res.params.K;
      muRet = Array.from({ length: K }, (_, k) => unscale(res.params.mu[k * fs.D], 0, scaler));
      nextP = filterHsmm(spanZ, spanRows, res.params).nextProb;
    } else {
      const res = fit(trainZ, trainRows, fs.D, { states, seed: opts.seed ?? 42, restarts: opts.restarts ?? 2 });
      K = res.params.K;
      muRet = Array.from({ length: K }, (_, k) => unscale(res.params.mu[k * fs.D], 0, scaler));
      const { alpha } = filter(spanZ, spanRows, res.params);
      nextP = new Float64Array(spanRows * K);
      for (let t = 0; t < spanRows; t++) {
        const nx = predictNext(alpha, t * K, res.params);
        for (let k = 0; k < K; k++) nextP[t * K + k] = nx[k];
      }
    }

    // Per-row readings, then broadcast to base bars through completed-bar alignment.
    const rowExp = new Float64Array(spanRows);
    const rowBull = new Float64Array(spanRows);
    const rowBear = new Float64Array(spanRows);
    for (let t = 0; t < spanRows; t++) {
      let e = 0;
      for (let k = 0; k < K; k++) e += nextP[t * K + k] * muRet[k];
      rowExp[t] = e;
      rowBull[t] = nextP[t * K + (K - 1)];
      rowBear[t] = nextP[t * K + 0];
    }

    for (let i = 0; i < spanBaseBars; i++) {
      const aggIdx = lastCompletedIndex(i, tf.factor);
      const row = aggIdx - rowOffset;
      // Only readings from a bar that has already closed, and only after the
      // model's own training window.
      if (row < trainRows || row >= spanRows) continue;
      expectedReturn[i] = rowExp[row];
      bullProb[i] = rowBull[row];
      bearProb[i] = rowBear[row];
      ready[i] = 1;
    }

    return { label: tf.label, role: tf.role, expectedReturn, bullProb, bearProb, ready };
  });
}

export interface ConfluenceRule {
  /** Bias/setup gates pass when their expected return clears this, in bps. */
  gateBps?: number;
  /** Trigger fires when its expected return clears this, in bps. */
  triggerBps?: number;
  /** Optional extra demand: the bias timeframe must also be this confident. */
  biasConfidence?: number;
  /** Exit as soon as the bias flips, rather than waiting for the trigger. */
  exitOnBiasFlip?: boolean;
  allowShort?: boolean;
}

/**
 * Combine the stack into a position series, one entry per base bar.
 *
 * Long needs all three to agree. Exit is deliberately looser than entry: a
 * scalp should be abandoned the moment the structure that justified it breaks,
 * so a single failing gate closes the position even though it took three to
 * open it.
 */
export function combineSignals(
  signals: TimeframeSignal[],
  n: number,
  rule: ConfluenceRule = {},
): { positions: number[]; agreement: number[] } {
  const gate = (rule.gateBps ?? 0) / 10_000;
  const trigger = (rule.triggerBps ?? 0) / 10_000;
  const biasConf = rule.biasConfidence ?? 0;
  const exitOnFlip = rule.exitOnBiasFlip ?? true;
  const allowShort = rule.allowShort ?? false;

  const bias = signals.find((s) => s.role === "bias");
  const setup = signals.find((s) => s.role === "setup");
  const trig = signals.find((s) => s.role === "trigger");
  if (!bias || !setup || !trig) throw new Error("stack needs a bias, a setup and a trigger timeframe");

  const positions = new Array<number>(n).fill(0);
  const agreement = new Array<number>(n).fill(0);
  let prev = 0;

  for (let i = 0; i < n; i++) {
    const allReady = bias.ready[i] && setup.ready[i] && trig.ready[i];
    if (!allReady) { positions[i] = 0; prev = 0; continue; }

    const bullVotes =
      (bias.expectedReturn[i] > gate ? 1 : 0) +
      (setup.expectedReturn[i] > gate ? 1 : 0) +
      (trig.expectedReturn[i] > trigger ? 1 : 0);
    const bearVotes =
      (bias.expectedReturn[i] < -gate ? 1 : 0) +
      (setup.expectedReturn[i] < -gate ? 1 : 0) +
      (trig.expectedReturn[i] < -trigger ? 1 : 0);
    agreement[i] = bullVotes - bearVotes;

    const biasLong = bias.expectedReturn[i] > gate && bias.bullProb[i] >= biasConf;
    const biasShort = bias.expectedReturn[i] < -gate && bias.bearProb[i] >= biasConf;

    let pos = 0;
    if (prev > 0) {
      // Hold while the structure survives.
      const broken = exitOnFlip ? !biasLong : bullVotes === 0;
      pos = broken ? 0 : 1;
    } else if (prev < 0) {
      const broken = exitOnFlip ? !biasShort : bearVotes === 0;
      pos = broken ? 0 : -1;
    } else {
      if (biasLong && bullVotes === 3) pos = 1;
      else if (allowShort && biasShort && bearVotes === 3) pos = -1;
    }
    positions[i] = pos;
    prev = pos;
  }
  return { positions, agreement };
}

export interface ConfluenceResult {
  positions: number[];
  signals: TimeframeSignal[];
  agreement: number[];
  /** Base feature rows; positions are indexed by these. */
  rows: number;
  firstTradableRow: number;
}

/**
 * Walk-forward over the base timeframe, refitting all three models each block.
 * Positions are indexed by BASE FEATURE ROW, matching walkForward's convention,
 * so the same backtest and diagnostics accounting applies unchanged.
 */
export function walkForwardConfluence(
  candles: Candle[],
  stack: TimeframeSpec[] = DEFAULT_STACK,
  opts: ConfluenceOptions & { trainSize?: number; testSize?: number } = {},
  rule: ConfluenceRule = {},
): ConfluenceResult {
  const featureConfig = opts.featureConfig ?? { window: 5 };
  const trainSize = opts.trainSize ?? 2000;
  const testSize = opts.testSize ?? 500;

  const baseFs = buildFeatures(candles, featureConfig);
  const T = baseFs.T;
  const rowOffset = candles.length - T;

  const positions = new Array<number>(T).fill(0);
  const agreement = new Array<number>(T).fill(0);
  let lastSignals: TimeframeSignal[] = [];

  for (let start = 0; start + trainSize < T - 1; start += testSize) {
    const trainEnd = start + trainSize;
    let testEnd = Math.min(trainEnd + testSize, T);
    if (T - testEnd < testSize / 2) testEnd = T;

    // Work in candle-index space so every timeframe agrees on where "now" is.
    const trainBaseBars = trainEnd + rowOffset;
    const spanBaseBars = testEnd + rowOffset;

    const signals = buildSignals(candles, stack, trainBaseBars, spanBaseBars, opts);
    lastSignals = signals;
    const combined = combineSignals(signals, spanBaseBars, rule);

    for (let i = trainEnd; i < testEnd; i++) {
      positions[i] = combined.positions[i + rowOffset] ?? 0;
      agreement[i] = combined.agreement[i + rowOffset] ?? 0;
    }
    if (testEnd >= T) break;
  }

  return { positions, signals: lastSignals, agreement, rows: T, firstTradableRow: trainSize };
}
