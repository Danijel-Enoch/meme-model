/**
 * Monte Carlo, for two questions the rest of the repo cannot answer.
 *
 * 1. HOW OFTEN DOES THIS PIPELINE CRY WOLF?
 *    Every p-value here is computed for a single strategy on a single series.
 *    But a research process involves choices — timeframe, states, window, model,
 *    which of ten coins to look at — and each one is another chance to get
 *    lucky. The only honest way to calibrate that is to run the whole procedure
 *    on data that provably contains no signal and count how often it declares
 *    victory. If the rate is 5%, a p-value of 0.04 means what it says. If it is
 *    30%, nothing reported anywhere in this repo should be believed.
 *
 * 2. WHAT IS THE SPREAD OF OUTCOMES, NOT JUST THE ONE THAT HAPPENED?
 *    A backtest returns one number from one path. A $50 account at 2x that
 *    ended at $76 might have had a 20% chance of ending under $30. Bootstrapping
 *    the realised trades gives the distribution the single run is a draw from,
 *    which is what position sizing actually needs.
 *
 * Neither creates edge. The first tells you whether an apparent edge is real;
 * the second tells you what to risk if it is.
 */

import { makeRng } from "./hmm";
import type { Candle } from "./features";
import type { PerpTrade } from "./perp";

/**
 * A null series: same marginal return distribution, same fat tails, same
 * return/volume relationship — but no temporal structure at all.
 *
 * Log returns are paired with their bar's volume and wick geometry, then the
 * pairs are shuffled and the price path rebuilt. Anything a regime model finds
 * in this is by construction an artefact, because consecutive bars are now
 * independent draws. Keeping the pairing matters: shuffling returns while
 * leaving volume in place would destroy a correlation the model legitimately
 * uses, making the null easier than reality and flattering the pipeline.
 */
export function shuffledSeries(candles: Candle[], seed: number): Candle[] {
  const rng = makeRng(seed);
  const n = candles.length;
  if (n < 3) return candles.slice();

  interface Step { logRet: number; volume: number; hiFrac: number; loFrac: number }
  const steps: Step[] = [];
  for (let i = 1; i < n; i++) {
    const prev = Math.max(candles[i - 1].close, 1e-18);
    const cur = Math.max(candles[i].close, 1e-18);
    const body = Math.max(candles[i].open, candles[i].close);
    const bodyLo = Math.min(candles[i].open, candles[i].close);
    steps.push({
      logRet: Math.log(cur / prev),
      volume: candles[i].volume,
      // Wick sizes as a fraction of price, so they rescale with the new path.
      hiFrac: body > 0 ? Math.max(candles[i].high - body, 0) / body : 0,
      loFrac: bodyLo > 0 ? Math.max(bodyLo - candles[i].low, 0) / bodyLo : 0,
    });
  }

  for (let i = steps.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [steps[i], steps[j]] = [steps[j], steps[i]];
  }

  const out: Candle[] = [{ ...candles[0] }];
  let price = candles[0].close;
  for (let i = 0; i < steps.length; i++) {
    const s = steps[i];
    const open = price;
    price = Math.max(price * Math.exp(s.logRet), 1e-18);
    const close = price;
    const bodyHi = Math.max(open, close);
    const bodyLo = Math.min(open, close);
    out.push({
      time: candles[i + 1].time,
      open,
      high: bodyHi * (1 + s.hiFrac),
      low: bodyLo * (1 - s.loFrac),
      close,
      volume: s.volume,
    });
  }
  return out;
}

/**
 * A gentler null that keeps volatility clustering: resample in contiguous
 * blocks rather than one bar at a time.
 *
 * An iid shuffle destroys the vol clustering that real markets have, which
 * could make the null unrealistically easy and understate the false-positive
 * rate. Blocks preserve short-range structure while still severing any
 * relationship between one block and the next.
 */
export function blockShuffledSeries(candles: Candle[], blockSize: number, seed: number): Candle[] {
  const rng = makeRng(seed);
  const n = candles.length;
  if (n < 3 || blockSize < 1) return candles.slice();

  const rets: number[] = [];
  const vols: number[] = [];
  for (let i = 1; i < n; i++) {
    rets.push(Math.log(Math.max(candles[i].close, 1e-18) / Math.max(candles[i - 1].close, 1e-18)));
    vols.push(candles[i].volume);
  }

  const picked: { r: number; v: number }[] = [];
  while (picked.length < rets.length) {
    const start = Math.floor(rng() * Math.max(rets.length - blockSize, 1));
    for (let k = 0; k < blockSize && picked.length < rets.length; k++) {
      picked.push({ r: rets[start + k], v: vols[start + k] });
    }
  }

  const out: Candle[] = [{ ...candles[0] }];
  let price = candles[0].close;
  for (let i = 0; i < picked.length; i++) {
    const open = price;
    price = Math.max(price * Math.exp(picked[i].r), 1e-18);
    out.push({
      time: candles[i + 1].time,
      open,
      high: Math.max(open, price),
      low: Math.min(open, price),
      close: price,
      volume: picked[i].v,
    });
  }
  return out;
}

export interface FalsePositiveResult {
  trials: number;
  /** Trials where the permutation test reported p < alpha. */
  significant: number;
  rate: number;
  alpha: number;
  /** Nominal rate if the test were perfectly calibrated. */
  expected: number;
  pValues: number[];
  /** ROI of each null trial, for context on how good "nothing" can look. */
  rois: number[];
  /** Trials where the strategy took no trades at all. */
  noTrades: number;
}

/**
 * Run a strategy-and-test procedure over many null series and count how often
 * it finds significance. `runner` returns the p-value and ROI for one series.
 */
export function falsePositiveRate(
  candles: Candle[],
  runner: (nullCandles: Candle[], trial: number) => { pValue: number; roi: number; trades: number } | null,
  opts: { trials?: number; alpha?: number; seed?: number; blockSize?: number } = {},
): FalsePositiveResult {
  const trials = opts.trials ?? 100;
  const alpha = opts.alpha ?? 0.05;
  const seed = opts.seed ?? 1234;
  const blockSize = opts.blockSize ?? 0;

  const pValues: number[] = [];
  const rois: number[] = [];
  let significant = 0;
  let noTrades = 0;

  for (let t = 0; t < trials; t++) {
    const nullSeries = blockSize > 0
      ? blockShuffledSeries(candles, blockSize, seed + t * 7919)
      : shuffledSeries(candles, seed + t * 7919);
    const res = runner(nullSeries, t);
    if (!res || res.trades === 0) { noTrades++; continue; }
    pValues.push(res.pValue);
    rois.push(res.roi);
    if (res.pValue < alpha) significant++;
  }

  return {
    trials,
    significant,
    rate: pValues.length > 0 ? significant / pValues.length : 0,
    alpha,
    expected: alpha,
    pValues,
    rois,
    noTrades,
  };
}

export interface BootstrapResult {
  trials: number;
  startingEquity: number;
  median: number;
  mean: number;
  p05: number;
  p25: number;
  p75: number;
  p95: number;
  worst: number;
  best: number;
  /** Share of resampled paths that ended below the starting equity. */
  probLoss: number;
  /** Share that lost at least half the account. */
  probHalved: number;
  /** Share that were wiped out (equity floored at zero). */
  probRuin: number;
}

/**
 * Resample the realised trades to see what else could have happened.
 *
 * Trades are drawn with replacement and compounded in a random order, which
 * treats each trade's return as an independent draw from the strategy's
 * distribution. That is the right null for "was this run lucky", though it
 * assumes trade outcomes are independent — if the strategy holds correlated
 * positions through one big move, the true spread is wider than this suggests.
 */
export function bootstrapTrades(
  trades: Pick<PerpTrade, "netPnl" | "equityAfter">[],
  startingEquity: number,
  opts: { trials?: number; seed?: number } = {},
): BootstrapResult {
  const trials = opts.trials ?? 5000;
  const rng = makeRng(opts.seed ?? 4242);

  // Each trade's return ON EQUITY, recovered exactly from the ledger:
  // equity before the trade is equityAfter minus the trade's P&L. Leverage is
  // already baked into netPnl, so it must not be applied again here.
  const returns: number[] = [];
  for (const t of trades) {
    const before = t.equityAfter - t.netPnl;
    if (before > 0) returns.push(t.netPnl / before);
  }
  if (returns.length === 0) {
    return {
      trials: 0, startingEquity, median: startingEquity, mean: startingEquity,
      p05: startingEquity, p25: startingEquity, p75: startingEquity, p95: startingEquity,
      worst: startingEquity, best: startingEquity, probLoss: 0, probHalved: 0, probRuin: 0,
    };
  }

  const finals: number[] = [];
  let ruin = 0;
  for (let i = 0; i < trials; i++) {
    let eq = startingEquity;
    for (let k = 0; k < returns.length; k++) {
      eq = eq * (1 + returns[Math.floor(rng() * returns.length)]);
      if (eq <= 0) { eq = 0; break; }
    }
    if (eq <= 0) ruin++;
    finals.push(eq);
  }
  finals.sort((a, b) => a - b);
  const q = (f: number) => finals[Math.min(finals.length - 1, Math.floor(f * finals.length))];

  return {
    trials,
    startingEquity,
    median: q(0.5),
    mean: finals.reduce((a, b) => a + b, 0) / finals.length,
    p05: q(0.05),
    p25: q(0.25),
    p75: q(0.75),
    p95: q(0.95),
    worst: finals[0],
    best: finals[finals.length - 1],
    probLoss: finals.filter((x) => x < startingEquity).length / finals.length,
    probHalved: finals.filter((x) => x < startingEquity / 2).length / finals.length,
    probRuin: ruin / finals.length,
  };
}
