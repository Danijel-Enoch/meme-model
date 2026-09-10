/**
 * Two tests that decide whether a backtest means anything.
 *
 * The permutation test asks: does the TIMING carry information, or is the
 * return just payment for being in the market? A long-only strategy in a rising
 * series makes money with no skill at all, and that is the single easiest way
 * to fool yourself with a regime model.
 *
 * The ceiling analysis asks the opposite question: if the model were PERFECT,
 * would there be enough money in this series to cover costs? If not, no amount
 * of modelling helps and the honest move is to stop.
 */

import { makeRng } from "./hmm";
import { buildFeatures, type Candle, type FeatureConfig } from "./features";

export type NullMethod = "shuffle" | "rotate";

export interface PermutationResult {
  method: NullMethod;
  actualReturn: number;
  exposure: number;
  medianRandom: number;
  p05: number;
  p95: number;
  /** Share of random shufflings that matched or beat the strategy. */
  pValue: number;
  trials: number;
}

/** Compound a position series against realized simple returns, net of costs. */
export function compound(positions: number[], returns: number[], costBps = 0): number {
  let eq = 1, prev = 0;
  for (let i = 0; i < positions.length; i++) {
    const cost = Math.abs(positions[i] - prev) * (costBps / 10_000);
    eq *= 1 + positions[i] * returns[i] - cost;
    prev = positions[i];
  }
  return eq - 1;
}

/**
 * Shuffle the same positions to random times and see how often luck wins.
 *
 * The null is "only the amount of time spent in the market matters" — the
 * shuffle preserves exposure exactly and destroys only the timing. A p-value
 * above 0.05 means the strategy has not demonstrated timing skill, whatever
 * its headline return.
 */
export function permutationTest(
  positions: number[],
  returns: number[],
  opts: { trials?: number; seed?: number; costBps?: number; method?: NullMethod } = {},
): PermutationResult {
  const trials = opts.trials ?? 2000;
  const rng = makeRng(opts.seed ?? 99);
  const cost = opts.costBps ?? 0;
  // With costs on, a free shuffle is the wrong null: it scatters long holds
  // into isolated bars, so the shuffled series pays an entry and an exit almost
  // every bar. Any low-turnover strategy then "beats random" for reasons that
  // have nothing to do with timing. A circular rotation keeps every run, and
  // therefore the exact turnover, and changes only WHEN the series is applied.
  const method: NullMethod = opts.method ?? (cost > 0 ? "rotate" : "shuffle");

  const actual = compound(positions, returns, cost);
  const n = positions.length;
  const draws: number[] = [];
  for (let t = 0; t < trials; t++) {
    let candidate: number[];
    if (method === "rotate") {
      const shift = 1 + Math.floor(rng() * Math.max(n - 1, 1));
      candidate = new Array(n);
      for (let i = 0; i < n; i++) candidate[i] = positions[(i + shift) % n];
    } else {
      candidate = [...positions];
      for (let i = candidate.length - 1; i > 0; i--) {
        const j = Math.floor(rng() * (i + 1));
        [candidate[i], candidate[j]] = [candidate[j], candidate[i]];
      }
    }
    draws.push(compound(candidate, returns, cost));
  }
  draws.sort((a, b) => a - b);
  const q = (f: number) => draws[Math.floor(f * (draws.length - 1))];
  const beat = draws.filter((d) => d >= actual).length;

  return {
    method,
    actualReturn: actual,
    exposure: positions.filter((p) => Math.abs(p) > 1e-9).length / positions.length,
    medianRandom: q(0.5),
    p05: q(0.05),
    p95: q(0.95),
    pValue: beat / trials,
    trials,
  };
}

export interface OracleRow {
  name: string;
  holdBars: number;
  costBps: number;
  roi: number;
  trades: number;
  exposure: number;
}

export interface CeilingResult {
  rows: OracleRow[];
  buyHold: number;
  bars: number;
}

/**
 * What perfect foresight is worth on this series, at each cost level.
 *
 * Two oracles, because they bound different things:
 *   - "next bar" knows the sign of the very next return. No causal model can
 *     beat it, so it is the absolute ceiling for a one-step-ahead predictor.
 *     It also trades constantly, which is why it dies fastest to costs.
 *   - "hold k bars" commits for k bars at a time based on the realized forward
 *     k-bar return. That is the honest ceiling for a REGIME model, which is
 *     supposed to make few, durable calls rather than flip every bar.
 *
 * If the k-bar oracle cannot clear your cost assumption, the series does not
 * contain a tradeable regime signal at that cost and the model is not the problem.
 */
export function ceilingAnalysis(
  candles: Candle[],
  featureConfig: FeatureConfig = {},
  opts: { costs?: number[]; holds?: number[]; skip?: number } = {},
): CeilingResult {
  const costs = opts.costs ?? [0, 10, 30];
  const holds = opts.holds ?? [1, 5, 20];
  const fs = buildFeatures(candles, featureConfig);
  const skip = opts.skip ?? 0;

  const returns: number[] = [];
  for (let i = skip; i < fs.T - 1; i++) returns.push(Math.exp(fs.rawReturn[i + 1]) - 1);
  const n = returns.length;

  const rows: OracleRow[] = [];
  for (const hold of holds) {
    // Commit in blocks of `hold` bars, using the realized forward return of the
    // whole block. Perfect knowledge, but only re-decided every `hold` bars.
    const positions = new Array(n).fill(0);
    for (let start = 0; start < n; start += hold) {
      let fwd = 1;
      for (let i = start; i < Math.min(start + hold, n); i++) fwd *= 1 + returns[i];
      const long = fwd > 1;
      for (let i = start; i < Math.min(start + hold, n); i++) positions[i] = long ? 1 : 0;
    }
    let trades = 0, prev = 0;
    for (const p of positions) { if (Math.abs(p - prev) > 1e-9) trades++; prev = p; }
    const exposure = positions.filter((p) => p > 0).length / n;

    for (const costBps of costs) {
      rows.push({
        name: hold === 1 ? "oracle: next bar" : `oracle: hold ${hold} bars`,
        holdBars: hold, costBps,
        roi: compound(positions, returns, costBps),
        trades, exposure,
      });
    }
  }

  let bh = 1;
  for (const r of returns) bh *= 1 + r;
  return { rows, buyHold: bh - 1, bars: n };
}

/** Extract the traded slice of a walk-forward run for the tests above. */
export function tradedSeries(
  candles: Candle[],
  featureConfig: FeatureConfig,
  positions: number[],
  firstTradedRow: number,
): { positions: number[]; returns: number[] } {
  const fs = buildFeatures(candles, featureConfig);
  const pos: number[] = [], ret: number[] = [];
  for (let i = firstTradedRow; i < fs.T - 1; i++) {
    pos.push(positions[i] ?? 0);
    ret.push(Math.exp(fs.rawReturn[i + 1]) - 1);
  }
  return { positions: pos, returns: ret };
}
