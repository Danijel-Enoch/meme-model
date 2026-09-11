/**
 * One hypothesis, every coin, tested once.
 *
 * The sweep asks 240 separate questions ("does this model beat chance on this
 * coin at this timeframe?") and each answer rests on ~1000 out-of-sample bars.
 * That is the worst possible arrangement of a fixed amount of data: the
 * multiple-comparison burden scales with the number of questions while the
 * evidence per question stays tiny, so the best cell in the table is a maximum
 * of 240 noisy draws and its p-value means nothing.
 *
 * The panel test rearranges the same data into the opposite shape. One
 * pre-specified configuration — no per-coin timeframe, no per-coin model — is
 * applied to every coin, and the question becomes a single one:
 *
 *     do this model's positions carry timing information, across the market?
 *
 * Thirty coins x ~1000 bars is ~30k observations against ONE hypothesis. The
 * standard error of the pooled mean falls by sqrt(30) — a factor of 5.5 — and
 * there is no selection to correct for, because nothing was selected.
 *
 * ── The trap this file exists to avoid ──────────────────────────────────────
 *
 * Crypto perps are not independent. They are one market factor plus noise: when
 * BTC moves, thirty coins move. A pooled test whose null draws each coin
 * independently destroys that dependence, so the null distribution comes out far
 * too narrow and any long-biased strategy looks significant. On a rising 45-day
 * window that failure mode manufactures p < 0.001 out of nothing.
 *
 * So the default null rotates every coin by the SAME offset. Each draw is the
 * whole market's position matrix, shifted in time as one object: the
 * cross-sectional correlation at every instant survives, only the alignment
 * between positions and the returns they were meant to predict is destroyed.
 * That is the null we want — "the same positions, held at the wrong times" —
 * and it is conservative in exactly the direction that matters.
 *
 * `nullMethod: "independent"` is kept only so the difference can be measured;
 * panel.test.ts asserts it is anti-conservative on market-factor data.
 */

import { buildFeatures, type Candle, type FeatureConfig } from "./features";
import { walkForward, type ModelType, type StrategyConfig } from "./backtest";
import { walkForwardSizes, barBudget } from "./sweep";
import { makeRng } from "./hmm";
import * as hl from "./hyperliquid";

/** One coin's out-of-sample record: what it held, and what happened next. */
export interface PanelSeries {
  coin: string;
  /** Bar close times, unix seconds, ascending. */
  times: number[];
  /** Target position in [-1, 1] decided at that bar. */
  positions: number[];
  /** Simple return of the FOLLOWING bar — what `positions[i]` earns. */
  returns: number[];
  /**
   * The model's own expected next-bar return at that bar, in real units, before
   * it was quantized into a position. Optional only because older cached panels
   * predate it.
   */
  forecasts?: number[];
}

export interface PanelConfig {
  coins?: string[];
  limit?: number;
  timeframe?: string;
  days?: number;
  /**
   * Build the panel out of an OLDER slice of history than the default recent
   * window: fetch `fetchBars` candles, then keep `[sliceFrom, sliceTo)`.
   *
   * This is how a held-out window is made. Hyperliquid retains ~5000 candles
   * per interval, so at 30m the most recent 2160 cover the 45 days every test
   * so far has used, and the ~2800 before them have never been looked at. A
   * pattern found in the first and confirmed in the second has been confirmed;
   * a pattern found and confirmed in the same bars has only been described.
   */
  fetchBars?: number;
  sliceFrom?: number;
  sliceTo?: number;
  costBps?: number;
  states?: number;
  modelType?: ModelType;
  maxDuration?: number;
  /** Jump model only: the price of one regime switch. */
  lambda?: number;
  seed?: number;
}

export interface PanelResult {
  coins: number;
  /** Total (coin, bar) observations behind the statistic. */
  observations: number;
  /** Pooled mean net return per bar, in bps. */
  actualBps: number;
  medianNullBps: number;
  p05Bps: number;
  p95Bps: number;
  pValue: number;
  trials: number;
  nullMethod: NullMethod;
  perCoin: { coin: string; bars: number; netBps: number; exposure: number }[];
}

export type NullMethod = "common" | "independent";

/**
 * Pooled mean net return per bar, in bps.
 *
 * Costs are charged on |Δposition| exactly as the backtest does, because a null
 * that rotates positions must pay the same turnover the real series paid —
 * otherwise the test rewards low turnover rather than good timing. (The repo
 * learned this once already: see the `rotate` null in diagnostics.ts.)
 */
export function pooledStatistic(
  positions: (number | null)[][],
  returns: (number | null)[][],
  costBps: number,
): { bps: number; observations: number } {
  const cost = costBps / 10_000;
  let total = 0;
  let n = 0;
  for (let c = 0; c < positions.length; c++) {
    const pos = positions[c];
    const ret = returns[c];
    let prev = 0;
    for (let t = 0; t < pos.length; t++) {
      const p = pos[t];
      const r = ret[t];
      if (p === null || r === null || !Number.isFinite(r)) { prev = p ?? prev; continue; }
      total += p * r - cost * Math.abs(p - prev);
      prev = p;
      n++;
    }
  }
  return { bps: n > 0 ? (total / n) * 10_000 : 0, observations: n };
}

/**
 * Put every coin on one time axis.
 *
 * A common grid is what makes a common rotation meaningful: shifting the whole
 * matrix by k has to shift every coin by the same wall-clock amount, not by k
 * of its own bars. Coins that lack a bar at some grid time get null there and
 * contribute nothing at that instant.
 */
export function alignPanel(series: PanelSeries[]): {
  times: number[];
  positions: (number | null)[][];
  returns: (number | null)[][];
} {
  const all = new Set<number>();
  for (const s of series) for (const t of s.times) all.add(t);
  const times = [...all].sort((a, b) => a - b);
  const index = new Map(times.map((t, i) => [t, i]));

  const positions: (number | null)[][] = [];
  const returns: (number | null)[][] = [];
  for (const s of series) {
    const p = new Array<number | null>(times.length).fill(null);
    const r = new Array<number | null>(times.length).fill(null);
    for (let i = 0; i < s.times.length; i++) {
      const at = index.get(s.times[i]);
      if (at === undefined) continue;
      p[at] = s.positions[i];
      r[at] = s.returns[i];
    }
    positions.push(p);
    returns.push(r);
  }
  return { times, positions, returns };
}

/** Circular shift by `k`. Preserves every run, and therefore turnover exactly. */
function rotate<T>(row: T[], k: number): T[] {
  const n = row.length;
  if (n === 0) return row;
  const s = ((k % n) + n) % n;
  return row.slice(s).concat(row.slice(0, s));
}

export interface PanelTestOptions {
  costBps?: number;
  trials?: number;
  seed?: number;
  nullMethod?: NullMethod;
  /**
   * Score several horizons at once and test their MEAN rank IC as one
   * statistic. Every horizon is recomputed inside each rotation draw, so the
   * null belongs to the composite rather than being six nulls stitched
   * together.
   *
   * The point is to use a pattern that showed up across horizons without
   * selecting the horizon where it looked best — picking the winner and then
   * testing it is the same error as picking the winning coin out of a sweep.
   */
  horizons?: number[];
  /**
   * Forecast horizon in bars, for `forecastTest` only. Default 1.
   *
   * A model that learns a nine-bar dwell is making a nine-bar claim; scoring it
   * one bar out scores something it never asserted. Overlapping h-bar windows
   * are autocorrelated by construction, which would wreck a naive t-stat — the
   * rotation null is immune, because it resamples the alignment while leaving
   * the return series, and therefore its autocorrelation, exactly as it is.
   */
  horizon?: number;
}

export function panelTest(series: PanelSeries[], opts: PanelTestOptions = {}): PanelResult {
  const costBps = opts.costBps ?? hl.HL_TAKER_BPS;
  const trials = opts.trials ?? 2000;
  const nullMethod = opts.nullMethod ?? "common";
  const rng = makeRng(opts.seed ?? 4242);

  const { times, positions, returns } = alignPanel(series);
  const actual = pooledStatistic(positions, returns, costBps);

  const draws: number[] = [];
  for (let i = 0; i < trials; i++) {
    let shifted: (number | null)[][];
    if (nullMethod === "common") {
      // One offset for the whole market: the cross-section stays intact.
      const k = 1 + Math.floor(rng() * Math.max(1, times.length - 1));
      shifted = positions.map((row) => rotate(row, k));
    } else {
      shifted = positions.map((row) => rotate(row, 1 + Math.floor(rng() * Math.max(1, row.length - 1))));
    }
    draws.push(pooledStatistic(shifted, returns, costBps).bps);
  }
  draws.sort((a, b) => a - b);
  const q = (f: number) => draws[Math.min(draws.length - 1, Math.max(0, Math.floor(f * draws.length)))];
  const atLeast = draws.filter((d) => d >= actual.bps).length;

  const perCoin = series.map((s, c) => {
    const one = pooledStatistic([positions[c]], [returns[c]], costBps);
    const held = s.positions.filter((p) => Math.abs(p) > 1e-9).length;
    return {
      coin: s.coin,
      bars: one.observations,
      netBps: one.bps,
      exposure: s.positions.length ? held / s.positions.length : 0,
    };
  });

  return {
    coins: series.length,
    observations: actual.observations,
    actualBps: actual.bps,
    medianNullBps: q(0.5),
    p05Bps: q(0.05),
    p95Bps: q(0.95),
    // +1 in both places: the observed arrangement is itself one of the
    // arrangements under the null, and omitting it can return p = 0, which is
    // never an honest answer from a finite resample.
    pValue: (atLeast + 1) / (trials + 1),
    trials,
    nullMethod,
    perCoin,
  };
}

/**
 * Does the model forecast returns at all?
 *
 * The P&L test above answers a compound question — "does the forecast beat the
 * cost of acting on it" — and it answers it with almost no power, because
 * turning a continuous forecast into a -1/0/1 position throws most of the
 * information away and then the round trip eats what is left. Two things are
 * being asked at once, and a null result cannot say which one failed.
 *
 * So ask the first one on its own. Regress the realized next-bar return on the
 * model's expected next-bar return across the whole panel (the Mincer-Zarnowitz
 * form: r = a + b f + e, with H0: b = 0) and report the pooled information
 * coefficient alongside it. Every bar contributes, including the ones the
 * strategy sat out, which is where most of the sample lives.
 *
 * The separation matters more than the significance:
 *
 *   b > 0 significantly, P&L not  ->  the model forecasts; costs eat it.
 *                                     Trade less, or trade a slower timeframe.
 *   neither                       ->  there is no forecast. No amount of
 *                                     position sizing, cost reduction or
 *                                     further searching creates one.
 *
 * The null is the same common rotation the P&L test uses, for the same reason:
 * these coins are one market factor, and a null that scrambles them
 * independently would report a correlation as significant when it is just beta.
 */
export interface ForecastResult {
  /** Bars ahead the forecast was scored against. */
  horizon: number;
  observations: number;
  /** OLS slope of realized on forecast. 1.0 would be a perfectly calibrated forecast. */
  slope: number;
  /** Pooled Pearson correlation between forecast and realization. */
  ic: number;
  /** Spearman, which survives the outliers a 40% candle produces. */
  rankIc: number;
  /** One-sided: how often the null's IC reached the observed one. */
  pValue: number;
  /**
   * TWO-SIDED p for the rank IC.
   *
   * Two-sided on purpose. A one-sided test picked after seeing which way the
   * statistic pointed is not a test, it is a description — and the rank IC's
   * sign is exactly the kind of thing you notice first and rationalise second.
   */
  pValueRank: number;
  /** One-sided lower tail. Only legitimate when the sign was predicted first. */
  pValueRankLower: number;
  medianNullIc: number;
  p05Ic: number;
  p95Ic: number;
  medianNullRankIc: number;
  p05RankIc: number;
  p95RankIc: number;
  trials: number;
  nullMethod: NullMethod;
}

function pearson(x: number[], y: number[]): number {
  const n = x.length;
  if (n < 3) return 0;
  let mx = 0, my = 0;
  for (let i = 0; i < n; i++) { mx += x[i]; my += y[i]; }
  mx /= n; my /= n;
  let sxy = 0, sxx = 0, syy = 0;
  for (let i = 0; i < n; i++) {
    const a = x[i] - mx, b = y[i] - my;
    sxy += a * b; sxx += a * a; syy += b * b;
  }
  return sxx > 0 && syy > 0 ? sxy / Math.sqrt(sxx * syy) : 0;
}

function ranks(v: number[]): number[] {
  const order = v.map((x, i) => [x, i] as const).sort((a, b) => a[0] - b[0]);
  const out = new Array(v.length).fill(0);
  let i = 0;
  while (i < order.length) {
    let j = i;
    while (j + 1 < order.length && order[j + 1][0] === order[i][0]) j++;
    const tied = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) out[order[k][1]] = tied;
    i = j + 1;
  }
  return out;
}

/** Flatten the panel into aligned (forecast, realized) pairs. */
function pairs(
  forecasts: (number | null)[][], returns: (number | null)[][],
): { f: number[]; r: number[] } {
  const f: number[] = [], r: number[] = [];
  for (let c = 0; c < forecasts.length; c++) {
    for (let t = 0; t < forecasts[c].length; t++) {
      const a = forecasts[c][t], b = returns[c][t];
      if (a === null || b === null || !Number.isFinite(a) || !Number.isFinite(b)) continue;
      f.push(a); r.push(b);
    }
  }
  return { f, r };
}

/**
 * Compound each coin's returns forward over `h` bars, so row t holds what a
 * position opened at t and held for h bars would have earned. Rows without a
 * full h bars ahead become null rather than being padded — a short final window
 * would understate the horizon it claims to measure.
 */
function forwardCumulative(returns: (number | null)[][], h: number): (number | null)[][] {
  if (h <= 1) return returns;
  return returns.map((row) => row.map((_, t) => {
    let acc = 1;
    for (let k = 0; k < h; k++) {
      const r = row[t + k];
      if (r === null || r === undefined || !Number.isFinite(r)) return null;
      acc *= 1 + r;
    }
    return acc - 1;
  }));
}

export function forecastTest(series: PanelSeries[], opts: PanelTestOptions = {}): ForecastResult {
  const trials = opts.trials ?? 2000;
  const nullMethod = opts.nullMethod ?? "common";
  const rng = makeRng(opts.seed ?? 4242);

  const withForecasts = series.filter((s) => s.forecasts && s.forecasts.length === s.times.length);
  if (withForecasts.length === 0) {
    throw new Error("no forecasts in this panel — recollect it (the cache predates the field)");
  }
  const aligned = alignPanel(withForecasts.map((s) => ({ ...s, positions: s.forecasts! })));
  const forecasts = aligned.positions; // the aligner does not care what it carries
  const returns = forwardCumulative(aligned.returns, opts.horizon ?? 1);

  const horizons = opts.horizons ?? null;
  const returnsByHorizon = horizons
    ? horizons.map((h) => forwardCumulative(aligned.returns, h))
    : null;

  /** Mean rank IC across the composite's horizons, for one arrangement. */
  const compositeRankIc = (fc: (number | null)[][]) => {
    let acc = 0;
    for (const rets of returnsByHorizon!) {
      const p = pairs(fc, rets);
      acc += pearson(ranks(p.f), ranks(p.r));
    }
    return acc / returnsByHorizon!.length;
  };

  const { f, r } = pairs(forecasts, returns);
  const ic = pearson(f, r);
  const rankIc = horizons ? compositeRankIc(forecasts) : pearson(ranks(f), ranks(r));

  let mf = 0, mr = 0;
  for (let i = 0; i < f.length; i++) { mf += f[i]; mr += r[i]; }
  mf /= f.length; mr /= r.length;
  let sfr = 0, sff = 0;
  for (let i = 0; i < f.length; i++) { sfr += (f[i] - mf) * (r[i] - mr); sff += (f[i] - mf) ** 2; }
  const slope = sff > 0 ? sfr / sff : 0;

  const draws: number[] = [];
  const rankDraws: number[] = [];
  for (let i = 0; i < trials; i++) {
    let shifted: (number | null)[][];
    if (nullMethod === "common") {
      const k = 1 + Math.floor(rng() * Math.max(1, aligned.times.length - 1));
      shifted = forecasts.map((row) => rotate(row, k));
    } else {
      shifted = forecasts.map((row) => rotate(row, 1 + Math.floor(rng() * Math.max(1, row.length - 1))));
    }
    const p = pairs(shifted, returns);
    draws.push(pearson(p.f, p.r));
    rankDraws.push(horizons ? compositeRankIc(shifted) : pearson(ranks(p.f), ranks(p.r)));
  }
  draws.sort((a, b) => a - b);
  rankDraws.sort((a, b) => a - b);
  const q = (arr: number[]) => (x: number) =>
    arr[Math.min(arr.length - 1, Math.max(0, Math.floor(x * arr.length)))];
  const qi = q(draws), qr = q(rankDraws);

  // Two-sided: distance from the null's own centre, in either direction.
  const centre = qr(0.5);
  const extreme = rankDraws.filter((d) => Math.abs(d - centre) >= Math.abs(rankIc - centre)).length;
  // One-sided lower tail. Only legitimate when the direction was fixed by a
  // previous window's result, never by this one's.
  const below = rankDraws.filter((d) => d <= rankIc).length;

  return {
    horizon: opts.horizon ?? 1,
    observations: f.length,
    slope, ic, rankIc,
    pValue: (draws.filter((d) => d >= ic).length + 1) / (trials + 1),
    pValueRank: (extreme + 1) / (trials + 1),
    pValueRankLower: (below + 1) / (trials + 1),
    medianNullIc: qi(0.5), p05Ic: qi(0.05), p95Ic: qi(0.95),
    medianNullRankIc: centre, p05RankIc: qr(0.05), p95RankIc: qr(0.95),
    trials, nullMethod,
  };
}

const FEATURES: FeatureConfig = { window: 5, useVolatility: true, useVolume: true };

/**
 * Run the SAME configuration over every coin and collect the out-of-sample
 * record. Nothing here is chosen per coin — that is the entire point.
 */
export async function collectPanel(
  cfg: PanelConfig = {},
  onProgress?: (done: number, total: number, coin: string) => void,
): Promise<PanelSeries[]> {
  const timeframe = cfg.timeframe ?? "30m";
  const days = cfg.days ?? 45;
  const costBps = cfg.costBps ?? hl.HL_TAKER_BPS;
  const states = cfg.states ?? 3;
  const modelType = cfg.modelType ?? "hsmm";
  const { bars } = barBudget(timeframe, days);

  const coins = cfg.coins ?? (await hl.topMarkets(cfg.limit ?? 30)).map((m) => m.coin);
  const strategy: StrategyConfig = { costBps, durationAware: true };
  const out: PanelSeries[] = [];

  for (let i = 0; i < coins.length; i++) {
    const coin = coins[i];
    onProgress?.(i, coins.length, coin);
    try {
      const fetched = await hl.fetchCandles(coin, timeframe, cfg.fetchBars ?? bars);
      const candles = cfg.sliceFrom !== undefined || cfg.sliceTo !== undefined
        ? fetched.candles.slice(cfg.sliceFrom ?? 0, cfg.sliceTo ?? fetched.candles.length)
        : fetched.candles;
      const fs = buildFeatures(candles, FEATURES);
      const sizes = walkForwardSizes(fs.T);
      if (!sizes.fits) continue;

      const res = walkForward(candles, FEATURES, strategy, {
        modelType, states, maxDuration: cfg.maxDuration ?? 30, lambda: cfg.lambda,
        trainSize: sizes.trainSize, testSize: sizes.testSize,
        seed: cfg.seed ?? 42, restarts: 4,
        barsPerYear: hl.hlBarsPerYear(timeframe),
      });

      const times: number[] = [];
      const positions: number[] = [];
      const returns: number[] = [];
      const forecasts: number[] = [];
      for (let r = sizes.trainSize; r < fs.T - 1; r++) {
        times.push(candles[fs.index[r]].time);
        positions.push(res.positions[r] ?? 0);
        returns.push(Math.exp(fs.rawReturn[r + 1]) - 1);
        forecasts.push(res.expectedReturns[r] ?? 0);
      }
      if (times.length > 0) out.push({ coin, times, positions, returns, forecasts });
    } catch {
      // One dead symbol must not take the panel with it.
      continue;
    }
  }
  onProgress?.(coins.length, coins.length, "done");
  return out;
}

// ---------------------------------------------------------------------------

/**
 * `bun run src/panel.ts [--timeframe 30m] [--days 45] [--limit 30] [--model hsmm]`
 *
 * Collects once and caches, because the point of the panel is to test ONE
 * hypothesis — refetching and refitting on every run would invite quietly
 * trying another configuration until one of them prints a small number.
 */
if (import.meta.main) {
  const arg = (n: string, d: string) => {
    const i = process.argv.indexOf(`--${n}`);
    return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : d;
  };
  const timeframe = arg("timeframe", "30m");
  const days = Number(arg("days", "45"));
  const limit = Number(arg("limit", "30"));
  const modelType = arg("model", "hsmm") as ModelType;
  const costBps = Number(arg("cost", String(hl.HL_TAKER_BPS)));
  const cache = `models/panel-${timeframe}-${days}d-${modelType}.json`;

  let series: PanelSeries[];
  const cached = Bun.file(cache);
  if (await cached.exists()) {
    series = await cached.json();
    console.log(`panel: ${series.length} coins from ${cache}`);
  } else {
    console.log(`collecting ${limit} coins at ${timeframe} over ${days}d with a fixed ${modelType.toUpperCase()}…`);
    series = await collectPanel({ timeframe, days, limit, modelType, costBps },
      (done, total, coin) => process.stderr.write(`  ${done}/${total} ${coin}\r`));
    await Bun.write(cache, JSON.stringify(series));
    console.log(`\npanel: ${series.length} coins collected -> ${cache}`);
  }

  for (const method of ["common", "independent"] as NullMethod[]) {
    const res = panelTest(series, { costBps, trials: 5000, seed: 4242, nullMethod: method });
    console.log(
      `\n${method === "common" ? "COMMON rotation (correct: keeps the market factor in the null)" : "independent rotation (anti-conservative, shown for contrast)"}\n` +
      `  coins ${res.coins}   observations ${res.observations}\n` +
      `  actual   ${res.actualBps.toFixed(3)} bps/bar\n` +
      `  null     median ${res.medianNullBps.toFixed(3)}   90% band [${res.p05Bps.toFixed(3)}, ${res.p95Bps.toFixed(3)}]\n` +
      `  p-value  ${res.pValue.toFixed(4)}${res.pValue < 0.05 ? "  <- below 0.05" : ""}`);
    if (method === "common") {
      const sorted = [...res.perCoin].sort((a, b) => b.netBps - a.netBps);
      console.log("  best/worst coins: " +
        sorted.slice(0, 3).map((c) => `${c.coin} ${c.netBps.toFixed(2)}`).join("  ") + "   …   " +
        sorted.slice(-3).map((c) => `${c.coin} ${c.netBps.toFixed(2)}`).join("  "));
      const positive = res.perCoin.filter((c) => c.netBps > 0).length;
      console.log(`  ${positive}/${res.perCoin.length} coins net positive after costs`);
    }
  }

  // The other half of the question: forget whether it is tradeable, does the
  // model forecast anything at all?
  try {
    const fc = forecastTest(series, { trials: 5000, seed: 4242 });
    console.log(
      `\nFORECAST content (all bars, not just the traded ones)\n` +
      `  observations ${fc.observations}\n` +
      `  IC       ${fc.ic.toFixed(5)}   rank IC ${fc.rankIc.toFixed(5)}   slope ${fc.slope.toFixed(4)}\n` +
      `  null     median ${fc.medianNullIc.toFixed(5)}   90% band [${fc.p05Ic.toFixed(5)}, ${fc.p95Ic.toFixed(5)}]\n` +
      `  p-value  ${fc.pValue.toFixed(4)}${fc.pValue < 0.05 ? "  <- below 0.05" : ""}`);
    console.log(fc.slope > 0 && fc.pValue < 0.05
      ? "  => the model forecasts; whether costs leave anything is the P&L question above."
      : "  => no forecasting content. Position sizing and cost reduction cannot create any.");
  } catch (e) {
    console.log(`\nFORECAST content: ${e instanceof Error ? e.message : e}`);
  }
}
