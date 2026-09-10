/**
 * Timeframe sweep: the same walk-forward, run across a whole universe.
 *
 * Every other command in this repo answers a question about ONE market at ONE
 * interval. That is one draw, and the README already documents what one draw is
 * worth here — removing four leading bars from a 5004-bar series flipped a
 * result from +33.4% to 0.0%. A sweep is the honest way to ask the broader
 * question ("does anything survive anywhere?"), but it changes the experiment,
 * so three things are built in rather than left to the reader:
 *
 *   Skill, not exposure. Every row carries `excessRoi` = ROI minus the median
 *   ROI of a rotated null at the same exposure and turnover. Raw ROI is the
 *   number that makes a sweep look like a discovery machine; it is mostly a
 *   ranking of which coins went up.
 *
 *   An exposure cap on the winner. A row sitting in the market 95% of the time
 *   is buy-and-hold wearing a model. It stays in `rows` because suppressing it
 *   would hide how often that happens, but it cannot be a coin's `best`.
 *
 *   A ceiling beside every result. If perfect foresight committing 20 bars at a
 *   time cannot clear the fee on this series, the model was never the problem.
 *
 * One thing the sweep CANNOT fix: `validate` is calibrated for a single test on
 * a single market. Running 30 coins x 4 timeframes x 2 models is 240 tests, so
 * the smallest p-values in the table are the ones you would expect from noise
 * alone. Divide your alpha by the number of rows, not by one.
 */

import { walkForward, type ModelType } from "./backtest";
import { permutationTest, ceilingAnalysis, tradedSeries } from "./diagnostics";
import { fit, serialize, type HmmParams } from "./hmm";
import { fitHsmm, serializeHsmm, type HsmmParams } from "./hsmm";
import { buildFeatures, fitScaler, applyScaler, type Candle, type FeatureConfig } from "./features";
import * as hl from "./hyperliquid";

export interface SweepConfig {
  coins?: string[];          // default: top `limit` perps by 24h volume via hl.topMarkets
  limit?: number;            // default 30
  timeframes?: string[];     // default ["15m", "30m", "1h", "2h"]
  days?: number;             // calendar window, default 45
  costBps?: number;          // default hl.HL_TAKER_BPS (4.5)
  states?: number;           // default 3
  modelTypes?: ModelType[];  // default ["hmm", "hsmm"]
  trials?: number;           // permutation trials, default 200
  seed?: number;             // default 42
}

export interface SweepRow {
  coin: string; timeframe: string; modelType: ModelType;
  bars: number;           // feature rows available
  trainSize: number; testSize: number; refits: number; barsTraded: number;
  roi: number; buyHold: number; sharpe: number; maxDD: number; exposure: number; trades: number;
  medianRandom: number;   // permutation median ROI at the same exposure
  excessRoi: number;      // roi - medianRandom  <- the skill estimate
  pValue: number;
  ceilingHold20: number;  // oracle "hold 20 bars" ROI net of costBps
  skipped?: string;       // set when this combination could not be evaluated, with the reason
}

export interface SweepResult {
  generatedAt: number; days: number; costBps: number;
  rows: SweepRow[];
  best: Record<string, SweepRow>;   // coin -> winning row (absent if no row qualified)
}

/**
 * Hyperliquid retains ~5000 candles per interval regardless of the window
 * asked for, so the interval decides the history, not the request. 5m tops out
 * near 17 days, which is why it is not in the default list: a 45-day sweep at
 * 5m would silently be a 17-day sweep.
 */
export const SWEEP_BAR_CAP = 5000;

/** Same lookback the CLI defaults to — long windows cannot see short regimes. */
export const SWEEP_WINDOW = 5;

/** Below this a coin's headline number is one or two coin flips, not a measurement. */
export const MIN_BEST_TRADES = 3;

/** Above this the "strategy" is buy-and-hold with extra steps. */
export const MAX_BEST_EXPOSURE = 0.90;

export const DEFAULT_TIMEFRAMES = ["15m", "30m", "1h", "2h"];
export const DEFAULT_MODEL_TYPES: ModelType[] = ["hmm", "hsmm"];
const DEFAULT_STATES = 3;

export function barsPerDay(timeframe: string): number {
  return 86_400 / hl.intervalSeconds(timeframe);
}

export interface BarBudget {
  /** Candles to request: what `days` needs, capped at what the venue retains. */
  bars: number;
  /** What `days` would need if retention were unlimited. */
  wanted: number;
  /** Calendar days the capped request actually reaches. */
  daysCovered: number;
  /** True when the cap bit, i.e. this timeframe cannot cover the window asked for. */
  truncated: boolean;
}

/** How many candles a (timeframe, days) pair needs, and whether it can get them. */
export function barBudget(timeframe: string, days: number, cap = SWEEP_BAR_CAP): BarBudget {
  const wanted = Math.ceil(days * barsPerDay(timeframe));
  const bars = Math.min(cap, wanted);
  return { bars, wanted, daysCovered: bars / barsPerDay(timeframe), truncated: wanted > cap };
}

export interface WalkForwardSizes {
  trainSize: number;
  testSize: number;
  /** Feature rows the harness demands for these sizes. */
  needed: number;
  fits: boolean;
}

/**
 * Train/test blocks scaled to the series rather than fixed.
 *
 * A 2h series over 45 days is ~540 bars; the repo's default 1500/500 would
 * simply throw. Half the series to train and a third of that to trade gives
 * every timeframe at least one refit, and the clamps keep both ends sane: under
 * ~300 training rows a 3-state Gaussian fit is estimating means from a handful
 * of observations, and over 1500 the extra history is mostly a different market.
 */
export function walkForwardSizes(bars: number): WalkForwardSizes {
  const clamp = (x: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, x));
  const trainSize = clamp(Math.round(bars * 0.5), 300, 1500);
  const testSize = clamp(Math.round(trainSize / 3), 100, 500);
  const needed = trainSize + testSize + 2;
  return { trainSize, testSize, needed, fits: bars >= needed };
}

/**
 * The winning row for one coin, or undefined if nothing qualified.
 *
 * Ranked on excess ROI over the null, never on ROI: the coin that went up the
 * most would otherwise win every sweep. Ties break on the lower p-value, which
 * only matters when two timeframes produce identical returns.
 */
export function pickBest(rows: SweepRow[]): SweepRow | undefined {
  const eligible = rows.filter(
    (r) => !r.skipped && r.trades >= MIN_BEST_TRADES && r.exposure <= MAX_BEST_EXPOSURE,
  );
  if (eligible.length === 0) return undefined;
  return eligible.reduce((a, b) =>
    b.excessRoi > a.excessRoi || (b.excessRoi === a.excessRoi && b.pValue < a.pValue) ? b : a,
  );
}

/** Why a coin produced no winner — for the CLI, which should say so out loud. */
export function rejectionReason(rows: SweepRow[]): string {
  if (rows.length === 0) return "no rows";
  const usable = rows.filter((r) => !r.skipped);
  if (usable.length === 0) return rows[0].skipped ?? "every row skipped";
  if (usable.every((r) => r.trades < MIN_BEST_TRADES)) {
    return `never traded more than ${Math.max(...usable.map((r) => r.trades))} times`;
  }
  if (usable.every((r) => r.exposure > MAX_BEST_EXPOSURE)) return "always above the 90% exposure cap";
  return `no row cleared ${MIN_BEST_TRADES} trades under ${(MAX_BEST_EXPOSURE * 100).toFixed(0)}% exposure`;
}

export function bestByCoin(rows: SweepRow[]): Record<string, SweepRow> {
  const byCoin = new Map<string, SweepRow[]>();
  for (const r of rows) {
    if (!byCoin.has(r.coin)) byCoin.set(r.coin, []);
    byCoin.get(r.coin)!.push(r);
  }
  const best: Record<string, SweepRow> = {};
  for (const [coin, coinRows] of byCoin) {
    const win = pickBest(coinRows);
    if (win) best[coin] = win;
  }
  return best;
}

const featureConfig: FeatureConfig = { window: SWEEP_WINDOW, useVolatility: true, useVolume: true };

function skippedRow(
  coin: string, timeframe: string, modelType: ModelType, skipped: string,
  bars = 0, trainSize = 0, testSize = 0,
): SweepRow {
  return {
    coin, timeframe, modelType, bars, trainSize, testSize, refits: 0, barsTraded: 0,
    roi: 0, buyHold: 0, sharpe: 0, maxDD: 0, exposure: 0, trades: 0,
    medianRandom: 0, excessRoi: 0, pValue: 1, ceilingHold20: 0, skipped,
  };
}

const errText = (e: unknown) => (e instanceof Error ? e.message : String(e));

export async function sweep(
  cfg: SweepConfig = {},
  onProgress?: (done: number, total: number, label: string) => void,
): Promise<SweepResult> {
  const limit = cfg.limit ?? 30;
  const timeframes = cfg.timeframes ?? DEFAULT_TIMEFRAMES;
  const days = cfg.days ?? 45;
  const costBps = cfg.costBps ?? hl.HL_TAKER_BPS;
  const states = cfg.states ?? DEFAULT_STATES;
  const modelTypes = cfg.modelTypes ?? DEFAULT_MODEL_TYPES;
  const trials = cfg.trials ?? 200;
  const seed = cfg.seed ?? 42;

  const coins = cfg.coins?.map((c) => c.toUpperCase())
    ?? (await hl.topMarkets(limit)).map((m) => m.coin);

  const rows: SweepRow[] = [];
  const total = coins.length * timeframes.length * modelTypes.length;
  let done = 0;
  const tick = (label: string) => { done++; onProgress?.(done, total, label); };

  for (const coin of coins) {
    const seen = new Set<string>();
    const push = (r: SweepRow) => {
      rows.push(r);
      seen.add(`${r.timeframe}/${r.modelType}`);
      tick(`${r.coin} ${r.timeframe} ${r.modelType}`);
    };

    try {
      for (const tf of timeframes) {
        const budget = barBudget(tf, days);
        // Run it anyway — a 17-day 5m result is still informative — but mark it,
        // because comparing it to a 45-day row as if they were the same
        // experiment is exactly the mistake this note exists to prevent.
        const short = budget.truncated
          ? `${tf} reaches only ${budget.daysCovered.toFixed(1)} of ${days} days ` +
            `at the ${SWEEP_BAR_CAP}-candle retention cap`
          : undefined;

        let candles: Candle[];
        try {
          candles = (await hl.fetchCandles(coin, tf, budget.bars)).candles;
        } catch (e) {
          for (const mt of modelTypes) push(skippedRow(coin, tf, mt, `fetch failed: ${errText(e)}`));
          continue;
        }

        let bars: number;
        try {
          bars = buildFeatures(candles, featureConfig).T;
        } catch (e) {
          for (const mt of modelTypes) push(skippedRow(coin, tf, mt, `no features: ${errText(e)}`));
          continue;
        }

        const { trainSize, testSize, needed, fits } = walkForwardSizes(bars);
        if (!fits) {
          const why = `only ${bars} feature rows, needs ${needed} for ${trainSize}/${testSize} walk-forward`;
          for (const mt of modelTypes) push(skippedRow(coin, tf, mt, why, bars, trainSize, testSize));
          continue;
        }

        // The oracle depends on the series and the cost, not on the model, so
        // it is computed once per (coin, timeframe) and shared by both rows.
        let ceilingHold20 = 0;
        try {
          const ceil = ceilingAnalysis(candles, featureConfig, {
            costs: [costBps], holds: [20], skip: trainSize,
          });
          ceilingHold20 = ceil.rows.find((r) => r.holdBars === 20)?.roi ?? 0;
        } catch { /* leave at 0; a missing ceiling must not kill the row */ }

        for (const mt of modelTypes) {
          try {
            const r = walkForward(
              candles, featureConfig,
              { costBps, durationAware: true },
              {
                modelType: mt, trainSize, testSize, states, seed,
                barsPerYear: hl.hlBarsPerYear(tf),
                // The CLI's default, not walkForward's 60. An HSMM sweep is
                // O(T x K x maxDuration) per EM sweep across 240 cells, and the
                // README's own measurement is that 30 was as good as 60.
                maxDuration: 30,
              },
            );
            // Same slice the `validate` command tests: positions from the first
            // traded row onward, against the return of the following bar.
            const ts = tradedSeries(candles, featureConfig, r.positions, trainSize);
            const perm = permutationTest(ts.positions, ts.returns, { trials, seed, costBps });

            push({
              coin, timeframe: tf, modelType: mt,
              bars, trainSize, testSize,
              refits: r.refits, barsTraded: r.metrics.bars,
              roi: r.metrics.totalReturn,
              buyHold: r.metrics.buyHoldReturn,
              sharpe: r.metrics.sharpe,
              maxDD: r.metrics.maxDrawdown,
              exposure: r.metrics.exposure,
              trades: r.metrics.trades,
              medianRandom: perm.medianRandom,
              excessRoi: r.metrics.totalReturn - perm.medianRandom,
              pValue: perm.pValue,
              ceilingHold20,
              skipped: short,
            });
          } catch (e) {
            push(skippedRow(coin, tf, mt, `backtest failed: ${errText(e)}`, bars, trainSize, testSize));
          }
        }
      }
    } catch (e) {
      // One bad symbol must not end the run. Fill in whatever this coin never
      // reached so the progress count and the row grid stay complete.
      const why = `coin failed: ${errText(e)}`;
      for (const tf of timeframes) {
        for (const mt of modelTypes) {
          if (!seen.has(`${tf}/${mt}`)) push(skippedRow(coin, tf, mt, why));
        }
      }
    }
  }

  return { generatedAt: Date.now(), days, costBps, rows, best: bestByCoin(rows) };
}

/**
 * The JSON `predict --model` reads. Written out as a type so the test can
 * assert the shape rather than assert that a particular string appears.
 */
export interface SavedModel {
  modelType: ModelType;
  params: unknown;
  scaler: { mean: number[]; std: number[] };
  names: string[];
  window: number;
  trainedBars: number;
  source: { coin: string; timeframe: string };
  label: string;
}

/** Field-for-field what `train --out` writes, so the two stay in step. */
export function buildSavedModel(
  row: SweepRow,
  params: HmmParams | HsmmParams,
  scaler: { mean: Float64Array; std: Float64Array },
  names: string[],
  trainedBars: number,
): SavedModel {
  return {
    modelType: row.modelType,
    params: JSON.parse(
      row.modelType === "hsmm"
        ? serializeHsmm(params as HsmmParams)
        : serialize(params as HmmParams),
    ),
    scaler: { mean: Array.from(scaler.mean), std: Array.from(scaler.std) },
    names,
    window: SWEEP_WINDOW,
    trainedBars,
    // Recorded so `predict --model` reloads the same market and interval
    // without the caller repeating the flags the sweep already chose.
    source: { coin: row.coin, timeframe: row.timeframe },
    label: `${row.coin}-PERP ${row.timeframe}`,
  };
}

export function modelFileName(row: SweepRow): string {
  return `${row.coin.toLowerCase()}-${row.timeframe}-${row.modelType}.model.json`;
}

/**
 * Fit the winning config on the full window and write it in the exact JSON
 * shape `predict --model` already reads. Returns the path written.
 *
 * This refits on ALL rows, unlike the walk-forward that selected it — the point
 * is a model to predict with tomorrow, so withholding the most recent third of
 * the history would be throwing away the bars that matter most. Nothing about
 * this fit is out-of-sample, and nothing should be read off it as evidence.
 *
 * `states` is not carried on a row, so it is passed separately; the sweep's own
 * default and the CLI's `--states` default are both 3.
 */
export async function saveBestModel(row: SweepRow, dir: string, states = DEFAULT_STATES): Promise<string> {
  // `row.bars` counts feature rows, which drop the first `window` candles.
  const res = await hl.fetchCandles(row.coin, row.timeframe, row.bars + SWEEP_WINDOW);
  const fs = buildFeatures(res.candles, featureConfig);
  const scaler = fitScaler(fs.X, fs.T, fs.D);
  const Z = applyScaler(fs.X, fs.T, fs.D, scaler);

  const params: HmmParams | HsmmParams = row.modelType === "hsmm"
    ? fitHsmm(Z, fs.T, fs.D, { states, seed: 42, restarts: 4, maxDuration: 30 }).params
    : fit(Z, fs.T, fs.D, { states, seed: 42, restarts: 8 }).params;

  const path = `${dir.replace(/\/$/, "")}/${modelFileName(row)}`;
  await Bun.write(path, JSON.stringify(buildSavedModel(row, params, scaler, fs.names, fs.T), null, 2));
  return path;
}
