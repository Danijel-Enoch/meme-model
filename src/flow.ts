/**
 * Non-price information from Hyperliquid's public info endpoint.
 *
 * WHY THIS FILE EXISTS
 *
 * Every feature in `features.ts` — log return, realized vol, volume surge — is
 * a deterministic function of the past close/volume series. The repo's own
 * conclusion is that nothing in that series times these regimes in advance, so
 * remodelling it cannot help: the only thing that can change the answer is new
 * information.
 *
 * WHAT THE PUBLIC API ACTUALLY SERVES, VERIFIED BY REQUEST (2026-09-10)
 *
 * Deep enough to backtest (>= 45 days):
 *   candleSnapshot .n        trades executed inside the bar. Same 5000-candle
 *                            depth as o/h/l/c/v, on the same grid, and NOT a
 *                            function of price or volume — v/n (average trade
 *                            size) is a separate axis entirely.
 *   fundingHistory .fundingRate   hourly, back to the coin's listing day
 *                            (SOL: 2023-05-12). Cross-market: the rate is set
 *                            by the perp's premium over an oracle built from
 *                            CEX spot, so it carries information that is not
 *                            in Hyperliquid's own candles.
 *   fundingHistory .premium  the raw perp-vs-oracle premium for that hour,
 *                            returned alongside the rate and unused by the
 *                            rest of this repo. It is NOT rate/8 — the rate is
 *                            computed from a TWAP of premium samples and then
 *                            clamped, so the two are related but distinct.
 *
 * Snapshot only, therefore useless for a 45-day backtest:
 *   metaAndAssetCtxs         openInterest, oraclePx, markPx, midPx, premium,
 *                            impactPxs, dayNtlVlm — current values, no history
 *                            endpoint of any kind. This is why there is no
 *                            open-interest feature below.
 *   l2Book                   20 levels a side, right now.
 *   predictedFundings        Binance/Bybit/HL funding, right now.
 *   perpsAtOpenInterestCap   a list, right now.
 *   recentTrades             undocumented; hard-capped at TEN trades (~15
 *                            seconds). Ignores startTime, n, nTrades. This is
 *                            the endpoint that would have given taker
 *                            buy/sell aggression, and it does not.
 *
 * Closed without credentials:
 *   s3://hyperliquid-archive          requester-pays: HTTP 403 anonymous.
 *   stats-data.hyperliquid.xyz        HTTP 403.
 *
 * So of the four things the README wanted to try next — order flow, holder
 * concentration, LP changes, liquidations — the public API supports exactly
 * none of them historically. What is left is funding, the premium, and the
 * trade tape's *count*.
 *
 * CAUSALITY
 *
 * The repo's cardinal rule: a feature at row i may use information available
 * at or before the close of bar i and nothing later. Hourly series are aligned
 * onto the candle grid with a strict step function — bar i sees the newest
 * print stamped STRICTLY BEFORE its close instant — forward-filled with a
 * staleness cap and never, ever interpolated backwards.
 *
 * WHAT THEY PREDICT (top 10 perps, 30m, 45 days to 2026-09-10, ~19,900 rows)
 *
 * Pooled rank correlation with the NEXT bar's return, per-coin z-scored:
 *
 *   logReturn        -0.0660   t -9.34   both halves agree   (short-horizon
 *                                                             reversal; the
 *                                                             calibration bar)
 *   avgTradeSizeZ     0.0100   t  1.41   0.0096 / 0.0097
 *   carryBps          0.0070   t  0.99
 *   volumeSurge       0.0053   t  0.74
 *   premiumBps        0.0024   t  0.34
 *   tradeCountSurge   0.0014   t  0.20  -0.0210 / 0.0215
 *   premiumZ         -0.0003   t -0.05
 *   fundingBps       -0.0041   t -0.57
 *   fundingZ         -0.0068   t -0.95
 *
 * Not one of them is distinguishable from zero, and the two largest flip sign
 * between halves. The same columns are NOT noise, though — against the same
 * bar's ABSOLUTE return they are strongly informative:
 *
 *   realizedVol 0.504   volumeSurge 0.283   tradeCountSurge 0.278
 *   avgTradeSizeZ 0.196   premiumBps 0.187   carryBps 0.139
 *
 * That is the whole finding, and it is the same one the microstructure
 * literature keeps reporting: order-flow and funding variables explain the
 * move that is happening, and do not forecast the sign of the next one.
 *
 * THE LITERATURE, AND WHAT IT ACTUALLY CLAIMS
 *
 * Order flow imbalance
 *   Cont, Kukanov & Stoikov (2014), "The Price Impact of Order Book Events",
 *   Journal of Financial Econometrics 12(1) 47-88 (arXiv:1011.6402). A linear
 *   relation between OFI and price changes with slope inversely proportional
 *   to depth. Read the verb: price changes are *driven by* OFI. It is a
 *   contemporaneous impact model, not a forecast, and it is routinely cited as
 *   if it were the latter.
 *   Makarov & Schoar (2020), JFE 135(2) 293-319, decompose signed volume
 *   across exchanges and find the common component "explains 80% of bitcoin
 *   returns" — again explains, same period.
 *   Bieganowski & Slepaczuk (2026), arXiv:2602.00776, get OFI to carry
 *   short-horizon predictive weight on Binance perps — at ONE-SECOND
 *   frequency, from the full order book. That is the horizon this effect lives
 *   at, and it is four orders of magnitude away from a 30m bar.
 *
 * VPIN
 *   Easley, Lopez de Prado & O'Hara (2012), "Flow Toxicity and Liquidity in a
 *   High-frequency World", RFS 25(5) 1457-1493, propose VPIN and claim it is
 *   "a useful indicator of short-term, toxicity-induced volatility" — of
 *   VOLATILITY, never of return direction.
 *   Andersen & Bondarenko (2014), "VPIN and the flash crash", Journal of
 *   Financial Markets 17 1-46, then find VPIN "a poor predictor of short run
 *   volatility", that it peaked AFTER rather than before the flash crash, and
 *   that its predictive content is "due primarily to a mechanical relation
 *   with the underlying trading intensity". Even the volatility claim is
 *   contested; the directional one was never made.
 *
 * Open interest
 *   Bessembinder & Seguin (1993), JFQA 28(1) 21-39: "large open interest
 *   mitigates volatility" — OI as a depth proxy, a volatility story.
 *   Hong & Yogo (2012), JFE 105(3) 473-490: movements in open interest DO
 *   predict commodity, bond and short-rate movements. Real, replicated, and at
 *   monthly frequency on macro hedging demand — nothing to do with the next
 *   thirty minutes of a perp.
 *   The trader's four-quadrant table (up+OI up = new longs = bullish, etc.)
 *   has no academic source at all. It is an accounting identity about who
 *   holds the contract dressed as a forecast.
 *   Giagkiozis & Said (2024), Ledger 9 1-15, add the practical insult:
 *   exchange-reported OI in BTC perps is "systematically misquoted by some of
 *   the largest derivatives exchanges".
 *
 * Funding rates
 *   He, Manela, Ross & von Wachter, "Fundamentals of Perpetual Futures"
 *   (arXiv:2212.06888): no-arbitrage prices for perps, deviations larger in
 *   crypto than in FX, comoving and shrinking over time, and "an implied
 *   arbitrage strategy yields high Sharpe ratios". That is CARRY — you get
 *   paid for holding a hedged basis position. It is not a claim that funding
 *   tells you which way the next bar goes, and the two are constantly
 *   conflated.
 *
 * Liquidation cascades
 *   Chitra (arXiv:2512.01112) on autodeleveraging, and Garcia Seuma
 *   (arXiv:2607.27070, arXiv:2608.03616) on seven major cascades 2022-2025 —
 *   whose own finding is that early-warning signals are "event-heterogeneous",
 *   i.e. no reproducible predictor across events. Mechanism, not forecast.
 *
 * Verdicts. Funding as carry: real, and irrelevant to timing. Funding as a
 * directional timing signal: unsupported. OI at macro horizons: real. The
 * four-quadrant OI framework: folklore. OFI: real at the tick, contemporaneous
 * at anything slower. VPIN: contested even on its own volatility claim.
 * Liquidation cascades: descriptive.
 *
 * The measurement above is what that literature predicts you would find, and
 * it is what was found.
 */

import type { Candle, FeatureSet } from "./features";

const API = "https://api.hyperliquid.xyz/info";

/** Same light spacing hyperliquid.ts uses; the weight limit is 1200/min. */
const MIN_INTERVAL_MS = 150;
let lastCall = 0;

export type Transport = (body: unknown) => Promise<any>;

async function httpPost(body: unknown, retries = 3): Promise<any> {
  for (let attempt = 0; attempt <= retries; attempt++) {
    const wait = lastCall + MIN_INTERVAL_MS - Date.now();
    if (wait > 0) await new Promise((r) => setTimeout(r, wait));
    lastCall = Date.now();

    let res: Response;
    try {
      res = await fetch(API, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(30_000),
      });
    } catch (e) {
      if (attempt === retries) throw new Error(`hyperliquid network error: ${e instanceof Error ? e.message : e}`);
      await new Promise((r) => setTimeout(r, 1000 * (attempt + 1)));
      continue;
    }
    if (res.status === 429) {
      await new Promise((r) => setTimeout(r, 3000 * (attempt + 1)));
      continue;
    }
    if (!res.ok) {
      if (attempt === retries) throw new Error(`hyperliquid HTTP ${res.status}`);
      await new Promise((r) => setTimeout(r, 1000 * (attempt + 1)));
      continue;
    }
    return res.json();
  }
  throw new Error("hyperliquid: retries exhausted");
}

// ---------------------------------------------------------------------------
// Candles carrying the trade count
// ---------------------------------------------------------------------------

/** A candle plus the one field the rest of the repo throws away. */
export interface FlowCandle extends Candle {
  /** `n` from candleSnapshot: trades executed inside the bar. */
  trades: number;
}

const INTERVALS: Record<string, number> = {
  "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
  "1h": 3600, "2h": 7200, "4h": 14400, "8h": 28800, "12h": 43200,
  "1d": 86400, "3d": 259200, "1w": 604800, "1M": 2592000,
};

export function flowIntervalSeconds(interval: string): number {
  const s = INTERVALS[interval];
  if (!s) throw new Error(`unsupported hyperliquid interval "${interval}"`);
  return s;
}

/**
 * Candles with `trades` kept. Structurally a `Candle[]`, so it can be handed
 * straight to `buildFeatures` — the extra field is ignored there and used here.
 */
export async function fetchFlowCandles(
  coin: string,
  interval: string,
  bars: number,
  opts: { transport?: Transport } = {},
): Promise<FlowCandle[]> {
  const post = opts.transport ?? httpPost;
  const secs = flowIntervalSeconds(interval);
  const byTime = new Map<number, FlowCandle>();
  let endTime = Date.now();

  while (byTime.size < bars) {
    const startTime = endTime - 5200 * secs * 1000;
    const rows: any[] = await post({ type: "candleSnapshot", req: { coin, interval, startTime, endTime } });
    if (!Array.isArray(rows) || rows.length === 0) break;

    let oldest = Infinity;
    for (const r of rows) {
      const t = Math.floor(Number(r.t) / 1000);
      const close = Number(r.c);
      if (!Number.isFinite(t) || !Number.isFinite(close) || close <= 0) continue;
      if (!byTime.has(t)) {
        byTime.set(t, {
          time: t,
          open: Number(r.o) || close,
          high: Number(r.h) || close,
          low: Number(r.l) || close,
          close,
          volume: Math.max(Number(r.v) || 0, 0),
          trades: Math.max(Number(r.n) || 0, 0),
        });
      }
      if (t < oldest) oldest = t;
    }
    if (!Number.isFinite(oldest)) break;
    const nextEnd = oldest * 1000 - 1;
    if (nextEnd >= endTime) break;
    endTime = nextEnd;
    if (rows.length < 100) break;
  }

  return [...byTime.values()].sort((a, b) => a.time - b.time).slice(-bars);
}

// ---------------------------------------------------------------------------
// Funding history, and the paging trap
// ---------------------------------------------------------------------------

export interface FundingBar {
  /** Unix MILLISECONDS of the hour the rate applied to. Kept in ms on purpose:
   *  the API stamps rows a few ms past the hour and rounding to seconds loses
   *  exactly the ordering the causality check depends on. */
  timeMs: number;
  /** Hourly funding rate as a fraction. Longs pay when positive. */
  rate: number;
  /** Perp-vs-oracle premium reported for that hour, as a fraction. */
  premium: number;
}

/** The endpoint serves at most this many hourly rows per request. Measured. */
export const FUNDING_PAGE_HOURS = 500;

/**
 * `fundingHistory` returns the FIRST ~500 hours at or after `startTime`, not
 * the most recent 500 — asking for 45 days back returns days 45..24 and leaves
 * the window you care about empty. Verified on SOL, 2026-09-10:
 *
 *   startTime = now - 45d, no endTime  ->  500 rows, 2026-07-28 .. 2026-08-17
 *   now                                ->  2026-09-10
 *
 * `endTime` IS honoured, though, which the existing fetcher does not exploit.
 * So rather than paging forward off the newest row seen (40 round trips worst
 * case), split the range into windows of < 500 hours and pin both ends. 45
 * days is three requests, and every one of them is a bounded, resumable ask.
 */
export function fundingChunks(
  startMs: number,
  endMs: number,
  maxHours = FUNDING_PAGE_HOURS - 20,
): Array<{ startTime: number; endTime: number }> {
  if (!(endMs > startMs)) return [];
  const span = maxHours * 3600_000;
  const out: Array<{ startTime: number; endTime: number }> = [];
  for (let s = startMs; s < endMs; s += span) {
    out.push({ startTime: s, endTime: Math.min(s + span, endMs) });
  }
  return out;
}

/** Hourly funding + premium covering [startMs, endMs], chunked past the trap. */
export async function fetchFundingRange(
  coin: string,
  startMs: number,
  endMs: number,
  opts: { transport?: Transport } = {},
): Promise<FundingBar[]> {
  const post = opts.transport ?? httpPost;
  const byTime = new Map<number, FundingBar>();

  for (const chunk of fundingChunks(startMs, endMs)) {
    const rows: any[] = await post({ type: "fundingHistory", coin, startTime: chunk.startTime, endTime: chunk.endTime });
    if (!Array.isArray(rows)) continue;
    for (const r of rows) {
      const timeMs = Number(r.time);
      const rate = Number(r.fundingRate);
      const premium = Number(r.premium);
      if (!Number.isFinite(timeMs) || !Number.isFinite(rate)) continue;
      byTime.set(timeMs, { timeMs, rate, premium: Number.isFinite(premium) ? premium : NaN });
    }
  }

  return [...byTime.values()].sort((a, b) => a.timeMs - b.timeMs);
}

// ---------------------------------------------------------------------------
// Aligning an irregular series onto the candle grid
// ---------------------------------------------------------------------------

export interface StepPoint {
  timeMs: number;
  value: number;
}

export interface AlignedSeries {
  /** One value per candle. NaN where nothing usable was available. */
  value: Float64Array;
  /** 1 where `value` is a real, non-stale observation. */
  valid: Uint8Array;
  /** Age in ms of the print used, measured from the bar's close instant. */
  ageMs: Float64Array;
}

/**
 * Step-function alignment. Bar i is assigned the newest point stamped STRICTLY
 * BEFORE bar i's close instant, forward-filled until a newer one arrives and
 * discarded once it is older than `maxStaleMs`.
 *
 * Strictly-before, not at-or-before: a funding row stamped 12:00:00.026 must
 * not reach the 11:30-12:00 bar, whose close instant is 12:00:00.000. Working
 * in milliseconds is what makes that distinction expressible at all.
 *
 * There is deliberately no backward fill. A bar before the first observation
 * gets `valid = 0`, never the first future value — that is lookahead wearing
 * the name "interpolation".
 */
export function alignStep(
  points: StepPoint[],
  candles: Candle[],
  opts: { intervalSeconds: number; maxStaleMs: number },
): AlignedSeries {
  const n = candles.length;
  const value = new Float64Array(n).fill(NaN);
  const valid = new Uint8Array(n);
  const ageMs = new Float64Array(n).fill(NaN);

  const pts = points
    .filter((p) => Number.isFinite(p.timeMs) && Number.isFinite(p.value))
    .sort((a, b) => a.timeMs - b.timeMs);
  if (pts.length === 0) return { value, valid, ageMs };

  let j = -1; // index of the newest point known to be usable
  for (let i = 0; i < n; i++) {
    const closeMs = (candles[i].time + opts.intervalSeconds) * 1000;
    while (j + 1 < pts.length && pts[j + 1].timeMs < closeMs) j++;
    if (j < 0) continue;
    const age = closeMs - pts[j].timeMs;
    ageMs[i] = age;
    if (age <= opts.maxStaleMs) {
      value[i] = pts[j].value;
      valid[i] = 1;
    }
  }
  return { value, valid, ageMs };
}

// ---------------------------------------------------------------------------
// Derived hourly series (computed at the funding grid, then aligned)
// ---------------------------------------------------------------------------

/** Trailing z-score, inclusive of the current point. NaN until the window fills. */
function rollingZ(values: number[], window: number): number[] {
  const out = new Array<number>(values.length).fill(NaN);
  for (let i = window - 1; i < values.length; i++) {
    let mean = 0;
    for (let j = i - window + 1; j <= i; j++) mean += values[j];
    mean /= window;
    let sq = 0;
    for (let j = i - window + 1; j <= i; j++) {
      const d = values[j] - mean;
      sq += d * d;
    }
    const sd = Math.sqrt(sq / window);
    out[i] = sd > 1e-12 ? (values[i] - mean) / sd : 0;
  }
  return out;
}

/** Trailing sum, inclusive of the current point. NaN until the window fills. */
function rollingSum(values: number[], window: number): number[] {
  const out = new Array<number>(values.length).fill(NaN);
  let acc = 0;
  for (let i = 0; i < values.length; i++) {
    acc += values[i];
    if (i >= window) acc -= values[i - window];
    if (i >= window - 1) out[i] = acc;
  }
  return out;
}

function points(times: number[], values: number[]): StepPoint[] {
  const out: StepPoint[] = [];
  for (let i = 0; i < times.length; i++) {
    if (Number.isFinite(values[i])) out.push({ timeMs: times[i], value: values[i] });
  }
  return out;
}

// ---------------------------------------------------------------------------
// The feature columns
// ---------------------------------------------------------------------------

export interface FlowConfig {
  /** Lookback for the candle-grid features. Match `buildFeatures`. */
  window?: number;
  /** Lookback in HOURS for the funding z-scores. */
  fundingWindowHours?: number;
  /** Hours of carry to accumulate for `fundingCum`. */
  carryHours?: number;
  /** How long a funding print may be forward-filled past its own timestamp. */
  maxStaleHours?: number;
  useFunding?: boolean;
  usePremium?: boolean;
  useTape?: boolean;
}

export interface FlowColumns {
  names: string[];
  /** One Float64Array per name, length = candles.length. NaN where invalid. */
  cols: Float64Array[];
  /** 1 where EVERY column is valid at that candle. */
  valid: Uint8Array;
  /** Per-column validity, same order as `names`. */
  colValid: Uint8Array[];
}

/**
 * Build the non-price columns on the candle grid.
 *
 * Units are chosen so the raw numbers are readable before standardization:
 * funding and premium in basis points, the tape features in logs.
 *
 *   fundingBps      rate for the last completed funding hour, x1e4
 *   fundingZ        that rate against its own trailing distribution
 *   carryBps        funding accrued over the trailing `carryHours`, x1e4
 *   premiumBps      perp-vs-oracle premium for that hour, x1e4
 *   premiumZ        that premium against its own trailing distribution
 *   tradeCountSurge log(n / trailing mean n) — the volume-surge idea, but on
 *                   the count rather than the size. Same bar, different axis.
 *   avgTradeSizeZ   z-score of log(volume / n). Rises when the same notional
 *                   arrives in fewer, larger prints — the standard cheap proxy
 *                   for a large participant working an order.
 */
export function buildFlowColumns(
  candles: FlowCandle[],
  funding: FundingBar[],
  interval: string,
  config: FlowConfig = {},
): FlowColumns {
  const window = config.window ?? 20;
  const fundingWindowHours = config.fundingWindowHours ?? 168;
  const carryHours = config.carryHours ?? 24;
  const maxStaleHours = config.maxStaleHours ?? 3;
  const useFunding = config.useFunding ?? true;
  const usePremium = config.usePremium ?? true;
  const useTape = config.useTape ?? true;

  const secs = flowIntervalSeconds(interval);
  const n = candles.length;
  const names: string[] = [];
  const cols: Float64Array[] = [];
  const colValid: Uint8Array[] = [];

  const maxStaleMs = maxStaleHours * 3600_000 + secs * 1000;
  const push = (name: string, s: AlignedSeries) => {
    names.push(name);
    cols.push(s.value);
    colValid.push(s.valid);
  };
  const pushRaw = (name: string, value: Float64Array, valid: Uint8Array) => {
    names.push(name);
    cols.push(value);
    colValid.push(valid);
  };

  if (useFunding || usePremium) {
    const times = funding.map((f) => f.timeMs);
    const rates = funding.map((f) => f.rate);
    const prem = funding.map((f) => f.premium);

    if (useFunding) {
      push("fundingBps", alignStep(points(times, rates.map((r) => r * 1e4)), candles, { intervalSeconds: secs, maxStaleMs }));
      push("fundingZ", alignStep(points(times, rollingZ(rates, fundingWindowHours)), candles, { intervalSeconds: secs, maxStaleMs }));
      push("carryBps", alignStep(points(times, rollingSum(rates, carryHours).map((v) => v * 1e4)), candles, { intervalSeconds: secs, maxStaleMs }));
    }
    if (usePremium) {
      push("premiumBps", alignStep(points(times, prem.map((p) => p * 1e4)), candles, { intervalSeconds: secs, maxStaleMs }));
      push("premiumZ", alignStep(points(times, rollingZ(prem, fundingWindowHours)), candles, { intervalSeconds: secs, maxStaleMs }));
    }
  }

  if (useTape) {
    const surge = new Float64Array(n).fill(NaN);
    const surgeOk = new Uint8Array(n);
    const sizeZ = new Float64Array(n).fill(NaN);
    const sizeOk = new Uint8Array(n);

    // log average trade size, per bar. Undefined on a bar with no trades.
    const logSize = new Float64Array(n).fill(NaN);
    for (let i = 0; i < n; i++) {
      const t = candles[i].trades;
      if (t > 0 && candles[i].volume > 0) logSize[i] = Math.log(candles[i].volume / t);
    }

    for (let i = window - 1; i < n; i++) {
      let cnt = 0;
      for (let j = i - window + 1; j <= i; j++) cnt += Math.max(candles[j].trades, 0);
      const baseline = cnt / window;
      surge[i] = Math.log((Math.max(candles[i].trades, 0) + 1e-8) / (baseline + 1e-8));
      surgeOk[i] = 1;

      // z-score of log size, skipping bars that had no trades at all.
      let m = 0, k = 0;
      for (let j = i - window + 1; j <= i; j++) if (Number.isFinite(logSize[j])) { m += logSize[j]; k++; }
      if (k >= 2 && Number.isFinite(logSize[i])) {
        m /= k;
        let sq = 0;
        for (let j = i - window + 1; j <= i; j++) if (Number.isFinite(logSize[j])) { const d = logSize[j] - m; sq += d * d; }
        const sd = Math.sqrt(sq / k);
        sizeZ[i] = sd > 1e-12 ? (logSize[i] - m) / sd : 0;
        sizeOk[i] = 1;
      }
    }
    pushRaw("tradeCountSurge", surge, surgeOk);
    pushRaw("avgTradeSizeZ", sizeZ, sizeOk);
  }

  const valid = new Uint8Array(n).fill(1);
  for (let i = 0; i < n; i++) {
    for (const v of colValid) if (!v[i]) { valid[i] = 0; break; }
  }
  return { names, cols, valid, colValid };
}

/** Project candle-grid columns onto the rows of a FeatureSet. */
export function projectToRows(cols: FlowColumns, index: Int32Array): FlowColumns {
  const T = index.length;
  const out: Float64Array[] = [];
  const outValid: Uint8Array[] = [];
  for (let c = 0; c < cols.cols.length; c++) {
    const v = new Float64Array(T);
    const ok = new Uint8Array(T);
    for (let r = 0; r < T; r++) {
      v[r] = cols.cols[c][index[r]];
      ok[r] = cols.colValid[c][index[r]];
    }
    out.push(v);
    outValid.push(ok);
  }
  const valid = new Uint8Array(T);
  for (let r = 0; r < T; r++) valid[r] = cols.valid[index[r]];
  return { names: cols.names.slice(), cols: out, valid, colValid: outValid };
}

/** Everything at once: candle-grid columns projected onto `fs`'s rows. */
export function flowFeatures(
  fs: FeatureSet,
  candles: FlowCandle[],
  funding: FundingBar[],
  interval: string,
  config: FlowConfig = {},
): FlowColumns {
  return projectToRows(buildFlowColumns(candles, funding, interval, config), fs.index);
}

// ---------------------------------------------------------------------------
// Appending columns to a FeatureSet
// ---------------------------------------------------------------------------

/**
 * Append columns to a FeatureSet. The input is never touched — every typed
 * array in the result is freshly allocated, so a caller holding `fs` cannot
 * observe the augmentation.
 *
 * Throws on any non-finite value. A NaN silently poisons a Gaussian emission
 * into `-Infinity` log-likelihood for every state at once, which shows up as
 * "EM diverged" three files away; refusing here is worth the inconvenience.
 * Use `dropInvalidRows` or `trimLeadingInvalid` to decide what to do first.
 */
export function augmentFeatures(
  fs: FeatureSet,
  extra: { names: string[]; cols: ArrayLike<number>[] },
): FeatureSet {
  if (extra.names.length !== extra.cols.length) {
    throw new Error(`augmentFeatures: ${extra.names.length} names but ${extra.cols.length} columns`);
  }
  for (const name of extra.names) {
    if (fs.names.includes(name)) throw new Error(`augmentFeatures: duplicate feature name "${name}"`);
  }
  for (let c = 0; c < extra.cols.length; c++) {
    if (extra.cols[c].length !== fs.T) {
      throw new Error(`augmentFeatures: column "${extra.names[c]}" has ${extra.cols[c].length} rows, expected ${fs.T}`);
    }
    for (let r = 0; r < fs.T; r++) {
      if (!Number.isFinite(extra.cols[c][r])) {
        throw new Error(`augmentFeatures: column "${extra.names[c]}" is not finite at row ${r}`);
      }
    }
  }

  const E = extra.cols.length;
  const D = fs.D + E;
  const X = new Float64Array(fs.T * D);
  for (let r = 0; r < fs.T; r++) {
    for (let d = 0; d < fs.D; d++) X[r * D + d] = fs.X[r * fs.D + d];
    for (let e = 0; e < E; e++) X[r * D + fs.D + e] = extra.cols[e][r];
  }
  return {
    X,
    T: fs.T,
    D,
    names: [...fs.names, ...extra.names],
    index: Int32Array.from(fs.index),
    rawReturn: Float64Array.from(fs.rawReturn),
  };
}

/** First row at which `valid` is 1, or -1 if there is none. */
export function firstValidRow(valid: ArrayLike<number>): number {
  for (let i = 0; i < valid.length; i++) if (valid[i]) return i;
  return -1;
}

/**
 * Drop the leading run of invalid rows and nothing else, so the surviving rows
 * stay contiguous in time — which is what a Markov chain with a fixed time step
 * requires. Throws if an invalid row survives, because silently keeping one
 * would mean the model steps over a hole it cannot see.
 */
export function trimLeadingInvalid(fs: FeatureSet, cols: FlowColumns): { fs: FeatureSet; cols: FlowColumns; dropped: number } {
  const start = firstValidRow(cols.valid);
  if (start < 0) throw new Error("trimLeadingInvalid: no valid rows at all");
  for (let r = start; r < cols.valid.length; r++) {
    if (!cols.valid[r]) throw new Error(`trimLeadingInvalid: row ${r} is invalid after the leading run; the series has a hole`);
  }
  const T = fs.T - start;
  const X = new Float64Array(T * fs.D);
  for (let r = 0; r < T; r++) for (let d = 0; d < fs.D; d++) X[r * fs.D + d] = fs.X[(r + start) * fs.D + d];
  const trimmed: FeatureSet = {
    X, T, D: fs.D, names: fs.names.slice(),
    index: fs.index.slice(start),
    rawReturn: fs.rawReturn.slice(start),
  };
  const outCols: FlowColumns = {
    names: cols.names.slice(),
    cols: cols.cols.map((c) => c.slice(start)),
    valid: cols.valid.slice(start),
    colValid: cols.colValid.map((c) => c.slice(start)),
  };
  return { fs: trimmed, cols: outCols, dropped: start };
}

// ---------------------------------------------------------------------------
// Predictive content — measurement only, no strategy anywhere near this
// ---------------------------------------------------------------------------

export function pearson(x: ArrayLike<number>, y: ArrayLike<number>): number {
  const n = x.length;
  if (n !== y.length) throw new Error("pearson: length mismatch");
  if (n < 3) return NaN;
  let mx = 0, my = 0;
  for (let i = 0; i < n; i++) { mx += x[i]; my += y[i]; }
  mx /= n; my /= n;
  let sxy = 0, sxx = 0, syy = 0;
  for (let i = 0; i < n; i++) {
    const a = x[i] - mx, b = y[i] - my;
    sxy += a * b; sxx += a * a; syy += b * b;
  }
  if (sxx <= 0 || syy <= 0) return NaN;
  return sxy / Math.sqrt(sxx * syy);
}

/** Average ranks, so ties do not tilt the correlation. */
export function ranks(v: ArrayLike<number>): Float64Array {
  const n = v.length;
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => v[a] - v[b]);
  const r = new Float64Array(n);
  let i = 0;
  while (i < n) {
    let j = i;
    while (j + 1 < n && v[order[j + 1]] === v[order[i]]) j++;
    const avg = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) r[order[k]] = avg;
    i = j + 1;
  }
  return r;
}

/** Spearman rank correlation — the information coefficient. */
export function spearman(x: ArrayLike<number>, y: ArrayLike<number>): number {
  return pearson(ranks(x), ranks(y));
}

export interface IcResult {
  n: number;
  /** Pearson correlation with the NEXT bar's return. */
  corr: number;
  /** Spearman rank correlation with the next bar's return — the IC. */
  ic: number;
  /** t-statistic of `ic` under iid, which returns are not. Read it as a scale. */
  t: number;
}

export function icStats(x: ArrayLike<number>, y: ArrayLike<number>): IcResult {
  const n = x.length;
  const corr = pearson(x, y);
  const ic = spearman(x, y);
  const t = Number.isFinite(ic) && n > 3 ? (ic * Math.sqrt(n - 2)) / Math.sqrt(Math.max(1 - ic * ic, 1e-12)) : NaN;
  return { n, corr, ic, t };
}

/**
 * The NEXT bar's return for each row: target[i] = rawReturn[i + 1].
 * The last row has no next bar and is marked invalid.
 *
 * This is the whole point of the exercise. Correlating a feature with the
 * CURRENT bar's return measures nothing but the feature's construction — a
 * volume surge is contemporaneous with the move that caused it.
 */
export function nextBarReturn(
  fs: FeatureSet,
  spacing?: { candles: Candle[]; intervalSeconds: number },
): { value: Float64Array; valid: Uint8Array } {
  const value = new Float64Array(fs.T).fill(NaN);
  const valid = new Uint8Array(fs.T);
  for (let i = 0; i + 1 < fs.T; i++) {
    // Only if row i+1 really is the next bar in TIME. `index` is a position in
    // the candle array and stays contiguous across a missing interval, so on a
    // gapped series it would quietly hand back a multi-bar return wearing a
    // one-bar label. Hyperliquid perps have no gaps; DEX series are full of them.
    if (spacing) {
      const a = spacing.candles[fs.index[i]];
      const b = spacing.candles[fs.index[i + 1]];
      if (!a || !b || b.time - a.time !== spacing.intervalSeconds) continue;
    }
    value[i] = fs.rawReturn[i + 1];
    valid[i] = 1;
  }
  return { value, valid };
}

/** Pairs where both sides are valid and finite. */
export function pairValid(
  x: ArrayLike<number>, xv: ArrayLike<number>,
  y: ArrayLike<number>, yv: ArrayLike<number>,
  from = 0, to = x.length,
): { x: Float64Array; y: Float64Array } {
  const xs: number[] = [], ys: number[] = [];
  for (let i = Math.max(from, 0); i < Math.min(to, x.length); i++) {
    if (xv[i] && yv[i] && Number.isFinite(x[i]) && Number.isFinite(y[i])) { xs.push(x[i]); ys.push(y[i]); }
  }
  return { x: Float64Array.from(xs), y: Float64Array.from(ys) };
}

export interface FeatureIc {
  name: string;
  full: IcResult;
  firstHalf: IcResult;
  secondHalf: IcResult;
}

/**
 * Correlation of each column with the next bar's return, over the whole sample
 * and over each half. The split is on ROW ORDER, so the second half is held out
 * in the only sense that matters here: it was not looked at when the feature
 * was designed.
 */
export function measureIc(
  fs: FeatureSet,
  cols: FlowColumns,
  extraNames: string[] = [],
  spacing?: { candles: Candle[]; intervalSeconds: number },
): FeatureIc[] {
  const target = nextBarReturn(fs, spacing);
  const mid = Math.floor(fs.T / 2);
  const out: FeatureIc[] = [];

  const add = (name: string, v: Float64Array, ok: Uint8Array) => {
    const all = pairValid(v, ok, target.value, target.valid);
    const a = pairValid(v, ok, target.value, target.valid, 0, mid);
    const b = pairValid(v, ok, target.value, target.valid, mid, fs.T);
    out.push({ name, full: icStats(all.x, all.y), firstHalf: icStats(a.x, a.y), secondHalf: icStats(b.x, b.y) });
  };

  // The existing price features, as a calibration bar.
  for (const name of extraNames) {
    const d = fs.names.indexOf(name);
    if (d < 0) continue;
    const v = new Float64Array(fs.T);
    for (let r = 0; r < fs.T; r++) v[r] = fs.X[r * fs.D + d];
    add(name, v, new Uint8Array(fs.T).fill(1));
  }
  for (let c = 0; c < cols.names.length; c++) add(cols.names[c], cols.cols[c], cols.colValid[c]);
  return out;
}
