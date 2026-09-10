/**
 * Hyperliquid perpetuals — market data from the public info API. No key.
 *
 * Why this matters for everything else in this repo: the DEX work kept dying
 * on two problems that do not exist here.
 *
 *   Data quality. GeckoTerminal omits untraded intervals, so a "5m bar" on a
 *   thin pool can span three hours — 26% of bars missing on a $5M pool, 63% on
 *   1m. A Hyperliquid perp is a continuous order book: 5000 consecutive 5m
 *   candles came back with zero irregular spacings.
 *
 *   Cost. A DEX round trip on a meme coin runs ~60bps once fee, priority fee
 *   and slippage are counted, against a per-bar signal of roughly 5bps. That
 *   gap is what made every strategy untradeable. Hyperliquid's base taker fee
 *   is 4.5bps a side, an order of magnitude cheaper.
 *
 * The model does not change. The economics it is being asked to clear do.
 */

import type { Candle } from "./features";

const API = "https://api.hyperliquid.xyz/info";

/** Documented base-tier fees, in basis points per side. */
export const HL_TAKER_BPS = 4.5;
export const HL_MAKER_BPS = 1.5;

/** Weight-based limit is generous (1200/min); a light spacing is plenty. */
const MIN_INTERVAL_MS = 150;
let lastCall = 0;

async function post(body: unknown, retries = 3): Promise<any> {
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

export interface HlMarket {
  coin: string;
  markPrice: number;
  dayNotionalVolume: number;
  openInterestUsd: number;
  maxLeverage: number;
  /** Funding rate for the current hour, as a fraction. */
  funding: number;
  prevDayPrice: number;
  change24h: number;
}

/** Live perp markets, sorted by 24h notional volume. Delisted ones are dropped. */
export async function topMarkets(limit = 10): Promise<HlMarket[]> {
  const [meta, ctxs] = await post({ type: "metaAndAssetCtxs" });
  const universe = meta?.universe ?? [];
  const out: HlMarket[] = [];

  for (let i = 0; i < universe.length; i++) {
    const m = universe[i];
    const c = ctxs[i];
    if (!m || !c || m.isDelisted) continue;
    const mark = Number(c.markPx ?? 0);
    const prev = Number(c.prevDayPx ?? 0);
    if (!Number.isFinite(mark) || mark <= 0) continue;
    out.push({
      coin: String(m.name),
      markPrice: mark,
      dayNotionalVolume: Number(c.dayNtlVlm ?? 0),
      openInterestUsd: Number(c.openInterest ?? 0) * mark,
      maxLeverage: Number(m.maxLeverage ?? 0),
      funding: Number(c.funding ?? 0),
      prevDayPrice: prev,
      change24h: prev > 0 ? mark / prev - 1 : 0,
    });
  }
  out.sort((a, b) => b.dayNotionalVolume - a.dayNotionalVolume);
  return out.slice(0, limit);
}

/** Intervals the candle endpoint accepts, with their length in seconds. */
const INTERVALS: Record<string, number> = {
  "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
  "1h": 3600, "2h": 7200, "4h": 14400, "8h": 28800, "12h": 43200,
  "1d": 86400, "3d": 259200, "1w": 604800, "1M": 2592000,
};

export function intervalSeconds(interval: string): number {
  const s = INTERVALS[interval];
  if (!s) throw new Error(`unsupported hyperliquid interval "${interval}". Use one of: ${Object.keys(INTERVALS).join(", ")}`);
  return s;
}

export interface HlFetchResult {
  candles: Candle[];
  coin: string;
  interval: string;
  intervalSeconds: number;
  gaps: number;
  requests: number;
}

/**
 * Pull `bars` candles, paging backwards.
 *
 * The endpoint caps a response at ~5000 candles regardless of the window asked
 * for, so long histories are assembled by walking `endTime` backwards. Bars are
 * keyed by open time and deduped, since consecutive pages overlap at the seam.
 */
export async function fetchCandles(
  coin: string,
  interval: string,
  bars: number,
  opts: { verbose?: boolean } = {},
): Promise<HlFetchResult> {
  const secs = intervalSeconds(interval);
  const byTime = new Map<number, Candle>();
  let endTime = Date.now();
  let requests = 0;

  while (byTime.size < bars) {
    // Ask for a little more than the cap so a full page always comes back.
    const startTime = endTime - 5200 * secs * 1000;
    const rows: any[] = await post({
      type: "candleSnapshot",
      req: { coin, interval, startTime, endTime },
    });
    requests++;
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
          // Base-asset volume; multiplied by price it is notional traded.
          volume: Math.max(Number(r.v) || 0, 0),
        });
      }
      if (t < oldest) oldest = t;
    }
    if (opts.verbose) console.error(`  ${coin}: ${byTime.size} bars after ${requests} requests`);
    if (!Number.isFinite(oldest)) break;
    const nextEnd = oldest * 1000 - 1;
    if (nextEnd >= endTime) break; // no progress; history exhausted
    endTime = nextEnd;
    if (rows.length < 100) break; // short page means we reached listing time
  }

  const candles = [...byTime.values()].sort((a, b) => a.time - b.time).slice(-bars);
  let gaps = 0;
  for (let i = 1; i < candles.length; i++) {
    gaps += Math.max(Math.round((candles[i].time - candles[i - 1].time) / secs) - 1, 0);
  }
  return { candles, coin, interval, intervalSeconds: secs, gaps, requests };
}

export interface FundingPoint {
  /** Unix seconds of the hour the rate applied to. */
  time: number;
  /** Rate for that hour, as a fraction. Longs pay when positive. */
  rate: number;
}

/**
 * Hourly funding history. Perps charge this every hour on the position's
 * notional, so at 2x leverage it is deducted from a $50 account twice as fast
 * as the headline rate suggests. Over a two-week hold it is small but not
 * negligible — roughly 0.34% of notional on BTC, more on the meme perps.
 */
export async function fetchFunding(coin: string, startTimeMs: number): Promise<FundingPoint[]> {
  // The endpoint returns roughly the first 500 hours AFTER startTime, not the
  // most recent 500. Requesting a long window therefore yields ancient data and
  // silently leaves the period you care about uncovered, so page forward.
  const byTime = new Map<number, number>();
  let cursor = startTimeMs;
  const now = Date.now();

  for (let page = 0; page < 40; page++) {
    const rows: any[] = await post({ type: "fundingHistory", coin, startTime: cursor });
    if (!Array.isArray(rows) || rows.length === 0) break;

    let newest = cursor;
    for (const r of rows) {
      const ms = Number(r.time);
      const rate = Number(r.fundingRate);
      if (!Number.isFinite(ms) || !Number.isFinite(rate)) continue;
      byTime.set(Math.floor(ms / 1000), rate);
      if (ms > newest) newest = ms;
    }
    if (newest <= cursor) break;          // no forward progress
    cursor = newest + 1;
    if (cursor >= now) break;             // caught up to the present
    if (rows.length < 100) break;         // short page: history exhausted
  }

  return [...byTime.entries()]
    .map(([time, rate]) => ({ time, rate }))
    .sort((a, b) => a.time - b.time);
}

/**
 * Maintenance margin fraction on Hyperliquid is half the initial margin
 * required at the asset's maximum leverage. A position is liquidated when
 * account equity falls below notional x this.
 */
export function maintenanceMarginFraction(maxLeverage: number): number {
  return maxLeverage > 0 ? 1 / (2 * maxLeverage) : 0.05;
}

export function hlBarsPerYear(interval: string): number {
  return (365.25 * 24 * 3600) / intervalSeconds(interval);
}
