/**
 * Free market data sources. No API keys.
 *
 *   GeckoTerminal  — OHLCV candle history. The only free source here that
 *                    actually serves candles: 1000 bars per request, paged
 *                    backwards with `before_timestamp`. ~30 req/min.
 *   DexScreener    — pair discovery and live liquidity. Its public API has no
 *                    candle endpoint, so it is used to find *which* pool to
 *                    pull, then GeckoTerminal supplies the history.
 *
 * Two things about this data that matter more than they look:
 *   - `before_timestamp` is inclusive, so consecutive pages share a bar.
 *     Un-deduped, that injects a fake zero return on every page boundary.
 *   - Illiquid coins have missing bars. GeckoTerminal omits untraded intervals
 *     entirely, so a "5m bar" can silently span three hours. An HMM assumes a
 *     fixed time step, so gaps have to be surfaced and dealt with, not ignored.
 */

import type { Candle } from "./features";

const GT = "https://api.geckoterminal.com/api/v2";
const DS = "https://api.dexscreener.com/latest/dex";
const GT_HEADERS = { Accept: "application/json;version=20230302" };

/** Free tier is ~30 requests/minute. Stay under it rather than eat 429s. */
const MIN_INTERVAL_MS = 2500;
let lastCall = 0;

async function throttle() {
  const wait = lastCall + MIN_INTERVAL_MS - Date.now();
  if (wait > 0) await new Promise((r) => setTimeout(r, wait));
  lastCall = Date.now();
}

async function getJson(url: string, headers: Record<string, string> = {}, retries = 3): Promise<any> {
  // 429s get their own budget. Being throttled is not a failure of the request,
  // and counting it against the error retries makes a slow window look like an
  // outage — which is exactly what happened before this was split out.
  let rateLimitWaits = 0;
  const MAX_RATE_LIMIT_WAITS = 6;

  for (let attempt = 0; attempt <= retries; ) {
    await throttle();
    let res: Response;
    try {
      res = await fetch(url, { headers, signal: AbortSignal.timeout(30_000) });
    } catch (e) {
      attempt++;
      if (attempt > retries) throw new Error(`network error for ${url}: ${e instanceof Error ? e.message : e}`);
      await new Promise((r) => setTimeout(r, 1500 * attempt));
      continue;
    }
    if (res.status === 429) {
      rateLimitWaits++;
      if (rateLimitWaits > MAX_RATE_LIMIT_WAITS) {
        throw new Error(`rate limited repeatedly by ${new URL(url).host} — wait a minute and retry`);
      }
      const wait = Math.min(5000 * rateLimitWaits, 30_000);
      console.error(`  rate limited, waiting ${wait / 1000}s...`);
      await new Promise((r) => setTimeout(r, wait));
      continue;
    }
    attempt++;
    if (res.status === 404) throw new Error(`not found (404): ${url}`);
    if (!res.ok) {
      if (attempt > retries) throw new Error(`HTTP ${res.status} for ${url}`);
      await new Promise((r) => setTimeout(r, 1500 * attempt));
      continue;
    }
    return res.json();
  }
  throw new Error(`gave up after ${retries + 1} attempts: ${url}`);
}

export interface Timeframe {
  unit: "minute" | "hour" | "day";
  aggregate: number;
  seconds: number;
  label: string;
}

/** GeckoTerminal only supports these aggregates; anything else is rejected upstream. */
const TIMEFRAMES: Record<string, Timeframe> = {
  "1m":  { unit: "minute", aggregate: 1,  seconds: 60,    label: "1m" },
  "5m":  { unit: "minute", aggregate: 5,  seconds: 300,   label: "5m" },
  "15m": { unit: "minute", aggregate: 15, seconds: 900,   label: "15m" },
  "1h":  { unit: "hour",   aggregate: 1,  seconds: 3600,  label: "1h" },
  "4h":  { unit: "hour",   aggregate: 4,  seconds: 14400, label: "4h" },
  "12h": { unit: "hour",   aggregate: 12, seconds: 43200, label: "12h" },
  "1d":  { unit: "day",    aggregate: 1,  seconds: 86400, label: "1d" },
};

export function parseTimeframe(tf: string): Timeframe {
  const t = TIMEFRAMES[tf.toLowerCase()];
  if (!t) throw new Error(`unsupported timeframe "${tf}". Use one of: ${Object.keys(TIMEFRAMES).join(", ")}`);
  return t;
}

/** Bars per year for a timeframe — feeds Sharpe annualization. */
export function barsPerYear(tf: Timeframe): number {
  return (365.25 * 24 * 3600) / tf.seconds;
}

/** Common aliases -> GeckoTerminal network ids. */
const NETWORK_ALIASES: Record<string, string> = {
  sol: "solana", solana: "solana",
  eth: "eth", ethereum: "eth", mainnet: "eth",
  base: "base",
  bsc: "bsc", bnb: "bsc",
  arb: "arbitrum", arbitrum: "arbitrum",
  poly: "polygon_pos", polygon: "polygon_pos", matic: "polygon_pos",
  avax: "avax", avalanche: "avax",
  op: "optimism", optimism: "optimism",
  ton: "ton", sui: "sui", tron: "tron", hyperevm: "hyperevm", abstract: "abstract",
  blast: "blast", linea: "linea", scroll: "scroll", zksync: "zksync",
  pulsechain: "pulsechain", berachain: "berachain", sonic: "sonic", unichain: "unichain",
};

export function resolveNetwork(name: string): string {
  const key = name.toLowerCase().trim();
  return NETWORK_ALIASES[key] ?? key; // unknown ids pass through; the API rejects bad ones
}

/** DexScreener uses its own chain ids, mostly matching the common alias. */
const DS_CHAIN: Record<string, string> = {
  solana: "solana", eth: "ethereum", base: "base", bsc: "bsc",
  arbitrum: "arbitrum", polygon_pos: "polygon", avax: "avalanche", optimism: "optimism",
};

export interface PairInfo {
  chain: string;
  network: string;
  pairAddress: string;
  baseSymbol: string;
  quoteSymbol: string;
  dex: string;
  priceUsd: number;
  liquidityUsd: number;
  volume24h: number;
  ageDays: number | null;
  url: string;
}

/**
 * Token names on these chains are attacker-controlled: they can carry control
 * characters, bidi overrides, or run to kilobytes (a real DexScreener search
 * result during development had a 10KB symbol). Strip and truncate before any
 * of it reaches a terminal.
 */
const UNSAFE_CHARS = new RegExp("[\\u0000-\\u001F\\u007F-\\u009F\\u200B-\\u200F\\u2028\\u2029\\u202A-\\u202E\\u2060\\uFEFF]", "g");

function sanitize(s: unknown, max = 24): string {
  if (typeof s !== "string") return "?";
  const clean = s.replace(UNSAFE_CHARS, "").trim();
  return clean.length > max ? clean.slice(0, max) + "..." : clean || "?";
}

/**
 * Rank pools by the geometric mean of liquidity and 24h volume.
 *
 * Neither number alone works. Ranking by liquidity picks stale pools — the
 * deepest dogwifhat pool holds $59M and trades $0, which yields a candle
 * series that is almost entirely gaps. Ranking by volume alone rewards
 * wash-traded shells with no real depth. Requiring both is the filter that
 * survives contact with this data.
 */
function poolScore(liquidityUsd: number, volume24h: number): number {
  return Math.sqrt(Math.max(liquidityUsd, 0) * Math.max(volume24h, 0));
}

/**
 * Find candidate pools for a token symbol or address, best first.
 *
 * DexScreener search is a literal text match on symbol and name, so a bare
 * ticker often misses the real token and returns only impersonators: querying
 * "WIF" does not find dogwifhat, whose symbol is "$WIF", but does find seven
 * copycats. Short queries are therefore tried in a few spellings and merged.
 */
export async function searchPairs(
  query: string,
  network?: string,
  minLiquidity = 10_000,
  minVolume24h = 1_000,
): Promise<PairInfo[]> {
  const q = query.trim();
  const variants = [q];
  // Only for ticker-shaped queries — never for addresses or multi-word names.
  if (q.length <= 12 && !q.includes(" ") && !/^0x|^[1-9A-HJ-NP-Za-km-z]{32,}$/.test(q)) {
    if (!q.startsWith("$")) variants.push(`$${q}`);
    else variants.push(q.slice(1));
  }

  const wanted = network ? DS_CHAIN[resolveNetwork(network)] ?? resolveNetwork(network) : null;
  const byPair = new Map<string, PairInfo>();

  for (const v of variants) {
    let data: any;
    try {
      data = await getJson(`${DS}/search?q=${encodeURIComponent(v)}`);
    } catch {
      continue; // one spelling failing should not sink the search
    }
    for (const p of data.pairs ?? []) {
      const liq = Number(p.liquidity?.usd ?? 0);
      const vol = Number(p.volume?.h24 ?? 0);
      if (!Number.isFinite(liq) || liq < minLiquidity) continue;
      if (!Number.isFinite(vol) || vol < minVolume24h) continue;
      if (wanted && p.chainId !== wanted) continue;
      if (!p.pairAddress || byPair.has(p.pairAddress)) continue;
      const gtNetwork = Object.entries(DS_CHAIN).find(([, val]) => val === p.chainId)?.[0] ?? p.chainId;
      byPair.set(p.pairAddress, {
        chain: p.chainId,
        network: gtNetwork,
        pairAddress: p.pairAddress,
        baseSymbol: sanitize(p.baseToken?.symbol),
        quoteSymbol: sanitize(p.quoteToken?.symbol),
        dex: sanitize(p.dexId, 16),
        priceUsd: Number(p.priceUsd ?? 0),
        liquidityUsd: liq,
        volume24h: vol,
        ageDays: p.pairCreatedAt ? (Date.now() - p.pairCreatedAt) / 86_400_000 : null,
        url: p.url ?? "",
      });
    }
  }

  return [...byPair.values()].sort(
    (a, b) => poolScore(b.liquidityUsd, b.volume24h) - poolScore(a.liquidityUsd, a.volume24h),
  );
}

export interface TrendingPool {
  network: string;
  address: string;
  name: string;
  priceUsd: number;
  volume24h: number;
  liquidityUsd: number;
  change24h: number;
}

export async function trendingPools(network: string, limit = 20): Promise<TrendingPool[]> {
  const net = resolveNetwork(network);
  const data = await getJson(`${GT}/networks/${net}/trending_pools?page=1`, GT_HEADERS);
  return (data.data ?? []).slice(0, limit).map((p: any) => ({
    network: net,
    address: p.attributes?.address ?? "",
    name: sanitize(p.attributes?.name, 32),
    priceUsd: Number(p.attributes?.base_token_price_usd ?? 0),
    volume24h: Number(p.attributes?.volume_usd?.h24 ?? 0),
    liquidityUsd: Number(p.attributes?.reserve_in_usd ?? 0),
    change24h: Number(p.attributes?.price_change_percentage?.h24 ?? 0),
  }));
}

export interface TopPool {
  network: string;
  address: string;
  name: string;
  baseSymbol: string;
  quoteSymbol: string;
  dex: string;
  priceUsd: number;
  liquidityUsd: number;
  volume24h: number;
  txns24h: number;
  ageDays: number | null;
  /** Geometric mean of liquidity and volume — see poolScore. */
  score: number;
  /** volume / liquidity. Extreme values mean wash trading, not a deep market. */
  turnover: number;
}

/**
 * Top pools on a network, by 24h volume, across several pages.
 *
 * Volume alone is not a ranking: the top "Solana" result by volume during
 * development had $106M of daily volume against $0.0000016 of liquidity.
 * The caller gets liquidity, turnover and age alongside, so obvious wash
 * trading can be screened out rather than traded.
 */
export async function topPools(network: string, pages = 2): Promise<TopPool[]> {
  const net = resolveNetwork(network);
  const out: TopPool[] = [];
  const seen = new Set<string>();

  for (let page = 1; page <= pages; page++) {
    let data: any;
    try {
      data = await getJson(`${GT}/networks/${net}/pools?page=${page}&sort=h24_volume_usd_desc`, GT_HEADERS);
    } catch {
      break; // a missing page just ends the listing
    }
    const rows = data?.data ?? [];
    if (rows.length === 0) break;
    for (const p of rows) {
      const a = p.attributes ?? {};
      const addr = a.address;
      if (!addr || seen.has(addr)) continue;
      seen.add(addr);
      const liq = Number(a.reserve_in_usd ?? 0);
      const vol = Number(a.volume_usd?.h24 ?? 0);
      const name: string = a.name ?? "?";
      const [base, quote] = name.split(" / ");
      const tx = a.transactions?.h24;
      out.push({
        network: net,
        address: addr,
        name: sanitize(name, 32),
        baseSymbol: sanitize(base, 16),
        quoteSymbol: sanitize(quote, 16),
        dex: sanitize(p.relationships?.dex?.data?.id, 16),
        priceUsd: Number(a.base_token_price_usd ?? 0),
        liquidityUsd: liq,
        volume24h: vol,
        txns24h: (Number(tx?.buys ?? 0) + Number(tx?.sells ?? 0)) || 0,
        ageDays: a.pool_created_at ? (Date.now() - Date.parse(a.pool_created_at)) / 86_400_000 : null,
        score: poolScore(liq, vol),
        turnover: liq > 0 ? vol / liq : Infinity,
      });
    }
  }
  return out.sort((a, b) => b.score - a.score);
}

export interface FetchResult {
  candles: Candle[];
  timeframe: Timeframe;
  network: string;
  pool: string;
  baseSymbol: string;
  quoteSymbol: string;
  /** Bars absent from the API response entirely (no trades in that interval). */
  gaps: number;
  /** Longest run of consecutive missing bars. */
  largestGap: number;
  duplicatesDropped: number;
  requests: number;
}

/**
 * Pull up to `bars` candles, walking backwards 1000 at a time.
 * Returns oldest-first, deduped, with gap statistics attached.
 */
export async function fetchOhlcv(
  network: string,
  pool: string,
  timeframeLabel: string,
  bars: number,
  opts: { currency?: "usd" | "token"; verbose?: boolean } = {},
): Promise<FetchResult> {
  const net = resolveNetwork(network);
  const tf = parseTimeframe(timeframeLabel);
  const currency = opts.currency ?? "usd";

  const byTime = new Map<number, Candle>();
  let duplicatesDropped = 0;
  let before: number | null = null;
  let requests = 0;
  let baseSymbol = "?", quoteSymbol = "?";

  while (byTime.size < bars) {
    const url =
      `${GT}/networks/${net}/pools/${pool}/ohlcv/${tf.unit}` +
      `?aggregate=${tf.aggregate}&limit=1000&currency=${currency}` +
      (before !== null ? `&before_timestamp=${before}` : "");

    const data = await getJson(url, GT_HEADERS);
    requests++;
    const list: number[][] = data?.data?.attributes?.ohlcv_list ?? [];
    baseSymbol = sanitize(data?.meta?.base?.symbol);
    quoteSymbol = sanitize(data?.meta?.quote?.symbol);

    if (list.length === 0) break; // reached the start of the pool's history

    let oldest = Infinity;
    for (const row of list) {
      const [time, open, high, low, close, volume] = row;
      if (!Number.isFinite(close) || close <= 0) continue; // drop unusable bars
      if (byTime.has(time)) { duplicatesDropped++; continue; }
      byTime.set(time, {
        time,
        open: Number(open) > 0 ? Number(open) : Number(close),
        high: Number(high) > 0 ? Number(high) : Number(close),
        low: Number(low) > 0 ? Number(low) : Number(close),
        close: Number(close),
        volume: Math.max(Number(volume) || 0, 0),
      });
      if (time < oldest) oldest = time;
    }

    if (opts.verbose) console.error(`  fetched ${byTime.size} bars (${requests} requests)`);
    if (!Number.isFinite(oldest) || oldest === before) break; // no further progress
    before = oldest;
    if (list.length < 1000) break; // a short page means history is exhausted
  }

  const candles = [...byTime.values()].sort((a, b) => a.time - b.time).slice(-bars);

  // Count intervals with no bar at all. These are not zeros in the data — the
  // API simply omits them, so the series silently changes its time step.
  let gaps = 0, largestGap = 0;
  for (let i = 1; i < candles.length; i++) {
    const missing = Math.round((candles[i].time - candles[i - 1].time) / tf.seconds) - 1;
    if (missing > 0) { gaps += missing; largestGap = Math.max(largestGap, missing); }
  }

  return { candles, timeframe: tf, network: net, pool, baseSymbol, quoteSymbol, gaps, largestGap, duplicatesDropped, requests };
}

/**
 * Insert flat bars for untraded intervals so the series has a constant time step,
 * which is what the Markov chain assumes. A filled bar is a real event — nobody
 * traded — so it carries a zero return and zero volume rather than an invented one.
 */
export function fillGaps(candles: Candle[], stepSeconds: number, maxFill = 12): { candles: Candle[]; filled: number; skipped: number } {
  if (candles.length === 0) return { candles, filled: 0, skipped: 0 };
  const out: Candle[] = [candles[0]];
  let filled = 0, skipped = 0;

  for (let i = 1; i < candles.length; i++) {
    const missing = Math.round((candles[i].time - candles[i - 1].time) / stepSeconds) - 1;
    if (missing > 0 && missing <= maxFill) {
      const prev = candles[i - 1];
      for (let m = 1; m <= missing; m++) {
        out.push({
          time: prev.time + m * stepSeconds,
          open: prev.close, high: prev.close, low: prev.close, close: prev.close,
          volume: 0,
        });
        filled++;
      }
    } else if (missing > maxFill) {
      // A gap this long is a different market, not a quiet patch. Forward-filling
      // hours of flat bars would fabricate a low-volatility regime that never existed.
      skipped += missing;
    }
    out.push(candles[i]);
  }
  return { candles: out, filled, skipped };
}

/** On-disk cache so repeated backtests do not re-hit the API. */
export function cachePath(network: string, pool: string, tf: string, bars: number): string {
  return `.cache/${resolveNetwork(network)}_${pool.slice(0, 12)}_${tf}_${bars}.json`;
}

export async function readCache(path: string, maxAgeMs: number): Promise<FetchResult | null> {
  try {
    const f = Bun.file(path);
    if (!(await f.exists())) return null;
    const raw = JSON.parse(await f.text());
    if (Date.now() - raw.savedAt > maxAgeMs) return null;
    return raw.result as FetchResult;
  } catch {
    return null;
  }
}

export async function writeCache(path: string, result: FetchResult): Promise<void> {
  await Bun.write(path, JSON.stringify({ savedAt: Date.now(), result }));
}
