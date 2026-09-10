/** Loading candles from CSV, and a synthetic regime-switching generator for the demo. */

import type { Candle } from "./features";
import { makeRng, randn } from "./hmm";

/**
 * Parse an OHLCV CSV. Header names are matched loosely, so the usual exports
 * (Binance, CoinGecko, Birdeye, DexScreener) work without editing.
 * A close column is the only hard requirement; OHLC fall back to close and
 * volume falls back to 0.
 */
export function parseCsv(text: string): Candle[] {
  const lines = text.trim().split(/\r?\n/).filter((l) => l.trim().length > 0);
  if (lines.length < 2) throw new Error("CSV has no data rows");

  const delim = (lines[0].match(/;/g)?.length ?? 0) > (lines[0].match(/,/g)?.length ?? 0) ? ";" : ",";
  const header = lines[0].split(delim).map((h) => h.trim().toLowerCase().replace(/^["']|["']$/g, ""));

  const find = (...cands: string[]) => {
    for (const c of cands) {
      const i = header.findIndex((h) => h === c);
      if (i >= 0) return i;
    }
    for (const c of cands) {
      const i = header.findIndex((h) => h.includes(c));
      if (i >= 0) return i;
    }
    return -1;
  };

  const iTime = find("time", "timestamp", "date", "open_time", "unix");
  const iOpen = find("open");
  const iHigh = find("high");
  const iLow = find("low");
  const iClose = find("close", "price", "last");
  const iVol = find("volume", "vol", "amount", "quote_volume");
  if (iClose < 0) throw new Error(`no close/price column found in header: ${header.join(", ")}`);

  const candles: Candle[] = [];
  for (let i = 1; i < lines.length; i++) {
    const parts = lines[i].split(delim);
    const num = (idx: number, fallback: number) => {
      if (idx < 0 || idx >= parts.length) return fallback;
      const v = parseFloat(parts[idx].replace(/["']/g, "").trim());
      return Number.isFinite(v) ? v : fallback;
    };
    const close = num(iClose, NaN);
    if (!Number.isFinite(close) || close <= 0) continue; // skip bad rows quietly
    let time = i;
    if (iTime >= 0) {
      const raw = parts[iTime].replace(/["']/g, "").trim();
      const asNum = parseFloat(raw);
      time = Number.isFinite(asNum) && raw.match(/^[\d.]+$/) ? asNum : Date.parse(raw) || i;
    }
    candles.push({
      time,
      open: num(iOpen, close),
      high: num(iHigh, close),
      low: num(iLow, close),
      close,
      volume: Math.max(num(iVol, 0), 0),
    });
  }
  if (candles.length === 0) throw new Error("no valid rows parsed");
  // Chronological order — some exports are newest-first.
  if (candles.length > 1 && candles[0].time > candles[candles.length - 1].time) candles.reverse();
  return candles;
}

export async function loadCsv(path: string): Promise<Candle[]> {
  const text = await Bun.file(path).text();
  return parseCsv(text);
}

export interface SynthConfig {
  bars?: number;
  seed?: number;
  startPrice?: number;
}

/**
 * Synthetic meme coin: a 3-regime Markov chain with the properties that make
 * these things distinctive — a rare, violent, short-lived pump regime, a long
 * grinding bleed, and a low-vol chop state that dominates the sample.
 * Returns are drawn from a normal/heavy mixture so the tails are not Gaussian.
 * This exists so `demo` runs with no data file; it is not real market data.
 */
export function generateSynthetic(config: SynthConfig = {}): { candles: Candle[]; trueStates: Int32Array } {
  const bars = config.bars ?? 6000;
  const rng = makeRng(config.seed ?? 7);
  let price = config.startPrice ?? 0.0001;

  // 0 = dump, 1 = chop, 2 = pump
  const A = [
    [0.94, 0.06, 0.00],
    [0.02, 0.96, 0.02],
    [0.06, 0.16, 0.78],
  ];
  const drift = [-0.0035, 0.0000, 0.0090];
  const vol = [0.030, 0.011, 0.055];
  const volScale = [1.6, 1.0, 4.2];

  const candles: Candle[] = [];
  const trueStates = new Int32Array(bars);
  let s = 1;
  const t0 = Date.UTC(2025, 0, 1) / 1000;

  for (let i = 0; i < bars; i++) {
    const u = rng();
    let acc = 0, next = 0;
    for (let j = 0; j < 3; j++) {
      acc += A[s][j];
      if (u <= acc) { next = j; break; }
    }
    s = next;
    trueStates[i] = s;

    // Heavy tails: 8% of bars come from a 3x-wide component.
    const shock = rng() < 0.08 ? 3 : 1;
    const r = drift[s] + vol[s] * shock * randn(rng);

    const open = price;
    price = Math.max(price * Math.exp(r), 1e-12);
    const close = price;
    const wick = vol[s] * Math.abs(randn(rng));
    const high = Math.max(open, close) * (1 + wick * 0.5);
    const low = Math.min(open, close) * (1 - wick * 0.5);
    // Volume tracks activity, with its own lognormal noise.
    const volume = Math.max(0, 50_000 * volScale[s] * Math.exp(0.6 * randn(rng)));

    candles.push({ time: t0 + i * 300, open, high, low, close, volume });
  }
  return { candles, trueStates };
}

export function toCsv(candles: Candle[]): string {
  const rows = ["time,open,high,low,close,volume"];
  for (const c of candles) {
    rows.push(`${c.time},${c.open},${c.high},${c.low},${c.close},${c.volume}`);
  }
  return rows.join("\n");
}
