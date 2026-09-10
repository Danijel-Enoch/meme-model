import { expect, test, describe } from "bun:test";
import { extractTrades, summarizeTrades } from "./backtest";
import { buildFeatures, type Candle } from "./features";

/** Candles with exact closes, so every expected number can be worked out by hand. */
function fromCloses(closes: number[]): Candle[] {
  return closes.map((c, i) => ({ time: 1_000_000 + i * 300, open: c, high: c, low: c, close: c, volume: 100 }));
}

const W = 2; // feature window: row r maps to candle index r + W
const cfg = { window: W, useVolatility: false, useVolume: false };

describe("trade extraction", () => {
  test("a single long trade earns the price move between entry and exit closes", () => {
    // candles: idx 0..5 -> closes below. Row r = candle r+2.
    const candles = fromCloses([100, 100, 100, 110, 121, 121]);
    // Long from row 0 (candle 2, close 100) through row 1; flat at row 2.
    const pos = [1, 1, 0, 0];
    const t = extractTrades(candles, cfg, pos, 0);
    expect(t.length).toBe(1);
    expect(t[0].entryPrice).toBe(100);
    // Held rows 0 and 1, earning the returns of rows 1 and 2 = 110/100 then 121/110.
    expect(t[0].grossReturn).toBeCloseTo(0.21, 10);
    expect(t[0].netReturn).toBeCloseTo(0.21, 10);
    expect(t[0].win).toBe(true);
    expect(t[0].barsHeld).toBe(2);
  });

  test("costs are charged on both legs", () => {
    const candles = fromCloses([100, 100, 100, 110, 110, 110]);
    const pos = [1, 0, 0, 0];
    const t = extractTrades(candles, cfg, pos, 100); // 100bps per side
    expect(t[0].grossReturn).toBeCloseTo(0.1, 10);
    expect(t[0].netReturn).toBeCloseTo(1.1 * 0.99 * 0.99 - 1, 10);
    expect(t[0].costPaid).toBeCloseTo(0.02, 10);
  });

  test("a losing trade is marked a loss", () => {
    const candles = fromCloses([100, 100, 100, 90, 90, 90]);
    const t = extractTrades(candles, cfg, [1, 0, 0, 0], 0);
    expect(t[0].netReturn).toBeCloseTo(-0.1, 10);
    expect(t[0].win).toBe(false);
  });

  test("a small gain becomes a loss once costs exceed it", () => {
    const candles = fromCloses([100, 100, 100, 100.5, 100.5, 100.5]);
    const free = extractTrades(candles, cfg, [1, 0, 0, 0], 0);
    const costly = extractTrades(candles, cfg, [1, 0, 0, 0], 50); // 50bps a side
    expect(free[0].win).toBe(true);
    expect(costly[0].win).toBe(false);
  });

  test("flipping long to short closes one trade and opens another", () => {
    const candles = fromCloses([100, 100, 100, 110, 100, 100, 100]);
    const t = extractTrades(candles, cfg, [1, -1, 0, 0, 0], 0);
    expect(t.length).toBe(2);
    expect(t[0].direction).toBe(1);
    expect(t[1].direction).toBe(-1);
  });

  test("a short profits when price falls", () => {
    const candles = fromCloses([100, 100, 100, 90, 90, 90]);
    const t = extractTrades(candles, cfg, [-1, 0, 0, 0], 0);
    expect(t[0].direction).toBe(-1);
    expect(t[0].netReturn).toBeCloseTo(0.1, 10);
    expect(t[0].win).toBe(true);
  });

  test("a position still open at the end is closed on the last bar", () => {
    const candles = fromCloses([100, 100, 100, 110, 120, 130]);
    const t = extractTrades(candles, cfg, [1, 1, 1, 1], 0);
    expect(t.length).toBe(1);
    const fs = buildFeatures(candles, cfg);
    expect(t[0].exitRow).toBe(fs.T - 1);
  });

  test("no positions means no trades", () => {
    const candles = fromCloses([100, 101, 102, 103, 104, 105]);
    expect(extractTrades(candles, cfg, [0, 0, 0, 0], 30).length).toBe(0);
  });

  test("entry and exit timestamps come from the traded candles", () => {
    const candles = fromCloses([100, 100, 100, 110, 110, 110]);
    const t = extractTrades(candles, cfg, [1, 0, 0, 0], 0);
    expect(t[0].entryTime).toBe(candles[W].time);
    expect(t[0].exitTime).toBe(candles[W + 1].time);
  });
});

describe("trade summary", () => {
  const candles = fromCloses([100, 100, 100, 110, 99, 99, 99, 99]);

  test("win rate and roi compound the individual trades", () => {
    const t = extractTrades(candles, cfg, [1, 0, 1, 0, 0, 0], 0);
    const s = summarizeTrades(t);
    expect(s.trades).toBe(2);
    expect(s.winRate).toBeCloseTo(s.wins / s.trades, 10);
    let eq = 1;
    for (const x of t) eq *= 1 + x.netReturn;
    expect(s.roi).toBeCloseTo(eq - 1, 12);
  });

  test("an empty ledger summarizes to zeroes, not NaN", () => {
    const s = summarizeTrades([]);
    expect(s.trades).toBe(0);
    expect(s.winRate).toBe(0);
    expect(s.roi).toBe(0);
    expect(Number.isNaN(s.profitFactor)).toBe(false);
  });

  test("profit factor exceeds 1 exactly when gross wins beat gross losses", () => {
    const t = extractTrades(candles, cfg, [1, 0, 0, 0, 0, 0], 0);
    const s = summarizeTrades(t);
    expect(s.profitFactor > 1).toBe(s.roi > 0);
  });
});
