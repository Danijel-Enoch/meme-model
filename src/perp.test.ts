import { expect, test, describe } from "bun:test";
import { simulatePerp, type PerpConfig } from "./perp";
import type { Candle } from "./features";
import type { FundingPoint } from "./hyperliquid";

const W = 2;
const cfg = { window: W, useVolatility: false, useVolume: false };
const base: PerpConfig = {
  startingEquity: 50, leverage: 2, takerBps: 4.5,
  maintenanceMarginFraction: 0.0125, minOrderUsd: 10,
};

/** Candles with explicit highs/lows so liquidation paths can be controlled. */
function mk(closes: number[], lows?: number[], highs?: number[]): Candle[] {
  return closes.map((c, i) => ({
    time: 1_000_000 + i * 3600,
    open: c,
    high: highs ? highs[i] : c,
    low: lows ? lows[i] : c,
    close: c,
    volume: 1,
  }));
}
const noFunding: FundingPoint[] = [];

describe("leveraged accounting", () => {
  test("pnl is on notional, so 2x doubles the move against equity", () => {
    // Row 0 = candle 2 (close 100). Long into row 1 = candle 3 (close 110): +10%.
    const candles = mk([100, 100, 100, 110, 110, 110]);
    const r = simulatePerp(candles, cfg, [1, 0, 0, 0], noFunding,
      { ...base, takerBps: 0 });
    expect(r.trades.length).toBe(1);
    // $50 equity, 2x = $100 notional, +10% = +$10 => +20% on equity.
    expect(r.trades[0].notional).toBeCloseTo(100, 10);
    expect(r.trades[0].grossPnl).toBeCloseTo(10, 8);
    expect(r.finalEquity).toBeCloseTo(60, 8);
  });

  test("fees are charged on notional both sides", () => {
    const candles = mk([100, 100, 100, 100, 100, 100]);
    const r = simulatePerp(candles, cfg, [1, 0, 0, 0], noFunding, base);
    // Flat price: pure cost. $100 notional x 4.5bps x 2 legs = $0.09.
    expect(r.totalFees).toBeCloseTo(0.09, 8);
    expect(r.finalEquity).toBeCloseTo(50 - 0.09, 8);
  });

  test("a 4.5bps fee costs 9bps of equity at 2x", () => {
    const candles = mk([100, 100, 100, 100, 100, 100]);
    const one = simulatePerp(candles, cfg, [1, 0, 0, 0], noFunding, { ...base, leverage: 1 });
    const two = simulatePerp(candles, cfg, [1, 0, 0, 0], noFunding, { ...base, leverage: 2 });
    expect((50 - two.finalEquity) / (50 - one.finalEquity)).toBeCloseTo(2, 6);
  });

  test("longs pay positive funding, on notional", () => {
    const candles = mk([100, 100, 100, 100, 100, 100]);
    // One hourly charge of 0.01 (absurdly large, to make it checkable).
    const f: FundingPoint[] = [{ time: candles[W].time + 1, rate: 0.01 }];
    const r = simulatePerp(candles, cfg, [1, 0, 0, 0], f, { ...base, takerBps: 0 });
    expect(r.totalFunding).toBeCloseTo(1.0, 8); // 0.01 x $100 notional
    expect(r.finalEquity).toBeCloseTo(49, 8);
  });

  test("shorts receive positive funding", () => {
    const candles = mk([100, 100, 100, 100, 100, 100]);
    const f: FundingPoint[] = [{ time: candles[W].time + 1, rate: 0.01 }];
    const r = simulatePerp(candles, cfg, [-1, 0, 0, 0], f, { ...base, takerBps: 0 });
    expect(r.totalFunding).toBeCloseTo(-1.0, 8);
    expect(r.finalEquity).toBeCloseTo(51, 8);
  });

  test("notional compounds with equity", () => {
    const candles = mk([100, 100, 100, 110, 110, 121, 121, 121]);
    const r = simulatePerp(candles, cfg, [1, 0, 1, 0, 0, 0], noFunding, { ...base, takerBps: 0 });
    expect(r.trades.length).toBe(2);
    // Second trade sizes off the grown equity, not the original $50.
    expect(r.trades[1].notional).toBeGreaterThan(r.trades[0].notional);
    expect(r.trades[1].notional).toBeCloseTo(r.trades[0].equityAfter * 2, 8);
  });
});

describe("liquidation", () => {
  test("an intrabar wick liquidates even if the close recovers", () => {
    // Close only ever dips to 99, but the low touches 45: a -55% excursion.
    const candles = mk([100, 100, 100, 99, 99, 99], [100, 100, 100, 45, 99, 99]);
    const r = simulatePerp(candles, cfg, [1, 1, 1, 1], noFunding, { ...base, takerBps: 0 });
    expect(r.liquidated).toBe(true);
    expect(r.finalEquity).toBe(0);
    expect(r.trades[0].liquidated).toBe(true);
  });

  test("the same series survives when only closes are inspected", () => {
    // Identical closes, no wick: proves the wick is what killed it above.
    const candles = mk([100, 100, 100, 99, 99, 99]);
    const r = simulatePerp(candles, cfg, [1, 1, 1, 1], noFunding, { ...base, takerBps: 0 });
    expect(r.liquidated).toBe(false);
    expect(r.finalEquity).toBeGreaterThan(0);
  });

  test("2x survives a 40% drawdown but not a 55% one", () => {
    const survives = mk([100, 100, 100, 60, 60, 60], [100, 100, 100, 60, 60, 60]);
    const dies = mk([100, 100, 100, 45, 45, 45], [100, 100, 100, 45, 45, 45]);
    expect(simulatePerp(survives, cfg, [1, 1, 1, 1], noFunding, { ...base, takerBps: 0 }).liquidated).toBe(false);
    expect(simulatePerp(dies, cfg, [1, 1, 1, 1], noFunding, { ...base, takerBps: 0 }).liquidated).toBe(true);
  });

  test("higher leverage liquidates sooner on the same path", () => {
    const candles = mk([100, 100, 100, 80, 80, 80], [100, 100, 100, 80, 80, 80]);
    const at2 = simulatePerp(candles, cfg, [1, 1, 1, 1], noFunding, { ...base, leverage: 2, takerBps: 0 });
    const at10 = simulatePerp(candles, cfg, [1, 1, 1, 1], noFunding, { ...base, leverage: 10, takerBps: 0 });
    expect(at2.liquidated).toBe(false);
    expect(at10.liquidated).toBe(true);
  });

  test("equity never goes negative", () => {
    const candles = mk([100, 100, 100, 10, 10, 10], [100, 100, 100, 10, 10, 10]);
    const r = simulatePerp(candles, cfg, [1, 1, 1, 1], noFunding, base);
    expect(r.finalEquity).toBeGreaterThanOrEqual(0);
  });
});

describe("order constraints", () => {
  test("a position below the exchange minimum is not opened", () => {
    const candles = mk([100, 100, 100, 110, 110, 110]);
    const r = simulatePerp(candles, cfg, [1, 0, 0, 0], noFunding,
      { ...base, startingEquity: 2, leverage: 2, minOrderUsd: 10 }); // $4 notional
    expect(r.trades.length).toBe(0);
    expect(r.finalEquity).toBeCloseTo(2, 10);
  });

  test("no positions means equity is untouched", () => {
    const candles = mk([100, 105, 110, 115, 120, 125]);
    const r = simulatePerp(candles, cfg, [0, 0, 0, 0], noFunding, base);
    expect(r.finalEquity).toBe(50);
    expect(r.totalFees).toBe(0);
  });
});
