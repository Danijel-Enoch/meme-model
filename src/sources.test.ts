import { expect, test, describe } from "bun:test";
import { parseTimeframe, barsPerYear, resolveNetwork, fillGaps } from "./sources";
import type { Candle } from "./features";

const bar = (time: number, close: number, volume = 100): Candle =>
  ({ time, open: close, high: close, low: close, close, volume });

describe("timeframes", () => {
  test("maps labels to GeckoTerminal unit/aggregate pairs", () => {
    expect(parseTimeframe("5m")).toMatchObject({ unit: "minute", aggregate: 5, seconds: 300 });
    expect(parseTimeframe("4h")).toMatchObject({ unit: "hour", aggregate: 4, seconds: 14400 });
    expect(parseTimeframe("1d")).toMatchObject({ unit: "day", aggregate: 1, seconds: 86400 });
    expect(parseTimeframe("1H")).toMatchObject({ unit: "hour", aggregate: 1 });
  });

  test("rejects timeframes the API does not support", () => {
    expect(() => parseTimeframe("3m")).toThrow();
    expect(() => parseTimeframe("2h")).toThrow();
  });

  test("bars per year is consistent across timeframes", () => {
    expect(barsPerYear(parseTimeframe("1d"))).toBeCloseTo(365.25, 5);
    expect(barsPerYear(parseTimeframe("1h"))).toBeCloseTo(365.25 * 24, 5);
    // A year of 5m bars is 12x the hourly count.
    expect(barsPerYear(parseTimeframe("5m"))).toBeCloseTo(barsPerYear(parseTimeframe("1h")) * 12, 5);
  });
});

describe("network aliases", () => {
  test("resolves common names to GeckoTerminal ids", () => {
    expect(resolveNetwork("sol")).toBe("solana");
    expect(resolveNetwork("SOLANA")).toBe("solana");
    expect(resolveNetwork("ethereum")).toBe("eth");
    expect(resolveNetwork("polygon")).toBe("polygon_pos");
    expect(resolveNetwork("bnb")).toBe("bsc");
  });

  test("passes unknown ids through unchanged", () => {
    expect(resolveNetwork("some_new_chain")).toBe("some_new_chain");
  });
});

describe("gap filling", () => {
  const step = 300;

  test("inserts flat zero-volume bars for short gaps", () => {
    const input = [bar(0, 10), bar(300, 11), bar(1500, 12)]; // 3 missing bars
    const { candles, filled, skipped } = fillGaps(input, step, 12);
    expect(filled).toBe(3);
    expect(skipped).toBe(0);
    expect(candles.length).toBe(6);
    expect(candles.map((c) => c.time)).toEqual([0, 300, 600, 900, 1200, 1500]);
    // Filled bars carry the prior close and no volume: nobody traded.
    for (const c of candles.slice(2, 5)) {
      expect(c.close).toBe(11);
      expect(c.open).toBe(11);
      expect(c.volume).toBe(0);
    }
  });

  test("leaves long gaps alone rather than fabricating a quiet regime", () => {
    const input = [bar(0, 10), bar(300 * 100, 11)]; // 99 missing bars
    const { candles, filled, skipped } = fillGaps(input, step, 12);
    expect(filled).toBe(0);
    expect(skipped).toBe(99);
    expect(candles.length).toBe(2);
  });

  test("produces a constant time step when every gap is fillable", () => {
    const input = [bar(0, 10), bar(600, 11), bar(1200, 12), bar(1800, 13)];
    const { candles } = fillGaps(input, step, 12);
    for (let i = 1; i < candles.length; i++) {
      expect(candles[i].time - candles[i - 1].time).toBe(step);
    }
  });

  test("filled bars contribute exactly zero return", () => {
    const input = [bar(0, 10), bar(1500, 10)];
    const { candles } = fillGaps(input, step, 12);
    for (let i = 1; i < candles.length; i++) {
      expect(Math.log(candles[i].close / candles[i - 1].close)).toBeCloseTo(0, 12);
    }
  });

  test("is a no-op on a series with no gaps", () => {
    const input = [bar(0, 10), bar(300, 11), bar(600, 12)];
    const { candles, filled } = fillGaps(input, step, 12);
    expect(filled).toBe(0);
    expect(candles).toEqual(input);
  });

  test("handles an empty series", () => {
    expect(fillGaps([], step).candles).toEqual([]);
  });
});
