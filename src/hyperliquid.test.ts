import { expect, test, describe } from "bun:test";
import { intervalSeconds, hlBarsPerYear, HL_TAKER_BPS, HL_MAKER_BPS } from "./hyperliquid";

describe("hyperliquid intervals", () => {
  test("maps every supported interval to its length", () => {
    expect(intervalSeconds("1m")).toBe(60);
    expect(intervalSeconds("5m")).toBe(300);
    expect(intervalSeconds("1h")).toBe(3600);
    expect(intervalSeconds("4h")).toBe(14400);
    expect(intervalSeconds("1d")).toBe(86400);
  });

  test("rejects intervals the API does not serve", () => {
    expect(() => intervalSeconds("7m")).toThrow();
    expect(() => intervalSeconds("90m")).toThrow();
  });

  test("bars per year is internally consistent", () => {
    expect(hlBarsPerYear("1d")).toBeCloseTo(365.25, 5);
    expect(hlBarsPerYear("1h")).toBeCloseTo(365.25 * 24, 5);
    expect(hlBarsPerYear("5m")).toBeCloseTo(hlBarsPerYear("1h") * 12, 5);
  });

  test("fees are an order of magnitude under a DEX round trip", () => {
    // A meme coin round trip runs ~60bps once slippage is counted.
    expect(2 * HL_TAKER_BPS).toBeLessThan(15);
    expect(HL_MAKER_BPS).toBeLessThan(HL_TAKER_BPS);
  });
});
