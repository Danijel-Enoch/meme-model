import { expect, test, describe } from "bun:test";
import {
  aggregate, lastCompletedIndex, buildSignals, combineSignals,
  walkForwardConfluence, DEFAULT_STACK, type TimeframeSignal,
} from "./confluence";
import { generateSynthetic } from "./data";
import type { Candle } from "./features";

const bar = (t: number, o: number, h: number, l: number, c: number, v = 10): Candle =>
  ({ time: t, open: o, high: h, low: l, close: c, volume: v });

describe("aggregate", () => {
  test("rolls OHLCV correctly", () => {
    const src = [
      bar(0, 10, 12, 9, 11, 100),
      bar(60, 11, 15, 10, 14, 200),
      bar(120, 14, 14, 8, 9, 300),
    ];
    const out = aggregate(src, 3);
    expect(out.length).toBe(1);
    expect(out[0].time).toBe(0);
    expect(out[0].open).toBe(10);   // first open
    expect(out[0].high).toBe(15);   // max high
    expect(out[0].low).toBe(8);     // min low
    expect(out[0].close).toBe(9);   // last close
    expect(out[0].volume).toBe(600); // summed
  });

  test("drops an incomplete trailing group rather than inventing a short bar", () => {
    const src = Array.from({ length: 7 }, (_, i) => bar(i * 60, 1, 1, 1, 1));
    expect(aggregate(src, 3).length).toBe(2); // 7 -> 2 complete groups, 1 dropped
  });

  test("factor 1 is the identity", () => {
    const src = [bar(0, 1, 2, 0.5, 1.5)];
    expect(aggregate(src, 1)).toBe(src);
  });
});

describe("completed-bar alignment (the lookahead guard)", () => {
  test("an hourly bar is not available until its final 5m bar closes", () => {
    // factor 12: bars 0..11 make hourly bar 0, which closes at base bar 11.
    for (let i = 0; i <= 10; i++) expect(lastCompletedIndex(i, 12)).toBe(-1);
    expect(lastCompletedIndex(11, 12)).toBe(0);
    expect(lastCompletedIndex(12, 12)).toBe(0); // mid-hour: still only bar 0
    expect(lastCompletedIndex(22, 12)).toBe(0);
    expect(lastCompletedIndex(23, 12)).toBe(1);
  });

  test("factor 1 always has the current bar available", () => {
    for (let i = 0; i < 5; i++) expect(lastCompletedIndex(i, 1)).toBe(i);
  });

  test("the index never points at a bar that has not closed", () => {
    for (const factor of [2, 3, 12, 48]) {
      for (let i = 0; i < 200; i++) {
        const j = lastCompletedIndex(i, factor);
        if (j >= 0) {
          // The last base bar of HTF bar j must be at or before i.
          expect(j * factor + factor - 1).toBeLessThanOrEqual(i);
        }
      }
    }
  });
});

describe("no lookahead end to end", () => {
  test("corrupting future candles leaves every earlier signal unchanged", () => {
    const { candles } = generateSynthetic({ bars: 3000, seed: 11 });
    const opts = { featureConfig: { window: 5 }, states: 3, seed: 1, restarts: 1 };
    const clean = buildSignals(candles, DEFAULT_STACK, 1200, 2400, opts);

    // Wreck everything from base bar 2000 on, including mid-hour bars.
    const tampered = candles.map((c, i) =>
      i >= 2000 ? { ...c, open: c.open * 7, high: c.high * 7, low: c.low * 7, close: c.close * 7, volume: c.volume * 13 } : c);
    const dirty = buildSignals(tampered, DEFAULT_STACK, 1200, 2400, opts);

    for (let s = 0; s < clean.length; s++) {
      for (let i = 0; i < 2000; i++) {
        expect(dirty[s].ready[i]).toBe(clean[s].ready[i]);
        if (clean[s].ready[i]) {
          expect(dirty[s].expectedReturn[i]).toBeCloseTo(clean[s].expectedReturn[i], 12);
          expect(dirty[s].bullProb[i]).toBeCloseTo(clean[s].bullProb[i], 12);
        }
      }
    }
  });

  test("walk-forward positions before the tampered region do not move", () => {
    const { candles } = generateSynthetic({ bars: 3000, seed: 12 });
    const opts = { featureConfig: { window: 5 }, states: 3, seed: 1, restarts: 1, trainSize: 1000, testSize: 400 };
    const a = walkForwardConfluence(candles, DEFAULT_STACK, opts);
    const tampered = candles.map((c, i) => (i >= 2400 ? { ...c, close: c.close * 5 } : c));
    const b = walkForwardConfluence(tampered, DEFAULT_STACK, opts);
    for (let i = 0; i < 2000; i++) expect(b.positions[i]).toBe(a.positions[i]);
  });
});

describe("signal combination", () => {
  const mk = (label: string, role: any, exp: number[], bull = 1): TimeframeSignal => ({
    label, role,
    expectedReturn: Float64Array.from(exp),
    bullProb: Float64Array.from(exp.map(() => bull)),
    bearProb: Float64Array.from(exp.map(() => 0)),
    ready: Uint8Array.from(exp.map(() => 1)),
  });

  test("all three must agree to open a long", () => {
    const n = 3;
    // bar 0: all positive. bar 1: setup disagrees. bar 2: all positive again.
    const sig = [
      mk("1h", "bias", [0.01, 0.01, 0.01]),
      mk("15m", "setup", [0.01, -0.01, 0.01]),
      mk("5m", "trigger", [0.01, 0.01, 0.01]),
    ];
    const { positions } = combineSignals(sig, n, { exitOnBiasFlip: true });
    expect(positions[0]).toBe(1);
    expect(positions[1]).toBe(1); // already long, bias still fine -> hold
    expect(positions[2]).toBe(1);
  });

  test("a disagreeing setup blocks a fresh entry", () => {
    const sig = [
      mk("1h", "bias", [0.01]),
      mk("15m", "setup", [-0.01]),
      mk("5m", "trigger", [0.01]),
    ];
    expect(combineSignals(sig, 1, {}).positions[0]).toBe(0);
  });

  test("a bias flip closes the position even though three votes opened it", () => {
    const sig = [
      mk("1h", "bias", [0.01, -0.01]),
      mk("15m", "setup", [0.01, 0.01]),
      mk("5m", "trigger", [0.01, 0.01]),
    ];
    const { positions } = combineSignals(sig, 2, { exitOnBiasFlip: true });
    expect(positions[0]).toBe(1);
    expect(positions[1]).toBe(0);
  });

  test("nothing trades until every timeframe is ready", () => {
    const sig = [
      mk("1h", "bias", [0.01, 0.01]),
      mk("15m", "setup", [0.01, 0.01]),
      mk("5m", "trigger", [0.01, 0.01]),
    ];
    sig[0].ready[0] = 0; // bias not warmed up on the first bar
    const { positions } = combineSignals(sig, 2, {});
    expect(positions[0]).toBe(0);
    expect(positions[1]).toBe(1);
  });

  test("shorts stay off unless explicitly enabled", () => {
    const sig = [
      mk("1h", "bias", [-0.01], 0),
      mk("15m", "setup", [-0.01], 0),
      mk("5m", "trigger", [-0.01], 0),
    ];
    sig.forEach((s) => (s.bearProb = Float64Array.from([1])));
    expect(combineSignals(sig, 1, { allowShort: false }).positions[0]).toBe(0);
    expect(combineSignals(sig, 1, { allowShort: true }).positions[0]).toBe(-1);
  });

  test("agreement score runs from -3 to +3", () => {
    const sig = [
      mk("1h", "bias", [0.01, -0.01]),
      mk("15m", "setup", [0.01, -0.01]),
      mk("5m", "trigger", [0.01, -0.01]),
    ];
    const { agreement } = combineSignals(sig, 2, {});
    expect(agreement[0]).toBe(3);
    expect(agreement[1]).toBe(-3);
  });
});
