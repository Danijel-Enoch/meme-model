import { expect, test, describe } from "bun:test";
import { closedCandles, nextCloseMs, pollOnce } from "./live";
import { newPaperState } from "../paper";
import { fit } from "../hmm";
import { buildFeatures, fitScaler, applyScaler } from "../features";
import type { Candle } from "../features";
import type { RuntimeModel } from "./jobs";

const MIN = 60;
const IV = 30 * MIN; // 30m bars

/** A deterministic trending series, long enough to fit a small model on. */
function series(n: number, startTime = 1_700_000_000): Candle[] {
  const out: Candle[] = [];
  let p = 100;
  for (let i = 0; i < n; i++) {
    p *= 1 + (i % 60 < 30 ? 0.0012 : -0.0009) + 0.0004 * Math.sin(i / 3);
    out.push({
      time: startTime + i * IV,
      open: p * 0.999, high: p * 1.002, low: p * 0.998, close: p, volume: 1000,
    });
  }
  return out;
}

function modelFor(candles: Candle[]): RuntimeModel {
  const featureConfig = { window: 5, useVolatility: true, useVolume: true };
  const fs = buildFeatures(candles, featureConfig);
  const scaler = fitScaler(fs.X, fs.T, fs.D);
  const Z = applyScaler(fs.X, fs.T, fs.D, scaler);
  const res = fit(Z, fs.T, fs.D, { states: 2, restarts: 2, seed: 7 });
  return {
    coin: "TEST", timeframe: "30m", modelType: "hmm",
    params: res.params, scaler, names: fs.names, window: 5,
  };
}

describe("closed bars", () => {
  const candles = series(10);
  const lastStart = candles[9].time;

  test("the bar still forming is never returned", () => {
    // One second before the last bar would close: it is still open.
    const now = (lastStart + IV - 1) * 1000;
    const closed = closedCandles(candles, IV, now);
    expect(closed.length).toBe(9);
    expect(closed[closed.length - 1].time).toBe(candles[8].time);
  });

  test("it becomes available the second its interval elapses", () => {
    const now = (lastStart + IV) * 1000;
    expect(closedCandles(candles, IV, now).length).toBe(10);
  });

  test("an empty series does not throw", () => {
    expect(closedCandles([], IV, Date.now())).toEqual([]);
    expect(nextCloseMs(undefined, IV)).toBeNull();
  });

  test("the countdown points at the next close, never the past", () => {
    const now = (lastStart + 5) * 1000;
    const t = nextCloseMs(candles[9], IV, now)!;
    expect(t).toBeGreaterThan(now);
    expect(t).toBe((lastStart + 2 * IV) * 1000);
  });
});

describe("polling", () => {
  const candles = series(400);
  const model = modelFor(candles);
  const strategy = { costBps: 4.5, durationAware: true };

  /** Serve the first `n` candles, as the API would at that moment in time. */
  const feed = (n: number) => async () => candles.slice(0, n);

  test("the forming bar is fetched but never traded", async () => {
    const state = newPaperState({ startingEquity: 1000, maxCoins: 1 });
    // 300 candles served, the 300th still forming.
    const now = (candles[299].time + 10) * 1000;
    const res = await pollOnce(state, {
      coin: "TEST", timeframe: "30m", model, strategy, now, fetch: feed(300),
    });
    expect(res.lastClosed!.time).toBe(candles[298].time);
    // The mark comes off the forming bar; the account's last applied bar does not.
    expect(res.mark).toBe(candles[299].close);
    expect(state.lastBar["TEST"]).toBe(candles[298].time);
  });

  test("a second poll inside the same bar changes nothing", async () => {
    const state = newPaperState({ startingEquity: 1000, maxCoins: 1 });
    const now = (candles[299].time + 10) * 1000;
    const opts = { coin: "TEST", timeframe: "30m", model, strategy, now, fetch: feed(300) };
    await pollOnce(state, opts);
    const fills = state.fills.length;
    const applied = await pollOnce(state, opts);
    expect(applied.applied).toBe(0);
    expect(state.fills.length).toBe(fills);
  });

  test("bars missed while the terminal was asleep are applied in order", async () => {
    const state = newPaperState({ startingEquity: 1000, maxCoins: 1 });
    await pollOnce(state, {
      coin: "TEST", timeframe: "30m", model, strategy,
      now: (candles[299].time + 10) * 1000, fetch: feed(300),
    });
    const seen = state.lastBar["TEST"];
    // Five bars later, in one poll.
    const res = await pollOnce(state, {
      coin: "TEST", timeframe: "30m", model, strategy,
      now: (candles[304].time + 10) * 1000, fetch: feed(305),
    });
    expect(res.applied).toBe(5);
    expect(state.lastBar["TEST"]).toBe(candles[303].time);
    expect(state.lastBar["TEST"]).toBeGreaterThan(seen);
  });

  test("a poll with nothing new still refreshes the mark", async () => {
    const state = newPaperState({ startingEquity: 1000, maxCoins: 1 });
    const base = { coin: "TEST", timeframe: "30m", model, strategy } as const;
    await pollOnce(state, { ...base, now: (candles[299].time + 10) * 1000, fetch: feed(300) });
    const res = await pollOnce(state, {
      ...base, now: (candles[299].time + 20) * 1000,
      fetch: async () => {
        // Same bars, but the forming bar has moved.
        const c = candles.slice(0, 300).map((x) => ({ ...x }));
        c[299] = { ...c[299], close: c[299].close * 1.05 };
        return c;
      },
    });
    expect(res.applied).toBe(0);
    expect(res.mark).toBeCloseTo(candles[299].close * 1.05, 8);
  });
});
