import { expect, test, describe } from "bun:test";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { rm } from "node:fs/promises";
import {
  newPaperState, onBar, markToMarket, replay, savePaper, loadPaper, paperStats,
  type PaperState,
} from "./paper";
import { walkForward, signalNow, type StrategyConfig } from "./backtest";
import { buildFeatures, type Candle, type FeatureConfig } from "./features";
import { makeRng, randn } from "./hmm";

/** Flat-price candles: any equity change is then unambiguously a fee. */
function flat(n: number, price = 100): Candle[] {
  return Array.from({ length: n }, (_, i) => ({
    time: 1_700_000_000 + i * 3600,
    open: price, high: price, low: price, close: price, volume: 1,
  }));
}

/** Candles with explicit closes and, optionally, wicks, so intrabar paths bite. */
function mk(closes: number[], lows?: number[], highs?: number[]): Candle[] {
  return closes.map((c, i) => ({
    time: 1_700_000_000 + i * 3600,
    open: c,
    high: highs ? highs[i] : c,
    low: lows ? lows[i] : c,
    close: c,
    volume: 1,
  }));
}

/**
 * A deterministic three-regime series — bleed, chop, pump. The models need
 * something with structure in it or the invariant test passes by taking no
 * trades at all, which would prove nothing.
 */
function regimeCandles(n: number, seed: number): Candle[] {
  const rng = makeRng(seed);
  const means = [-0.004, 0.0, 0.006];
  const sds = [0.010, 0.004, 0.014];
  let s = 1;
  let price = 100;
  const out: Candle[] = [];
  for (let i = 0; i < n; i++) {
    if (rng() > 0.9) s = Math.min(2, Math.floor(rng() * 3));
    const r = means[s] + sds[s] * randn(rng);
    const open = price;
    price *= Math.exp(r);
    out.push({
      time: 1_700_000_000 + i * 3600,
      open,
      high: Math.max(open, price) * 1.001,
      low: Math.min(open, price) * 0.999,
      close: price,
      volume: 100 + 50 * rng(),
    });
  }
  return out;
}

const relErr = (a: number, b: number) => Math.abs(a - b) / Math.max(Math.abs(b), 1e-12);

describe("the walk-forward invariant", () => {
  /**
   * THE test. If paper trading does not reproduce the backtest bar for bar,
   * paper results say nothing about the model and every other number in the
   * engine is decoration.
   */
  function checkReplayMatches(
    candles: Candle[], featureConfig: FeatureConfig, strategy: StrategyConfig,
    wf: { trainSize: number; testSize: number; states: number; seed: number; restarts: number;
          modelType: "hmm" | "hsmm"; maxDuration?: number },
  ) {
    const res = walkForward(candles, featureConfig, strategy, wf);
    expect(res.refits).toBe(1); // one block, so one frozen model to replay with
    const model = res.lastModel!;
    const names = buildFeatures(candles, featureConfig).names;

    const state = newPaperState({ startingEquity: 1, leverage: 1, costBps: strategy.costBps, maxCoins: 1 });
    replay(state, "TEST", candles,
      { ...model, modelType: wf.modelType, names, window: featureConfig.window! },
      strategy, wf.trainSize);

    expect(state.equityCurve.length).toBe(res.equity.length);
    let worst = 0;
    for (let i = 0; i < res.equity.length; i++) {
      worst = Math.max(worst, relErr(state.equityCurve[i].equity, res.equity[i]));
      worst = Math.max(worst, relErr(state.equityCurve[i].buyHold, res.buyHoldEquity[i]));
    }
    return { res, state, worst };
  }

  test("replay reproduces walkForward's out-of-sample equity curve", () => {
    const candles = regimeCandles(600, 7);
    const fc: FeatureConfig = { window: 5, useVolatility: false, useVolume: false };
    const strategy: StrategyConfig = { costBps: 4.5, entryBps: 2, allowShort: true };
    const { res, state, worst } = checkReplayMatches(candles, fc, strategy, {
      trainSize: 400, testSize: 190, states: 3, seed: 42, restarts: 2, modelType: "hmm",
    });

    // Not vacuous: the strategy has to have actually traded.
    expect(res.metrics.trades).toBeGreaterThan(3);
    expect(worst).toBeLessThan(1e-9);
    expect(relErr(state.equity, 1 + res.metrics.totalReturn)).toBeLessThan(1e-3);
  });

  test("the same holds for the semi-Markov model", () => {
    const candles = regimeCandles(300, 11);
    const fc: FeatureConfig = { window: 5, useVolatility: false, useVolume: false };
    const strategy: StrategyConfig = { costBps: 4.5, entryBps: 1, allowShort: true };
    const { res, worst } = checkReplayMatches(candles, fc, strategy, {
      trainSize: 200, testSize: 93, states: 2, seed: 3, restarts: 1,
      modelType: "hsmm", maxDuration: 10,
    });
    expect(res.metrics.trades).toBeGreaterThan(0);
    expect(worst).toBeLessThan(1e-9);
  });

  test("signalNow reproduces the position walkForward took on the last bar", () => {
    const candles = regimeCandles(600, 7);
    const fc: FeatureConfig = { window: 5, useVolatility: false, useVolume: false };
    const strategy: StrategyConfig = { costBps: 4.5, entryBps: 2, allowShort: true };
    const res = walkForward(candles, fc, strategy,
      { trainSize: 400, testSize: 190, states: 3, seed: 42, restarts: 2, modelType: "hmm" });
    const fs = buildFeatures(candles, fc);
    const last = fs.T - 1;

    const sig = signalNow(res.lastModel!.params, "hmm", res.lastModel!.scaler, fs.names,
      candles, fc, strategy, res.positions[last - 1]);
    expect(sig.target).toBe(res.positions[last]);
    expect(relErr(sig.expectedReturn, res.expectedReturns[last])).toBeLessThan(1e-12);
    expect(sig.stateProbs.length).toBe(3);
    expect(sig.state).toBeGreaterThanOrEqual(0);
    expect(sig.state).toBeLessThan(3);
    expect(sig.expectedHoldBars).toBeGreaterThan(0);
  });

  test("signalNow never refits the scaler — a tampered future cannot move it", () => {
    // Same guard walkForward carries: standardizing on the live series would
    // leak the future through the mean and standard deviation.
    const candles = regimeCandles(300, 5);
    const fc: FeatureConfig = { window: 5, useVolatility: true, useVolume: true };
    const res = walkForward(candles, fc, { costBps: 4.5, entryBps: 2 },
      { trainSize: 200, testSize: 93, states: 2, seed: 1, restarts: 1 });
    const names = buildFeatures(candles, fc).names;
    const prefix = candles.slice(0, 260);

    const a = signalNow(res.lastModel!.params, "hmm", res.lastModel!.scaler, names, prefix, fc, {}, 0);
    const longer = candles.map((c, i) => (i >= 260 ? { ...c, close: c.close * 4, high: c.high * 4 } : c));
    const b = signalNow(res.lastModel!.params, "hmm", res.lastModel!.scaler, names,
      longer.slice(0, 260), fc, {}, 0);
    expect(b.expectedReturn).toBe(a.expectedReturn);
    expect(b.target).toBe(a.target);
  });
});

describe("replay warm-up", () => {
  test("replaying from the start does not fall off the feature lookback", () => {
    const candles = regimeCandles(120, 13);
    const fc: FeatureConfig = { window: 5, useVolatility: false, useVolume: false };
    const strategy: StrategyConfig = { costBps: 4.5, entryBps: 2, allowShort: true };
    const res = walkForward(candles, fc, strategy,
      { trainSize: 60, testSize: 50, states: 2, seed: 1, restarts: 1 });
    const names = buildFeatures(candles, fc).names;
    const s = newPaperState({ startingEquity: 100, costBps: 4.5 });
    expect(() => replay(s, "T", candles, { ...res.lastModel!, modelType: "hmm", names, window: 5 },
      strategy, 0)).not.toThrow();
    expect(s.equityCurve.length).toBe(buildFeatures(candles, fc).T - 1);
  });
});

describe("fees", () => {
  test("a round trip at 4.5bps on 1x costs 9bps of the notional", () => {
    const c = flat(4);
    const s = newPaperState({ startingEquity: 1000, leverage: 1, costBps: 4.5 });
    onBar(s, "X", c[0], 1);
    onBar(s, "X", c[1], 0);
    // Leg 1: $1000 notional x 4.5bps = $0.45. Leg 2 sizes off the equity left
    // after leg 1: $999.55 x 4.5bps = $0.4497975.
    expect(s.fills.length).toBe(2);
    expect(s.fills[0].feeUsd).toBeCloseTo(0.45, 12);
    expect(s.fills[1].feeUsd).toBeCloseTo(0.4497975, 12);
    expect(s.feesUsd).toBeCloseTo(0.8997975, 12);
    expect(s.equity).toBeCloseTo(999.1002025, 10);
    // 9bps of the traded notional, to within the second-order term.
    expect(s.feesUsd / 1000).toBeCloseTo(0.0009, 6);
  });

  test("at 5x the same round trip costs 45bps of equity", () => {
    const c = flat(4);
    const s = newPaperState({ startingEquity: 1000, leverage: 5, costBps: 4.5 });
    onBar(s, "X", c[0], 1);
    onBar(s, "X", c[1], 0);
    // $5000 notional x 4.5bps = $2.25, then $4988.75 x 4.5bps = $2.2449375.
    expect(s.fills[0].sizeUsd).toBeCloseTo(5000, 10);
    expect(s.feesUsd).toBeCloseTo(4.4949375, 10);
    expect(s.equity).toBeCloseTo(995.5050625, 10);
    expect(s.feesUsd / 1000).toBeCloseTo(0.0045, 4);
  });

  test("holding costs nothing — only changes in position pay", () => {
    const c = flat(6);
    const s = newPaperState({ startingEquity: 1000, leverage: 1, costBps: 4.5 });
    for (let i = 0; i < 5; i++) onBar(s, "X", c[i], 1);
    expect(s.fills.length).toBe(1);
    expect(s.feesUsd).toBeCloseTo(0.45, 12);
  });
});

describe("flips", () => {
  test("a flip trades two units of notional and pays for both", () => {
    const c = flat(4);
    const s = newPaperState({ startingEquity: 1000, leverage: 1, costBps: 4.5 });
    onBar(s, "X", c[0], 1);
    const equityAtFlip = s.equity; // 999.55
    onBar(s, "X", c[1], -1);

    expect(s.fills[1].reason).toBe("flip");
    expect(s.fills[1].side).toBe("sell");
    expect(s.fills[1].sizeUsd).toBeCloseTo(2 * equityAtFlip, 10);
    expect(s.fills[1].feeUsd).toBeCloseTo(2 * equityAtFlip * 0.00045, 12);
    // One unit closes the long, one opens the short.
    expect(s.positions.X.position).toBe(-1);
    expect(s.positions.X.notionalUsd).toBeCloseTo(equityAtFlip, 10);
  });

  test("a note annotates the reason without destroying the vocabulary", () => {
    const c = flat(3);
    const s = newPaperState({ startingEquity: 1000, costBps: 0 });
    onBar(s, "X", c[0], 1, "bias flipped");
    onBar(s, "X", c[1], 0, "manual");
    expect(s.fills[0].reason).toBe("entry bias flipped");
    expect(s.fills[1].reason).toBe("exit manual");
    // Still readable as a completed round trip.
    expect(paperStats(s).trades).toBe(1);
  });

  test("a flip earns the next bar's return in the NEW direction", () => {
    const c = mk([100, 100, 90, 90]);
    const s = newPaperState({ startingEquity: 1000, leverage: 1, costBps: 0 });
    onBar(s, "X", c[0], 1);
    onBar(s, "X", c[1], -1);   // still flat-priced, so the long earned nothing
    onBar(s, "X", c[2], 0);    // -10% bar, held short
    expect(s.equity).toBeCloseTo(1100, 8);
  });
});

describe("idempotency", () => {
  test("re-applying the same bar timestamp changes nothing", () => {
    const c = mk([100, 110, 121]);
    const s = newPaperState({ startingEquity: 1000, leverage: 1, costBps: 4.5 }, "fixed");
    onBar(s, "X", c[0], 1);
    onBar(s, "X", c[1], 1);
    const snapshot = JSON.stringify({ ...s, updatedAt: 0 });

    onBar(s, "X", c[1], 1);          // the exact bar again
    onBar(s, "X", c[1], -1);         // and with a different target
    onBar(s, "X", c[0], 0);          // and a stale one
    expect(JSON.stringify({ ...s, updatedAt: 0 })).toBe(snapshot);

    // The next real bar still applies.
    onBar(s, "X", c[2], 0);
    expect(s.equityCurve.length).toBe(3);
  });

  test("a replayed span is unchanged by replaying it again", () => {
    const candles = regimeCandles(300, 2);
    const fc: FeatureConfig = { window: 5, useVolatility: false, useVolume: false };
    const strategy: StrategyConfig = { costBps: 4.5, entryBps: 2, allowShort: true };
    const res = walkForward(candles, fc, strategy,
      { trainSize: 200, testSize: 93, states: 2, seed: 9, restarts: 1 });
    const names = buildFeatures(candles, fc).names;
    const model = { ...res.lastModel!, modelType: "hmm" as const, names, window: 5 };

    const s = newPaperState({ startingEquity: 1000, costBps: 4.5 }, "fixed");
    replay(s, "T", candles, model, strategy, 200);
    const once = JSON.stringify({ ...s, updatedAt: 0 });
    replay(s, "T", candles, model, strategy, 200);
    expect(JSON.stringify({ ...s, updatedAt: 0 })).toBe(once);
  });
});

describe("no lookahead", () => {
  /**
   * The bar that produces a signal must not pay it. Prices double on bar 2, so
   * a position opened at bar 2's close earning bar 2's move would be free money
   * and completely obvious in the equity.
   */
  const jump = mk([100, 100, 200, 200]);

  test("a position taken BEFORE the jump earns it", () => {
    const s = newPaperState({ startingEquity: 1000, leverage: 1, costBps: 0 });
    onBar(s, "X", jump[0], 0);
    onBar(s, "X", jump[1], 1);   // decided at close 100
    onBar(s, "X", jump[2], 0);   // realizes the +100% bar
    expect(s.equity).toBeCloseTo(2000, 9);
  });

  test("a position taken ON the jump does not", () => {
    const s = newPaperState({ startingEquity: 1000, leverage: 1, costBps: 0 });
    onBar(s, "X", jump[0], 0);
    onBar(s, "X", jump[1], 0);
    onBar(s, "X", jump[2], 1);   // decided at the close that already jumped
    onBar(s, "X", jump[3], 0);   // flat bar: nothing earned
    expect(s.equity).toBeCloseTo(1000, 9);
    expect(s.realizedUsd).toBeCloseTo(0, 9);
  });

  test("the equity curve point for a bar is the mark of what was held through it", () => {
    const s = newPaperState({ startingEquity: 1000, leverage: 1, costBps: 0 });
    onBar(s, "X", jump[1], 1);
    onBar(s, "X", jump[2], 1);
    expect(s.equityCurve[0].equity).toBeCloseTo(1000, 9);
    expect(s.equityCurve[1].equity).toBeCloseTo(2000, 9);
  });
});

describe("liquidation", () => {
  // 5x on an asset the venue lists at 10x: maintenance is 1/(2*10) = 5% of
  // notional, so $1000 of equity behind $5000 of notional dies below $250.
  const cfg = { startingEquity: 1000, leverage: 5, costBps: 0, maxLeverage: 10 };

  test("an intrabar wick liquidates even though the close recovers", () => {
    const c = mk([100, 100, 100], [100, 80, 100]);  // low 80 on bar 1, close 100
    const s = newPaperState(cfg);
    onBar(s, "X", c[0], 1);
    onBar(s, "X", c[1], 1);
    expect(s.liquidated).toBe(true);
    expect(s.equity).toBe(0);
    const liq = s.fills[s.fills.length - 1];
    expect(liq.reason).toBe("liquidation");
    expect(liq.price).toBe(80);   // filled at the extreme, not the close
  });

  test("the same closes survive without the wick", () => {
    const c = mk([100, 100, 100], [100, 95, 100]);  // -5% worst tick: $750 left
    const s = newPaperState(cfg);
    onBar(s, "X", c[0], 1);
    onBar(s, "X", c[1], 1);
    expect(s.liquidated).toBe(false);
    expect(s.equity).toBeCloseTo(1000, 9);
  });

  test("shorts are liquidated by the high, not the low", () => {
    const c = mk([100, 100, 100], [100, 100, 100], [100, 120, 100]);
    const s = newPaperState(cfg);
    onBar(s, "X", c[0], -1);
    onBar(s, "X", c[1], -1);
    expect(s.liquidated).toBe(true);
    expect(s.fills[s.fills.length - 1].price).toBe(120);
  });

  test("trading stops after a liquidation", () => {
    const c = mk([100, 100, 100, 100], [100, 80, 100, 100]);
    const s = newPaperState(cfg);
    onBar(s, "X", c[0], 1);
    onBar(s, "X", c[1], 1);
    const fills = s.fills.length;
    const curve = s.equityCurve.length;

    onBar(s, "X", c[2], 1);
    onBar(s, "X", c[3], -1);
    markToMarket(s, { X: 500 }, c[3].time);
    expect(s.fills.length).toBe(fills);
    expect(s.equityCurve.length).toBe(curve);
    expect(s.equity).toBe(0);
    expect(Object.keys(s.positions).length).toBe(0);
  });
});

describe("funding", () => {
  test("longs pay on open notional, per hour of bar time", () => {
    const c = flat(3);   // hourly bars
    const s = newPaperState({ startingEquity: 1000, leverage: 2, costBps: 0, fundingPerHour: 0.001 });
    onBar(s, "X", c[0], 1);
    onBar(s, "X", c[1], 1);
    // $2000 notional x 0.1% x 1 hour.
    expect(s.fundingUsd).toBeCloseTo(2, 9);
    expect(s.equity).toBeCloseTo(998, 9);
  });

  test("shorts receive it", () => {
    const c = flat(3);
    const s = newPaperState({ startingEquity: 1000, leverage: 2, costBps: 0, fundingPerHour: 0.001 });
    onBar(s, "X", c[0], -1);
    onBar(s, "X", c[1], -1);
    expect(s.fundingUsd).toBeCloseTo(-2, 9);
    expect(s.equity).toBeCloseTo(1002, 9);
  });
});

describe("marking between bars", () => {
  test("a tick moves equity but does not trade or extend the curve", () => {
    const c = mk([100, 100]);
    const s = newPaperState({ startingEquity: 1000, leverage: 2, costBps: 0 });
    onBar(s, "X", c[0], 1);
    const curve = s.equityCurve.length;

    markToMarket(s, { X: 110 }, c[0].time + 60);
    expect(s.positions.X.unrealizedUsd).toBeCloseTo(200, 9);  // $2000 notional, +10%
    expect(s.equity).toBeCloseTo(1200, 9);
    expect(s.cash).toBeCloseTo(1000, 9);
    expect(s.equityCurve.length).toBe(curve);
    expect(s.fills.length).toBe(1);
  });

  test("the next bar realizes the move once, not twice", () => {
    const c = mk([100, 110]);
    const s = newPaperState({ startingEquity: 1000, leverage: 2, costBps: 0 });
    onBar(s, "X", c[0], 1);
    markToMarket(s, { X: 110 }, c[0].time + 60);
    onBar(s, "X", c[1], 1);
    expect(s.equity).toBeCloseTo(1200, 9);
    expect(s.cash).toBeCloseTo(1200, 9);
    expect(s.positions.X.unrealizedUsd).toBe(0);
  });
});

describe("persistence", () => {
  test("save/load round-trips exactly, floats included", async () => {
    const candles = regimeCandles(300, 4);
    const fc: FeatureConfig = { window: 5, useVolatility: false, useVolume: false };
    const strategy: StrategyConfig = { costBps: 4.5, entryBps: 2, allowShort: true };
    const res = walkForward(candles, fc, strategy,
      { trainSize: 200, testSize: 93, states: 2, seed: 9, restarts: 1 });
    const names = buildFeatures(candles, fc).names;

    const s = newPaperState({ startingEquity: 1000, costBps: 4.5, fundingPerHour: 1e-7 }, "round-trip");
    replay(s, "T", candles, { ...res.lastModel!, modelType: "hmm", names, window: 5 }, strategy, 200);

    const path = join(tmpdir(), `paper-${Date.now().toString(36)}.json`);
    try {
      await savePaper(path, s);
      const back: PaperState = await loadPaper(path);
      expect(JSON.stringify(back)).toBe(JSON.stringify(s));
      expect(back.equity).toBe(s.equity);
      expect(back.feesUsd).toBe(s.feesUsd);
      expect(back.equityCurve[3].equity).toBe(s.equityCurve[3].equity);
      expect(paperStats(back)).toEqual(paperStats(s));
    } finally {
      await rm(path, { force: true });
    }
  });
});

describe("stats", () => {
  test("roi, fee drag and buy-and-hold are read off the same run", () => {
    const c = mk([100, 100, 120, 120, 120]);
    const s = newPaperState({ startingEquity: 1000, leverage: 1, costBps: 0 });
    onBar(s, "X", c[0], 1);
    onBar(s, "X", c[1], 1);
    onBar(s, "X", c[2], 0);   // out after capturing the +20% bar
    onBar(s, "X", c[3], 0);
    const st = paperStats(s);
    expect(st.roi).toBeCloseTo(0.2, 9);
    expect(st.trades).toBe(1);
    expect(st.winRate).toBe(1);
    expect(st.feeDrag).toBe(0);
    // Held the whole way for bars 0 and 1, flat for 2 and 3.
    expect(st.exposure).toBeCloseTo(0.5, 9);
    // Buy and hold caught the same move, so timing added nothing.
    expect(st.vsBuyHold).toBeCloseTo(0, 9);
  });

  test("sitting out a fall beats holding it", () => {
    const c = mk([100, 100, 50, 50]);
    const s = newPaperState({ startingEquity: 1000, leverage: 1, costBps: 0 });
    for (const bar of c) onBar(s, "X", bar, 0);
    const st = paperStats(s);
    expect(st.roi).toBe(0);
    expect(st.exposure).toBe(0);
    expect(st.vsBuyHold).toBeCloseTo(0.5, 9);
    expect(st.maxDD).toBe(0);
  });
});
