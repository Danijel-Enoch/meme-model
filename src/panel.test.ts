import { expect, test, describe } from "bun:test";
import { alignPanel, forecastTest, panelTest, pooledStatistic, type PanelSeries } from "./panel";
import { makeRng, randn } from "./hmm";

/**
 * A market: one common factor every coin shares, plus idiosyncratic noise.
 * This is what crypto perps actually look like, and it is the shape that breaks
 * a naive pooled test.
 */
function market(coins: number, bars: number, seed: number, beta = 0.9) {
  const rng = makeRng(seed);
  const factor = Array.from({ length: bars }, () => 0.004 * randn(rng) + 0.0004);
  const out: PanelSeries[] = [];
  for (let c = 0; c < coins; c++) {
    const returns = factor.map((f) => beta * f + 0.002 * randn(rng));
    out.push({
      coin: `C${c}`,
      times: Array.from({ length: bars }, (_, t) => 1_700_000_000 + t * 1800),
      positions: new Array(bars).fill(0),
      returns,
    });
  }
  return out;
}

describe("pooled statistic", () => {
  test("charges turnover the way the backtest does", () => {
    // Flat, long for two bars, flat. Two position changes at 10bps a side.
    const positions = [[0, 1, 1, 0]];
    const returns = [[0, 0, 0, 0]];
    const { bps } = pooledStatistic(positions, returns, 10);
    // Two units of |delta| x 10bps, spread over 4 observations.
    expect(bps).toBeCloseTo((-2 * 0.001 / 4) * 10_000, 9);
  });

  test("earns the position's own return, never the next coin's", () => {
    const { bps } = pooledStatistic([[1, 1]], [[0.01, 0.02]], 0);
    expect(bps).toBeCloseTo(150, 9);
  });

  test("nulls are skipped without shifting anything", () => {
    const a = pooledStatistic([[1, null, 1]], [[0.01, 0.5, 0.01]], 0);
    expect(a.observations).toBe(2);
    expect(a.bps).toBeCloseTo(100, 9);
  });
});

describe("alignment", () => {
  test("coins on different grids land on one axis", () => {
    const s: PanelSeries[] = [
      { coin: "A", times: [10, 20, 30], positions: [1, 1, 0], returns: [0.1, 0.2, 0.3] },
      { coin: "B", times: [20, 30, 40], positions: [0, 1, 1], returns: [0.4, 0.5, 0.6] },
    ];
    const { times, positions, returns } = alignPanel(s);
    expect(times).toEqual([10, 20, 30, 40]);
    expect(positions[0]).toEqual([1, 1, 0, null]);
    expect(positions[1]).toEqual([null, 0, 1, 1]);
    expect(returns[1][0]).toBeNull();
  });
});

describe("the panel test", () => {
  test("finds a planted edge", () => {
    // Positions that know the next bar's sign, on half the bars.
    const base = market(8, 900, 11);
    const series = base.map((s) => ({
      ...s,
      positions: s.returns.map((r, i) => (i % 2 === 0 ? (r > 0 ? 1 : 0) : 0)),
    }));
    const res = panelTest(series, { costBps: 0, trials: 400, seed: 5 });
    expect(res.actualBps).toBeGreaterThan(res.p95Bps);
    expect(res.pValue).toBeLessThan(0.01);
  });

  test("does not find one that is not there", () => {
    const rng = makeRng(77);
    const series = market(8, 900, 12).map((s) => ({
      ...s,
      // Positions independent of returns, but persistent, so turnover is realistic.
      positions: s.returns.map((_, i) => (Math.floor(i / 20) % 3 === 0 ? 1 : 0)),
    }));
    const res = panelTest(series, { costBps: 4.5, trials: 400, seed: 6 });
    expect(res.pValue).toBeGreaterThan(0.05);
  });

  test("rejects at roughly the nominal rate on noise", () => {
    // The property that makes a p-value mean anything: under the null it is
    // uniform, so 5% of noise panels reject at 5%. Anything much above this and
    // the test is manufacturing significance.
    let rejects = 0;
    const runs = 40;
    for (let s = 0; s < runs; s++) {
      const series = market(6, 500, 100 + s).map((x, c) => ({
        ...x,
        positions: x.returns.map((_, i) => (Math.floor((i + c * 7) / 15) % 2 === 0 ? 1 : 0)),
      }));
      if (panelTest(series, { costBps: 4.5, trials: 200, seed: 900 + s }).pValue < 0.05) rejects++;
    }
    expect(rejects / runs).toBeLessThan(0.20);
  });

  test("THE TRAP: an independent null understates how correlated the evidence is", () => {
    // Every coin runs the SAME position schedule — long in fixed blocks — on a
    // market that is one factor plus noise. There is no per-coin skill here:
    // twenty coins holding the same view at the same time is ONE bet, made
    // twenty times over, not twenty independent confirmations.
    //
    // A null that rotates each coin separately scatters those blocks across
    // different times, so the common factor averages away and the null
    // distribution collapses to a sliver. The real statistic still contains the
    // factor, so it lands far outside that sliver and the test "discovers"
    // something. Rotating the whole market together keeps the factor in the
    // null, where it belongs.
    const bars = 1200;
    const schedule = Array.from({ length: bars }, (_, t) => (Math.floor(t / 30) % 3 === 0 ? 1 : 0));
    const series = market(20, bars, 21).map((s) => ({ ...s, positions: [...schedule] }));

    const common = panelTest(series, { costBps: 0, trials: 400, seed: 7, nullMethod: "common" });
    const independent = panelTest(series, { costBps: 0, trials: 400, seed: 7, nullMethod: "independent" });

    // Same observed statistic — only the null differs.
    expect(common.actualBps).toBeCloseTo(independent.actualBps, 9);

    const commonSpread = common.p95Bps - common.p05Bps;
    const independentSpread = independent.p95Bps - independent.p05Bps;
    // The sliver: the independent null is several times too tight.
    expect(independentSpread).toBeLessThan(commonSpread / 2);
    // And so it can only ever produce a p-value at least as small.
    expect(independent.pValue).toBeLessThanOrEqual(common.pValue);
  });

  test("p can never be reported as zero", () => {
    const series = market(4, 300, 33).map((s) => ({ ...s, positions: s.returns.map((r) => (r > 0 ? 1 : -1)) }));
    const res = panelTest(series, { costBps: 0, trials: 100, seed: 8 });
    expect(res.pValue).toBeGreaterThan(0);
    expect(res.pValue).toBeCloseTo(1 / 101, 6);
  });
});

describe("the forecast test", () => {
  test("finds forecasting content the P&L test misses", () => {
    // Real but weak information: 2% of the next bar's return, buried in noise.
    // The P&L test cannot see it and the forecast test can, which is the whole
    // argument for running both.
    //
    // Worth knowing while reading this: the rotation null pays the same
    // turnover the real series paid, so raising costs does NOT make the P&L
    // test fail to reject — cost cancels between the statistic and its null.
    // That test measures timing skill, not profitability. The gap being
    // demonstrated here is power, not economics.
    const rng = makeRng(4242);
    const series = market(10, 1500, 55).map((s) => {
      const forecasts = s.returns.map((r) => 0.02 * r + 0.0006 * randn(rng));
      // The forecast is continuous and every bar carries a little of it; the
      // position is a threshold crossing that fires rarely. Quantizing throws
      // away most of the sample, and with it most of the power — which is the
      // whole reason to test the forecast directly.
      return { ...s, forecasts, positions: forecasts.map((f) => (f > 0.0018 ? 1 : 0)) };
    });
    const pnl = panelTest(series, { costBps: 4.5, trials: 300, seed: 3 });
    const fc = forecastTest(series, { trials: 300, seed: 3 });

    expect(fc.ic).toBeGreaterThan(fc.p95Ic);
    expect(fc.pValue).toBeLessThan(0.05);
    // Forecast yes, tradeable no — and the P&L test alone would have called the
    // whole model worthless.
    expect(pnl.pValue).toBeGreaterThan(0.05);
  });

  test("does not find content in a forecast that has none", () => {
    // Independent AR(1) noise, not a periodic wave: rotating a sinusoid only
    // ever produces other phases of the same sinusoid, which makes the null
    // degenerate and the test meaningless.
    const rng = makeRng(99);
    const series = market(10, 1200, 56).map((s) => {
      const f: number[] = [];
      let prev = 0;
      for (let i = 0; i < s.returns.length; i++) {
        prev = 0.9 * prev + 0.001 * randn(rng);
        f.push(prev);
      }
      return { ...s, forecasts: f, positions: s.returns.map(() => 0) };
    });
    const fc = forecastTest(series, { trials: 300, seed: 4 });
    expect(fc.pValue).toBeGreaterThan(0.05);
  });

  test("refuses to run on a panel with no forecasts rather than inventing them", () => {
    const series = market(3, 200, 57);
    expect(() => forecastTest(series, { trials: 10 })).toThrow(/no forecasts/);
  });

  test("the slope reads as calibration", () => {
    // A forecast equal to the realization must give slope 1 and ic 1.
    const s = market(4, 400, 58).map((x) => ({ ...x, forecasts: [...x.returns] }));
    const fc = forecastTest(s, { trials: 50, seed: 5 });
    expect(fc.slope).toBeCloseTo(1, 6);
    expect(fc.ic).toBeCloseTo(1, 6);
  });
});

describe("forecast horizons", () => {
  test("a forecast about h bars ahead is invisible at one bar and clear at h", () => {
    // The signal is spread over the next 8 bars, a little in each, so no single
    // bar carries enough of it to see. This is what a dwell-based regime claim
    // looks like, and scoring it one bar out is scoring the wrong claim.
    const rng = makeRng(31);
    const bars = 1600, h = 8;
    const series = market(8, bars, 61).map((s) => {
      const forecasts = s.returns.map(() => 0.001 * randn(rng));
      const returns = s.returns.map((r, i) => {
        let carried = 0;
        for (let k = 1; k <= h; k++) carried += 0.10 * (forecasts[i - k] ?? 0);
        return r + carried;
      });
      return { ...s, returns, forecasts, positions: returns.map(() => 0) };
    });
    const one = forecastTest(series, { trials: 300, seed: 9, horizon: 1 });
    const many = forecastTest(series, { trials: 300, seed: 9, horizon: h });
    expect(one.pValue).toBeGreaterThan(0.05);
    expect(many.pValue).toBeLessThan(0.05);
    expect(many.ic).toBeGreaterThan(one.ic);
  });

  test("the horizon window never runs off the end of the series", () => {
    const s = market(3, 60, 62).map((x) => ({ ...x, forecasts: x.returns.map(() => 0.001) }));
    const res = forecastTest(s, { trials: 20, seed: 10, horizon: 10 });
    // The last h-1 rows of each coin cannot have a full window and are dropped.
    expect(res.observations).toBeLessThanOrEqual(3 * (60 - 9));
    expect(res.observations).toBeGreaterThan(0);
    expect(Number.isFinite(res.ic)).toBe(true);
  });
});
