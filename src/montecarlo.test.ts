import { expect, test, describe } from "bun:test";
import { shuffledSeries, blockShuffledSeries, bootstrapTrades, falsePositiveRate } from "./montecarlo";
import { generateSynthetic } from "./data";
import { buildFeatures } from "./features";
import { fit, viterbi, makeRng } from "./hmm";
import { fitScaler, applyScaler } from "./features";

const { candles } = generateSynthetic({ bars: 2000, seed: 3 });

function logReturns(cs: typeof candles) {
  const r: number[] = [];
  for (let i = 1; i < cs.length; i++) r.push(Math.log(cs[i].close / cs[i - 1].close));
  return r;
}
const mean = (a: number[]) => a.reduce((x, y) => x + y, 0) / a.length;
const sd = (a: number[]) => { const m = mean(a); return Math.sqrt(a.reduce((x, y) => x + (y - m) ** 2, 0) / a.length); };
/** Lag-1 autocorrelation of squared returns — the volatility-clustering signature. */
function volClustering(r: number[]) {
  const sq = r.map((x) => x * x);
  const m = mean(sq);
  let num = 0, den = 0;
  for (let i = 1; i < sq.length; i++) num += (sq[i] - m) * (sq[i - 1] - m);
  for (let i = 0; i < sq.length; i++) den += (sq[i] - m) ** 2;
  return num / den;
}

describe("null series generation", () => {
  test("preserves the marginal return distribution exactly", () => {
    const orig = logReturns(candles).slice().sort((a, b) => a - b);
    const shuf = logReturns(shuffledSeries(candles, 1)).slice().sort((a, b) => a - b);
    expect(shuf.length).toBe(orig.length);
    // Same multiset of returns, just reordered.
    for (let i = 0; i < orig.length; i += 50) expect(shuf[i]).toBeCloseTo(orig[i], 9);
    expect(sd(shuf)).toBeCloseTo(sd(orig), 9);
  });

  test("keeps the fat tails that make this asset class hard", () => {
    const orig = logReturns(candles);
    const shuf = logReturns(shuffledSeries(candles, 2));
    expect(Math.max(...shuf)).toBeCloseTo(Math.max(...orig), 9);
    expect(Math.min(...shuf)).toBeCloseTo(Math.min(...orig), 9);
  });

  test("destroys volatility clustering, which is the point", () => {
    const orig = volClustering(logReturns(candles));
    const shuf = volClustering(logReturns(shuffledSeries(candles, 3)));
    expect(orig).toBeGreaterThan(0.05);      // the real series has clustering
    expect(Math.abs(shuf)).toBeLessThan(orig); // the shuffle removes it
  });

  test("destroys the regime structure a model could find", () => {
    // Fit on the real series and on a null; the null's states should be far
    // less persistent, because consecutive bars are now independent.
    const persistence = (cs: typeof candles) => {
      const fs = buildFeatures(cs, { window: 5 });
      const sc = fitScaler(fs.X, fs.T, fs.D);
      const r = fit(applyScaler(fs.X, fs.T, fs.D, sc), fs.T, fs.D, { states: 3, restarts: 2, seed: 1 });
      let diag = 0;
      for (let k = 0; k < 3; k++) diag += r.params.A[k * 3 + k];
      return diag / 3;
    };
    expect(persistence(candles)).toBeGreaterThan(persistence(shuffledSeries(candles, 4)));
  });

  test("block shuffling retains more clustering than iid shuffling", () => {
    const iid = Math.abs(volClustering(logReturns(shuffledSeries(candles, 5))));
    const blk = Math.abs(volClustering(logReturns(blockShuffledSeries(candles, 50, 5))));
    expect(blk).toBeGreaterThan(iid);
  });

  test("output length and timestamps match the input", () => {
    const out = shuffledSeries(candles, 6);
    expect(out.length).toBe(candles.length);
    expect(out[0].time).toBe(candles[0].time);
    expect(out[out.length - 1].time).toBe(candles[candles.length - 1].time);
  });

  test("prices stay positive and OHLC stays consistent", () => {
    for (const c of shuffledSeries(candles, 7)) {
      expect(c.close).toBeGreaterThan(0);
      expect(c.high).toBeGreaterThanOrEqual(Math.max(c.open, c.close) - 1e-12);
      expect(c.low).toBeLessThanOrEqual(Math.min(c.open, c.close) + 1e-12);
    }
  });
});

describe("false positive rate harness", () => {
  test("a runner that always reports p=0.5 yields a zero rate", () => {
    const r = falsePositiveRate(candles, () => ({ pValue: 0.5, roi: 0, trades: 5 }), { trials: 20 });
    expect(r.rate).toBe(0);
    expect(r.pValues.length).toBe(20);
  });

  test("uniform p-values give a rate near alpha", () => {
    const rng = makeRng(11);
    const r = falsePositiveRate(candles, () => ({ pValue: rng(), roi: 0, trades: 5 }), { trials: 400, alpha: 0.05 });
    expect(Math.abs(r.rate - 0.05)).toBeLessThan(0.04);
  });

  test("trials that take no trades are excluded, not counted as passes", () => {
    const r = falsePositiveRate(candles, (_c, t) => ({ pValue: 0.01, roi: 0, trades: t % 2 === 0 ? 0 : 3 }), { trials: 20 });
    expect(r.noTrades).toBe(10);
    expect(r.pValues.length).toBe(10);
    expect(r.rate).toBe(1);
  });
});

describe("trade bootstrap", () => {
  const mk = (pnls: number[], start = 50) => {
    let eq = start;
    return pnls.map((p) => { eq += p; return { netPnl: p, equityAfter: eq }; });
  };

  test("all-positive trades cannot produce a loss", () => {
    const r = bootstrapTrades(mk([5, 3, 4]), 50, { trials: 500, seed: 1 });
    expect(r.probLoss).toBe(0);
    expect(r.p05).toBeGreaterThan(50);
  });

  test("all-negative trades cannot produce a gain", () => {
    const r = bootstrapTrades(mk([-5, -3, -4]), 50, { trials: 500, seed: 1 });
    expect(r.probLoss).toBe(1);
    expect(r.p95).toBeLessThan(50);
  });

  test("percentiles are ordered and bracket the median", () => {
    const r = bootstrapTrades(mk([8, -6, 4, -3, 9, -7]), 50, { trials: 2000, seed: 1 });
    expect(r.worst).toBeLessThanOrEqual(r.p05);
    expect(r.p05).toBeLessThanOrEqual(r.p25);
    expect(r.p25).toBeLessThanOrEqual(r.median);
    expect(r.median).toBeLessThanOrEqual(r.p75);
    expect(r.p75).toBeLessThanOrEqual(r.p95);
    expect(r.p95).toBeLessThanOrEqual(r.best);
  });

  test("leverage is not applied twice — a +10% equity trade compounds as +10%", () => {
    // One trade: $50 -> $55 is +10% on equity. Two draws must give $60.50.
    const r = bootstrapTrades(mk([5]), 50, { trials: 10, seed: 1 });
    expect(r.median).toBeCloseTo(55, 6);
    const two = bootstrapTrades([...mk([5]), ...mk([5], 55)], 50, { trials: 10, seed: 1 });
    expect(two.median).toBeCloseTo(60.5, 6);
  });

  test("a catastrophic trade can wipe the account out", () => {
    const r = bootstrapTrades(mk([-50]), 50, { trials: 100, seed: 1 });
    expect(r.probRuin).toBe(1);
    expect(r.median).toBe(0);
  });

  test("an empty ledger returns the starting equity, not NaN", () => {
    const r = bootstrapTrades([], 50, { trials: 100 });
    expect(r.median).toBe(50);
    expect(Number.isNaN(r.mean)).toBe(false);
  });
});
