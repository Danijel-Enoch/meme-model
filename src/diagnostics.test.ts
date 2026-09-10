import { expect, test, describe } from "bun:test";
import { permutationTest, ceilingAnalysis, compound } from "./diagnostics";
import { generateSynthetic } from "./data";
import { makeRng } from "./hmm";

describe("compound", () => {
  test("multiplies returns and charges on position changes", () => {
    expect(compound([1, 1], [0.1, 0.1], 0)).toBeCloseTo(0.21, 10);
    // One entry at 100bps: 1.1 * (1 - 0.01) - 1
    expect(compound([1], [0.1], 100)).toBeCloseTo(1.1 - 0.01 - 1, 10);
    expect(compound([0, 0], [0.5, 0.5], 30)).toBe(0);
  });
});

describe("permutation test", () => {
  const rng = makeRng(3);
  const returns = Array.from({ length: 600 }, () => (rng() - 0.5) * 0.1);

  test("random positions on random returns look like luck", () => {
    const pos = returns.map(() => (rng() > 0.5 ? 1 : 0));
    const r = permutationTest(pos, returns, { trials: 500, seed: 11 });
    expect(r.pValue).toBeGreaterThan(0.05);
  });

  test("perfect foresight is detected as skill", () => {
    const pos = returns.map((x) => (x > 0 ? 1 : 0));
    const r = permutationTest(pos, returns, { trials: 500, seed: 11 });
    expect(r.pValue).toBeLessThan(0.01);
    expect(r.actualReturn).toBeGreaterThan(r.p95);
  });

  test("exposure is preserved exactly by the shuffle", () => {
    const pos = returns.map((_, i) => (i % 4 === 0 ? 1 : 0));
    const r = permutationTest(pos, returns, { trials: 100, seed: 11 });
    expect(r.exposure).toBeCloseTo(0.25, 10);
  });

  test("a flat strategy has no timing to test", () => {
    const r = permutationTest(returns.map(() => 0), returns, { trials: 50, seed: 11 });
    expect(r.actualReturn).toBe(0);
    expect(r.exposure).toBe(0);
  });

  test("quantiles are ordered", () => {
    const pos = returns.map(() => (rng() > 0.5 ? 1 : 0));
    const r = permutationTest(pos, returns, { trials: 300, seed: 11 });
    expect(r.p05).toBeLessThanOrEqual(r.medianRandom);
    expect(r.medianRandom).toBeLessThanOrEqual(r.p95);
  });
});

describe("ceiling analysis", () => {
  const { candles } = generateSynthetic({ bars: 2000, seed: 4 });

  test("the oracle beats buy and hold at zero cost", () => {
    const r = ceilingAnalysis(candles, { window: 5 }, { costs: [0], holds: [1] });
    expect(r.rows[0].roi).toBeGreaterThan(r.buyHold);
  });

  test("higher costs never help the oracle", () => {
    const r = ceilingAnalysis(candles, { window: 5 }, { costs: [0, 10, 30], holds: [5] });
    expect(r.rows[1].roi).toBeLessThanOrEqual(r.rows[0].roi + 1e-9);
    expect(r.rows[2].roi).toBeLessThanOrEqual(r.rows[1].roi + 1e-9);
  });

  test("committing for longer means fewer trades", () => {
    const r = ceilingAnalysis(candles, { window: 5 }, { costs: [0], holds: [1, 5, 20] });
    const byHold = new Map(r.rows.map((x) => [x.holdBars, x.trades]));
    expect(byHold.get(20)!).toBeLessThan(byHold.get(5)!);
    expect(byHold.get(5)!).toBeLessThan(byHold.get(1)!);
  });

  test("the next-bar oracle is the highest ceiling at zero cost", () => {
    const r = ceilingAnalysis(candles, { window: 5 }, { costs: [0], holds: [1, 5, 20] });
    const byHold = new Map(r.rows.map((x) => [x.holdBars, x.roi]));
    expect(byHold.get(1)!).toBeGreaterThan(byHold.get(5)!);
    expect(byHold.get(5)!).toBeGreaterThan(byHold.get(20)!);
  });

  test("skip excludes the training region from the ceiling", () => {
    const all = ceilingAnalysis(candles, { window: 5 }, { costs: [0], holds: [5] });
    const tail = ceilingAnalysis(candles, { window: 5 }, { costs: [0], holds: [5], skip: 1000 });
    expect(tail.bars).toBeLessThan(all.bars);
  });
});

describe("null method", () => {
  const rng = makeRng(5);
  const returns = Array.from({ length: 800 }, () => (rng() - 0.5) * 0.08);
  // One long hold: 2 legs of cost in total.
  const blocky = returns.map((_, i) => (i >= 100 && i < 500 ? 1 : 0));

  test("rotation preserves turnover; a free shuffle destroys it", () => {
    const rotated = permutationTest(blocky, returns, { trials: 50, seed: 1, costBps: 30, method: "rotate" });
    const shuffled = permutationTest(blocky, returns, { trials: 50, seed: 1, costBps: 30, method: "shuffle" });
    // Shuffling scatters the block, so the null pays vastly more in fees and
    // its median return collapses — which would fake significance.
    expect(shuffled.medianRandom).toBeLessThan(rotated.medianRandom);
  });

  test("rotation is the default once costs are on", () => {
    expect(permutationTest(blocky, returns, { trials: 10, costBps: 30 }).method).toBe("rotate");
    expect(permutationTest(blocky, returns, { trials: 10, costBps: 0 }).method).toBe("shuffle");
  });

  test("a rotated null still finds genuine foresight significant", () => {
    const perfect = returns.map((x) => (x > 0 ? 1 : 0));
    const r = permutationTest(perfect, returns, { trials: 300, seed: 1, costBps: 0, method: "rotate" });
    expect(r.pValue).toBeLessThan(0.05);
  });
});
