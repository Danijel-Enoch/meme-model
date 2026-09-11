import { expect, test, describe } from "bun:test";
import { makeRng, randn } from "./hmm";
import {
  stationaryBootstrapIndices, automaticBlockLength, stationaryBootstrapVariance,
  spaTest, realityCheck, romanoWolf, deflatedSharpe, probabilisticSharpe,
  expectedMaxSharpe, normalCdf, normalInv, alignOnGrid, seriesMoments,
  judgeSweep, effectiveTrials, type SeriesCache, type CellSeries,
} from "./spa";

/**
 * Everything here is tested against data whose answer is known by construction.
 * A data-snooping correction that is only checked on real returns cannot be
 * checked at all — there is no ground truth to check it against, which is the
 * entire problem it exists to solve.
 */

/** iid standard normal series. */
function noise(n: number, rng: () => number, sd = 1): number[] {
  return Array.from({ length: n }, () => randn(rng) * sd);
}

/** AR(1): x_t = phi * x_{t-1} + eps, scaled to unit unconditional variance. */
function ar1(n: number, phi: number, rng: () => number): number[] {
  const sd = Math.sqrt(1 - phi * phi);
  const x: number[] = [];
  let prev = randn(rng);
  for (let i = 0; i < n; i++) {
    prev = phi * prev + sd * randn(rng);
    x.push(prev);
  }
  return x;
}

function lag1Corr(x: number[]): number {
  const n = x.length;
  const mean = x.reduce((a, b) => a + b, 0) / n;
  let c0 = 0, c1 = 0;
  for (let i = 0; i < n; i++) c0 += (x[i] - mean) ** 2;
  for (let i = 0; i < n - 1; i++) c1 += (x[i] - mean) * (x[i + 1] - mean);
  return c1 / c0;
}

const mean = (x: number[]) => x.reduce((a, b) => a + b, 0) / x.length;

describe("normal helpers", () => {
  test("cdf and inverse agree and hit the textbook quantiles", () => {
    expect(normalCdf(0)).toBeCloseTo(0.5, 12);
    expect(normalCdf(1.959963985)).toBeCloseTo(0.975, 8);
    expect(normalInv(0.975)).toBeCloseTo(1.959963985, 7);
    expect(normalInv(0.5)).toBeCloseTo(0, 10);
    for (const p of [1e-6, 0.001, 0.05, 0.4, 0.9, 0.999, 1 - 1e-6]) {
      expect(normalCdf(normalInv(p))).toBeCloseTo(p, 9);
    }
  });
});

describe("stationary bootstrap (Politis & Romano 1994)", () => {
  test("indices stay in range and are the right length", () => {
    const rng = makeRng(1);
    const idx = stationaryBootstrapIndices(100, 10, rng);
    expect(idx.length).toBe(100);
    for (const i of idx) { expect(i).toBeGreaterThanOrEqual(0); expect(i).toBeLessThan(100); }
  });

  test("mean block length 1 degenerates to the iid bootstrap", () => {
    // p = 1 means every step starts a fresh block, so no index is ever the
    // successor of the previous one except by chance (probability 1/n).
    const rng = makeRng(2);
    const idx = stationaryBootstrapIndices(500, 1, rng);
    let consecutive = 0;
    for (let i = 1; i < idx.length; i++) if (idx[i] === (idx[i - 1] + 1) % 500) consecutive++;
    expect(consecutive).toBeLessThan(20); // ~1 expected by chance
  });

  test("long blocks produce long runs, and the run length matches 1/p", () => {
    const rng = makeRng(3);
    const b = 20, n = 20_000;
    const idx = stationaryBootstrapIndices(n, b, rng);
    let breaks = 1;
    for (let i = 1; i < n; i++) if (idx[i] !== (idx[i - 1] + 1) % n) breaks++;
    // Geometric(1/b) blocks, so n/b breaks in expectation.
    expect(n / breaks).toBeGreaterThan(b * 0.8);
    expect(n / breaks).toBeLessThan(b * 1.25);
  });

  test("(c) it preserves AR(1) dependence where an iid shuffle destroys it", () => {
    const rng = makeRng(11);
    const phi = 0.7;
    const x = ar1(4000, phi, rng);
    const truth = lag1Corr(x);
    expect(truth).toBeGreaterThan(0.6); // sanity: the series really is persistent

    const b = automaticBlockLength(x);
    const boot: number[] = [], shuffled: number[] = [];
    for (let r = 0; r < 40; r++) {
      const idx = stationaryBootstrapIndices(x.length, b, rng);
      boot.push(lag1Corr(Array.from(idx, (i) => x[i])));
      // The iid alternative: sample bars independently, i.e. block length 1.
      const iid = stationaryBootstrapIndices(x.length, 1, rng);
      shuffled.push(lag1Corr(Array.from(iid, (i) => x[i])));
    }
    const bootRho = mean(boot), shufRho = mean(shuffled);
    // The block bootstrap keeps most of the dependence; the shuffle keeps none.
    expect(bootRho).toBeGreaterThan(0.55);
    expect(Math.abs(shufRho)).toBeLessThan(0.05);
    expect(Math.abs(bootRho - truth)).toBeLessThan(Math.abs(shufRho - truth));
  });

  test("the variance of the sample mean is inflated by persistence, as it must be", () => {
    const rng = makeRng(12);
    const x = ar1(3000, 0.7, rng);
    const gamma0 = x.reduce((a, v) => a + (v - mean(x)) ** 2, 0) / x.length;
    const b = automaticBlockLength(x);
    const lrv = stationaryBootstrapVariance(x, b);
    // Long-run variance of an AR(1) is gamma0 * (1+phi)/(1-phi) ~ 5.7 * gamma0.
    // An iid variance estimate would report gamma0 and understate every
    // standard error by a factor of ~2.4, i.e. halve every p-value.
    expect(lrv).toBeGreaterThan(2 * gamma0);
    expect(stationaryBootstrapVariance(x, 1)).toBeCloseTo(gamma0, 8);
  });
});

describe("automatic block length (Politis & White 2004)", () => {
  test("(d) it grows with the AR coefficient", () => {
    // Averaged over seeds: the rule is a bandwidth selector, so a single draw
    // is noisy even though the ordering is systematic.
    const lengths = [0, 0.3, 0.6, 0.85].map((phi) => {
      const runs = Array.from({ length: 12 }, (_, s) => automaticBlockLength(ar1(1500, phi, makeRng(100 + s))));
      return mean(runs);
    });
    for (let i = 1; i < lengths.length; i++) {
      expect(lengths[i]).toBeGreaterThan(lengths[i - 1]);
    }
    // iid data needs no blocks at all; strong persistence needs many bars.
    expect(lengths[0]).toBeLessThan(3);
    expect(lengths[3]).toBeGreaterThan(8);
  });

  test("degenerate inputs fall back to iid instead of exploding", () => {
    expect(automaticBlockLength([])).toBe(1);
    expect(automaticBlockLength(new Array(500).fill(0.01))).toBe(1);
    expect(Number.isFinite(automaticBlockLength(noise(20, makeRng(5))))).toBe(true);
  });

  test("it never exceeds the paper's cap of min(3 sqrt n, n/3)", () => {
    for (const phi of [0.5, 0.9, 0.97]) {
      const n = 600;
      const b = automaticBlockLength(ar1(n, phi, makeRng(7)));
      expect(b).toBeLessThanOrEqual(Math.ceil(Math.min(3 * Math.sqrt(n), n / 3)));
    }
  });
});

describe("SPA calibration under the null", () => {
  /**
   * (a) The headline claim. Ten strategies, all pure noise, no skill anywhere.
   * A researcher who tests each one at 5% and reports the best rejects far more
   * than 5% of the time; SPA, which knows all ten were tried, does not.
   */
  test("(a) N noise strategies: SPA holds its size where naive min-p does not", () => {
    const seeds = 160, n = 400, m = 10, B = 200;
    const spaP: number[] = [];
    let spaRej = 0, naiveRej = 0, rcRej = 0;

    for (let s = 0; s < seeds; s++) {
      const rng = makeRng(1000 + s);
      const cells: Record<string, number[]> = {};
      for (let k = 0; k < m; k++) cells[`noise${k}`] = noise(n, rng, 0.01);

      const spa = spaTest(cells, { bootstraps: B, seed: 500 + s });
      spaP.push(spa.pConsistent);
      if (spa.pConsistent <= 0.05) spaRej++;
      if (realityCheck(cells, { bootstraps: B, seed: 500 + s }).pValue <= 0.05) rcRej++;

      // The naive procedure this repo's sweep table currently invites: test each
      // cell on its own at 5% and report the smallest p. Independent cells make
      // that 1 - 0.95^10 = 40% by construction.
      let minP = 1;
      for (const name of Object.keys(cells)) {
        const t = spa.tStats[name];
        minP = Math.min(minP, 1 - normalCdf(t));
      }
      if (minP <= 0.05) naiveRej++;
    }

    const spaRate = spaRej / seeds, naiveRate = naiveRej / seeds, rcRate = rcRej / seeds;
    // Size control: nominal 5%, allow bootstrap noise (se ~ 1.7pp at 160 seeds).
    expect(spaRate).toBeLessThan(0.12);
    expect(rcRate).toBeLessThan(0.12);
    // The uncorrected procedure is off by a factor of several.
    expect(naiveRate).toBeGreaterThan(0.25);
    expect(naiveRate).toBeGreaterThan(spaRate * 2.5);

    // Uniform-ish: the p-value should not pile up at either end.
    const avg = mean(spaP);
    expect(avg).toBeGreaterThan(0.25);
    expect(avg).toBeLessThan(0.75);
    const below = (q: number) => spaP.filter((p) => p <= q).length / seeds;
    expect(below(0.25)).toBeLessThan(0.45);
    expect(below(0.75)).toBeGreaterThan(0.55);
  }, 120_000);

  test("(c, applied) an iid bootstrap wrecks SPA's size on dependent data", () => {
    // The reason the whole stationary-bootstrap apparatus is here, measured.
    // Same null (no skill anywhere), but the strategies' returns are serially
    // correlated the way a real trend-follower's are. Forcing blockLength = 1
    // is the iid bootstrap: it estimates the variance of the sample mean as
    // gamma0/n instead of the long-run variance/n, so its critical values are
    // far too small and it rejects a true null most of the time.
    const seeds = 60, n = 400, m = 8, B = 200;
    let iidRej = 0, blockRej = 0;
    for (let s = 0; s < seeds; s++) {
      const rng = makeRng(2000 + s);
      const cells: Record<string, number[]> = {};
      for (let k = 0; k < m; k++) cells[`c${k}`] = ar1(n, 0.6, rng).map((v) => v * 0.01);
      if (spaTest(cells, { bootstraps: B, seed: 900 + s, blockLength: 1 }).pConsistent <= 0.05) iidRej++;
      if (spaTest(cells, { bootstraps: B, seed: 900 + s }).pConsistent <= 0.05) blockRej++;
    }
    expect(iidRej / seeds).toBeGreaterThan(0.4);   // catastrophic, ~0.6 in practice
    expect(blockRej / seeds).toBeLessThan(iidRej / seeds / 2);
  }, 60_000);

  test("the three Hansen p-values are ordered lower <= consistent <= upper", () => {
    const rng = makeRng(31);
    const cells: Record<string, number[]> = {};
    // A deliberately mixed family: a couple of decent cells and a pile of duds.
    for (let k = 0; k < 3; k++) cells[`ok${k}`] = noise(600, rng, 0.01).map((v) => v + 0.0004);
    for (let k = 0; k < 20; k++) cells[`dud${k}`] = noise(600, rng, 0.01).map((v) => v - 0.003);
    const spa = spaTest(cells, { bootstraps: 500, seed: 3 });
    expect(spa.pLower).toBeLessThanOrEqual(spa.pConsistent);
    expect(spa.pConsistent).toBeLessThanOrEqual(spa.pUpper);
  });

  test("both admissible thresholds land inside the lower/upper bracket", () => {
    // Hansen's published rule is the LIL rate; his working paper used
    // (1/4) n^(-1/4). Both are valid, they differ in finite samples, and the
    // whole reason he reports p^l and p^u is to bound that ambiguity.
    const rng = makeRng(32);
    const cells: Record<string, number[]> = {};
    for (let k = 0; k < 4; k++) cells[`ok${k}`] = noise(600, rng, 0.01).map((v) => v + 0.0005);
    for (let k = 0; k < 25; k++) cells[`dud${k}`] = noise(600, rng, 0.01).map((v) => v - 0.002);
    const lil = spaTest(cells, { bootstraps: 800, seed: 4 });
    const qtr = spaTest(cells, { bootstraps: 800, seed: 4, threshold: "quarter" });
    expect(lil.tStat).toBeCloseTo(qtr.tStat, 12);           // same statistic
    for (const r of [lil, qtr]) {
      expect(r.pConsistent).toBeGreaterThanOrEqual(r.pLower);
      expect(r.pConsistent).toBeLessThanOrEqual(r.pUpper);
    }
    // The looser working-paper rule drops more models, so its p-value is the
    // smaller of the two.
    expect(qtr.pConsistent).toBeLessThanOrEqual(lil.pConsistent);
  });

  test("adding hopeless models inflates White's RC far more than Hansen's SPA", () => {
    // Hansen's central criticism, reproduced. The extra cells are so bad that
    // no one would claim them, yet the RC keeps them on the null boundary and
    // lets them raise the bootstrap maximum, diluting the real signal.
    const rng = makeRng(41);
    const n = 800;
    const base: Record<string, number[]> = {
      real: noise(n, rng, 0.01).map((v) => v + 0.0013),
    };
    for (let k = 0; k < 4; k++) base[`mild${k}`] = noise(n, rng, 0.01);

    const padded: Record<string, number[]> = { ...base };
    for (let k = 0; k < 60; k++) padded[`awful${k}`] = noise(n, rng, 0.02).map((v) => v - 0.01);

    const spaBase = spaTest(base, { bootstraps: 500, seed: 9 });
    const spaPad = spaTest(padded, { bootstraps: 500, seed: 9 });
    const rcBase = realityCheck(base, { bootstraps: 500, seed: 9 });
    const rcPad = realityCheck(padded, { bootstraps: 500, seed: 9 });

    // SPA is essentially untouched: the awful cells are dropped from the null.
    expect(spaPad.pConsistent - spaBase.pConsistent).toBeLessThan(0.05);
    // The RC degrades much more.
    expect(rcPad.pValue - rcBase.pValue).toBeGreaterThan(spaPad.pConsistent - spaBase.pConsistent);
    expect(rcPad.pValue).toBeGreaterThan(spaPad.pConsistent);
  });
});

describe("SPA power", () => {
  /** (b) One real edge buried in 99 fakes still has to be findable. */
  test("(b) it finds the one profitable strategy hidden among 99 noise ones", () => {
    const rng = makeRng(77);
    const n = 800;
    const cells: Record<string, number[]> = {};
    for (let k = 0; k < 99; k++) cells[`noise${k}`] = noise(n, rng, 0.01);
    // ~0.14 per-bar Sharpe: not a fantasy, but well outside what the max of 100
    // noise cells produces (that max sits near t = 2.7).
    cells["real"] = noise(n, rng, 0.01).map((v) => v + 0.0014);

    const spa = spaTest(cells, { bootstraps: 500, seed: 21 });
    expect(spa.best).toBe("real");
    expect(spa.pConsistent).toBeLessThan(0.05);

    // Romano-Wolf must single it out, and must not sweep in the noise with it.
    const rw = romanoWolf(cells, { bootstraps: 500, seed: 21, alpha: 0.05 });
    expect(rw.rejected).toContain("real");
    expect(rw.rejected.length).toBeLessThanOrEqual(3);
    expect(rw.adjustedP["real"]).toBeLessThan(0.05);
  }, 60_000);

  test("Romano-Wolf rejects nothing when nothing is real", () => {
    const rng = makeRng(78);
    const cells: Record<string, number[]> = {};
    for (let k = 0; k < 60; k++) cells[`noise${k}`] = noise(500, rng, 0.01);
    const rw = romanoWolf(cells, { bootstraps: 400, seed: 22, alpha: 0.05 });
    expect(rw.rejected.length).toBe(0);
  });

  test("Romano-Wolf adjusted p-values are monotone in the statistic", () => {
    const rng = makeRng(79);
    const cells: Record<string, number[]> = {};
    for (let k = 0; k < 8; k++) cells[`c${k}`] = noise(400, rng, 0.01).map((v) => v + k * 0.0004);
    const rw = romanoWolf(cells, { bootstraps: 300, seed: 23 });
    const order = Object.keys(rw.tStats).sort((a, b) => rw.tStats[b] - rw.tStats[a]);
    for (let i = 1; i < order.length; i++) {
      expect(rw.adjustedP[order[i]]).toBeGreaterThanOrEqual(rw.adjustedP[order[i - 1]]);
    }
    // Every adjusted p-value is at least the raw single-test p-value.
    for (const name of order) {
      expect(rw.adjustedP[name]).toBeGreaterThanOrEqual(1 - normalCdf(rw.tStats[name]) - 0.02);
    }
  });

  test("a single strategy makes SPA collapse to an ordinary one-sided test", () => {
    // With m = 1 there is nothing to correct for, so the multiple-testing
    // machinery must not charge anything: p should track the plain t-test.
    const rng = makeRng(80);
    const x = noise(2000, rng, 0.01).map((v) => v + 0.00055);
    const spa = spaTest({ only: x }, { bootstraps: 2000, seed: 24 });
    const naive = 1 - normalCdf(spa.tStats["only"]);
    expect(Math.abs(spa.pConsistent - naive)).toBeLessThan(0.05);
  });
});

describe("input handling", () => {
  test("misaligned series are refused rather than silently truncated", () => {
    expect(() => spaTest({ a: [1, 2, 3], b: [1, 2] })).toThrow(/aligned in time/);
    expect(() => spaTest({})).toThrow(/no strategies/);
  });

  test("a never-traded cell contributes nothing instead of dividing by zero", () => {
    const rng = makeRng(90);
    const cells = {
      flat: new Array(300).fill(0),
      live: noise(300, rng, 0.01),
    };
    const spa = spaTest(cells, { bootstraps: 200, seed: 25 });
    expect(Number.isFinite(spa.pConsistent)).toBe(true);
    expect(spa.tStats["flat"]).toBe(0);
  });
});

describe("Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014)", () => {
  test("the expected maximum Sharpe under the null grows with the trial count", () => {
    const v = 0.04; // sd of trial Sharpes = 0.2
    const one = expectedMaxSharpe(2, v);
    const many = expectedMaxSharpe(240, v);
    expect(many).toBeGreaterThan(one);
    // 240 trials: the max of 240 standard normals sits near 2.9 sd.
    expect(many / Math.sqrt(v)).toBeGreaterThan(2.5);
    expect(many / Math.sqrt(v)).toBeLessThan(3.3);
  });

  test("the Gumbel approximation for E[max Sharpe] matches Monte Carlo", () => {
    // Bailey & Lopez de Prado's SR0 is an approximation to the expected maximum
    // of N draws. Worth checking it is actually right at the N this repo uses,
    // rather than trusting the closed form: simulate N noise strategies, take
    // the best, and compare to the formula in units of the trial spread.
    const rng = makeRng(4242);
    for (const [N, bars] of [[228, 269], [50, 500]] as Array<[number, number]>) {
      const maxes: number[] = [];
      for (let r = 0; r < 60; r++) {
        const srs = Array.from({ length: N }, () => seriesMoments(noise(bars, rng, 0.01)).sharpe);
        const mu = mean(srs);
        const v = srs.reduce((a, b) => a + (b - mu) * (b - mu), 0) / (N - 1);
        maxes.push(Math.max(...srs) / Math.sqrt(v));
      }
      expect(mean(maxes) / expectedMaxSharpe(N, 1)).toBeCloseTo(1, 1);
    }
  }, 60_000);

  test("PSR falls when returns are negatively skewed or fat-tailed", () => {
    const normal = probabilisticSharpe(0.1, 0, 500, 0, 3);
    const skewed = probabilisticSharpe(0.1, 0, 500, -1.5, 3);
    const fat = probabilisticSharpe(0.1, 0, 500, 0, 9);
    expect(skewed).toBeLessThan(normal);
    expect(fat).toBeLessThan(normal);
  });

  test("the same Sharpe is convincing after 1 trial and worthless after 500", () => {
    // Every trial has sd 0.05 in per-bar Sharpe; the winner posted 0.12.
    const rng = makeRng(55);
    // Fixed spread for the small search so the sample variance of five draws
    // is not itself the thing under test.
    const few = [-0.06, -0.03, 0, 0.03, 0.06];
    const many = Array.from({ length: 500 }, () => randn(rng) * 0.05);
    const dFew = deflatedSharpe(few, 0.12, 1000);
    const dMany = deflatedSharpe(many, 0.12, 1000);
    expect(dFew).toBeGreaterThan(0.9);
    // At 500 trials the expected maximum Sharpe under the null is ~3 sd of the
    // trial spread, i.e. above 0.12: the winner is below its own null.
    expect(dMany).toBeLessThan(0.5);
  });

  test("the winner of a pure-noise search is deflated to nothing", () => {
    // Simulate the actual procedure: 240 cells of 500 bars of pure noise, take
    // the best. Its raw Sharpe looks fine; its DSR must not.
    const rng = makeRng(56);
    const bars = 500;
    const sharpes = Array.from({ length: 240 }, () => seriesMoments(noise(bars, rng, 0.01)).sharpe);
    const best = Math.max(...sharpes);
    expect(best * Math.sqrt(bars)).toBeGreaterThan(2); // "t-stat over 2", the classic trap
    expect(deflatedSharpe(sharpes, best, bars)).toBeLessThan(0.95);
  });

  test("correlated trials count for less than independent ones", () => {
    const rng = makeRng(57);
    const M = 30, n = 400;
    const indep = Array.from({ length: M }, () => noise(n, rng));
    expect(effectiveTrials(indep).effective).toBeGreaterThan(M * 0.9);
    expect(Math.abs(effectiveTrials(indep).rhoBar)).toBeLessThan(0.05);

    // Thirty copies of one series are one trial, not thirty.
    const one = noise(n, rng);
    const clones = Array.from({ length: M }, () => [...one]);
    expect(effectiveTrials(clones).rhoBar).toBeCloseTo(1, 6);
    expect(effectiveTrials(clones).effective).toBeCloseTo(1, 6);

    // Halfway: a common factor plus idiosyncratic noise.
    const factor = noise(n, rng);
    const mixed = Array.from({ length: M }, () => {
      const e = noise(n, rng);
      return factor.map((f, i) => f + e[i]);
    });
    const mix = effectiveTrials(mixed);
    expect(mix.rhoBar).toBeGreaterThan(0.3);
    expect(mix.rhoBar).toBeLessThan(0.7);
    expect(mix.effective).toBeGreaterThan(1);
    expect(mix.effective).toBeLessThan(M);
  });

  test("degenerate inputs do not produce a number", () => {
    expect(Number.isNaN(deflatedSharpe([], 1, 100))).toBe(true);
    expect(Number.isNaN(probabilisticSharpe(1, 0, 500, 3, 1))).toBe(true); // 1 - 3*1 + 0 < 0
  });
});

describe("putting cells on one clock", () => {
  test("log returns are summed into buckets, not averaged", () => {
    const s = [
      { name: "fast", times: [0, 900, 1800, 2700], values: [0.01, 0.02, 0.03, 0.04] },
      { name: "slow", times: [0, 3600], values: [0.5, 0.6] },
    ];
    const { grid, aligned } = alignOnGrid(s, 3600);
    expect(grid.length).toBe(2);
    expect(aligned["fast"][0]).toBeCloseTo(0.1, 12);  // 0.01+0.02+0.03+0.04, all in bucket 0
    expect(aligned["fast"][1]).toBe(0);               // fast has no bars in bucket 1
    expect(aligned["slow"]).toEqual([0.5, 0.6]);
  });

  test("padding a short cell with zeros barely moves its studentised statistic", () => {
    // This is what licenses the union-of-windows alignment: a cell that is not
    // deployed is flat, which IS the benchmark, so its loss differential is 0.
    const rng = makeRng(61);
    const short = noise(300, rng, 0.01).map((v) => v + 0.0008);
    const padded = [...new Array(300).fill(0), ...short];
    const a = spaTest({ x: short }, { bootstraps: 400, seed: 26 });
    const b = spaTest({ x: padded }, { bootstraps: 400, seed: 26 });
    expect(Math.abs(a.tStats["x"] - b.tStats["x"])).toBeLessThan(0.35);
  });

  test("seriesMoments recovers the moments it claims to", () => {
    const rng = makeRng(62);
    const x = noise(20_000, rng, 1).map((v) => v + 0.25);
    const m = seriesMoments(x);
    expect(m.sharpe).toBeCloseTo(0.25, 1);
    expect(Math.abs(m.skew)).toBeLessThan(0.1);
    expect(m.kurtosis).toBeCloseTo(3, 0);
  });
});

describe("judging a whole sweep end to end", () => {
  /**
   * The full pipeline on a synthetic sweep whose answer is known: cells on four
   * different timeframes, aligned onto one clock, judged by all four tests.
   */
  function fakeSweep(edge: number, seed: number): SeriesCache {
    const rng = makeRng(seed);
    const tfs: Array<[string, number]> = [["15m", 900], ["30m", 1800], ["1h", 3600], ["2h", 7200]];
    const t0 = 1_750_000_000;
    const days = 20;
    const cells: CellSeries[] = [];
    for (const [tf, secs] of tfs) {
      const bars = Math.floor(days * 86_400 / secs);
      for (let k = 0; k < 12; k++) {
        // Per-bar vol scales with sqrt(bar length), as real returns do.
        const sd = 0.01 * Math.sqrt(secs / 3600);
        const logNet = noise(bars, rng, sd);
        cells.push({
          name: `C${k}/${tf}/hmm`, coin: `C${k}`, timeframe: tf, modelType: "hmm",
          times: Array.from({ length: bars }, (_, i) => t0 + i * secs),
          logNet, roiRebuilt: 0, roiStored: 0,
        });
      }
    }
    if (edge > 0) {
      // Plant one real edge on the 1h clock.
      const secs = 3600, bars = Math.floor(days * 86_400 / secs);
      cells.push({
        name: "REAL/1h/hsmm", coin: "REAL", timeframe: "1h", modelType: "hsmm",
        times: Array.from({ length: bars }, (_, i) => t0 + i * secs),
        logNet: noise(bars, rng, 0.01).map((v) => v + edge), roiRebuilt: 0, roiStored: 0,
      });
    }
    return { generatedAt: 0, sweepGeneratedAt: 0, costBps: 4.5, days, cells };
  }

  test("a sweep of pure noise is judged as pure noise", () => {
    const v = judgeSweep(fakeSweep(0, 401), { bootstraps: 400, seed: 402 });
    expect(v.cells).toBe(48);
    expect(v.spa.pConsistent).toBeGreaterThan(0.05);
    expect(v.rc.pValue).toBeGreaterThan(0.05);
    expect(v.rw.rejected.length).toBe(0);
    expect(v.dsr).toBeLessThan(0.95);
    // The best of 48 noise cells still has a respectable-looking t-statistic.
    // That is the whole point: the raw number is not the evidence.
    expect(v.spa.tStat).toBeGreaterThan(1);
  });

  test("a real edge planted in the same sweep is still found", () => {
    const v = judgeSweep(fakeSweep(0.0035, 401), { bootstraps: 400, seed: 402 });
    expect(v.spa.best).toBe("REAL/1h/hsmm");
    expect(v.spa.pConsistent).toBeLessThan(0.05);
    expect(v.rw.rejected).toContain("REAL/1h/hsmm");
    expect(v.winner.name).toBe("REAL/1h/hsmm");
    expect(v.dsr).toBeGreaterThan(0.95);
  });
});
