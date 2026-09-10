import { expect, test, describe } from "bun:test";
import {
  runMcmc, ffbs, sampleGamma, sampleDirichlet, sampleInvGamma,
  posteriorStateMeans, posteriorSignal, posteriorMeanParams, DEFAULT_PRIORS,
} from "./mcmc";
import { fit, posteriors, makeRng, randn, type HmmParams } from "./hmm";

function toyModel(): HmmParams {
  return {
    K: 3, D: 1,
    pi: new Float64Array([0.5, 0.3, 0.2]),
    A: new Float64Array([0.9, 0.07, 0.03, 0.05, 0.9, 0.05, 0.03, 0.07, 0.9]),
    mu: new Float64Array([-1.5, 0.0, 1.5]),
    vari: new Float64Array([0.4, 0.4, 0.4]),
  };
}

function sample(p: HmmParams, T: number, seed: number) {
  const rng = makeRng(seed);
  const X = new Float64Array(T);
  const states = new Int32Array(T);
  let s = 0, u = rng(), acc = 0;
  for (let k = 0; k < p.K; k++) { acc += p.pi[k]; if (u <= acc) { s = k; break; } }
  for (let t = 0; t < T; t++) {
    if (t > 0) {
      u = rng(); acc = 0;
      for (let k = 0; k < p.K; k++) { acc += p.A[s * p.K + k]; if (u <= acc) { s = k; break; } }
    }
    states[t] = s;
    X[t] = p.mu[s] + Math.sqrt(p.vari[s]) * randn(rng);
  }
  return { X, states };
}

const mean = (a: number[]) => a.reduce((x, y) => x + y, 0) / a.length;

describe("random samplers", () => {
  test("Gamma(a,1) has mean a and variance a", () => {
    const rng = makeRng(7);
    for (const a of [0.5, 2, 9]) {
      const draws = Array.from({ length: 20000 }, () => sampleGamma(a, rng));
      const m = mean(draws);
      const v = mean(draws.map((x) => (x - m) ** 2));
      expect(Math.abs(m - a) / a).toBeLessThan(0.05);
      expect(Math.abs(v - a) / a).toBeLessThan(0.1);
    }
  });

  test("Gamma draws are strictly positive", () => {
    const rng = makeRng(8);
    for (let i = 0; i < 2000; i++) expect(sampleGamma(0.3, rng)).toBeGreaterThan(0);
  });

  test("Dirichlet draws are simplex points with the right mean", () => {
    const rng = makeRng(9);
    const draws = Array.from({ length: 8000 }, () => sampleDirichlet([2, 3, 5], rng));
    for (const d of draws.slice(0, 200)) {
      expect(d.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 8);
      for (const v of d) expect(v).toBeGreaterThanOrEqual(0);
    }
    // E[x_i] = alpha_i / sum(alpha)
    for (let i = 0; i < 3; i++) {
      expect(Math.abs(mean(draws.map((d) => d[i])) - [0.2, 0.3, 0.5][i])).toBeLessThan(0.02);
    }
  });

  test("InvGamma(a,b) has mean b/(a-1) for a > 1", () => {
    const rng = makeRng(10);
    const draws = Array.from({ length: 20000 }, () => sampleInvGamma(4, 6, rng));
    expect(Math.abs(mean(draws) - 6 / 3) / 2).toBeLessThan(0.06);
  });
});

describe("forward-filter backward-sample", () => {
  test("averaged FFBS draws converge to the smoothed posterior", () => {
    // The strongest check available: posteriors() is already validated against
    // brute-force enumeration, so matching it pins down the sampler.
    const p = toyModel();
    const T = 60;
    const { X } = sample(p, T, 42);
    const exact = posteriors(X, T, p).gamma;

    const rng = makeRng(5);
    const counts = new Float64Array(T * p.K);
    const draws = 8000;
    for (let i = 0; i < draws; i++) {
      const path = ffbs(X, T, p, rng);
      for (let t = 0; t < T; t++) counts[t * p.K + path[t]] += 1;
    }
    let maxErr = 0;
    for (let t = 0; t < T; t++) {
      for (let k = 0; k < p.K; k++) {
        maxErr = Math.max(maxErr, Math.abs(counts[t * p.K + k] / draws - exact[t * p.K + k]));
      }
    }
    expect(maxErr).toBeLessThan(0.03);
  });

  test("paths are valid state indices of the right length", () => {
    const p = toyModel();
    const { X } = sample(p, 200, 3);
    const path = ffbs(X, 200, p, makeRng(1));
    expect(path.length).toBe(200);
    for (const s of path) { expect(s).toBeGreaterThanOrEqual(0); expect(s).toBeLessThan(p.K); }
  });
});

describe("posterior inference", () => {
  const p = toyModel();
  const { X } = sample(p, 1500, 77);
  const em = fit(X, 1500, 1, { states: 3, restarts: 4, seed: 3 });
  const res = runMcmc(X, 1500, 1, { states: 3, iterations: 400, burnIn: 200, thin: 2, seed: 11, init: em.params });

  test("retains draws and reports a finite log-likelihood", () => {
    expect(res.samples.length).toBeGreaterThan(50);
    expect(Number.isFinite(res.meanLogLik)).toBe(true);
  });

  test("labels stay ordered, so the posterior is not label-switching mush", () => {
    for (const s of res.samples) {
      expect(s.mu[0]).toBeLessThanOrEqual(s.mu[1]);
      expect(s.mu[1]).toBeLessThanOrEqual(s.mu[2]);
    }
  });

  test("credible intervals cover the true state means", () => {
    const ints = posteriorStateMeans(res, 0, (v) => v, 0.95);
    for (let k = 0; k < 3; k++) {
      expect(ints[k].lower).toBeLessThan(p.mu[k] + 0.35);
      expect(ints[k].upper).toBeGreaterThan(p.mu[k] - 0.35);
    }
  });

  test("intervals are ordered and non-degenerate", () => {
    for (const i of posteriorStateMeans(res)) {
      expect(i.lower).toBeLessThanOrEqual(i.median);
      expect(i.median).toBeLessThanOrEqual(i.upper);
      expect(i.upper - i.lower).toBeGreaterThan(0);
    }
  });

  test("posterior mean parameters form a valid model", () => {
    const pm = posteriorMeanParams(res);
    for (let i = 0; i < pm.K; i++) {
      let row = 0;
      for (let j = 0; j < pm.K; j++) row += pm.A[i * pm.K + j];
      expect(row).toBeCloseTo(1, 6);
      for (let d = 0; d < pm.D; d++) expect(pm.vari[i * pm.D + d]).toBeGreaterThan(0);
    }
  });
});

describe("shrinkage", () => {
  test("a weakly evidenced state is pulled toward the prior mean", () => {
    // 400 bars of pure noise: there is no real 3-regime structure, so a strong
    // prior at zero should keep the fitted state means near zero, while a
    // near-flat prior lets them chase noise.
    const rng = makeRng(21);
    const T = 400;
    const X = new Float64Array(T);
    for (let t = 0; t < T; t++) X[t] = randn(rng);

    const spread = (kappa0: number) => {
      const r = runMcmc(X, T, 1, {
        states: 3, iterations: 300, burnIn: 150, thin: 2, seed: 5,
        priors: { ...DEFAULT_PRIORS, kappa0 },
      });
      const ints = posteriorStateMeans(r, 0);
      return ints[2].mean - ints[0].mean;
    };
    // Heavy shrinkage must not widen the gap relative to a near-flat prior.
    expect(spread(200)).toBeLessThan(spread(0.01));
  });
});

describe("signal posterior", () => {
  const p = toyModel();
  const { X } = sample(p, 1200, 5);
  const res = runMcmc(X, 1200, 1, { states: 3, iterations: 300, burnIn: 150, thin: 2, seed: 2 });

  test("all mass on the bullish state gives a positive signal", () => {
    const s = posteriorSignal(res, [0, 0, 1], (v) => v, 0);
    expect(s.pPositive).toBeGreaterThan(0.9);
    expect(s.median).toBeGreaterThan(0);
  });

  test("the hurdle is compared against edge x hold, not one bar's edge", () => {
    const s = posteriorSignal(res, [0, 0, 1], (v) => v, 0);
    // A persistent state holds for many bars, so the trade-level edge must
    // exceed the per-bar edge by roughly the dwell time.
    expect(s.medianHold).toBeGreaterThan(2);
    expect(s.median).toBeGreaterThan(s.perBar.median);
    expect(s.median / s.perBar.median).toBeCloseTo(s.medianHold, 0);
  });

  test("an explicit holdBars overrides the posterior dwell", () => {
    const a = posteriorSignal(res, [0, 0, 1], (v) => v, 0, 0.9, { holdBars: 1 });
    const b = posteriorSignal(res, [0, 0, 1], (v) => v, 0, 0.9, { holdBars: 10 });
    expect(a.medianHold).toBe(1);
    expect(b.median / a.median).toBeCloseTo(10, 4);
  });

  test("an unreachable cost makes P(above cost) zero", () => {
    const s = posteriorSignal(res, [0, 0, 1], (v) => v, 1e9);
    expect(s.pAboveCost).toBe(0);
  });

  test("a higher cost hurdle never raises P(above cost)", () => {
    const lo = posteriorSignal(res, [0.2, 0.3, 0.5], (v) => v, 0);
    const hi = posteriorSignal(res, [0.2, 0.3, 0.5], (v) => v, 0.5);
    expect(hi.pAboveCost).toBeLessThanOrEqual(lo.pAboveCost);
  });
});
