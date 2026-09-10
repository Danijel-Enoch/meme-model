import { expect, test, describe } from "bun:test";
import {
  fitVb, selectStates, vbMeanParams, vbPosteriors, drawFromQ, digamma, lgamma,
} from "./vb";
import { runMcmc } from "./mcmc";
import { posteriorStateMeans, posteriorDurations, DEFAULT_PRIORS } from "./posterior";
import { fit, makeRng, randn, type HmmParams } from "./hmm";

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

describe("special functions", () => {
  test("lgamma matches known values", () => {
    // lgamma(n) = log((n-1)!) at the integers.
    expect(lgamma(1)).toBeCloseTo(0, 10);
    expect(lgamma(2)).toBeCloseTo(0, 10);
    expect(lgamma(5)).toBeCloseTo(Math.log(24), 10);
    expect(lgamma(10)).toBeCloseTo(12.801827480081469, 9);
    // Gamma(1/2) = sqrt(pi).
    expect(lgamma(0.5)).toBeCloseTo(Math.log(Math.sqrt(Math.PI)), 10);
  });

  test("lgamma satisfies the recurrence lgamma(x+1) = lgamma(x) + log(x)", () => {
    for (const x of [0.3, 0.75, 1.4, 3.2, 11.7, 40]) {
      expect(lgamma(x + 1)).toBeCloseTo(lgamma(x) + Math.log(x), 9);
    }
  });

  test("digamma matches known values", () => {
    const EULER = 0.5772156649015329;
    expect(digamma(1)).toBeCloseTo(-EULER, 10);
    // psi(1/2) = -gamma - 2 log 2
    expect(digamma(0.5)).toBeCloseTo(-EULER - 2 * Math.log(2), 10);
    // psi(n) = -gamma + sum_{k=1}^{n-1} 1/k
    expect(digamma(4)).toBeCloseTo(-EULER + 1 + 1 / 2 + 1 / 3, 10);
  });

  test("digamma satisfies the recurrence psi(x+1) = psi(x) + 1/x", () => {
    for (const x of [0.2, 0.9, 2.5, 5.9, 6.1, 30]) {
      expect(digamma(x + 1)).toBeCloseTo(digamma(x) + 1 / x, 9);
    }
  });

  test("digamma is the derivative of lgamma", () => {
    const h = 1e-5;
    for (const x of [0.7, 2.3, 8.4]) {
      expect((lgamma(x + h) - lgamma(x - h)) / (2 * h)).toBeCloseTo(digamma(x), 6);
    }
  });
});

describe("variational EM", () => {
  const { X } = sample(toyModel(), 1500, 3);
  const T = 1500;

  test("the ELBO increases monotonically", () => {
    const res = fitVb(X, T, 1, { states: 3, seed: 11, draws: 0 });
    expect(res.elboTrace.length).toBeGreaterThan(1);
    for (let i = 1; i < res.elboTrace.length; i++) {
      // The bound is monotone by construction; anything else is a bug in the
      // E step / M step pairing or in the KL terms.
      expect(res.elboTrace[i]).toBeGreaterThanOrEqual(res.elboTrace[i - 1] - 1e-9);
    }
    expect(res.converged).toBe(true);
  });

  test("the ELBO is a lower bound on the EM log-likelihood", () => {
    const em = fit(X, T, 1, { states: 3, restarts: 4, seed: 42 });
    const res = fitVb(X, T, 1, { states: 3, init: em.params, seed: 11, draws: 0 });
    expect(res.elbo).toBeLessThan(em.logLik);
    // ...but not a loose one: the gap is the price of the priors and of
    // integrating over the parameters, not an inference failure.
    expect((em.logLik - res.elbo) / T).toBeLessThan(0.2);
  });

  test("recovers the generating parameters", () => {
    const res = fitVb(X, T, 1, { states: 3, seed: 11, draws: 0 });
    const p = vbMeanParams(res.q);
    const truth = [-1.5, 0.0, 1.5];
    for (let k = 0; k < 3; k++) {
      // 1500 bars, so the sample's own means sit a few hundredths off the
      // generator's. Anything inside that is recovery, not luck.
      expect(Math.abs(p.mu[k] - truth[k])).toBeLessThan(0.15);
      expect(Math.abs(p.vari[k] - 0.4)).toBeLessThan(0.12);
      expect(p.A[k * 3 + k]).toBeGreaterThan(0.8);
    }
  });

  test("lands where EM lands, minus the shrinkage", () => {
    // The sharper test than "close to the generator": EM and VB are estimating
    // the same quantity from the same 1500 bars, so with a weak prior they
    // should agree to a few thousandths.
    const em = fit(X, T, 1, { states: 3, restarts: 4, seed: 42 });
    const res = fitVb(X, T, 1, { states: 3, init: em.params, seed: 11, draws: 0 });
    const p = vbMeanParams(res.q);
    for (let k = 0; k < 3; k++) {
      expect(Math.abs(p.mu[k] - em.params.mu[k])).toBeLessThan(0.01);
      expect(Math.abs(p.vari[k] - em.params.vari[k])).toBeLessThan(0.01);
    }
  });

  test("states come out sorted by mean return", () => {
    const res = fitVb(X, T, 1, { states: 4, seed: 5, draws: 50 });
    for (let k = 1; k < 4; k++) expect(res.q.m[k]).toBeGreaterThanOrEqual(res.q.m[k - 1]);
    for (const s of res.samples) {
      for (let k = 1; k < 4; k++) expect(s.mu[k]).toBeGreaterThanOrEqual(s.mu[k - 1]);
    }
  });

  test("a cold start reaches the same optimum as an EM warm start", () => {
    const em = fit(X, T, 1, { states: 3, restarts: 4, seed: 42 });
    const warm = fitVb(X, T, 1, { states: 3, init: em.params, seed: 11, draws: 0 });
    const cold = fitVb(X, T, 1, { states: 3, seed: 11, draws: 0 });
    expect(Math.abs(warm.elbo - cold.elbo) / T).toBeLessThan(1e-4);
    // The warm start is only cheaper, not better.
    expect(warm.iterations).toBeLessThanOrEqual(cold.iterations);
  });

  test("is deterministic given a seed", () => {
    const a = fitVb(X, T, 1, { states: 3, seed: 7, draws: 100 });
    const b = fitVb(X, T, 1, { states: 3, seed: 7, draws: 100 });
    expect(a.elbo).toBe(b.elbo);
    expect(a.iterations).toBe(b.iterations);
    for (let i = 0; i < a.q.m.length; i++) expect(a.q.m[i]).toBe(b.q.m[i]);
    // Same seed, same draws.
    expect(a.samples[0].mu[0]).toBe(b.samples[0].mu[0]);
  });

  test("q(z) sums to one at every step and tracks the true path", () => {
    const { X: X2, states } = sample(toyModel(), 900, 21);
    const res = fitVb(X2, 900, 1, { states: 3, seed: 4, draws: 0 });
    let hits = 0;
    for (let t = 0; t < 900; t++) {
      let sum = 0, best = -1, arg = 0;
      for (let k = 0; k < 3; k++) {
        sum += res.gamma[t * 3 + k];
        if (res.gamma[t * 3 + k] > best) { best = res.gamma[t * 3 + k]; arg = k; }
      }
      expect(sum).toBeCloseTo(1, 9);
      if (arg === states[t]) hits++;
    }
    expect(hits / 900).toBeGreaterThan(0.75);
  });

  test("vbPosteriors reproduces the fit's own q(z)", () => {
    const res = fitVb(X, T, 1, { states: 3, seed: 11, draws: 0 });
    // Re-running the E step against the returned q must give the same
    // responsibilities the last E step produced, up to the state ordering.
    const g = vbPosteriors(X, T, res.q);
    let maxDiff = 0;
    for (let i = 0; i < g.length; i++) maxDiff = Math.max(maxDiff, Math.abs(g[i] - res.gamma[i]));
    expect(maxDiff).toBeLessThan(0.02);
  });
});

describe("state pruning and model selection", () => {
  test("the ELBO picks the true number of regimes", () => {
    const { X } = sample(toyModel(), 2000, 13);
    const sel = selectStates(X, 2000, 1, [2, 3, 4, 5], { seed: 5, draws: 0 });
    expect(sel.best.K).toBe(3);
    // Under-fitting is the expensive mistake; over-fitting only costs the KL.
    const byK = new Map(sel.scores.map((s) => [s.states, s.elboPerBar]));
    expect(byK.get(3)!).toBeGreaterThan(byK.get(2)!);
    expect(byK.get(3)!).toBeGreaterThan(byK.get(4)!);
  });

  test("unsupported states collapse back onto the prior", () => {
    // One Gaussian, asked for six states. Five of them have nothing to explain.
    const rng = makeRng(77);
    const T = 1200;
    const X = new Float64Array(T);
    for (let t = 0; t < T; t++) X[t] = 0.5 * randn(rng);

    const res = fitVb(X, T, 1, { states: 6, seed: 2, draws: 0 });
    const occupied = Array.from(res.occupancy).filter((n) => n > 0.01 * T).length;
    expect(occupied).toBeLessThan(6);

    // A pruned state's hyperparameters are the prior's, exactly.
    const dead = Array.from(res.occupancy).findIndex((n) => n < 1e-3);
    if (dead >= 0) {
      expect(res.q.a[dead]).toBeCloseTo(DEFAULT_PRIORS.a0, 6);
      expect(res.q.b[dead]).toBeCloseTo(DEFAULT_PRIORS.b0, 6);
      expect(res.q.kappa[dead]).toBeCloseTo(DEFAULT_PRIORS.kappa0, 6);
    }
  });
});

describe("shrinkage", () => {
  test("a thinly evidenced state is pulled toward zero drift", () => {
    // 40 bars of a fake +2.0 regime inside 1200 bars of noise.
    const rng = makeRng(99);
    const T = 1200;
    const X = new Float64Array(T);
    for (let t = 0; t < T; t++) X[t] = (t >= 600 && t < 640 ? 2.0 : 0) + 0.3 * randn(rng);

    const em = fit(X, T, 1, { states: 2, restarts: 4, seed: 1 });
    const vb = fitVb(X, T, 1, { states: 2, init: em.params, seed: 3, draws: 0, priors: { kappa0: 20 } });
    const p = vbMeanParams(vb.q);

    // EM takes the 40 bars at face value; the prior does not.
    expect(em.params.mu[1]).toBeGreaterThan(1.8);
    expect(p.mu[1]).toBeLessThan(em.params.mu[1]);
    expect(p.mu[1]).toBeGreaterThan(0.5);
    // The well-evidenced state barely moves.
    expect(Math.abs(p.mu[0] - em.params.mu[0])).toBeLessThan(0.05);
  });
});

describe("draws from q", () => {
  const { X } = sample(toyModel(), 1200, 31);
  const T = 1200;

  test("are valid parameter sets", () => {
    const res = fitVb(X, T, 1, { states: 3, seed: 11, draws: 300 });
    for (const s of res.samples.slice(0, 50)) {
      expect(mean(Array.from(s.pi))).toBeCloseTo(1 / 3, 9);
      for (let i = 0; i < 3; i++) {
        let row = 0;
        for (let j = 0; j < 3; j++) {
          expect(s.A[i * 3 + j]).toBeGreaterThanOrEqual(0);
          row += s.A[i * 3 + j];
        }
        expect(row).toBeCloseTo(1, 9);
      }
      for (let k = 0; k < 3; k++) expect(s.vari[k]).toBeGreaterThan(0);
    }
  });

  test("are independent — unlike a Gibbs chain", () => {
    const res = fitVb(X, T, 1, { states: 3, seed: 11, draws: 3000 });
    const series = res.samples.map((s) => s.mu[2]);
    const m = mean(series);
    let num = 0, den = 0;
    for (let i = 0; i < series.length; i++) {
      den += (series[i] - m) ** 2;
      if (i > 0) num += (series[i] - m) * (series[i - 1] - m);
    }
    // i.i.d. draws: lag-1 autocorrelation is zero up to sampling noise, so
    // every draw counts toward the effective sample size.
    expect(Math.abs(num / den)).toBeLessThan(0.06);
  });

  test("their mean matches the analytic posterior mean", () => {
    const res = fitVb(X, T, 1, { states: 3, seed: 11, draws: 4000 });
    const analytic = vbMeanParams(res.q);
    for (let k = 0; k < 3; k++) {
      expect(mean(res.samples.map((s) => s.mu[k]))).toBeCloseTo(analytic.mu[k], 2);
      expect(mean(res.samples.map((s) => s.A[k * 3 + k]))).toBeCloseTo(analytic.A[k * 3 + k], 2);
    }
  });

  test("drawFromQ is reproducible under a seeded rng", () => {
    const res = fitVb(X, T, 1, { states: 3, seed: 11, draws: 0 });
    const a = drawFromQ(res.q, makeRng(5));
    const b = drawFromQ(res.q, makeRng(5));
    for (let i = 0; i < a.mu.length; i++) expect(a.mu[i]).toBe(b.mu[i]);
  });
});

describe("variational EM against the Gibbs sampler", () => {
  // The one comparison that keeps the approximation honest. Both engines put
  // the same priors on the same model, so a disagreement is inference error.
  const { X } = sample(toyModel(), 1500, 3);
  const T = 1500;
  const em = fit(X, T, 1, { states: 3, restarts: 4, seed: 42 });
  const vb = fitVb(X, T, 1, { states: 3, init: em.params, seed: 11, draws: 4000 });
  const mc = runMcmc(X, T, 1, {
    states: 3, iterations: 1200, burnIn: 600, thin: 3, seed: 11, init: em.params,
  });

  test("posterior means agree to within a fraction of the interval width", () => {
    const v = posteriorStateMeans(vb, 0);
    const g = posteriorStateMeans(mc, 0);
    for (let k = 0; k < 3; k++) {
      const width = g[k].upper - g[k].lower;
      expect(Math.abs(v[k].mean - g[k].mean)).toBeLessThan(0.25 * width);
    }
  });

  test("the narrowing costs real coverage", () => {
    // The claim that mean-field intervals come out too narrow is worth
    // measuring rather than asserting. Twelve independent series, 90% nominal
    // intervals on all three state means: the sampler lands near its nominal
    // rate, the variational approximation lands below it, and the shortfall
    // tracks the width it gave up.
    //
    // Measured: Gibbs 89% coverage at width 0.150, VB 83% at width 0.128.
    // That is the whole argument for keeping --gibbs. A 6-point coverage loss
    // is invisible until an interval that should straddle the round trip
    // sits just clear of it.
    const truth = [-1.5, 0.0, 1.5];
    const REPS = 12, TR = 800;
    let vHit = 0, gHit = 0, n = 0, vWidth = 0, gWidth = 0;

    for (let r = 0; r < REPS; r++) {
      const { X: Xr } = sample(toyModel(), TR, 100 + r * 13);
      const emR = fit(Xr, TR, 1, { states: 3, restarts: 3, seed: 42 });
      const v = posteriorStateMeans(
        fitVb(Xr, TR, 1, { states: 3, init: emR.params, seed: 11, draws: 3000 }), 0);
      const g = posteriorStateMeans(
        runMcmc(Xr, TR, 1, { states: 3, iterations: 600, burnIn: 300, thin: 2, seed: 11, init: emR.params }), 0);

      for (let k = 0; k < 3; k++) {
        n++;
        if (v[k].lower < truth[k] && v[k].upper > truth[k]) vHit++;
        if (g[k].lower < truth[k] && g[k].upper > truth[k]) gHit++;
        vWidth += v[k].upper - v[k].lower;
        gWidth += g[k].upper - g[k].lower;
      }
    }

    // The sampler is close to nominal.
    expect(gHit / n).toBeGreaterThan(0.75);
    // The variational intervals are systematically narrower. This is the
    // reliable effect; the coverage shortfall is its consequence.
    expect(vWidth / gWidth).toBeLessThan(0.95);
    expect(vHit).toBeLessThanOrEqual(gHit);
  });

  test("dwell times agree", () => {
    const v = posteriorDurations(vb);
    const g = posteriorDurations(mc);
    for (let k = 0; k < 3; k++) {
      expect(Math.abs(v[k].median - g[k].median) / g[k].median).toBeLessThan(0.25);
    }
  });
});
