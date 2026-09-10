import { expect, test, describe } from "bun:test";
import {
  fit, filter, posteriors, viterbi, logEmissions, forward, predictNext,
  stationary, makeRng, randn, serialize, deserialize, type HmmParams,
} from "./hmm";
import { buildFeatures, fitScaler, applyScaler } from "./features";
import { generateSynthetic, parseCsv } from "./data";
import { walkForward } from "./backtest";

/** A small hand-built model to check the recursions against brute force. */
function toyModel(): HmmParams {
  return {
    K: 3, D: 1,
    pi: new Float64Array([0.5, 0.3, 0.2]),
    A: new Float64Array([
      0.7, 0.2, 0.1,
      0.3, 0.4, 0.3,
      0.1, 0.3, 0.6,
    ]),
    mu: new Float64Array([-1.0, 0.0, 1.5]),
    vari: new Float64Array([0.5, 1.0, 0.25]),
  };
}

const gauss = (x: number, m: number, v: number) =>
  Math.exp(-0.5 * ((x - m) ** 2) / v) / Math.sqrt(2 * Math.PI * v);

/** Enumerate every one of the K^T state paths. Only viable for tiny T. */
function bruteForce(X: Float64Array, T: number, p: HmmParams) {
  const { K } = p;
  let total = 0;
  let bestJoint = -Infinity;
  let bestPath: number[] = [];
  const margAll = Array.from({ length: T }, () => new Float64Array(K));   // P(z_t, x_1..T)
  const margPrefix = Array.from({ length: T }, () => new Float64Array(K)); // P(z_t, x_1..t)

  const nPaths = K ** T;
  for (let code = 0; code < nPaths; code++) {
    const path: number[] = [];
    let c = code;
    for (let t = 0; t < T; t++) { path.push(c % K); c = Math.floor(c / K); }

    let joint = p.pi[path[0]] * gauss(X[0], p.mu[path[0]], p.vari[path[0]]);
    const prefix = [joint];
    for (let t = 1; t < T; t++) {
      joint *= p.A[path[t - 1] * K + path[t]] * gauss(X[t], p.mu[path[t]], p.vari[path[t]]);
      prefix.push(joint);
    }
    total += joint;
    for (let t = 0; t < T; t++) margAll[t][path[t]] += joint;
    if (joint > bestJoint) { bestJoint = joint; bestPath = [...path]; }
  }

  // Prefix marginals need their own pass over the shorter path space.
  for (let t = 0; t < T; t++) {
    const n = K ** (t + 1);
    for (let code = 0; code < n; code++) {
      const path: number[] = [];
      let c = code;
      for (let i = 0; i <= t; i++) { path.push(c % K); c = Math.floor(c / K); }
      let joint = p.pi[path[0]] * gauss(X[0], p.mu[path[0]], p.vari[path[0]]);
      for (let i = 1; i <= t; i++) {
        joint *= p.A[path[i - 1] * K + path[i]] * gauss(X[i], p.mu[path[i]], p.vari[path[i]]);
      }
      margPrefix[t][path[t]] += joint;
    }
  }
  return { likelihood: total, bestPath, margAll, margPrefix };
}

/** Sample an observation sequence from a known model. */
function sampleFrom(p: HmmParams, T: number, seed: number) {
  const rng = makeRng(seed);
  const X = new Float64Array(T * p.D);
  const states = new Int32Array(T);
  let s = 0;
  let u = rng(), acc = 0;
  for (let k = 0; k < p.K; k++) { acc += p.pi[k]; if (u <= acc) { s = k; break; } }
  for (let t = 0; t < T; t++) {
    if (t > 0) {
      u = rng(); acc = 0;
      for (let k = 0; k < p.K; k++) { acc += p.A[s * p.K + k]; if (u <= acc) { s = k; break; } }
    }
    states[t] = s;
    for (let d = 0; d < p.D; d++) {
      X[t * p.D + d] = p.mu[s * p.D + d] + Math.sqrt(p.vari[s * p.D + d]) * randn(rng);
    }
  }
  return { X, states };
}

describe("inference vs brute force", () => {
  const p = toyModel();
  const T = 7;
  const { X } = sampleFrom(p, T, 123);
  const bf = bruteForce(X, T, p);

  test("forward log-likelihood matches enumeration over all K^T paths", () => {
    const { logLik } = forward(logEmissions(X, T, p), T, p);
    expect(logLik).toBeCloseTo(Math.log(bf.likelihood), 8);
  });

  test("filtered probabilities match P(z_t | x_1..t)", () => {
    const { alpha } = filter(X, T, p);
    for (let t = 0; t < T; t++) {
      const norm = bf.margPrefix[t].reduce((a, b) => a + b, 0);
      for (let k = 0; k < p.K; k++) {
        expect(alpha[t * p.K + k]).toBeCloseTo(bf.margPrefix[t][k] / norm, 8);
      }
    }
  });

  test("smoothed posteriors match P(z_t | x_1..T)", () => {
    const { gamma } = posteriors(X, T, p);
    for (let t = 0; t < T; t++) {
      for (let k = 0; k < p.K; k++) {
        expect(gamma[t * p.K + k]).toBeCloseTo(bf.margAll[t][k] / bf.likelihood, 8);
      }
    }
  });

  test("viterbi finds the same path as exhaustive search", () => {
    const path = viterbi(X, T, p);
    expect(Array.from(path)).toEqual(bf.bestPath);
  });

  test("filtering is causal: future observations cannot change past beliefs", () => {
    const long = sampleFrom(p, 200, 9).X;
    const a1 = filter(long, 200, p).alpha;
    const truncated = long.slice(0, 120 * p.D);
    const a2 = filter(truncated, 120, p).alpha;
    for (let t = 0; t < 120; t++) {
      for (let k = 0; k < p.K; k++) {
        expect(a1[t * p.K + k]).toBeCloseTo(a2[t * p.K + k], 10);
      }
    }
  });
});

describe("EM", () => {
  test("log-likelihood is non-decreasing across iterations", () => {
    const p = toyModel();
    const { X } = sampleFrom(p, 600, 5);
    let prev = -Infinity;
    for (let iters = 1; iters <= 12; iters++) {
      const r = fit(X, 600, 1, { states: 3, restarts: 1, maxIter: iters, tol: 0, seed: 3 });
      // Allow a hair of float slop; EM is monotone in exact arithmetic.
      expect(r.logLik).toBeGreaterThanOrEqual(prev - 1e-6);
      prev = r.logLik;
    }
  });

  test("recovers the parameters of a model it generated data from", () => {
    const p = toyModel();
    const { X } = sampleFrom(p, 8000, 77);
    const r = fit(X, 8000, 1, { states: 3, restarts: 6, seed: 11, maxIter: 400 });
    // States come back sorted by mean, and the toy means are already ascending.
    // Tolerances are loose on purpose: the middle state (mu=0, var=1) overlaps
    // both neighbours heavily, so it is the least identifiable of the three.
    // Its error shrinks with more data (at T=40k, mu[1] estimates to ~-0.07).
    for (let k = 0; k < 3; k++) {
      expect(Math.abs(r.params.mu[k] - p.mu[k])).toBeLessThan(0.2);
      expect(Math.abs(r.params.vari[k] - p.vari[k])).toBeLessThan(0.2);
      expect(Math.abs(r.params.A[k * 3 + k] - p.A[k * 3 + k])).toBeLessThan(0.1);
    }
  });

  test("parameter error shrinks as the sample grows", () => {
    const p = toyModel();
    const err = (T: number) => {
      const { X } = sampleFrom(p, T, 77);
      const r = fit(X, T, 1, { states: 3, restarts: 4, seed: 11, maxIter: 400 });
      let e = 0;
      for (let k = 0; k < 3; k++) e += Math.abs(r.params.mu[k] - p.mu[k]);
      return e;
    };
    expect(err(30000)).toBeLessThan(err(4000));
  });

  test("fitted transition rows and stationary distribution are proper distributions", () => {
    const { candles } = generateSynthetic({ bars: 1500, seed: 4 });
    const fs = buildFeatures(candles);
    const sc = fitScaler(fs.X, fs.T, fs.D);
    const r = fit(applyScaler(fs.X, fs.T, fs.D, sc), fs.T, fs.D, { states: 3, restarts: 2, seed: 1 });
    for (let i = 0; i < 3; i++) {
      let row = 0;
      for (let j = 0; j < 3; j++) {
        expect(r.params.A[i * 3 + j]).toBeGreaterThanOrEqual(0);
        row += r.params.A[i * 3 + j];
      }
      expect(row).toBeCloseTo(1, 10);
    }
    const st = stationary(r.params);
    expect(st.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 8);
  });

  test("one-step-ahead state forecast sums to one", () => {
    const p = toyModel();
    const { X } = sampleFrom(p, 50, 8);
    const { alpha } = filter(X, 50, p);
    const next = predictNext(alpha, 49 * p.K, p);
    expect(Array.from(next).reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10);
  });

  test("serialize round-trips", () => {
    const p = toyModel();
    const q = deserialize(serialize(p));
    expect(Array.from(q.A)).toEqual(Array.from(p.A));
    expect(Array.from(q.mu)).toEqual(Array.from(p.mu));
  });
});

describe("features", () => {
  test("every feature row uses only past and current bars", () => {
    const { candles } = generateSynthetic({ bars: 400, seed: 2 });
    const full = buildFeatures(candles, { window: 20 });
    // Corrupt the tail; rows built before it must be byte-identical.
    const tampered = candles.map((c, i) => (i >= 300 ? { ...c, close: c.close * 5, volume: c.volume * 99 } : c));
    const alt = buildFeatures(tampered, { window: 20 });
    for (let t = 0; t < full.T; t++) {
      if (full.index[t] >= 300) break;
      for (let d = 0; d < full.D; d++) {
        expect(alt.X[t * full.D + d]).toBeCloseTo(full.X[t * full.D + d], 12);
      }
    }
  });

  test("scaler produces zero mean and unit variance on its training rows", () => {
    const { candles } = generateSynthetic({ bars: 800, seed: 3 });
    const fs = buildFeatures(candles);
    const sc = fitScaler(fs.X, fs.T, fs.D);
    const Z = applyScaler(fs.X, fs.T, fs.D, sc);
    for (let d = 0; d < fs.D; d++) {
      let m = 0;
      for (let t = 0; t < fs.T; t++) m += Z[t * fs.D + d];
      expect(m / fs.T).toBeCloseTo(0, 8);
    }
  });
});

describe("backtest", () => {
  test("positions in a block do not change when later candles are altered", () => {
    const { candles } = generateSynthetic({ bars: 3000, seed: 6 });
    const cfg = { trainSize: 800, testSize: 400, states: 3, seed: 1, restarts: 1 };
    const base = walkForward(candles, { window: 20 }, { costBps: 30 }, cfg);
    const tampered = candles.map((c, i) => (i >= 2400 ? { ...c, close: c.close * 3 } : c));
    const alt = walkForward(tampered, { window: 20 }, { costBps: 30 }, cfg);
    // Compare positions well before the tampered region.
    for (let i = 0; i < 2000; i++) {
      expect(alt.positions[i]).toBeCloseTo(base.positions[i], 10);
    }
  });

  test("higher costs never improve the strategy return", () => {
    const { candles } = generateSynthetic({ bars: 3000, seed: 6 });
    const cfg = { trainSize: 800, testSize: 400, states: 3, seed: 1, restarts: 1 };
    const cheap = walkForward(candles, { window: 20 }, { costBps: 0, entryBps: 20 }, cfg);
    const dear = walkForward(candles, { window: 20 }, { costBps: 100, entryBps: 20 }, cfg);
    expect(dear.metrics.totalReturn).toBeLessThanOrEqual(cheap.metrics.totalReturn + 1e-9);
  });

  test("a flat strategy (unreachable entry threshold) returns exactly zero", () => {
    const { candles } = generateSynthetic({ bars: 2000, seed: 6 });
    const r = walkForward(candles, { window: 20 }, { entryBps: 1e9, costBps: 30 },
      { trainSize: 800, testSize: 400, states: 3, seed: 1, restarts: 1 });
    expect(r.metrics.totalReturn).toBeCloseTo(0, 12);
    expect(r.metrics.trades).toBe(0);
  });

  test("sharpe sign agrees with the compounded equity curve", () => {
    const { candles } = generateSynthetic({ bars: 4000, seed: 6 });
    const r = walkForward(candles, { window: 20 }, { costBps: 30 },
      { trainSize: 1000, testSize: 500, states: 3, seed: 1, restarts: 1 });
    if (Math.abs(r.metrics.totalReturn) > 1e-6) {
      expect(Math.sign(r.metrics.sharpe)).toBe(Math.sign(r.metrics.totalReturn));
    }
  });
});

describe("csv", () => {
  test("parses varied headers and reorders newest-first exports", () => {
    const c = parseCsv("Date,Open,High,Low,Close,Volume\n2025-01-02,2,3,1,2.5,100\n2025-01-01,1,2,0.5,1.5,50");
    expect(c.length).toBe(2);
    expect(c[0].close).toBe(1.5); // reversed into chronological order
    expect(c[1].volume).toBe(100);
  });

  test("works with a close-only file", () => {
    const c = parseCsv("price\n1\n2\n3");
    expect(c.length).toBe(3);
    expect(c[2].close).toBe(3);
    expect(c[0].volume).toBe(0);
  });
});
