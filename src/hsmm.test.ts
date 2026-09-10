import { expect, test, describe } from "bun:test";
import {
  fitHsmm, filterHsmm, viterbiHsmm, fromHmm, expectedDurations,
  serializeHsmm, deserializeHsmm, type HsmmParams,
} from "./hsmm";
import { fit, filter, makeRng, randn } from "./hmm";

function toyHsmm(): HsmmParams {
  const K = 3, maxDuration = 4;
  // Durations deliberately non-geometric: state 0 favours length 3.
  const dur = new Float64Array([
    0.1, 0.2, 0.6, 0.1,
    0.5, 0.3, 0.1, 0.1,
    0.2, 0.2, 0.2, 0.4,
  ]);
  return {
    K, D: 1, maxDuration,
    pi: new Float64Array([0.5, 0.3, 0.2]),
    A: new Float64Array([0, 0.6, 0.4, 0.7, 0, 0.3, 0.5, 0.5, 0]),
    mu: new Float64Array([-1.0, 0.0, 1.5]),
    vari: new Float64Array([0.5, 1.0, 0.25]),
    dur,
  };
}

const gauss = (x: number, m: number, v: number) =>
  Math.exp(-0.5 * ((x - m) ** 2) / v) / Math.sqrt(2 * Math.PI * v);

/**
 * Enumerate every segmentation of [0,T) into consecutive runs of length
 * 1..maxDuration with no two adjacent runs sharing a state, and sum the joint
 * probability. This is the definition the forward recursion has to reproduce.
 */
function bruteForceHsmm(X: Float64Array, T: number, p: HsmmParams) {
  const { K, maxDuration } = p;
  let total = 0;
  let bestJoint = -Infinity;
  let bestPath: number[] = [];

  const walk = (pos: number, prevState: number, acc: number, path: number[]) => {
    if (pos === T) {
      total += acc;
      if (acc > bestJoint) { bestJoint = acc; bestPath = [...path]; }
      return;
    }
    for (let j = 0; j < K; j++) {
      if (j === prevState) continue; // self-transitions are forbidden
      for (let d = 1; d <= Math.min(maxDuration, T - pos); d++) {
        const pd = p.dur[j * maxDuration + (d - 1)];
        if (pd <= 0) continue;
        let seg = 1;
        for (let t = pos; t < pos + d; t++) seg *= gauss(X[t], p.mu[j], p.vari[j]);
        const trans = prevState < 0 ? p.pi[j] : p.A[prevState * K + j];
        if (trans <= 0) continue;
        walk(pos + d, j, acc * trans * pd * seg, [...path, ...Array(d).fill(j)]);
      }
    }
  };
  walk(0, -1, 1, []);
  return { likelihood: total, bestPath };
}

function sampleHsmm(p: HsmmParams, T: number, seed: number) {
  const rng = makeRng(seed);
  const X = new Float64Array(T);
  const states = new Int32Array(T);
  let t = 0, prev = -1;
  while (t < T) {
    let j = 0;
    const u = rng();
    let acc = 0;
    if (prev < 0) {
      for (let k = 0; k < p.K; k++) { acc += p.pi[k]; if (u <= acc) { j = k; break; } }
    } else {
      for (let k = 0; k < p.K; k++) { acc += p.A[prev * p.K + k]; if (u <= acc) { j = k; break; } }
    }
    const ud = rng();
    let da = 0, d = 1;
    for (let dd = 1; dd <= p.maxDuration; dd++) {
      da += p.dur[j * p.maxDuration + (dd - 1)];
      if (ud <= da) { d = dd; break; }
    }
    for (let k = 0; k < d && t < T; k++, t++) {
      states[t] = j;
      X[t] = p.mu[j] + Math.sqrt(p.vari[j]) * randn(rng);
    }
    prev = j;
  }
  return { X, states };
}

describe("HSMM inference vs brute force", () => {
  const p = toyHsmm();
  const T = 7;
  const { X } = sampleHsmm(p, T, 321);
  const bf = bruteForceHsmm(X, T, p);

  test("forward log-likelihood matches enumeration of all segmentations", () => {
    const { logLik } = filterHsmm(X, T, p);
    expect(logLik).toBeCloseTo(Math.log(bf.likelihood), 8);
  });

  test("segmental viterbi finds the same segmentation as exhaustive search", () => {
    expect(Array.from(viterbiHsmm(X, T, p))).toEqual(bf.bestPath);
  });

  test("filtered and predicted distributions are proper", () => {
    const { stateProb, nextProb } = filterHsmm(X, T, p);
    for (let t = 0; t < T; t++) {
      let a = 0, b = 0;
      for (let k = 0; k < p.K; k++) { a += stateProb[t * p.K + k]; b += nextProb[t * p.K + k]; }
      expect(a).toBeCloseTo(1, 8);
      expect(b).toBeCloseTo(1, 8);
    }
  });

  test("filtering is causal: later bars cannot change earlier beliefs", () => {
    const long = sampleHsmm(p, 120, 55).X;
    const a1 = filterHsmm(long, 120, p).stateProb;
    const a2 = filterHsmm(long.slice(0, 70), 70, p).stateProb;
    for (let t = 0; t < 70; t++) {
      for (let k = 0; k < p.K; k++) {
        expect(a1[t * p.K + k]).toBeCloseTo(a2[t * p.K + k], 8);
      }
    }
  });
});

describe("HSMM fitting", () => {
  test("log-likelihood is non-decreasing across EM iterations", () => {
    const p = toyHsmm();
    const { X } = sampleHsmm(p, 500, 12);
    let prev = -Infinity;
    for (let iters = 1; iters <= 8; iters++) {
      const r = fitHsmm(X, 500, 1, { states: 3, maxDuration: 6, maxIter: iters, tol: 0, restarts: 1, seed: 4 });
      expect(r.logLik).toBeGreaterThanOrEqual(prev - 1e-6);
      prev = r.logLik;
    }
  });

  test("recovers non-geometric durations a plain HMM cannot represent", () => {
    // Every regime lasts exactly 8 bars. A geometric dwell law cannot express
    // that; an explicit duration pmf can.
    const K = 2, maxDuration = 16;
    const dur = new Float64Array(K * maxDuration);
    for (let j = 0; j < K; j++) dur[j * maxDuration + 7] = 1; // duration 8, always
    const truth: HsmmParams = {
      K, D: 1, maxDuration,
      pi: new Float64Array([0.5, 0.5]),
      A: new Float64Array([0, 1, 1, 0]),
      mu: new Float64Array([-1.5, 1.5]),
      vari: new Float64Array([0.3, 0.3]),
      dur,
    };
    const { X } = sampleHsmm(truth, 4000, 8);
    const r = fitHsmm(X, 4000, 1, { states: 2, maxDuration, restarts: 3, seed: 7, maxIter: 60 });
    const means = expectedDurations(r.params);
    for (const m of means) expect(Math.abs(m - 8)).toBeLessThan(1.5);

    // The learned pmf should concentrate near 8, not decay from 1 like a geometric.
    for (let j = 0; j < 2; j++) {
      const atEight = r.params.dur[j * maxDuration + 7];
      const atOne = r.params.dur[j * maxDuration + 0];
      expect(atEight).toBeGreaterThan(atOne);
    }
  });

  test("predicts regime ends more sharply than an equivalent HMM", () => {
    // Fixed 8-bar regimes again: by bar 8 the HSMM should be confident a switch
    // is imminent, while the HMM's hazard is constant by construction.
    const K = 2, maxDuration = 16;
    const dur = new Float64Array(K * maxDuration);
    for (let j = 0; j < K; j++) dur[j * maxDuration + 7] = 1;
    const truth: HsmmParams = {
      K, D: 1, maxDuration,
      pi: new Float64Array([0.5, 0.5]),
      A: new Float64Array([0, 1, 1, 0]),
      mu: new Float64Array([-1.5, 1.5]),
      vari: new Float64Array([0.3, 0.3]),
      dur,
    };
    const { X, states } = sampleHsmm(truth, 3000, 21);

    const hs = fitHsmm(X, 3000, 1, { states: 2, maxDuration, restarts: 3, seed: 7, maxIter: 60 });
    const hm = fit(X, 3000, 1, { states: 2, restarts: 4, seed: 7 });
    const fh = filterHsmm(X, 3000, hs.params);
    const fm = filter(X, 3000, hm.params);

    // On the last bar of each run, how much mass does each model put on switching?
    let hsmmSwitch = 0, hmmSwitch = 0, n = 0;
    for (let t = 1; t < 2999; t++) {
      if (states[t] === states[t + 1]) continue; // not a final bar of a run
      const cur = states[t];
      const other = 1 - cur;
      hsmmSwitch += fh.nextProb[t * 2 + other];
      // The HMM's one-step forecast: alpha_t propagated through A.
      let m = 0;
      for (let i = 0; i < 2; i++) m += fm.alpha[t * 2 + i] * hm.params.A[i * 2 + other];
      hmmSwitch += m;
      n++;
    }
    hsmmSwitch /= n; hmmSwitch /= n;
    console.log(`      P(switch) on true final bars — HSMM ${hsmmSwitch.toFixed(3)}, HMM ${hmmSwitch.toFixed(3)}`);
    expect(hsmmSwitch).toBeGreaterThan(hmmSwitch);
  });

  test("serialize round-trips", () => {
    const p = toyHsmm();
    const q = deserializeHsmm(serializeHsmm(p));
    expect(Array.from(q.dur)).toEqual(Array.from(p.dur));
    expect(q.maxDuration).toBe(p.maxDuration);
  });

  test("fromHmm forbids self-transitions and normalizes rows", () => {
    const { X } = sampleHsmm(toyHsmm(), 400, 3);
    const h = fit(X, 400, 1, { states: 3, restarts: 2, seed: 1 });
    const p = fromHmm(h.params, 20);
    for (let i = 0; i < 3; i++) {
      expect(p.A[i * 3 + i]).toBe(0);
      let row = 0;
      for (let j = 0; j < 3; j++) row += p.A[i * 3 + j];
      expect(row).toBeCloseTo(1, 10);
      let dsum = 0;
      for (let d = 0; d < 20; d++) dsum += p.dur[i * 20 + d];
      expect(dsum).toBeCloseTo(1, 8);
    }
  });
});
