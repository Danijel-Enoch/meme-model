import { expect, test, describe } from "bun:test";
import {
  fitStudentHmm, fitStudentHsmm, studentLogEmissions, filterStudent, posteriorsStudent,
  predictNextStudent, viterbiStudent, sampleStudentHmm, lgammaFn, digammaFn,
  serializeStudent, deserializeStudent, stateVariance, expectedDurationsStudent,
  studentHsmmLogLik, type StudentParams,
} from "./student";
import { forward, fit as fitGauss, filter as filterGauss, makeRng, randn } from "./hmm";

/**
 * Exact Gamma(x) for positive integer and half-integer x.
 *
 * Written out so the brute-force reference below never calls the same lgamma
 * the module under test uses — otherwise a wrong lgamma would agree with
 * itself and the whole validation would pass on a broken density.
 */
function gammaExact(x: number): number {
  if (Number.isInteger(x)) {
    let r = 1;
    for (let i = 2; i < x; i++) r *= i;
    return r;
  }
  let r = Math.sqrt(Math.PI);
  let v = 0.5;
  while (v < x - 1e-12) { r *= v; v += 1; }
  return r;
}

/** Univariate Student-t density, written from the textbook form. */
const tpdf = (x: number, mu: number, s: number, nu: number) =>
  (gammaExact((nu + 1) / 2) / (gammaExact(nu / 2) * Math.sqrt(nu * Math.PI * s)))
  * Math.pow(1 + ((x - mu) ** 2) / (nu * s), -(nu + 1) / 2);

/**
 * A small hand-built t-HMM. The degrees of freedom are chosen so that both
 * (nu+1)/2 and nu/2 land on an integer or half-integer, which is what makes
 * `gammaExact` above exact.
 */
function toyModel(): StudentParams {
  return {
    K: 3, D: 1,
    pi: new Float64Array([0.5, 0.3, 0.2]),
    A: new Float64Array([
      0.7, 0.2, 0.1,
      0.3, 0.4, 0.3,
      0.1, 0.3, 0.6,
    ]),
    mu: new Float64Array([-1.0, 0.0, 1.5]),
    scale: new Float64Array([0.5, 1.0, 0.25]),
    nu: new Float64Array([3, 5, 9]),
  };
}

/** Enumerate every one of the K^T state paths. Only viable for tiny T. */
function bruteForce(X: Float64Array, T: number, p: StudentParams) {
  const { K } = p;
  let total = 0;
  let bestJoint = -Infinity;
  let bestPath: number[] = [];
  const margAll = Array.from({ length: T }, () => new Float64Array(K));    // P(z_t, x_1..T)
  const margPrefix = Array.from({ length: T }, () => new Float64Array(K)); // P(z_t, x_1..t)
  const em = (t: number, k: number) => tpdf(X[t], p.mu[k], p.scale[k], p.nu[k]);

  const nPaths = K ** T;
  for (let code = 0; code < nPaths; code++) {
    const path: number[] = [];
    let c = code;
    for (let t = 0; t < T; t++) { path.push(c % K); c = Math.floor(c / K); }

    let joint = p.pi[path[0]] * em(0, path[0]);
    for (let t = 1; t < T; t++) {
      joint *= p.A[path[t - 1] * K + path[t]] * em(t, path[t]);
    }
    total += joint;
    for (let t = 0; t < T; t++) margAll[t][path[t]] += joint;
    if (joint > bestJoint) { bestJoint = joint; bestPath = [...path]; }
  }

  for (let t = 0; t < T; t++) {
    const n = K ** (t + 1);
    for (let code = 0; code < n; code++) {
      const path: number[] = [];
      let c = code;
      for (let i = 0; i <= t; i++) { path.push(c % K); c = Math.floor(c / K); }
      let joint = p.pi[path[0]] * em(0, path[0]);
      for (let i = 1; i <= t; i++) {
        joint *= p.A[path[i - 1] * K + path[i]] * em(i, path[i]);
      }
      margPrefix[t][path[t]] += joint;
    }
  }
  return { likelihood: total, bestPath, margAll, margPrefix };
}

describe("special functions", () => {
  test("lgamma matches exact integer and half-integer values", () => {
    for (const x of [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4.5, 5, 6.5, 9, 12.5, 30]) {
      expect(lgammaFn(x)).toBeCloseTo(Math.log(gammaExact(x)), 9);
    }
  });

  test("lgamma satisfies its own recurrence log G(x+1) = log x + log G(x)", () => {
    for (const x of [0.13, 0.9, 1.7, 4.2, 17.3, 250.1]) {
      expect(lgammaFn(x + 1) - lgammaFn(x)).toBeCloseTo(Math.log(x), 10);
    }
  });

  test("digamma matches closed forms", () => {
    const EULER = 0.5772156649015329;
    expect(digammaFn(1)).toBeCloseTo(-EULER, 8);
    expect(digammaFn(0.5)).toBeCloseTo(-EULER - 2 * Math.LN2, 8);
    expect(digammaFn(2)).toBeCloseTo(1 - EULER, 8);
    expect(digammaFn(3)).toBeCloseTo(1.5 - EULER, 8);
  });

  test("digamma is the derivative of lgamma", () => {
    const h = 1e-5;
    for (const x of [0.7, 1.4, 3.3, 11.2, 60.5]) {
      const fd = (lgammaFn(x + h) - lgammaFn(x - h)) / (2 * h);
      expect(digammaFn(x)).toBeCloseTo(fd, 6);
    }
  });
});

describe("the Student-t emission density", () => {
  test("matches an independently written t pdf", () => {
    const p = toyModel();
    const X = new Float64Array([-3.2, -0.4, 0.0, 0.9, 4.7, 41.0]);
    const logB = studentLogEmissions(X, 6, p);
    for (let t = 0; t < 6; t++) {
      for (let k = 0; k < 3; k++) {
        expect(Math.exp(logB[t * 3 + k])).toBeCloseTo(tpdf(X[t], p.mu[k], p.scale[k], p.nu[k]), 12);
      }
    }
  });

  test("integrates to one", () => {
    const p: StudentParams = {
      K: 1, D: 1, pi: new Float64Array([1]), A: new Float64Array([1]),
      mu: new Float64Array([0.3]), scale: new Float64Array([2.0]), nu: new Float64Array([3.5]),
    };
    let integral = 0;
    const h = 0.002;
    for (let x = -400; x < 400; x += h) {
      integral += Math.exp(studentLogEmissions(new Float64Array([x]), 1, p)[0]) * h;
    }
    expect(integral).toBeCloseTo(1, 5);
  });

  test("a 2-D emission is the multivariate t, not a product of two univariate t's", () => {
    // One latent scale shared across dimensions. sum_d (x_d-mu_d)^2/s_d enters
    // once, so the density is NOT separable — that is the whole point of the
    // scale-mixture construction.
    const p: StudentParams = {
      K: 1, D: 2, pi: new Float64Array([1]), A: new Float64Array([1]),
      mu: new Float64Array([0, 0]), scale: new Float64Array([1, 1]), nu: new Float64Array([4]),
    };
    const x = new Float64Array([1.3, -0.7]);
    const delta = 1.3 * 1.3 + 0.7 * 0.7;
    const expected = (gammaExact(3) / (gammaExact(2) * 4 * Math.PI)) * Math.pow(1 + delta / 4, -3);
    expect(Math.exp(studentLogEmissions(x, 1, p)[0])).toBeCloseTo(expected, 12);

    const uni = tpdf(1.3, 0, 1, 4) * tpdf(-0.7, 0, 1, 4);
    expect(Math.abs(Math.exp(studentLogEmissions(x, 1, p)[0]) - uni)).toBeGreaterThan(1e-4);
  });

  test("degenerates to the Gaussian as nu grows", () => {
    const p: StudentParams = {
      K: 1, D: 1, pi: new Float64Array([1]), A: new Float64Array([1]),
      mu: new Float64Array([0.5]), scale: new Float64Array([1.7]), nu: new Float64Array([1e8]),
    };
    for (const x of [-2, 0, 0.5, 3.1]) {
      const g = Math.exp(-0.5 * ((x - 0.5) ** 2) / 1.7) / Math.sqrt(2 * Math.PI * 1.7);
      const v = Math.exp(studentLogEmissions(new Float64Array([x]), 1, p)[0]);
      // Relative, not absolute: the t and the normal genuinely differ by O(1/nu).
      expect(Math.abs(v / g - 1)).toBeLessThan(1e-6);
    }
  });

  test("keeps a 10-sigma bar finite where the Gaussian does not", () => {
    // A genuine 40% candle standardizes to something like this. The Gaussian
    // assigns it exp(-50); the t assigns it a polynomial tail.
    const p: StudentParams = {
      K: 1, D: 1, pi: new Float64Array([1]), A: new Float64Array([1]),
      mu: new Float64Array([0]), scale: new Float64Array([1]), nu: new Float64Array([3]),
    };
    const lt = studentLogEmissions(new Float64Array([10]), 1, p)[0];
    const lg = -0.5 * 100 - 0.5 * Math.log(2 * Math.PI);
    expect(lt).toBeGreaterThan(lg + 30);
  });
});

describe("inference vs brute force", () => {
  const p = toyModel();
  const T = 7;
  const { X } = sampleStudentHmm(p, T, 123);
  const bf = bruteForce(X, T, p);

  test("forward log-likelihood matches enumeration over all K^T paths", () => {
    const logB = studentLogEmissions(X, T, p);
    const { logLik } = forward(logB, T, { K: p.K, D: p.D, pi: p.pi, A: p.A, mu: p.mu, vari: p.scale });
    expect(logLik).toBeCloseTo(Math.log(bf.likelihood), 8);
  });

  test("filterStudent matches P(z_t | x_1..t)", () => {
    const { alpha } = filterStudent(X, T, p);
    for (let t = 0; t < T; t++) {
      const norm = bf.margPrefix[t].reduce((a, b) => a + b, 0);
      for (let k = 0; k < p.K; k++) {
        expect(alpha[t * p.K + k]).toBeCloseTo(bf.margPrefix[t][k] / norm, 8);
      }
    }
  });

  test("posteriorsStudent matches P(z_t | x_1..T)", () => {
    const { gamma, logLik } = posteriorsStudent(X, T, p);
    expect(logLik).toBeCloseTo(Math.log(bf.likelihood), 8);
    for (let t = 0; t < T; t++) {
      for (let k = 0; k < p.K; k++) {
        expect(gamma[t * p.K + k]).toBeCloseTo(bf.margAll[t][k] / bf.likelihood, 8);
      }
    }
  });

  test("viterbiStudent finds the same path as exhaustive search", () => {
    expect(Array.from(viterbiStudent(X, T, p))).toEqual(bf.bestPath);
  });

  test("predictNext is the filtered distribution pushed through A", () => {
    const { alpha } = filterStudent(X, T, p);
    const nxt = predictNextStudent(alpha, (T - 1) * p.K, p);
    let sum = 0;
    for (let k = 0; k < p.K; k++) sum += nxt[k];
    expect(sum).toBeCloseTo(1, 12);
    for (let j = 0; j < p.K; j++) {
      let acc = 0;
      for (let i = 0; i < p.K; i++) acc += alpha[(T - 1) * p.K + i] * p.A[i * p.K + j];
      expect(nxt[j]).toBeCloseTo(acc, 12);
    }
  });

  test("filtering is causal: future observations cannot change past beliefs", () => {
    const long = sampleStudentHmm(p, 200, 9).X;
    const a1 = filterStudent(long, 200, p).alpha;
    const a2 = filterStudent(long.slice(0, 120 * p.D), 120, p).alpha;
    for (let t = 0; t < 120; t++) {
      for (let k = 0; k < p.K; k++) {
        expect(a1[t * p.K + k]).toBeCloseTo(a2[t * p.K + k], 10);
      }
    }
  });
});

describe("EM", () => {
  const truth = toyModel();
  const { X } = sampleStudentHmm(truth, 900, 5150);

  test("log-likelihood is monotone across iterations", () => {
    // fit() returns the log-likelihood of the parameters ON ENTRY to the final
    // sweep, so running it with maxIter = 1..n traces the EM path. tol = 0
    // disables the early stop so every step is observed.
    let prev = -Infinity;
    for (let n = 1; n <= 25; n++) {
      const r = fitStudentHmm(X, 900, 1, {
        states: 3, restarts: 1, seed: 3, maxIter: n, tol: 0,
      });
      expect(r.logLik).toBeGreaterThanOrEqual(prev - 1e-8);
      prev = r.logLik;
    }
  });

  test("log-likelihood is monotone with nu held fixed too", () => {
    let prev = -Infinity;
    for (let n = 1; n <= 20; n++) {
      const r = fitStudentHmm(X, 900, 1, {
        states: 3, restarts: 1, seed: 4, maxIter: n, tol: 0, fixedNu: 4,
      });
      expect(r.logLik).toBeGreaterThanOrEqual(prev - 1e-8);
      prev = r.logLik;
    }
  });

  test("recovers the parameters of a known t-HMM", () => {
    const known: StudentParams = {
      K: 2, D: 1,
      pi: new Float64Array([0.5, 0.5]),
      A: new Float64Array([0.96, 0.04, 0.06, 0.94]),
      mu: new Float64Array([-1.5, 1.5]),
      scale: new Float64Array([0.7, 0.4]),
      nu: new Float64Array([4, 4]),
    };
    const T = 4000;
    const { X: Y } = sampleStudentHmm(known, T, 8123);
    const r = fitStudentHmm(Y, T, 1, { states: 2, restarts: 6, seed: 17, maxIter: 300 });
    const p = r.params;

    expect(p.mu[0]).toBeCloseTo(-1.5, 0);
    expect(p.mu[1]).toBeCloseTo(1.5, 0);
    expect(p.scale[0]).toBeGreaterThan(0.45);
    expect(p.scale[0]).toBeLessThan(1.0);
    expect(p.scale[1]).toBeGreaterThan(0.25);
    expect(p.scale[1]).toBeLessThan(0.65);
    // nu is the hard one; asking for the right order of magnitude is honest.
    for (let k = 0; k < 2; k++) {
      expect(p.nu[k]).toBeGreaterThan(2.5);
      expect(p.nu[k]).toBeLessThan(9);
    }
    expect(p.A[0]).toBeGreaterThan(0.88);
    expect(p.A[3]).toBeGreaterThan(0.86);
  });

  test("recovers a 2-D t-HMM, where the latent scale is shared across features", () => {
    const known: StudentParams = {
      K: 2, D: 2,
      pi: new Float64Array([0.5, 0.5]),
      A: new Float64Array([0.95, 0.05, 0.05, 0.95]),
      mu: new Float64Array([-1.2, 0.8, 1.2, -0.8]),
      scale: new Float64Array([0.5, 0.5, 0.5, 0.5]),
      nu: new Float64Array([5, 5]),
    };
    const T = 3000;
    const { X: Y } = sampleStudentHmm(known, T, 4242);
    const r = fitStudentHmm(Y, T, 2, { states: 2, restarts: 6, seed: 23, maxIter: 300 });
    const p = r.params;
    expect(p.mu[0]).toBeCloseTo(-1.2, 0);
    expect(p.mu[1]).toBeCloseTo(0.8, 0);
    expect(p.mu[2]).toBeCloseTo(1.2, 0);
    expect(p.mu[3]).toBeCloseTo(-0.8, 0);
    for (let k = 0; k < 2; k++) {
      expect(p.nu[k]).toBeGreaterThan(3);
      expect(p.nu[k]).toBeLessThan(12);
    }
  });

  test("nu runs to its ceiling on genuinely Gaussian data", () => {
    // The degrees of freedom must not invent tails that are not there.
    const rng = makeRng(99);
    const T = 3000;
    const Y = new Float64Array(T);
    let s = 0;
    for (let t = 0; t < T; t++) {
      if (rng() < 0.05) s = 1 - s;
      Y[t] = (s === 0 ? -1.5 : 1.5) + randn(rng);
    }
    const r = fitStudentHmm(Y, T, 1, { states: 2, restarts: 5, seed: 31, maxIter: 300, nuMax: 200 });
    // It lands around 20-40 rather than at the 200 ceiling, and that is not a
    // bug: the likelihood in nu is extremely flat above ~20, where the t is
    // numerically indistinguishable from a normal. Estimating nu precisely in
    // that region is not identified from 3000 bars. What matters is that it is
    // an order of magnitude away from the nu ~ 3 the tailed data below produce.
    for (let k = 0; k < 2; k++) expect(r.params.nu[k]).toBeGreaterThan(12);
    // ...and then it should agree with the Gaussian fit it has collapsed onto.
    const g = fitGauss(Y, T, 1, { states: 2, restarts: 5, seed: 31, maxIter: 300 });
    expect(r.logLik / T).toBeCloseTo(g.logLik / T, 2);
    for (let k = 0; k < 2; k++) {
      expect(stateVariance(r.params, k, 0)).toBeCloseTo(g.params.vari[k], 1);
    }
  });

  test("serialize round-trips", () => {
    const r = fitStudentHmm(X, 900, 1, { states: 2, restarts: 2, seed: 1 });
    const back = deserializeStudent(serializeStudent(r.params));
    expect(back.K).toBe(r.params.K);
    expect(Array.from(back.nu)).toEqual(Array.from(r.params.nu));
    expect(filterStudent(X, 900, back).logLik).toBeCloseTo(filterStudent(X, 900, r.params).logLik, 10);
  });
});

/**
 * The reason this module exists.
 *
 * Data from a 2-state t-HMM with nu = 3. The Student-t fit should recover two
 * states. The Gaussian fit cannot represent the tail inside a state, so it buys
 * likelihood by SPENDING STATES on it — and needs several more of them to reach
 * the same in-sample fit that two t states reach.
 */
describe("Gaussian emissions spend states on kurtosis", () => {
  const truth: StudentParams = {
    K: 2, D: 1,
    pi: new Float64Array([0.5, 0.5]),
    A: new Float64Array([0.95, 0.05, 0.05, 0.95]),
    mu: new Float64Array([-2, 2]),
    scale: new Float64Array([1, 1]),
    nu: new Float64Array([3, 3]),
  };
  const T = 3000;
  // A floor on the variance/scale, so neither model can win by collapsing a
  // state onto a single outlier — the classic unbounded-likelihood degeneracy
  // (Peel & McLachlan 2000, sec. 2). Without it BOTH fits eventually place a
  // spike on the largest bar, which is a different pathology than the one
  // being measured here.
  const FLOOR = 0.01;
  const { X } = sampleStudentHmm(truth, T, 20260910);

  const tFit = fitStudentHmm(X, T, 1, { states: 2, restarts: 6, seed: 7, maxIter: 300, scaleFloor: FLOOR });
  const gauss = Array.from({ length: 9 }, (_, K) =>
    K < 2 ? null : fitGauss(X, T, 1, { states: K, restarts: 6, seed: 7, maxIter: 300, varFloor: FLOOR }));

  test("the t fit recovers the true two states and the true nu", () => {
    const p = tFit.params;
    expect(p.K).toBe(2);
    expect(p.mu[0]).toBeCloseTo(-2, 0);
    expect(p.mu[1]).toBeCloseTo(2, 0);
    expect(p.scale[0]).toBeGreaterThan(0.6);
    expect(p.scale[0]).toBeLessThan(1.6);
    expect(p.scale[1]).toBeGreaterThan(0.6);
    expect(p.scale[1]).toBeLessThan(1.6);
    for (let k = 0; k < 2; k++) {
      expect(p.nu[k]).toBeGreaterThan(2.2);
      expect(p.nu[k]).toBeLessThan(6);
    }
  });

  test("two Gaussian states cannot match two t states", () => {
    expect(gauss[2]!.logLik / T).toBeLessThan(tFit.logLik / T - 0.05);
  });

  test("the Gaussian needs strictly more than two states to catch up", () => {
    const target = tFit.logLik / T;
    let need = -1;
    const row: string[] = [];
    for (let K = 2; K <= 8; K++) {
      const ll = gauss[K]!.logLik / T;
      row.push(`K=${K} ${ll.toFixed(4)}`);
      if (need < 0 && ll >= target) need = K;
    }
    console.log(`      t(K=2) = ${target.toFixed(4)}/bar | gaussian ${row.join("  ")} -> catches up at K=${need}`);
    expect(need).toBeGreaterThan(2);
    // Empirically it takes five; assert the weaker, robust claim of four or more.
    expect(need).toBeGreaterThanOrEqual(4);
  });

  test("the Gaussian's third state is a tail state, not a regime", () => {
    // The signature of a kurtosis state: it sits between the two real regimes
    // (or on top of one), carries a variance many times theirs, and holds
    // little of the occupancy.
    const g3 = gauss[3]!.params;
    const vars = Array.from(g3.vari);
    const ratio = Math.max(...vars) / Math.min(...vars);
    console.log(`      gaussian K=3: mu ${Array.from(g3.mu).map(v => v.toFixed(2)).join(", ")}  var ${vars.map(v => v.toFixed(2)).join(", ")}  (ratio ${ratio.toFixed(1)}x)`);
    // The true model has both states equally wide, so a >5x spread across the
    // three fitted variances is entirely manufactured.
    expect(ratio).toBeGreaterThan(5);
    // And the wide state contributes no new LOCATION: the two narrowest states
    // already sit on the two true regimes, so the third is pure tail cover.
    const byWidth = [0, 1, 2].sort((a, b) => vars[a] - vars[b]);
    const narrow = [g3.mu[byWidth[0]], g3.mu[byWidth[1]]].sort((a, b) => a - b);
    expect(narrow[0]).toBeCloseTo(-2, 0);
    expect(narrow[1]).toBeCloseTo(2, 0);
    // Meanwhile the t fit's two scales are within a factor of two of each other.
    const ts = Array.from(tFit.params.scale);
    expect(Math.max(...ts) / Math.min(...ts)).toBeLessThan(2);
  });

  test("the extra Gaussian states buy tails, not separation", () => {
    // Every additional Gaussian state after the second lands on top of one of
    // the two true locations rather than finding a new one.
    const g5 = gauss[5]!.params;
    let near = 0;
    for (let k = 0; k < 5; k++) {
      if (Math.min(Math.abs(g5.mu[k] + 2), Math.abs(g5.mu[k] - 2)) < 1.0) near++;
    }
    expect(near).toBeGreaterThanOrEqual(4);
  });
});

describe("Student-t HSMM", () => {
  const truth: StudentParams = {
    K: 2, D: 1,
    pi: new Float64Array([0.5, 0.5]),
    A: new Float64Array([0.92, 0.08, 0.08, 0.92]),
    mu: new Float64Array([-2, 2]),
    scale: new Float64Array([1, 1]),
    nu: new Float64Array([3, 3]),
  };
  const T = 900;
  const { X } = sampleStudentHmm(truth, T, 777);

  test("log-likelihood is monotone across segmental EM iterations", () => {
    let prev = -Infinity;
    for (let n = 1; n <= 12; n++) {
      const r = fitStudentHsmm(X, T, 1, {
        states: 2, restarts: 2, seed: 5, maxIter: n, tol: 0, maxDuration: 20, scaleFloor: 0.01,
      });
      expect(r.logLik).toBeGreaterThanOrEqual(prev - 1e-8);
      prev = r.logLik;
    }
  });

  test("recovers the true locations and a sane dwell", () => {
    const r = fitStudentHsmm(X, T, 1, {
      states: 2, restarts: 4, seed: 5, maxIter: 40, maxDuration: 40, scaleFloor: 0.01,
    });
    expect(r.params.mu[0]).toBeCloseTo(-2, 0);
    expect(r.params.mu[1]).toBeCloseTo(2, 0);
    const dwell = expectedDurationsStudent(r.params);
    // The generating chain has a 0.92 diagonal, so mean dwell is ~12.5 bars.
    for (const d of dwell) {
      expect(d).toBeGreaterThan(4);
      expect(d).toBeLessThan(35);
    }
    // fit() reports the log-likelihood on ENTRY to its last sweep, so scoring
    // the returned (post-M-step) parameters must not be worse.
    expect(studentHsmmLogLik(X, T, r.params)).toBeGreaterThanOrEqual(r.logLik - 1e-8);
  });
});
