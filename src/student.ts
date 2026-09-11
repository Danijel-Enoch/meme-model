/**
 * Student-t emission HMM (and HSMM), as a drop-in alternative to `hmm.ts`.
 *
 * WHY
 * ---
 * With Gaussian emissions the EM fit spends states on kurtosis. A single
 * heavy-tailed return distribution is cheaper to approximate with a mixture of
 * several Gaussians than with one, so extra states buy likelihood by modelling
 * the tails rather than by finding regimes — which is exactly what the README's
 * "the ELBO does not want three states" section measures. States that exist to
 * model kurtosis are not regimes.
 *
 * A Student-t emission absorbs the tail inside the state, so the state count is
 * free to mean what it is supposed to mean.
 *
 *   z_t | z_{t-1} ~ Categorical(A[z_{t-1}])
 *   x_t | z_t     ~ t_D(mu[z_t], Sigma[z_t] = diag(scale[z_t]), nu[z_t])
 *
 * The multivariate t is used (one latent scale per bar, shared across the D
 * features), not a product of independent univariate t's. That is the form with
 * the clean scale-mixture-of-normals representation:
 *
 *   u_t | z_t = k ~ Gamma(nu_k / 2, nu_k / 2)
 *   x_t | z_t = k, u_t ~ N(mu_k, Sigma_k / u_t)
 *
 * and it degenerates exactly to `hmm.ts`'s diagonal Gaussian as nu -> Infinity.
 *
 * Storage mirrors `hmm.ts` exactly — flat Float64Arrays, states sorted ascending
 * by the mean of feature 0, log-space forward-backward, `makeRng` seeding,
 * random restarts:
 *   A       K*K   transition matrix, A[i*K+j] = P(z_t=j | z_{t-1}=i)
 *   pi      K     initial distribution
 *   mu      K*D   per-state location
 *   scale   K*D   per-state SQUARED scale, i.e. diag(Sigma_k). This is not the
 *                 variance: Var[x] = Sigma * nu/(nu-2), finite only for nu > 2.
 *                 As nu -> Infinity, scale -> the Gaussian `vari`.
 *   nu      K     per-state degrees of freedom
 *
 * REFERENCES for the update equations implemented below
 * -----------------------------------------------------
 * Peel, D. & McLachlan, G.J. (2000). "Robust mixture modelling using the t
 *   distribution." Statistics and Computing 10(4), 339-348.  -- the scale
 *   mixture E-step weight u, the weighted location/scatter M-step, and the
 *   1-D digamma equation for nu.
 * Liu, C. & Rubin, D.B. (1995). "ML estimation of the t distribution using EM
 *   and its extensions, ECM and ECME." Statistica Sinica 5, 19-39.  -- the ECME
 *   variant, in which nu is chosen by maximising the ACTUAL (u-marginalised)
 *   likelihood rather than the u-augmented Q function. That is what is done
 *   here, because for an HMM the u-marginalised weighted likelihood is the
 *   exact EM objective, so the resulting sweep is a GEM step and the sequence
 *   log-likelihood is monotone.
 * Bulla, J. (2011). "Hidden Markov models with t components. Increased
 *   persistence and other aspects." Quantitative Finance 11(3), 459-475.  --
 *   the same model applied to financial return series.
 * Ryden, T., Terasvirta, T. & Asbrink, S. (1998). "Stylized facts of daily
 *   return series and the hidden Markov model." Journal of Applied Econometrics
 *   13(3), 217-244.  -- why a Gaussian HMM reproduces unconditional kurtosis by
 *   mixing, which is the failure mode this file exists to remove.
 */

import {
  forward, backward, makeRng, randn,
  type FitOptions, type HmmParams,
} from "./hmm";
import { emissionPrefix, hsmmExpectations, type HsmmChain } from "./hsmm";

export interface StudentParams {
  K: number;
  D: number;
  pi: Float64Array;
  A: Float64Array;
  mu: Float64Array;
  /** K*D squared scales — the diagonal of Sigma_k, NOT the variance. */
  scale: Float64Array;
  /** K degrees of freedom, one per state. */
  nu: Float64Array;
}

export interface StudentFitOptions extends FitOptions {
  /** Starting degrees of freedom for every state. */
  nuInit?: number;
  /** Lower bound on nu. Below 2 the t has no variance; 2.02 keeps it finite. */
  nuMin?: number;
  /** Upper bound on nu. Past ~100 the t is numerically Gaussian and the
   *  likelihood in nu is flat, so estimating it is not identified in practice. */
  nuMax?: number;
  /** Hold nu at this value for every state instead of estimating it. */
  fixedNu?: number;
  /** Lower bound on the squared scale, mirrors hmm.ts's varFloor. */
  scaleFloor?: number;
  /** Warm start instead of k-means++. */
  init?: StudentParams;
}

export interface StudentFitResult {
  params: StudentParams;
  logLik: number;
  iterations: number;
  converged: boolean;
}

// ---------------------------------------------------------------------------
// Special functions. Kept local so this module has the same zero-dependency,
// zero-cross-import profile as hmm.ts.
// ---------------------------------------------------------------------------

const LANCZOS = [
  676.5203681218851, -1259.1392167224028, 771.32342877765313,
  -176.61502916214059, 12.507343278686905, -0.13857109526572012,
  9.9843695780195716e-6, 1.5056327351493116e-7,
];

/** log Gamma(x), Lanczos g=7. Accurate to ~1e-13 relative on the range used. */
export function lgammaFn(x: number): number {
  if (x < 0.5) {
    // Reflection: Gamma(x) Gamma(1-x) = pi / sin(pi x)
    return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - lgammaFn(1 - x);
  }
  const z = x - 1;
  let a = 0.99999999999980993;
  for (let i = 0; i < 8; i++) a += LANCZOS[i] / (z + i + 1);
  const t = z + 7.5;
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(a);
}

/** digamma(x) = d/dx log Gamma(x). Recurrence up to x >= 6, then asymptotic. */
export function digammaFn(x: number): number {
  let r = 0;
  let v = x;
  while (v < 6) { r -= 1 / v; v += 1; }
  const f = 1 / (v * v);
  return r + Math.log(v) - 0.5 / v
    + f * (-1 / 12 + f * (1 / 120 + f * (-1 / 252 + f * (1 / 240 + f * (-1 / 132)))));
}

// ---------------------------------------------------------------------------
// Emissions
// ---------------------------------------------------------------------------

/**
 * log t_D(x_t | mu_k, diag(scale_k), nu_k) for every (t, k). Returns T*K flat.
 *
 *   log p = lgamma((nu+D)/2) - lgamma(nu/2) - (D/2) log(nu pi)
 *           - (1/2) sum_d log scale_d - ((nu+D)/2) log(1 + delta / nu)
 *
 * with delta the squared Mahalanobis distance sum_d (x_d - mu_d)^2 / scale_d.
 * The log1p form is what keeps the tail accurate: for a 10-sigma bar delta/nu
 * is enormous and the Gaussian's -delta/2 underflows to -Infinity, while the t
 * decays only logarithmically. That is the whole point.
 */
export function studentLogEmissions(X: Float64Array, T: number, p: StudentParams): Float64Array {
  const { K, D, mu, scale, nu } = p;
  const logB = new Float64Array(T * K);
  const norm = new Float64Array(K);
  const half = new Float64Array(K);
  for (let k = 0; k < K; k++) {
    const n = nu[k];
    let s = 0;
    for (let d = 0; d < D; d++) s += Math.log(scale[k * D + d]);
    norm[k] = lgammaFn((n + D) / 2) - lgammaFn(n / 2)
      - (D / 2) * Math.log(n * Math.PI) - 0.5 * s;
    half[k] = (n + D) / 2;
  }
  for (let t = 0; t < T; t++) {
    for (let k = 0; k < K; k++) {
      let delta = 0;
      for (let d = 0; d < D; d++) {
        const diff = X[t * D + d] - mu[k * D + d];
        delta += (diff * diff) / scale[k * D + d];
      }
      logB[t * K + k] = norm[k] - half[k] * Math.log1p(delta / nu[k]);
    }
  }
  return logB;
}

/**
 * hmm.ts's `forward`/`backward` read only K, pi and A off their params object,
 * so the scaled recursions are shared rather than reimplemented — one
 * implementation, validated once against brute force.
 */
function asChain(p: StudentParams): HmmParams {
  return { K: p.K, D: p.D, pi: p.pi, A: p.A, mu: p.mu, vari: p.scale };
}

/** Smoothed state probabilities P(z_t = k | x_1..T). Uses the whole series. */
export function posteriorsStudent(X: Float64Array, T: number, p: StudentParams): { gamma: Float64Array; logLik: number } {
  const logB = studentLogEmissions(X, T, p);
  const chain = asChain(p);
  const { alpha, logLik } = forward(logB, T, chain);
  const beta = backward(logB, T, chain);
  const { K } = p;
  const gamma = new Float64Array(T * K);
  for (let t = 0; t < T; t++) {
    let sum = 0;
    for (let k = 0; k < K; k++) {
      const v = alpha[t * K + k] * beta[t * K + k];
      gamma[t * K + k] = v;
      sum += v;
    }
    for (let k = 0; k < K; k++) gamma[t * K + k] = sum > 0 ? gamma[t * K + k] / sum : 1 / K;
  }
  return { gamma, logLik };
}

/** Filtered probabilities P(z_t = k | x_1..t) — the causal ones a strategy may see. */
export function filterStudent(X: Float64Array, T: number, p: StudentParams): { alpha: Float64Array; logLik: number } {
  const logB = studentLogEmissions(X, T, p);
  const { alpha, logLik } = forward(logB, T, asChain(p));
  return { alpha, logLik };
}

/** One-step-ahead state forecast P(z_{t+1} | x_1..t) from a filtered distribution. */
export function predictNextStudent(filtered: Float64Array, offset: number, p: StudentParams): Float64Array {
  const { K, A } = p;
  const out = new Float64Array(K);
  for (let j = 0; j < K; j++) {
    let acc = 0;
    for (let i = 0; i < K; i++) acc += filtered[offset + i] * A[i * K + j];
    out[j] = acc;
  }
  return out;
}

/** Viterbi: the single most likely state path. Log domain throughout. */
export function viterbiStudent(X: Float64Array, T: number, p: StudentParams): Int32Array {
  const { K, pi, A } = p;
  const logB = studentLogEmissions(X, T, p);
  const logA = new Float64Array(K * K);
  for (let i = 0; i < K * K; i++) logA[i] = Math.log(Math.max(A[i], 1e-300));

  const delta = new Float64Array(T * K);
  const psi = new Int32Array(T * K);
  for (let k = 0; k < K; k++) delta[k] = Math.log(Math.max(pi[k], 1e-300)) + logB[k];

  for (let t = 1; t < T; t++) {
    for (let j = 0; j < K; j++) {
      let best = -Infinity, arg = 0;
      for (let i = 0; i < K; i++) {
        const v = delta[(t - 1) * K + i] + logA[i * K + j];
        if (v > best) { best = v; arg = i; }
      }
      delta[t * K + j] = best + logB[t * K + j];
      psi[t * K + j] = arg;
    }
  }

  const path = new Int32Array(T);
  let best = -Infinity;
  for (let k = 0; k < K; k++) {
    if (delta[(T - 1) * K + k] > best) { best = delta[(T - 1) * K + k]; path[T - 1] = k; }
  }
  for (let t = T - 2; t >= 0; t--) path[t] = psi[(t + 1) * K + path[t + 1]];
  return path;
}

// ---------------------------------------------------------------------------
// The nu sub-problem
// ---------------------------------------------------------------------------

/**
 * d/dnu of  sum_t w_t log t_D(x_t | mu, Sigma, nu),  holding mu and Sigma fixed.
 *
 * Only the nu-dependent part matters:
 *   sum_t w_t [ lgamma((nu+D)/2) - lgamma(nu/2) - (D/2) log nu
 *               - ((nu+D)/2) log(1 + delta_t/nu) ]
 * whose derivative is
 *   (1/2) sum_t w_t [ psi((nu+D)/2) - psi(nu/2) - D/nu
 *                     - log(1 + delta_t/nu) + (nu+D) delta_t / (nu (nu+delta_t)) ]
 *
 * This is the ECME form of the nu step (Liu & Rubin 1995): nu maximises the
 * u-marginalised weighted likelihood, not the u-augmented Q. For a mixture or
 * an HMM the u-marginalised weighted likelihood IS the exact EM objective given
 * the responsibilities, so this step is exact rather than merely a bound.
 */
function dNuLogLik(delta: Float64Array, w: Float64Array, T: number, D: number, nu: number): number {
  const a = digammaFn((nu + D) / 2) - digammaFn(nu / 2) - D / nu;
  let acc = 0;
  for (let t = 0; t < T; t++) {
    const wt = w[t];
    if (wt <= 0) continue;
    const dl = delta[t];
    acc += wt * (a - Math.log1p(dl / nu) + ((nu + D) * dl) / (nu * (nu + dl)));
  }
  return 0.5 * acc;
}

/** The same objective's value, up to a constant in nu. Used to pick an endpoint. */
function nuLogLik(delta: Float64Array, w: Float64Array, T: number, D: number, nu: number): number {
  const c = lgammaFn((nu + D) / 2) - lgammaFn(nu / 2) - (D / 2) * Math.log(nu);
  const half = (nu + D) / 2;
  let acc = 0;
  for (let t = 0; t < T; t++) {
    const wt = w[t];
    if (wt <= 0) continue;
    acc += wt * (c - half * Math.log1p(delta[t] / nu));
  }
  return acc;
}

/**
 * Bisect the derivative for the maximising nu; clamp when it has no interior root.
 *
 * `warm` is the previous sweep's estimate. Once EM settles, nu barely moves, so
 * bracketing [warm/2, 2*warm] first turns a 20-step bisection over [2, 200] into
 * a handful of steps. Each derivative evaluation costs O(T), and this runs once
 * per state per sweep, so it is the difference between the t fit costing the
 * same as the Gaussian one and costing ten times as much.
 *
 * The tolerance is deliberately loose. nu is not identified to more than a few
 * significant figures from a few thousand bars — the likelihood in nu is very
 * flat once nu is large (Liu & Rubin 1995; Fernandez & Steel 1999) — so solving
 * the root to 1e-10 would be spending real time on noise.
 */
function solveNu(
  delta: Float64Array, w: Float64Array, T: number, D: number,
  lo: number, hi: number, warm: number,
): number {
  let a = lo, b = hi;
  let bracketed = false;

  if (warm > lo && warm < hi) {
    const l = Math.max(lo, warm / 2);
    const h = Math.min(hi, warm * 2);
    if (l < h
      && dNuLogLik(delta, w, T, D, l) > 0
      && dNuLogLik(delta, w, T, D, h) < 0) {
      a = l; b = h; bracketed = true;
    }
  }

  if (!bracketed) {
    const dLo = dNuLogLik(delta, w, T, D, lo);
    const dHi = dNuLogLik(delta, w, T, D, hi);
    if (!Number.isFinite(dLo) || !Number.isFinite(dHi)) return hi;
    // Increasing at the top => the data want a Gaussian; decreasing at the
    // bottom => they want the heaviest tail allowed.
    if (dLo <= 0 && dHi <= 0) return lo;
    if (dLo >= 0 && dHi >= 0) return hi;
    if (dLo < 0 && dHi > 0) {
      // Non-concave in nu (rare). Fall back to whichever endpoint is higher.
      return nuLogLik(delta, w, T, D, lo) >= nuLogLik(delta, w, T, D, hi) ? lo : hi;
    }
  }

  for (let it = 0; it < 60; it++) {
    const m = 0.5 * (a + b);
    if (dNuLogLik(delta, w, T, D, m) > 0) a = m; else b = m;
    if (b - a < 1e-6 * Math.max(1, b)) break;
  }
  return 0.5 * (a + b);
}

// ---------------------------------------------------------------------------
// Initialization — k-means++ then Lloyd, matching hmm.ts's seeding exactly.
// ---------------------------------------------------------------------------

function kmeansInit(X: Float64Array, T: number, D: number, K: number, rng: () => number) {
  const centers = new Float64Array(K * D);
  const first = Math.floor(rng() * T);
  for (let d = 0; d < D; d++) centers[d] = X[first * D + d];

  const dist = new Float64Array(T).fill(Infinity);
  for (let k = 1; k < K; k++) {
    let total = 0;
    for (let t = 0; t < T; t++) {
      let dd = 0;
      for (let d = 0; d < D; d++) {
        const diff = X[t * D + d] - centers[(k - 1) * D + d];
        dd += diff * diff;
      }
      if (dd < dist[t]) dist[t] = dd;
      total += dist[t];
    }
    let target = rng() * total, idx = T - 1;
    for (let t = 0; t < T; t++) {
      target -= dist[t];
      if (target <= 0) { idx = t; break; }
    }
    for (let d = 0; d < D; d++) centers[k * D + d] = X[idx * D + d];
  }

  const assign = new Int32Array(T);
  for (let iter = 0; iter < 15; iter++) {
    let moved = 0;
    for (let t = 0; t < T; t++) {
      let best = Infinity, arg = 0;
      for (let k = 0; k < K; k++) {
        let dd = 0;
        for (let d = 0; d < D; d++) {
          const diff = X[t * D + d] - centers[k * D + d];
          dd += diff * diff;
        }
        if (dd < best) { best = dd; arg = k; }
      }
      if (assign[t] !== arg) moved++;
      assign[t] = arg;
    }
    const sums = new Float64Array(K * D);
    const counts = new Float64Array(K);
    for (let t = 0; t < T; t++) {
      counts[assign[t]]++;
      for (let d = 0; d < D; d++) sums[assign[t] * D + d] += X[t * D + d];
    }
    for (let k = 0; k < K; k++) {
      if (counts[k] === 0) {
        const t = Math.floor(rng() * T);
        for (let d = 0; d < D; d++) centers[k * D + d] = X[t * D + d];
      } else {
        for (let d = 0; d < D; d++) centers[k * D + d] = sums[k * D + d] / counts[k];
      }
    }
    if (moved === 0) break;
  }
  return { centers, assign };
}

function initParams(
  X: Float64Array, T: number, D: number, K: number,
  rng: () => number, scaleFloor: number, nuInit: number,
): StudentParams {
  const { centers, assign } = kmeansInit(X, T, D, K, rng);
  const scale = new Float64Array(K * D);
  const counts = new Float64Array(K);
  for (let t = 0; t < T; t++) {
    counts[assign[t]]++;
    for (let d = 0; d < D; d++) {
      const diff = X[t * D + d] - centers[assign[t] * D + d];
      scale[assign[t] * D + d] += diff * diff;
    }
  }
  for (let k = 0; k < K; k++) {
    for (let d = 0; d < D; d++) {
      // Cluster variance is an estimate of Sigma * nu/(nu-2), so deflate it to
      // land on Sigma rather than starting every scale too wide.
      const v = counts[k] > 1 ? scale[k * D + d] / counts[k] : 1;
      const deflate = nuInit > 2 ? (nuInit - 2) / nuInit : 1;
      scale[k * D + d] = Math.max(v * deflate, scaleFloor);
    }
  }

  const A = new Float64Array(K * K);
  for (let i = 0; i < K; i++) {
    const stay = 0.85 + 0.1 * rng();
    for (let j = 0; j < K; j++) A[i * K + j] = i === j ? stay : (1 - stay) / (K - 1);
  }
  const pi = new Float64Array(K);
  for (let k = 0; k < K; k++) pi[k] = counts[k] / T || 1 / K;
  let s = 0;
  for (let k = 0; k < K; k++) s += pi[k];
  for (let k = 0; k < K; k++) pi[k] /= s;

  const nu = new Float64Array(K).fill(nuInit);
  return { K, D, pi, A, mu: centers, scale, nu };
}

// ---------------------------------------------------------------------------
// EM
// ---------------------------------------------------------------------------

interface ResolvedOptions {
  states: number; maxIter: number; tol: number; restarts: number;
  scaleFloor: number; transitionPrior: number; seed: number; verbose: boolean;
  nuInit: number; nuMin: number; nuMax: number; fixedNu: number | null;
}

/**
 * One EM sweep, in place. Returns the log-likelihood of the params ON ENTRY,
 * exactly like hmm.ts's emStep, so the caller's monotonicity check is the same.
 *
 * E step
 *   gamma_t(k), xi_t(i,j)   forward-backward on the t log-densities
 *   u_t(k) = (nu_k + D) / (nu_k + delta_t(k))          [Peel & McLachlan 2000]
 *            = E[u_t | x_t, z_t = k] under u ~ Gamma(nu/2, nu/2), evaluated at
 *            the CURRENT mu, Sigma, nu.
 *
 * M step
 *   pi_k    = gamma_1(k)
 *   A_ij    = (sum_t xi_t(i,j) + prior) / row sum
 *   mu_k    = sum_t gamma_t(k) u_t(k) x_t / sum_t gamma_t(k) u_t(k)
 *   Sigma_k = sum_t gamma_t(k) u_t(k) (x_t - mu_k)(x_t - mu_k)' / sum_t gamma_t(k)
 *             ^ note the denominator is sum gamma, NOT sum gamma*u. That is the
 *               exact EM update (Peel & McLachlan 2000); dividing by sum gamma*u
 *               instead is the common bug, and it biases Sigma downwards.
 *   nu_k    = argmax_nu sum_t gamma_t(k) log t_D(x_t | mu_k, Sigma_k, nu)
 *             by bisection on the derivative (the ECME step, Liu & Rubin 1995).
 */
function emStepStudent(X: Float64Array, T: number, p: StudentParams, opts: ResolvedOptions): number {
  const { K, D } = p;
  const logB = studentLogEmissions(X, T, p);
  const chain = asChain(p);
  const { alpha, logLik } = forward(logB, T, chain);
  const beta = backward(logB, T, chain);

  const gamma = new Float64Array(T * K);
  for (let t = 0; t < T; t++) {
    let sum = 0;
    for (let k = 0; k < K; k++) {
      const v = alpha[t * K + k] * beta[t * K + k];
      gamma[t * K + k] = v;
      sum += v;
    }
    for (let k = 0; k < K; k++) gamma[t * K + k] = sum > 0 ? gamma[t * K + k] / sum : 1 / K;
  }

  const xiSum = new Float64Array(K * K);
  const b = new Float64Array(K);
  const tmp = new Float64Array(K * K);
  for (let t = 0; t < T - 1; t++) {
    let m = -Infinity;
    for (let k = 0; k < K; k++) if (logB[(t + 1) * K + k] > m) m = logB[(t + 1) * K + k];
    for (let k = 0; k < K; k++) b[k] = Math.exp(logB[(t + 1) * K + k] - m);

    let sum = 0;
    tmp.fill(0);
    for (let i = 0; i < K; i++) {
      for (let j = 0; j < K; j++) {
        const v = alpha[t * K + i] * p.A[i * K + j] * b[j] * beta[(t + 1) * K + j];
        tmp[i * K + j] = v;
        sum += v;
      }
    }
    if (sum > 0 && Number.isFinite(sum)) {
      for (let i = 0; i < K * K; i++) xiSum[i] += tmp[i] / sum;
    }
  }

  // --- M step: chain ---
  for (let k = 0; k < K; k++) p.pi[k] = gamma[k];
  for (let i = 0; i < K; i++) {
    let rowSum = 0;
    for (let j = 0; j < K; j++) rowSum += xiSum[i * K + j] + opts.transitionPrior;
    for (let j = 0; j < K; j++) {
      p.A[i * K + j] = rowSum > 0 ? (xiSum[i * K + j] + opts.transitionPrior) / rowSum : 1 / K;
    }
  }

  // --- M step: emissions ---
  // u weights at the CURRENT parameters.
  const U = new Float64Array(T * K);
  const gsum = new Float64Array(K);   // sum_t gamma
  const wsum = new Float64Array(K);   // sum_t gamma * u
  const muNew = new Float64Array(K * D);
  for (let k = 0; k < K; k++) {
    const nk = p.nu[k];
    for (let t = 0; t < T; t++) {
      let delta = 0;
      for (let d = 0; d < D; d++) {
        const diff = X[t * D + d] - p.mu[k * D + d];
        delta += (diff * diff) / p.scale[k * D + d];
      }
      const u = (nk + D) / (nk + delta);
      U[t * K + k] = u;
      const g = gamma[t * K + k];
      gsum[k] += g;
      const w = g * u;
      wsum[k] += w;
      for (let d = 0; d < D; d++) muNew[k * D + d] += w * X[t * D + d];
    }
  }
  for (let k = 0; k < K; k++) {
    if (wsum[k] > 1e-12) for (let d = 0; d < D; d++) muNew[k * D + d] /= wsum[k];
    else for (let d = 0; d < D; d++) muNew[k * D + d] = p.mu[k * D + d];
  }

  const scaleNew = new Float64Array(K * D);
  for (let t = 0; t < T; t++) {
    for (let k = 0; k < K; k++) {
      const w = gamma[t * K + k] * U[t * K + k];
      if (w === 0) continue;
      for (let d = 0; d < D; d++) {
        const diff = X[t * D + d] - muNew[k * D + d];
        scaleNew[k * D + d] += w * diff * diff;
      }
    }
  }
  for (let k = 0; k < K; k++) {
    for (let d = 0; d < D; d++) {
      const v = gsum[k] > 1e-8 ? scaleNew[k * D + d] / gsum[k] : p.scale[k * D + d];
      p.scale[k * D + d] = Math.max(v, opts.scaleFloor);
    }
  }
  p.mu.set(muNew);

  // --- M step: degrees of freedom, at the freshly updated mu and Sigma ---
  if (opts.fixedNu === null) {
    const delta = new Float64Array(T);
    const w = new Float64Array(T);
    for (let k = 0; k < K; k++) {
      if (gsum[k] <= 1e-6) continue;
      for (let t = 0; t < T; t++) {
        let dl = 0;
        for (let d = 0; d < D; d++) {
          const diff = X[t * D + d] - p.mu[k * D + d];
          dl += (diff * diff) / p.scale[k * D + d];
        }
        delta[t] = dl;
        w[t] = gamma[t * K + k];
      }
      p.nu[k] = solveNu(delta, w, T, D, opts.nuMin, opts.nuMax, p.nu[k]);
    }
  }

  return logLik;
}

export function fitStudentHmm(
  X: Float64Array, T: number, D: number, options: StudentFitOptions = {},
): StudentFitResult {
  const opts: ResolvedOptions = {
    states: options.states ?? 3,
    maxIter: options.maxIter ?? 200,
    tol: options.tol ?? 1e-6,
    restarts: options.restarts ?? 5,
    scaleFloor: options.scaleFloor ?? options.varFloor ?? 1e-8,
    transitionPrior: options.transitionPrior ?? 0.1,
    seed: options.seed ?? 42,
    verbose: options.verbose ?? false,
    nuInit: options.nuInit ?? 8,
    nuMin: options.nuMin ?? 2.02,
    nuMax: options.nuMax ?? 200,
    fixedNu: options.fixedNu ?? null,
  };
  const K = opts.states;
  if (T < K * 10) throw new Error(`need at least ${K * 10} observations for ${K} states, got ${T}`);
  if (opts.fixedNu !== null) opts.nuInit = opts.fixedNu;

  let best: StudentFitResult | null = null;
  const restarts = options.init ? 1 : opts.restarts;
  for (let r = 0; r < restarts; r++) {
    const rng = makeRng(opts.seed + r * 7919);
    const p = options.init ? cloneParams(options.init) : initParams(X, T, D, K, rng, opts.scaleFloor, opts.nuInit);
    if (opts.fixedNu !== null) p.nu.fill(opts.fixedNu);

    let prev = -Infinity;
    let iter = 0;
    let converged = false;
    let logLik = -Infinity;
    for (; iter < opts.maxIter; iter++) {
      logLik = emStepStudent(X, T, p, opts);
      if (!Number.isFinite(logLik)) break;
      if (Math.abs(logLik - prev) / T < opts.tol) { converged = true; break; }
      prev = logLik;
    }
    if (opts.verbose) {
      console.error(`  restart ${r}: logLik/T = ${(logLik / T).toFixed(6)} after ${iter} iters, nu = ${Array.from(p.nu).map(v => v.toFixed(1)).join(",")}`);
    }
    if (!best || logLik > best.logLik) {
      best = { params: p, logLik, iterations: iter, converged };
    }
  }
  return sortStudentStates(best!);
}

export function cloneParams(p: StudentParams): StudentParams {
  return {
    K: p.K, D: p.D,
    pi: Float64Array.from(p.pi), A: Float64Array.from(p.A),
    mu: Float64Array.from(p.mu), scale: Float64Array.from(p.scale),
    nu: Float64Array.from(p.nu),
  };
}

/** Relabel states ascending by mean of feature 0, matching hmm.ts's convention. */
export function relabelStudent(p: StudentParams): StudentParams {
  const { K, D } = p;
  const order = Array.from({ length: K }, (_, k) => k).sort((a, b) => p.mu[a * D] - p.mu[b * D]);
  const pi = new Float64Array(K);
  const A = new Float64Array(K * K);
  const mu = new Float64Array(K * D);
  const scale = new Float64Array(K * D);
  const nu = new Float64Array(K);
  for (let ni = 0; ni < K; ni++) {
    const oi = order[ni];
    pi[ni] = p.pi[oi];
    nu[ni] = p.nu[oi];
    for (let d = 0; d < D; d++) {
      mu[ni * D + d] = p.mu[oi * D + d];
      scale[ni * D + d] = p.scale[oi * D + d];
    }
    for (let nj = 0; nj < K; nj++) A[ni * K + nj] = p.A[oi * K + order[nj]];
  }
  return { K, D, pi, A, mu, scale, nu };
}

export function sortStudentStates(res: StudentFitResult): StudentFitResult {
  return { ...res, params: relabelStudent(res.params) };
}

/** Implied variance of state k, feature d. Infinite for nu <= 2. */
export function stateVariance(p: StudentParams, k: number, d: number): number {
  const n = p.nu[k];
  return n > 2 ? p.scale[k * p.D + d] * (n / (n - 2)) : Infinity;
}

/** Stationary distribution of the chain, by power iteration. */
export function stationaryStudent(p: StudentParams): Float64Array {
  const { K, A } = p;
  let v = new Float64Array(K).fill(1 / K);
  for (let it = 0; it < 500; it++) {
    const nv = new Float64Array(K);
    for (let j = 0; j < K; j++) {
      let acc = 0;
      for (let i = 0; i < K; i++) acc += v[i] * A[i * K + j];
      nv[j] = acc;
    }
    let diff = 0;
    for (let k = 0; k < K; k++) diff += Math.abs(nv[k] - v[k]);
    v = nv;
    if (diff < 1e-12) break;
  }
  return v;
}

/** Expected number of bars a state persists: 1 / (1 - A[k][k]). */
export function expectedDurationStudent(p: StudentParams, k: number): number {
  const stay = p.A[k * p.K + k];
  return stay >= 1 ? Infinity : 1 / (1 - stay);
}

/** Free parameters, for BIC/AIC comparisons against the Gaussian fit. */
export function studentParamCount(K: number, D: number, estimateNu = true): number {
  // pi (K-1) + A K(K-1) + mu KD + scale KD + nu K
  return (K - 1) + K * (K - 1) + 2 * K * D + (estimateNu ? K : 0);
}

export function serializeStudent(p: StudentParams): string {
  return JSON.stringify({
    K: p.K, D: p.D,
    pi: Array.from(p.pi), A: Array.from(p.A),
    mu: Array.from(p.mu), scale: Array.from(p.scale), nu: Array.from(p.nu),
  }, null, 2);
}

export function deserializeStudent(json: string): StudentParams {
  const o = JSON.parse(json);
  return {
    K: o.K, D: o.D,
    pi: new Float64Array(o.pi), A: new Float64Array(o.A),
    mu: new Float64Array(o.mu), scale: new Float64Array(o.scale),
    nu: new Float64Array(o.nu),
  };
}

// ---------------------------------------------------------------------------
// Sampling from a known t-HMM — needed to test recovery, and to demonstrate
// what Gaussian emissions do when the truth has tails.
// ---------------------------------------------------------------------------

/** Gamma(shape, 1) by Marsaglia & Tsang (2000), with Johnk's boost for shape < 1. */
export function randGamma(shape: number, rng: () => number): number {
  if (shape < 1) {
    const u = Math.max(rng(), 1e-300);
    return randGamma(shape + 1, rng) * Math.pow(u, 1 / shape);
  }
  const d = shape - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  for (;;) {
    let x = 0, v = 0;
    do {
      x = randn(rng);
      v = 1 + c * x;
    } while (v <= 0);
    v = v * v * v;
    const u = Math.max(rng(), 1e-300);
    if (u < 1 - 0.0331 * x * x * x * x) return d * v;
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
  }
}

/**
 * Sample T observations from a known t-HMM, via the scale mixture:
 *   u ~ Gamma(nu/2, nu/2),  x | u ~ N(mu, Sigma / u).
 * One u per bar, shared across the D features — the multivariate t.
 */
export function sampleStudentHmm(p: StudentParams, T: number, seed: number): { X: Float64Array; states: Int32Array } {
  const rng = makeRng(seed);
  const { K, D } = p;
  const X = new Float64Array(T * D);
  const states = new Int32Array(T);
  let s = 0;
  let u = rng(), acc = 0;
  for (let k = 0; k < K; k++) { acc += p.pi[k]; if (u <= acc) { s = k; break; } }
  for (let t = 0; t < T; t++) {
    if (t > 0) {
      u = rng(); acc = 0;
      for (let k = 0; k < K; k++) { acc += p.A[s * K + k]; if (u <= acc) { s = k; break; } }
    }
    states[t] = s;
    const nu = p.nu[s];
    const w = randGamma(nu / 2, rng) / (nu / 2); // Gamma(nu/2, rate nu/2)
    const sd = 1 / Math.sqrt(Math.max(w, 1e-300));
    for (let d = 0; d < D; d++) {
      X[t * D + d] = p.mu[s * D + d] + Math.sqrt(p.scale[s * D + d]) * sd * randn(rng);
    }
  }
  return { X, states };
}

// ---------------------------------------------------------------------------
// Student-t HSMM
//
// The segmental recursions in hsmm.ts consume a chain of LOG parameters plus a
// prefix-summed emission matrix, and nothing else — which is what lets a
// different emission family drop in without touching them. The E step below is
// hsmm.ts's `hsmmExpectations` verbatim; only the emissions and the emission M
// step change.
// ---------------------------------------------------------------------------

export interface StudentHsmmParams extends StudentParams {
  /** K*maxDuration duration pmf; dur[j*maxDuration + (d-1)] = P(duration d | j). */
  dur: Float64Array;
  maxDuration: number;
}

export interface StudentHsmmFitOptions extends StudentFitOptions {
  maxDuration?: number;
  durationPrior?: number;
}

export interface StudentHsmmFitResult {
  params: StudentHsmmParams;
  logLik: number;
  iterations: number;
  converged: boolean;
}

const safeLog = (x: number) => (x > 0 ? Math.log(x) : -Infinity);

function studentChain(p: StudentHsmmParams): HsmmChain {
  const logPi = new Float64Array(p.K);
  for (let j = 0; j < p.K; j++) logPi[j] = safeLog(p.pi[j]);
  const logA = new Float64Array(p.K * p.K);
  for (let i = 0; i < p.K * p.K; i++) logA[i] = safeLog(p.A[i]);
  const logDur = new Float64Array(p.K * p.maxDuration);
  for (let i = 0; i < logDur.length; i++) logDur[i] = safeLog(p.dur[i]);
  return { K: p.K, maxDuration: p.maxDuration, logPi, logA, logDur };
}

/** Seed a t-HSMM from a fitted t-HMM: same emissions, geometric durations. */
export function studentHsmmFromHmm(h: StudentParams, maxDuration: number): StudentHsmmParams {
  const { K } = h;
  const A = new Float64Array(K * K);
  for (let i = 0; i < K; i++) {
    let off = 0;
    for (let j = 0; j < K; j++) if (i !== j) off += h.A[i * K + j];
    for (let j = 0; j < K; j++) A[i * K + j] = i === j ? 0 : off > 0 ? h.A[i * K + j] / off : 1 / (K - 1);
  }
  const dur = new Float64Array(K * maxDuration);
  for (let j = 0; j < K; j++) {
    const stay = Math.min(Math.max(h.A[j * K + j], 1e-6), 1 - 1e-6);
    let total = 0;
    for (let d = 1; d <= maxDuration; d++) {
      const pr = Math.pow(stay, d - 1) * (1 - stay);
      dur[j * maxDuration + (d - 1)] = pr;
      total += pr;
    }
    for (let d = 0; d < maxDuration; d++) dur[j * maxDuration + d] /= total;
  }
  return { ...cloneParams(h), A, dur, maxDuration };
}

function emStepStudentHsmm(
  X: Float64Array, T: number, p: StudentHsmmParams,
  scaleFloor: number, durationPrior: number, transitionPrior: number,
  nuMin: number, nuMax: number, fixedNu: number | null,
): number {
  const { K, D, maxDuration } = p;
  const logB = studentLogEmissions(X, T, p);
  const P = emissionPrefix(logB, T, K);
  const { gamma, xi, durCount, piCount, logZ: logLik } = hsmmExpectations(P, T, studentChain(p));
  if (!Number.isFinite(logLik)) return logLik;

  let piTotal = 0;
  for (let j = 0; j < K; j++) piTotal += piCount[j];
  for (let j = 0; j < K; j++) p.pi[j] = piTotal > 0 ? piCount[j] / piTotal : 1 / K;

  for (let i = 0; i < K; i++) {
    let row = 0;
    for (let j = 0; j < K; j++) if (i !== j) row += xi[i * K + j] + transitionPrior;
    for (let j = 0; j < K; j++) {
      p.A[i * K + j] = i === j ? 0 : row > 0 ? (xi[i * K + j] + transitionPrior) / row : 1 / (K - 1);
    }
  }

  for (let j = 0; j < K; j++) {
    let total = 0;
    for (let d = 0; d < maxDuration; d++) total += durCount[j * maxDuration + d] + durationPrior;
    for (let d = 0; d < maxDuration; d++) {
      p.dur[j * maxDuration + d] = total > 0 ? (durCount[j * maxDuration + d] + durationPrior) / total : 1 / maxDuration;
    }
  }

  // Same weighted-t emission M step as the HMM, driven by the segmental gamma.
  const U = new Float64Array(T * K);
  const gsum = new Float64Array(K);
  const wsum = new Float64Array(K);
  const muNew = new Float64Array(K * D);
  for (let k = 0; k < K; k++) {
    const nk = p.nu[k];
    for (let t = 0; t < T; t++) {
      let delta = 0;
      for (let d = 0; d < D; d++) {
        const diff = X[t * D + d] - p.mu[k * D + d];
        delta += (diff * diff) / p.scale[k * D + d];
      }
      const u = (nk + D) / (nk + delta);
      U[t * K + k] = u;
      const g = gamma[t * K + k];
      gsum[k] += g;
      const w = g * u;
      wsum[k] += w;
      for (let d = 0; d < D; d++) muNew[k * D + d] += w * X[t * D + d];
    }
  }
  for (let k = 0; k < K; k++) {
    if (wsum[k] > 1e-12) for (let d = 0; d < D; d++) muNew[k * D + d] /= wsum[k];
    else for (let d = 0; d < D; d++) muNew[k * D + d] = p.mu[k * D + d];
  }
  const scaleNew = new Float64Array(K * D);
  for (let t = 0; t < T; t++) {
    for (let k = 0; k < K; k++) {
      const w = gamma[t * K + k] * U[t * K + k];
      if (w === 0) continue;
      for (let d = 0; d < D; d++) {
        const diff = X[t * D + d] - muNew[k * D + d];
        scaleNew[k * D + d] += w * diff * diff;
      }
    }
  }
  for (let k = 0; k < K; k++) {
    for (let d = 0; d < D; d++) {
      const v = gsum[k] > 1e-8 ? scaleNew[k * D + d] / gsum[k] : p.scale[k * D + d];
      p.scale[k * D + d] = Math.max(v, scaleFloor);
    }
  }
  p.mu.set(muNew);

  if (fixedNu === null) {
    const delta = new Float64Array(T);
    const w = new Float64Array(T);
    for (let k = 0; k < K; k++) {
      if (gsum[k] <= 1e-6) continue;
      for (let t = 0; t < T; t++) {
        let dl = 0;
        for (let d = 0; d < D; d++) {
          const diff = X[t * D + d] - p.mu[k * D + d];
          dl += (diff * diff) / p.scale[k * D + d];
        }
        delta[t] = dl;
        w[t] = gamma[t * K + k];
      }
      p.nu[k] = solveNu(delta, w, T, D, nuMin, nuMax, p.nu[k]);
    }
  }

  return logLik;
}

export function fitStudentHsmm(
  X: Float64Array, T: number, D: number, options: StudentHsmmFitOptions = {},
): StudentHsmmFitResult {
  const K = options.states ?? 3;
  const maxDuration = options.maxDuration ?? 60;
  const maxIter = options.maxIter ?? 100;
  const tol = options.tol ?? 1e-6;
  const scaleFloor = options.scaleFloor ?? options.varFloor ?? 1e-8;
  const durationPrior = options.durationPrior ?? 0.05;
  const transitionPrior = options.transitionPrior ?? 0.1;
  const nuMin = options.nuMin ?? 2.02;
  const nuMax = options.nuMax ?? 200;
  const fixedNu = options.fixedNu ?? null;

  // Warm start from a t-HMM, exactly as hsmm.ts warm starts from a Gaussian HMM.
  const base = fitStudentHmm(X, T, D, {
    ...options,
    states: K,
    restarts: options.restarts ?? 4,
    maxIter: options.maxIter ?? 200,
  });
  const p = studentHsmmFromHmm(base.params, maxDuration);

  let prev = -Infinity;
  let iter = 0;
  let converged = false;
  let logLik = -Infinity;
  for (; iter < maxIter; iter++) {
    logLik = emStepStudentHsmm(X, T, p, scaleFloor, durationPrior, transitionPrior, nuMin, nuMax, fixedNu);
    if (!Number.isFinite(logLik)) break;
    if (Math.abs(logLik - prev) / T < tol) { converged = true; break; }
    prev = logLik;
  }
  return sortStudentHsmmStates({ params: p, logLik, iterations: iter, converged });
}

export function relabelStudentHsmm(p: StudentHsmmParams): StudentHsmmParams {
  const { K, D, maxDuration } = p;
  const order = Array.from({ length: K }, (_, k) => k).sort((a, b) => p.mu[a * D] - p.mu[b * D]);
  const base = relabelStudent(p);
  const dur = new Float64Array(K * maxDuration);
  for (let ni = 0; ni < K; ni++) {
    const oi = order[ni];
    for (let d = 0; d < maxDuration; d++) dur[ni * maxDuration + d] = p.dur[oi * maxDuration + d];
  }
  return { ...base, dur, maxDuration };
}

export function sortStudentHsmmStates(res: StudentHsmmFitResult): StudentHsmmFitResult {
  return { ...res, params: relabelStudentHsmm(res.params) };
}

/** Mean dwell time implied by each state's learned duration pmf. */
export function expectedDurationsStudent(p: StudentHsmmParams): number[] {
  return Array.from({ length: p.K }, (_, j) => {
    let m = 0;
    for (let d = 1; d <= p.maxDuration; d++) m += d * p.dur[j * p.maxDuration + (d - 1)];
    return m;
  });
}

/** Sequence log-likelihood of a t-HSMM. Useful for held-out comparisons. */
export function studentHsmmLogLik(X: Float64Array, T: number, p: StudentHsmmParams): number {
  const logB = studentLogEmissions(X, T, p);
  const P = emissionPrefix(logB, T, p.K);
  return hsmmExpectations(P, T, studentChain(p)).logZ;
}
