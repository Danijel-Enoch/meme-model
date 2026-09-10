/**
 * Bayesian Gaussian HMM by Gibbs sampling.
 *
 * `vb.ts` is the default posterior engine and is far faster. This one is the
 * reference it is checked against: Gibbs is asymptotically exact, variational
 * EM is not, and the only way to know how much the approximation costs on this
 * data is to have both.
 *
 * Everything else in this repo takes EM's point estimates and treats them as
 * fact. The trading signal is
 *
 *     E[r] = sum_k P(z = k) * mu_k
 *
 * with mu_k plugged straight in. But mu_k is estimated from however many bars
 * happened to land in state k, and a rare pump state might own 200 of 3000
 * bars. Its mean return has a standard error, and if that error is comparable
 * to the round-trip cost then the signal is noise wearing a point estimate.
 *
 * A posterior answers this directly. Two things fall out:
 *
 *   Credible intervals on mu_k. If the 90% interval for the bullish state
 *   straddles zero, there is no direction to trade, whatever EM reported.
 *
 *   Shrinkage. A Normal-Inverse-Gamma prior centred at zero pulls poorly
 *   evidenced states toward "no drift", in proportion to how little data
 *   supports them. EM cannot do this; it will happily fit a 90bps drift to
 *   twenty observations.
 *
 * Sampler: Gibbs, alternating
 *   1. state path  | parameters   via forward-filter backward-sample
 *   2. parameters  | state path   via conjugate Dirichlet and NIG draws
 */

import { logEmissions, forward, makeRng, randn, type HmmParams } from "./hmm";
import {
  DEFAULT_PRIORS, relabel, sampleDirichlet, sampleInvGamma, type Priors,
} from "./posterior";

export type { Priors as McmcPriors } from "./posterior";

export interface McmcOptions {
  states?: number;
  iterations?: number;
  burnIn?: number;
  /** Keep every nth post-burn-in draw, to cut autocorrelation. */
  thin?: number;
  seed?: number;
  priors?: Partial<Priors>;
  /** Optional EM fit to initialise from; otherwise a random path is used. */
  init?: HmmParams;
}

export interface McmcResult {
  K: number;
  D: number;
  /** Retained draws: samples[i] is one full parameter set. */
  samples: HmmParams[];
  /** Mean log-likelihood across retained draws. */
  meanLogLik: number;
  logLikTrace: number[];
  iterations: number;
  burnIn: number;
}

/**
 * Forward-filter backward-sample: draw a whole state path from its exact
 * conditional distribution given the parameters.
 *
 * The forward pass gives P(z_t | x_1..t). Sampling backwards from T then
 * corrects each draw by the transition into the state already sampled for
 * t+1, which yields a draw from the joint P(z_1..T | x_1..T) rather than a
 * sequence of independent marginal draws.
 */
export function ffbs(X: Float64Array, T: number, p: HmmParams, rng: () => number): Int32Array {
  const { K, A } = p;
  const logB = logEmissions(X, T, p);
  const { alpha } = forward(logB, T, p);
  const path = new Int32Array(T);

  const draw = (weights: number[]) => {
    let total = 0;
    for (const w of weights) total += w;
    if (!(total > 0)) return Math.floor(rng() * weights.length);
    let u = rng() * total;
    for (let k = 0; k < weights.length; k++) {
      u -= weights[k];
      if (u <= 0) return k;
    }
    return weights.length - 1;
  };

  const last: number[] = [];
  for (let k = 0; k < K; k++) last.push(alpha[(T - 1) * K + k]);
  path[T - 1] = draw(last);

  for (let t = T - 2; t >= 0; t--) {
    const w: number[] = [];
    for (let k = 0; k < K; k++) w.push(alpha[t * K + k] * A[k * K + path[t + 1]]);
    path[t] = draw(w);
  }
  return path;
}

export function runMcmc(X: Float64Array, T: number, D: number, options: McmcOptions = {}): McmcResult {
  const K = options.states ?? 3;
  const iterations = options.iterations ?? 2000;
  const burnIn = options.burnIn ?? Math.floor(iterations / 2);
  const thin = options.thin ?? 2;
  const rng = makeRng(options.seed ?? 20240);
  const pri: Priors = { ...DEFAULT_PRIORS, ...(options.priors ?? {}) };

  // Initialise from an EM fit when given; otherwise spread the states over the
  // data's own scale so the first FFBS pass is not degenerate.
  let cur: HmmParams;
  if (options.init) {
    cur = relabel({
      K, D,
      pi: Float64Array.from(options.init.pi),
      A: Float64Array.from(options.init.A),
      mu: Float64Array.from(options.init.mu),
      vari: Float64Array.from(options.init.vari),
    });
  } else {
    const mu = new Float64Array(K * D);
    const vari = new Float64Array(K * D).fill(1);
    for (let k = 0; k < K; k++) {
      for (let d = 0; d < D; d++) mu[k * D + d] = (k - (K - 1) / 2) * 0.8;
    }
    const A = new Float64Array(K * K);
    for (let i = 0; i < K; i++) {
      for (let j = 0; j < K; j++) A[i * K + j] = i === j ? 0.9 : 0.1 / (K - 1);
    }
    cur = { K, D, pi: new Float64Array(K).fill(1 / K), A, mu, vari };
  }

  const samples: HmmParams[] = [];
  const logLikTrace: number[] = [];

  for (let iter = 0; iter < iterations; iter++) {
    // --- 1. state path | parameters ---
    const z = ffbs(X, T, cur, rng);

    // --- 2. parameters | state path ---
    const counts = new Float64Array(K * K);
    const nk = new Float64Array(K);
    const sum = new Float64Array(K * D);
    const sumSq = new Float64Array(K * D);
    const firstCount = new Float64Array(K);
    firstCount[z[0]] += 1;

    for (let t = 0; t < T; t++) {
      const k = z[t];
      nk[k] += 1;
      for (let d = 0; d < D; d++) {
        const x = X[t * D + d];
        sum[k * D + d] += x;
        sumSq[k * D + d] += x * x;
      }
      if (t > 0) counts[z[t - 1] * K + k] += 1;
    }

    const pi = sampleDirichlet(Array.from({ length: K }, (_, k) => pri.alpha + firstCount[k]), rng);
    const A = new Float64Array(K * K);
    for (let i = 0; i < K; i++) {
      const alphas = Array.from({ length: K }, (_, j) =>
        counts[i * K + j] + pri.alpha + (i === j ? pri.alphaSelf : 0));
      const row = sampleDirichlet(alphas, rng);
      for (let j = 0; j < K; j++) A[i * K + j] = row[j];
    }

    const mu = new Float64Array(K * D);
    const vari = new Float64Array(K * D);
    for (let k = 0; k < K; k++) {
      for (let d = 0; d < D; d++) {
        const n = nk[k];
        const s = sum[k * D + d];
        const ss = sumSq[k * D + d];
        const xbar = n > 0 ? s / n : pri.mu0;
        const sse = n > 0 ? Math.max(ss - n * xbar * xbar, 0) : 0;

        // Normal-Inverse-Gamma conjugate update. kappaN in the denominator of
        // muN is what produces shrinkage: a state with few observations is
        // pulled toward mu0 in proportion to how little evidence it has.
        const kappaN = pri.kappa0 + n;
        const muN = (pri.kappa0 * pri.mu0 + s) / kappaN;
        const aN = pri.a0 + n / 2;
        const bN = pri.b0 + 0.5 * sse + (pri.kappa0 * n * (xbar - pri.mu0) ** 2) / (2 * kappaN);

        const v = Math.max(sampleInvGamma(aN, bN, rng), 1e-10);
        vari[k * D + d] = v;
        mu[k * D + d] = muN + Math.sqrt(v / kappaN) * randn(rng);
      }
    }

    cur = relabel({ K, D, pi: Float64Array.from(pi), A, mu, vari });

    const { logLik } = forward(logEmissions(X, T, cur), T, cur);
    logLikTrace.push(logLik);
    if (iter >= burnIn && (iter - burnIn) % thin === 0) {
      samples.push({
        K, D,
        pi: Float64Array.from(cur.pi),
        A: Float64Array.from(cur.A),
        mu: Float64Array.from(cur.mu),
        vari: Float64Array.from(cur.vari),
      });
    }
  }

  const kept = logLikTrace.slice(burnIn);
  return {
    K, D, samples,
    meanLogLik: kept.reduce((a, b) => a + b, 0) / Math.max(kept.length, 1),
    logLikTrace,
    iterations,
    burnIn,
  };
}
