/**
 * Bayesian Gaussian HMM by variational expectation maximization.
 *
 * `mcmc.ts` answers the same question by Gibbs sampling. That works, but it
 * costs what samplers cost: 1200 sweeps of forward-filter-backward-sample, a
 * burn-in you have to guess at, thinning to fight autocorrelation, and no
 * signal that any of it converged beyond eyeballing a trace. Every run gives a
 * different answer.
 *
 * Variational EM replaces the sampling with optimization. Approximate the
 * intractable posterior p(z, theta | x) by a factorized q(z) q(pi) q(A) q(mu,
 * lambda), then maximize a lower bound on log p(x) — the ELBO — by alternating
 * exactly like Baum-Welch does:
 *
 *   E step   q(z)     given the parameter posteriors, via forward-backward
 *            run on EXPECTED log parameters rather than point values
 *   M step   q(theta) given q(z), via the same conjugate updates the Gibbs
 *            sampler uses, but with expected counts instead of a sampled path
 *
 * Three things fall out that the sampler cannot give:
 *
 *   Determinism and speed. The ELBO is monotone by construction, so "has it
 *   converged" is a number, not a judgement call, and it gets there in tens of
 *   iterations rather than a thousand-plus sweeps.
 *
 *   Independent draws. q is a product of Dirichlets and Normal-Gammas, so the
 *   credible intervals come from i.i.d. draws — no burn-in, no thinning, no
 *   autocorrelation to discount the effective sample size by.
 *
 *   Model selection. The ELBO bounds log p(x) for a given K, so ELBOs are
 *   comparable across K and `selectStates` can pick the number of regimes
 *   rather than having 3 asserted at the command line. Unsupported states
 *   collapse back onto the prior on their own — a state that explains nothing
 *   takes N_k -> 0 and contributes no KL.
 *
 * What it costs: mean-field factorization assumes q(z) and q(theta) are
 * independent, which they are not. The classic consequence is that variational
 * credible intervals are too NARROW — the approximation is confident in
 * proportion to how wrong the independence assumption is. Since the whole point
 * of the posterior command is to ask whether an interval clears the round trip,
 * that bias points the dangerous way, and it is exactly why `--gibbs` is still
 * here and why vb.test.ts checks the two against each other.
 *
 * Priors are the same conjugate family the sampler uses, reparameterized from
 * variance to precision: lambda = 1/var ~ Gamma(a0, b0) and
 * mu | lambda ~ N(mu0, 1/(kappa0 lambda)). Same prior, same posterior, two
 * different ways of getting at it.
 */

import { backward, forward, logEmissions, makeRng, randn, type HmmParams } from "./hmm";
import {
  DEFAULT_PRIORS, relabel, sampleDirichlet, sampleGamma, type Priors,
} from "./posterior";

const LOG_2PI = Math.log(2 * Math.PI);

// ---------------------------------------------------------------------------
// Special functions. Needed because the E step works in expected log space:
// E[log pi_k] under a Dirichlet is a digamma difference, not log of the mean.
// ---------------------------------------------------------------------------

const LANCZOS = [
  0.99999999999980993, 676.5203681218851, -1259.1392167224028,
  771.32342877765313, -176.61502916214059, 12.507343278686905,
  -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
];

/** log Gamma(x) by the Lanczos approximation, g = 7. Accurate to ~15 digits. */
export function lgamma(x: number): number {
  if (x < 0.5) {
    // Reflection, for the left half plane.
    return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - lgamma(1 - x);
  }
  const z = x - 1;
  let acc = LANCZOS[0];
  for (let i = 1; i < 9; i++) acc += LANCZOS[i] / (z + i);
  const t = z + 7.5;
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(acc);
}

/**
 * digamma(x) = d/dx log Gamma(x). Recurrence up to x >= 6, then the asymptotic
 * series. Every argument here is a positive hyperparameter, so no reflection.
 */
export function digamma(x: number): number {
  let v = Math.max(x, 1e-12);
  let acc = 0;
  while (v < 6) { acc -= 1 / v; v += 1; }
  const inv = 1 / v;
  const inv2 = inv * inv;
  acc += Math.log(v) - 0.5 * inv;
  acc -= inv2 * (1 / 12 - inv2 * (1 / 120 - inv2 * (1 / 252 - inv2 * (1 / 240 - inv2 / 132))));
  return acc;
}

// ---------------------------------------------------------------------------

/**
 * The variational posterior q(theta), stored as its hyperparameters.
 *
 *   q(pi)          = Dirichlet(alphaPi)
 *   q(A[i])        = Dirichlet(alphaA[i])
 *   q(mu_kd, lam)  = NormalGamma(m, kappa, a, b)
 *                    lam ~ Gamma(a, rate b), mu | lam ~ N(m, 1/(kappa lam))
 */
export interface VbPosterior {
  K: number;
  D: number;
  alphaPi: Float64Array;  // K
  alphaA: Float64Array;   // K*K
  m: Float64Array;        // K*D
  kappa: Float64Array;    // K*D
  a: Float64Array;        // K*D
  b: Float64Array;        // K*D
}

export interface VbOptions {
  states?: number;
  maxIter?: number;
  /** Stop when the per-observation ELBO improves by less than this. */
  tol?: number;
  seed?: number;
  priors?: Partial<Priors>;
  /**
   * Parameters to seed the first E step from. An EM fit is the obvious choice
   * and is what the CLI passes; without one, a diffuse spread over the data's
   * own scale is used, which converges more slowly but to the same place on
   * anything well identified.
   */
  init?: HmmParams;
  /** Independent draws from q, used for the credible intervals. */
  draws?: number;
}

export interface VbResult {
  K: number;
  D: number;
  q: VbPosterior;
  /** Final evidence lower bound, in nats. Comparable across K. */
  elbo: number;
  elboTrace: number[];
  iterations: number;
  converged: boolean;
  /** i.i.d. draws from q. No burn-in, no thinning, no autocorrelation. */
  samples: HmmParams[];
  /** q(z_t = k), the variational analogue of the smoothed state posterior. */
  gamma: Float64Array;
  /** Expected occupancy of each state, sum_t q(z_t = k). Near zero = pruned. */
  occupancy: Float64Array;
}

/** A parameter-shaped object for forward/backward, which only reads K, pi, A. */
function chainOnly(K: number, pi: Float64Array, A: Float64Array): HmmParams {
  return { K, D: 1, pi, A, mu: new Float64Array(0), vari: new Float64Array(0) };
}

interface EStep {
  gamma: Float64Array;  // T*K
  xi: Float64Array;     // K*K, summed over t
  logZ: number;
}

/**
 * One E step: forward-backward over a given emission table and chain, plus the
 * expected transition counts. Identical in structure to the Baum-Welch E step
 * in hmm.ts — the only difference is what gets passed in. Here pi and A are
 * exp(E[log .]) rather than probabilities, so their rows sum to less than one
 * and `logZ` is the log of a sub-normalized mass rather than a likelihood.
 * That is the point: that quantity is the first term of the ELBO.
 */
function eStep(logB: Float64Array, T: number, K: number, pi: Float64Array, A: Float64Array): EStep {
  const chain = chainOnly(K, pi, A);
  const { alpha, logLik: logZ } = forward(logB, T, chain);
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

  const xi = new Float64Array(K * K);
  const b = new Float64Array(K);
  const tmp = new Float64Array(K * K);
  for (let t = 0; t < T - 1; t++) {
    let mx = -Infinity;
    for (let k = 0; k < K; k++) if (logB[(t + 1) * K + k] > mx) mx = logB[(t + 1) * K + k];
    for (let k = 0; k < K; k++) b[k] = Math.exp(logB[(t + 1) * K + k] - mx);

    let sum = 0;
    for (let i = 0; i < K; i++) {
      for (let j = 0; j < K; j++) {
        const v = alpha[t * K + i] * A[i * K + j] * b[j] * beta[(t + 1) * K + j];
        tmp[i * K + j] = v;
        sum += v;
      }
    }
    if (sum > 0 && Number.isFinite(sum)) {
      for (let i = 0; i < K * K; i++) xi[i] += tmp[i] / sum;
    }
  }
  return { gamma, xi, logZ };
}

/**
 * M step: the conjugate posterior updates, driven by expected counts.
 *
 * These are the same Normal-Inverse-Gamma and Dirichlet updates the Gibbs
 * sampler applies, with one substitution — the sampler conditions on a single
 * drawn path z and counts hard assignments; this conditions on the whole
 * distribution q(z) and counts responsibilities. A state with no support gets
 * N_k = 0 and its hyperparameters revert exactly to the prior, which is how
 * variational HMMs prune states they do not need.
 */
function mStep(
  X: Float64Array, T: number, D: number, K: number,
  e: EStep, pri: Priors,
): VbPosterior {
  const alphaPi = new Float64Array(K);
  for (let k = 0; k < K; k++) alphaPi[k] = pri.alpha + e.gamma[k];

  const alphaA = new Float64Array(K * K);
  for (let i = 0; i < K; i++) {
    for (let j = 0; j < K; j++) {
      alphaA[i * K + j] = e.xi[i * K + j] + pri.alpha + (i === j ? pri.alphaSelf : 0);
    }
  }

  const N = new Float64Array(K);
  const sum = new Float64Array(K * D);
  const sumSq = new Float64Array(K * D);
  for (let t = 0; t < T; t++) {
    for (let k = 0; k < K; k++) {
      const w = e.gamma[t * K + k];
      if (w <= 0) continue;
      N[k] += w;
      for (let d = 0; d < D; d++) {
        const x = X[t * D + d];
        sum[k * D + d] += w * x;
        sumSq[k * D + d] += w * x * x;
      }
    }
  }

  const m = new Float64Array(K * D);
  const kappa = new Float64Array(K * D);
  const a = new Float64Array(K * D);
  const b = new Float64Array(K * D);
  for (let k = 0; k < K; k++) {
    const n = N[k];
    for (let d = 0; d < D; d++) {
      const s = sum[k * D + d];
      const xbar = n > 1e-12 ? s / n : pri.mu0;
      // Responsibility-weighted scatter, floored at zero against roundoff.
      const scatter = n > 1e-12 ? Math.max(sumSq[k * D + d] - n * xbar * xbar, 0) : 0;

      const kap = pri.kappa0 + n;
      kappa[k * D + d] = kap;
      // kappa0 in the denominator is the shrinkage: little evidence, little
      // movement away from mu0.
      m[k * D + d] = (pri.kappa0 * pri.mu0 + s) / kap;
      a[k * D + d] = pri.a0 + n / 2;
      b[k * D + d] = pri.b0 + 0.5 * scatter + (pri.kappa0 * n * (xbar - pri.mu0) ** 2) / (2 * kap);
    }
  }
  return { K, D, alphaPi, alphaA, m, kappa, a, b };
}

/** exp(E_q[log p]) for a Dirichlet row: exp(psi(alpha_i) - psi(sum alpha)). */
function expectedLogSimplex(alphas: Float64Array, offset: number, K: number, out: Float64Array) {
  let total = 0;
  for (let k = 0; k < K; k++) total += alphas[offset + k];
  const psiTotal = digamma(total);
  for (let k = 0; k < K; k++) out[k] = Math.exp(digamma(alphas[offset + k]) - psiTotal);
}

/**
 * E_q[ log N(x_td | mu_kd, 1/lambda_kd) ] for every (t, k).
 *
 * This is where variational EM stops being EM-with-priors. Baum-Welch plugs
 * point estimates into the Gaussian density; this integrates the density
 * against q(mu, lambda). The extra -1/(2 kappa) term is the penalty for not
 * knowing mu, and it is what stops a state with three observations from
 * claiming a razor-sharp emission.
 */
function expectedLogEmissions(X: Float64Array, T: number, q: VbPosterior): Float64Array {
  const { K, D, m, kappa, a, b } = q;
  const logB = new Float64Array(T * K);
  // Everything not involving x, precomputed per (k, d).
  const constant = new Float64Array(K);
  const prec = new Float64Array(K * D);
  for (let k = 0; k < K; k++) {
    let c = 0;
    for (let d = 0; d < D; d++) {
      const i = k * D + d;
      prec[i] = a[i] / b[i];                               // E[lambda]
      c += digamma(a[i]) - Math.log(b[i]) - LOG_2PI - 1 / kappa[i];  // E[log lambda] - ...
    }
    constant[k] = 0.5 * c;
  }
  for (let t = 0; t < T; t++) {
    for (let k = 0; k < K; k++) {
      let quad = 0;
      for (let d = 0; d < D; d++) {
        const i = k * D + d;
        const diff = X[t * D + d] - m[i];
        quad += prec[i] * diff * diff;
      }
      logB[t * K + k] = constant[k] - 0.5 * quad;
    }
  }
  return logB;
}

/** KL( Dirichlet(alpha) || Dirichlet(beta) ), each read from its own offset. */
function klDirichlet(
  alpha: Float64Array, aOff: number,
  beta: Float64Array, bOff: number,
  K: number,
): number {
  let sa = 0, sb = 0;
  for (let k = 0; k < K; k++) { sa += alpha[aOff + k]; sb += beta[bOff + k]; }
  let kl = lgamma(sa) - lgamma(sb);
  const psiSa = digamma(sa);
  for (let k = 0; k < K; k++) {
    const ak = alpha[aOff + k], bk = beta[bOff + k];
    kl += lgamma(bk) - lgamma(ak) + (ak - bk) * (digamma(ak) - psiSa);
  }
  return kl;
}

/** KL( NormalGamma(m, kappa, a, b) || NormalGamma(m0, kappa0, a0, b0) ). */
function klNormalGamma(
  m: number, kappa: number, a: number, b: number,
  m0: number, kappa0: number, a0: number, b0: number,
): number {
  // Gamma part, rate parameterization.
  const klGamma = (a - a0) * digamma(a) - lgamma(a) + lgamma(a0)
    + a0 * (Math.log(b) - Math.log(b0)) + (a * (b0 - b)) / b;
  // Normal part, averaged over lambda ~ q: E[lambda] = a/b.
  const klNormal = 0.5 * (Math.log(kappa / kappa0) + kappa0 / kappa - 1
    + kappa0 * (a / b) * (m - m0) ** 2);
  return klGamma + klNormal;
}

/**
 * ELBO = log Z~ - KL(q(theta) || p(theta)).
 *
 * log Z~ is the normalizer that falls out of running forward-backward on the
 * expected log parameters — it already carries the E_q[log p(x, z)] and the
 * q(z) entropy. Subtracting the parameter KLs completes the bound on log p(x).
 */
function elboOf(logZ: number, q: VbPosterior, pri: Priors): number {
  const { K, D } = q;
  const priorPi = new Float64Array(K).fill(pri.alpha);
  let kl = klDirichlet(q.alphaPi, 0, priorPi, 0, K);

  const priorRow = new Float64Array(K);
  for (let i = 0; i < K; i++) {
    for (let j = 0; j < K; j++) priorRow[j] = pri.alpha + (i === j ? pri.alphaSelf : 0);
    kl += klDirichlet(q.alphaA, i * K, priorRow, 0, K);
  }

  for (let k = 0; k < K; k++) {
    for (let d = 0; d < D; d++) {
      const i = k * D + d;
      kl += klNormalGamma(q.m[i], q.kappa[i], q.a[i], q.b[i],
        pri.mu0, pri.kappa0, pri.a0, pri.b0);
    }
  }
  return logZ - kl;
}

/** Start q at the prior. */
function priorPosterior(K: number, D: number, pri: Priors): VbPosterior {
  const alphaA = new Float64Array(K * K);
  for (let i = 0; i < K; i++) {
    for (let j = 0; j < K; j++) alphaA[i * K + j] = pri.alpha + (i === j ? pri.alphaSelf : 0);
  }
  return {
    K, D,
    alphaPi: new Float64Array(K).fill(pri.alpha),
    alphaA,
    m: new Float64Array(K * D).fill(pri.mu0),
    kappa: new Float64Array(K * D).fill(pri.kappa0),
    a: new Float64Array(K * D).fill(pri.a0),
    b: new Float64Array(K * D).fill(pri.b0),
  };
}

/** A diffuse starting point when no EM fit is supplied. */
function diffuseInit(K: number, D: number): HmmParams {
  const mu = new Float64Array(K * D);
  for (let k = 0; k < K; k++) {
    for (let d = 0; d < D; d++) mu[k * D + d] = (k - (K - 1) / 2) * 0.8;
  }
  const A = new Float64Array(K * K);
  for (let i = 0; i < K; i++) {
    for (let j = 0; j < K; j++) A[i * K + j] = i === j ? 0.9 : 0.1 / (K - 1);
  }
  return { K, D, pi: new Float64Array(K).fill(1 / K), A, mu, vari: new Float64Array(K * D).fill(1) };
}

/** Permutation of q's states into ascending order of m[.][0]. */
function sortPosterior(q: VbPosterior): VbPosterior {
  const { K, D } = q;
  const order = Array.from({ length: K }, (_, k) => k).sort((x, y) => q.m[x * D] - q.m[y * D]);
  const out = priorPosterior(K, D, DEFAULT_PRIORS);
  for (let ni = 0; ni < K; ni++) {
    const oi = order[ni];
    out.alphaPi[ni] = q.alphaPi[oi];
    for (let nj = 0; nj < K; nj++) out.alphaA[ni * K + nj] = q.alphaA[oi * K + order[nj]];
    for (let d = 0; d < D; d++) {
      out.m[ni * D + d] = q.m[oi * D + d];
      out.kappa[ni * D + d] = q.kappa[oi * D + d];
      out.a[ni * D + d] = q.a[oi * D + d];
      out.b[ni * D + d] = q.b[oi * D + d];
    }
  }
  return { ...out, K, D };
}

function permute(v: Float64Array, T: number, K: number, order: number[]): Float64Array {
  const out = new Float64Array(T * K);
  for (let t = 0; t < T; t++) for (let k = 0; k < K; k++) out[t * K + k] = v[t * K + order[k]];
  return out;
}

/** One independent draw from q. */
export function drawFromQ(q: VbPosterior, rng: () => number): HmmParams {
  const { K, D } = q;
  const pi = Float64Array.from(sampleDirichlet(Array.from(q.alphaPi), rng));
  const A = new Float64Array(K * K);
  for (let i = 0; i < K; i++) {
    const row = sampleDirichlet(Array.from(q.alphaA.slice(i * K, i * K + K)), rng);
    for (let j = 0; j < K; j++) A[i * K + j] = row[j];
  }
  const mu = new Float64Array(K * D);
  const vari = new Float64Array(K * D);
  for (let k = 0; k < K; k++) {
    for (let d = 0; d < D; d++) {
      const i = k * D + d;
      // lambda ~ Gamma(a, rate b); Gamma(a,1)/b is the same thing.
      const lambda = Math.max(sampleGamma(q.a[i], rng) / q.b[i], 1e-300);
      const v = 1 / lambda;
      vari[i] = Math.max(v, 1e-10);
      mu[i] = q.m[i] + Math.sqrt(v / q.kappa[i]) * randn(rng);
    }
  }
  return { K, D, pi, A, mu, vari };
}

/** Posterior mean parameters. E[var] = b/(a-1) exists only for a > 1. */
export function vbMeanParams(q: VbPosterior): HmmParams {
  const { K, D } = q;
  const pi = new Float64Array(K);
  let piTotal = 0;
  for (let k = 0; k < K; k++) piTotal += q.alphaPi[k];
  for (let k = 0; k < K; k++) pi[k] = q.alphaPi[k] / piTotal;

  const A = new Float64Array(K * K);
  for (let i = 0; i < K; i++) {
    let rowTotal = 0;
    for (let j = 0; j < K; j++) rowTotal += q.alphaA[i * K + j];
    for (let j = 0; j < K; j++) A[i * K + j] = q.alphaA[i * K + j] / rowTotal;
  }

  const mu = Float64Array.from(q.m);
  const vari = new Float64Array(K * D);
  for (let i = 0; i < K * D; i++) {
    vari[i] = q.a[i] > 1 ? q.b[i] / (q.a[i] - 1) : q.b[i] / Math.max(q.a[i], 1e-6);
  }
  return { K, D, pi, A, mu, vari };
}

/**
 * Fit the variational posterior by EM on the ELBO.
 *
 * The bound is monotone, so `converged` means the improvement fell below
 * tolerance rather than "the trace looked flat". A decrease would be a bug and
 * vb.test.ts asserts it never happens.
 */
export function fitVb(X: Float64Array, T: number, D: number, options: VbOptions = {}): VbResult {
  const K = options.states ?? 3;
  const maxIter = options.maxIter ?? 300;
  const tol = options.tol ?? 1e-7;
  const drawCount = options.draws ?? 4000;
  const pri: Priors = { ...DEFAULT_PRIORS, ...(options.priors ?? {}) };
  const rng = makeRng(options.seed ?? 20240);
  if (T < K * 10) throw new Error(`need at least ${K * 10} observations for ${K} states, got ${T}`);

  // Seed the first M step with responsibilities from the initial parameters,
  // exactly as if they had come out of an E step.
  const seed = options.init ?? diffuseInit(K, D);
  const seedChain = chainOnly(K, seed.pi, seed.A);
  const seedE = eStep(logEmissions(X, T, seed), T, K, seedChain.pi, seedChain.A);
  let q = mStep(X, T, D, K, seedE, pri);

  const elboTrace: number[] = [];
  let prev = -Infinity;
  let iter = 0;
  let converged = false;
  let last: EStep = seedE;

  const piT = new Float64Array(K);
  const AT = new Float64Array(K * K);

  for (; iter < maxIter; iter++) {
    // --- E step: forward-backward on expected log parameters ---
    expectedLogSimplex(q.alphaPi, 0, K, piT);
    for (let i = 0; i < K; i++) {
      const row = new Float64Array(K);
      expectedLogSimplex(q.alphaA, i * K, K, row);
      for (let j = 0; j < K; j++) AT[i * K + j] = row[j];
    }
    const logB = expectedLogEmissions(X, T, q);
    last = eStep(logB, T, K, piT, AT);

    // The bound is evaluated at (q(z) just computed, q(theta) it came from),
    // which is the pairing that makes the following M step monotone.
    const elbo = elboOf(last.logZ, q, pri);
    elboTrace.push(elbo);

    // --- M step ---
    q = mStep(X, T, D, K, last, pri);

    if (Math.abs(elbo - prev) / T < tol) { converged = true; iter++; break; }
    prev = elbo;
  }

  // Order states by ascending mean return, and carry gamma along with them.
  const order = Array.from({ length: K }, (_, k) => k).sort((x, y) => q.m[x * D] - q.m[y * D]);
  const sorted = sortPosterior(q);
  const gamma = permute(last.gamma, T, K, order);

  const occupancy = new Float64Array(K);
  for (let t = 0; t < T; t++) for (let k = 0; k < K; k++) occupancy[k] += gamma[t * K + k];

  const samples: HmmParams[] = [];
  for (let i = 0; i < drawCount; i++) samples.push(relabel(drawFromQ(sorted, rng)));

  return {
    K, D,
    q: sorted,
    elbo: elboTrace.length ? elboTrace[elboTrace.length - 1] : -Infinity,
    elboTrace,
    iterations: iter,
    converged,
    samples,
    gamma,
    occupancy,
  };
}

export interface StateSelection {
  best: VbResult;
  /** One row per candidate K, in the order tried. */
  scores: { states: number; elbo: number; elboPerBar: number; occupied: number }[];
}

/**
 * Pick the number of regimes by ELBO.
 *
 * The ELBO lower-bounds log p(x) for whatever K it was fitted at, and the
 * parameter KL grows with K, so the bound stops improving once extra states
 * stop paying for themselves. That is model selection for free — no held-out
 * split, no information criterion with an ad hoc penalty.
 *
 * Read `occupied` alongside it: a state whose expected occupancy is a rounding
 * error has been pruned back onto the prior, which says the same thing the
 * ELBO does but more directly.
 */
export function selectStates(
  X: Float64Array, T: number, D: number,
  candidates: number[] = [2, 3, 4, 5],
  options: Omit<VbOptions, "states" | "init"> & { init?: (k: number) => HmmParams } = {},
): StateSelection {
  const { init, ...rest } = options;
  let best: VbResult | null = null;
  const scores: StateSelection["scores"] = [];

  for (const k of candidates) {
    if (T < k * 10) continue;
    const res = fitVb(X, T, D, { ...rest, states: k, init: init?.(k) });
    const occupied = Array.from(res.occupancy).filter((n) => n > 0.01 * T).length;
    scores.push({ states: k, elbo: res.elbo, elboPerBar: res.elbo / T, occupied });
    if (!best || res.elbo > best.elbo) best = res;
  }
  if (!best) throw new Error(`no candidate K fits in ${T} observations`);
  return { best, scores };
}

/**
 * Smoothed state posterior from an already-fitted q, for new data.
 * Uses the same expected-log-parameter emissions as the fit did.
 */
export function vbPosteriors(X: Float64Array, T: number, q: VbPosterior): Float64Array {
  const { K } = q;
  const piT = new Float64Array(K);
  const AT = new Float64Array(K * K);
  expectedLogSimplex(q.alphaPi, 0, K, piT);
  for (let i = 0; i < K; i++) {
    const row = new Float64Array(K);
    expectedLogSimplex(q.alphaA, i * K, K, row);
    for (let j = 0; j < K; j++) AT[i * K + j] = row[j];
  }
  return eStep(expectedLogEmissions(X, T, q), T, K, piT, AT).gamma;
}
