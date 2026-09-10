/**
 * The parts shared by both posterior engines.
 *
 * Two of them exist. `vb.ts` fits a variational approximation by expectation
 * maximization; `mcmc.ts` draws from the exact posterior by Gibbs sampling.
 * They put the same conjugate priors on the same model, so they share the
 * priors, the conjugate random draws, and everything downstream of "here is a
 * bag of parameter draws" — credible intervals on the state means, on dwell
 * times, and on the trading signal itself.
 *
 * Keeping the summary layer here is what makes the two comparable: any
 * disagreement between them is inference, not accounting.
 */

import { randn, type HmmParams } from "./hmm";

export interface Priors {
  /** Prior mean for each state's feature mean. Zero = "assume no drift". */
  mu0: number;
  /**
   * Prior strength on mu0, in pseudo-observations. Larger shrinks harder;
   * 0.01 is nearly flat, 10 meaningfully doubts a weakly-evidenced state.
   */
  kappa0: number;
  /** Inverse-Gamma shape and scale on the emission variance. */
  a0: number;
  b0: number;
  /** Dirichlet concentration on transition rows. */
  alpha: number;
  /** Extra concentration on the diagonal, encoding "regimes persist". */
  alphaSelf: number;
}

export const DEFAULT_PRIORS: Priors = {
  mu0: 0, kappa0: 0.5, a0: 2, b0: 1, alpha: 1, alphaSelf: 8,
};

/** Gamma(shape, 1) by Marsaglia-Tsang. Exported for testing. */
export function sampleGamma(shape: number, rng: () => number): number {
  if (shape < 1) {
    // Boost a sub-unit shape, then correct.
    return sampleGamma(shape + 1, rng) * Math.pow(Math.max(rng(), 1e-300), 1 / shape);
  }
  const d = shape - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  for (let guard = 0; guard < 1000; guard++) {
    let x: number, v: number;
    do {
      x = randn(rng);
      v = 1 + c * x;
    } while (v <= 0);
    v = v * v * v;
    const u = rng();
    if (u < 1 - 0.0331 * x * x * x * x) return d * v;
    if (Math.log(Math.max(u, 1e-300)) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
  }
  return d; // pathological fallback
}

export function sampleDirichlet(alphas: number[], rng: () => number): number[] {
  const draws = alphas.map((a) => sampleGamma(Math.max(a, 1e-6), rng));
  const total = draws.reduce((x, y) => x + y, 0);
  return total > 0 ? draws.map((d) => d / total) : alphas.map(() => 1 / alphas.length);
}

/** Inverse-Gamma(a, b): if g ~ Gamma(a, 1) then b/g ~ InvGamma(a, b). */
export function sampleInvGamma(a: number, b: number, rng: () => number): number {
  const g = sampleGamma(a, rng);
  return g > 0 ? b / g : b / Math.max(a, 1e-6);
}

/**
 * What the summary functions need: a bag of parameter draws, however they were
 * produced. Gibbs sweeps and variational draws both satisfy this.
 */
export interface PosteriorDraws {
  K: number;
  D: number;
  samples: HmmParams[];
}

export interface Interval {
  mean: number;
  median: number;
  lower: number;
  upper: number;
  /** Posterior probability the quantity is above zero. */
  pPositive: number;
}

function summarize(values: number[], credible = 0.9): Interval {
  const s = [...values].sort((a, b) => a - b);
  const q = (f: number) => s[Math.min(s.length - 1, Math.max(0, Math.floor(f * s.length)))];
  const tail = (1 - credible) / 2;
  return {
    mean: values.reduce((a, b) => a + b, 0) / values.length,
    median: q(0.5),
    lower: q(tail),
    upper: q(1 - tail),
    pPositive: values.filter((v) => v > 0).length / values.length,
  };
}

/**
 * Posterior for each state's mean of feature `d`, in the units the scaler
 * was fitted on. Pass an unscale function to read them as real returns.
 */
export function posteriorStateMeans(
  res: PosteriorDraws,
  d = 0,
  unscale: (v: number) => number = (v) => v,
  credible = 0.9,
): Interval[] {
  return Array.from({ length: res.K }, (_, k) =>
    summarize(res.samples.map((s) => unscale(s.mu[k * res.D + d])), credible));
}

/** Posterior for each state's expected dwell time, 1 / (1 - A[k][k]). */
export function posteriorDurations(res: PosteriorDraws, credible = 0.9): Interval[] {
  return Array.from({ length: res.K }, (_, k) =>
    summarize(res.samples.map((s) => {
      const stay = Math.min(s.A[k * res.K + k], 1 - 1e-9);
      return 1 / (1 - stay);
    }), credible));
}

/**
 * Posterior of the trading signal itself.
 *
 * Rather than plugging point estimates into sum_k P(k) * mu_k, this evaluates
 * that sum under every draw, so the spread is honest uncertainty on the edge
 * rather than a single number pretending to be known.
 *
 * THE COMPARISON THAT MATTERS. `edge` is a PER-BAR quantity; the round trip is
 * paid ONCE. Testing one bar's expected return against the whole round trip is
 * a category error — the same one that silently produced zero trades before
 * `--duration-aware` existed. A position held for its regime's dwell time
 * accumulates `edge * holdBars`, and that is what has to clear the cost.
 *
 * The hold is drawn from the posterior too: each draw has its own transition
 * matrix, hence its own expected dwell 1 / (1 - A[k][k]) for the state being
 * traded. So both the size of the edge and how long it persists are integrated
 * over, rather than fixed at a point estimate.
 */
export function posteriorSignal(
  res: PosteriorDraws,
  stateProbs: number[],
  unscale: (v: number) => number,
  roundTripCost: number,
  credible = 0.9,
  opts: { holdBars?: number; tradedState?: number } = {},
): Interval & { pAboveCost: number; perBar: Interval; medianHold: number } {
  const traded = opts.tradedState ?? res.K - 1;

  const perBarDraws: number[] = [];
  const tradeDraws: number[] = [];
  const holds: number[] = [];

  for (const s of res.samples) {
    let e = 0;
    for (let k = 0; k < res.K; k++) e += stateProbs[k] * unscale(s.mu[k * res.D]);
    perBarDraws.push(e);

    const stay = Math.min(s.A[traded * res.K + traded], 1 - 1e-9);
    const hold = opts.holdBars ?? 1 / (1 - stay);
    holds.push(hold);
    tradeDraws.push(e * hold);
  }

  holds.sort((a, b) => a - b);
  return {
    ...summarize(tradeDraws, credible),
    perBar: summarize(perBarDraws, credible),
    medianHold: holds[Math.floor(holds.length / 2)],
    pAboveCost: tradeDraws.filter((v) => v > roundTripCost).length / tradeDraws.length,
  };
}

/** Posterior mean parameters, usable anywhere an HmmParams is expected. */
export function posteriorMeanParams(res: PosteriorDraws): HmmParams {
  const { K, D } = res;
  const pi = new Float64Array(K);
  const A = new Float64Array(K * K);
  const mu = new Float64Array(K * D);
  const vari = new Float64Array(K * D);
  for (const s of res.samples) {
    for (let k = 0; k < K; k++) {
      pi[k] += s.pi[k] / res.samples.length;
      for (let j = 0; j < K; j++) A[k * K + j] += s.A[k * K + j] / res.samples.length;
      for (let d = 0; d < D; d++) {
        mu[k * D + d] += s.mu[k * D + d] / res.samples.length;
        vari[k * D + d] += s.vari[k * D + d] / res.samples.length;
      }
    }
  }
  return { K, D, pi, A, mu, vari };
}

/**
 * Relabel by ascending mean of feature 0.
 *
 * Both engines need this, for different reasons. Gibbs suffers label
 * switching — nothing in the likelihood distinguishes "state 1" from
 * "state 2", so the chain permutes them between sweeps and the averaged
 * posterior becomes mush. VB does not switch labels, but its states still
 * come out in whatever order the initialisation happened to put them, and
 * every caller downstream assumes state K-1 is the bullish one.
 */
export function relabel(p: HmmParams): HmmParams {
  const { K, D } = p;
  const order = Array.from({ length: K }, (_, k) => k).sort((a, b) => p.mu[a * D] - p.mu[b * D]);
  const pi = new Float64Array(K);
  const A = new Float64Array(K * K);
  const mu = new Float64Array(K * D);
  const vari = new Float64Array(K * D);
  for (let ni = 0; ni < K; ni++) {
    const oi = order[ni];
    pi[ni] = p.pi[oi];
    for (let d = 0; d < D; d++) {
      mu[ni * D + d] = p.mu[oi * D + d];
      vari[ni * D + d] = p.vari[oi * D + d];
    }
    for (let nj = 0; nj < K; nj++) A[ni * K + nj] = p.A[oi * K + order[nj]];
  }
  return { K, D, pi, A, mu, vari };
}
