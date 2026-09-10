/**
 * Gaussian HMM with diagonal covariance emissions.
 *
 * Latent state z_t in {0..K-1} follows a Markov chain; the observed feature
 * vector x_t in R^D is drawn from N(mu[z_t], diag(var[z_t])).
 *
 * Everything is stored as flat Float64Array for speed:
 *   A      K*K   transition matrix, A[i*K+j] = P(z_t=j | z_{t-1}=i)
 *   pi     K     initial distribution
 *   mu     K*D   per-state means
 *   vari   K*D   per-state variances (diagonal covariance)
 */

const LOG_2PI = Math.log(2 * Math.PI);

export interface HmmParams {
  K: number;
  D: number;
  pi: Float64Array;
  A: Float64Array;
  mu: Float64Array;
  vari: Float64Array;
}

export interface FitOptions {
  /** Number of hidden states. 3 is the sane default: pump / chop / dump. */
  states?: number;
  maxIter?: number;
  /** Stop when the per-observation log-likelihood improves by less than this. */
  tol?: number;
  /** Random restarts; the best log-likelihood wins. */
  restarts?: number;
  /** Lower bound on variance, guards against a state collapsing onto one point. */
  varFloor?: number;
  /** Dirichlet-style smoothing added to transition counts. */
  transitionPrior?: number;
  seed?: number;
  verbose?: boolean;
}

export interface FitResult {
  params: HmmParams;
  logLik: number;
  iterations: number;
  converged: boolean;
}

/** Deterministic PRNG so runs are reproducible. */
export function makeRng(seed: number): () => number {
  let s = seed >>> 0 || 1;
  return () => {
    s ^= s << 13; s >>>= 0;
    s ^= s >> 17;
    s ^= s << 5; s >>>= 0;
    return s / 4294967296;
  };
}

/** Standard normal via Box-Muller. */
export function randn(rng: () => number): number {
  let u = 0, v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/**
 * log N(x_t | mu_k, diag(var_k)) for every (t, k). Returns a T*K flat array.
 * Working in log space first, then exponentiating with a per-row max removed,
 * is what keeps the recursions stable when variances are tiny (they are —
 * minute-bar log returns live around 1e-3).
 */
export function logEmissions(X: Float64Array, T: number, p: HmmParams): Float64Array {
  const { K, D, mu, vari } = p;
  const logB = new Float64Array(T * K);
  // Precompute the normalizing constant -0.5 * (D*log(2pi) + sum log var) per state.
  const norm = new Float64Array(K);
  for (let k = 0; k < K; k++) {
    let s = D * LOG_2PI;
    for (let d = 0; d < D; d++) s += Math.log(vari[k * D + d]);
    norm[k] = -0.5 * s;
  }
  for (let t = 0; t < T; t++) {
    for (let k = 0; k < K; k++) {
      let quad = 0;
      for (let d = 0; d < D; d++) {
        const diff = X[t * D + d] - mu[k * D + d];
        quad += (diff * diff) / vari[k * D + d];
      }
      logB[t * K + k] = norm[k] - 0.5 * quad;
    }
  }
  return logB;
}

interface ForwardResult {
  /** T*K, alpha[t][k] = P(z_t = k | x_1..t) — normalized, so these are filtered probabilities. */
  alpha: Float64Array;
  /** Per-step log scaling factors; their sum is the total log-likelihood. */
  logScale: Float64Array;
  logLik: number;
}

/** Scaled forward pass. alpha rows are normalized, so no underflow over long series. */
export function forward(logB: Float64Array, T: number, p: HmmParams): ForwardResult {
  const { K, pi, A } = p;
  const alpha = new Float64Array(T * K);
  const logScale = new Float64Array(T);
  const b = new Float64Array(K);

  for (let t = 0; t < T; t++) {
    // Shift the log-densities by their row max before exponentiating.
    let m = -Infinity;
    for (let k = 0; k < K; k++) if (logB[t * K + k] > m) m = logB[t * K + k];
    for (let k = 0; k < K; k++) b[k] = Math.exp(logB[t * K + k] - m);

    let sum = 0;
    if (t === 0) {
      for (let k = 0; k < K; k++) {
        const v = pi[k] * b[k];
        alpha[k] = v;
        sum += v;
      }
    } else {
      for (let k = 0; k < K; k++) {
        let acc = 0;
        for (let j = 0; j < K; j++) acc += alpha[(t - 1) * K + j] * A[j * K + k];
        const v = acc * b[k];
        alpha[t * K + k] = v;
        sum += v;
      }
    }
    if (sum <= 0 || !Number.isFinite(sum)) {
      // Degenerate step (every state assigns ~zero mass): fall back to uniform.
      for (let k = 0; k < K; k++) alpha[t * K + k] = 1 / K;
      logScale[t] = m;
    } else {
      for (let k = 0; k < K; k++) alpha[t * K + k] /= sum;
      logScale[t] = m + Math.log(sum);
    }
  }

  let logLik = 0;
  for (let t = 0; t < T; t++) logLik += logScale[t];
  return { alpha, logScale, logLik };
}

/** Scaled backward pass, paired with the normalized alphas above. */
function backward(logB: Float64Array, T: number, p: HmmParams): Float64Array {
  const { K, A } = p;
  const beta = new Float64Array(T * K);
  const b = new Float64Array(K);
  for (let k = 0; k < K; k++) beta[(T - 1) * K + k] = 1;

  for (let t = T - 2; t >= 0; t--) {
    let m = -Infinity;
    for (let k = 0; k < K; k++) if (logB[(t + 1) * K + k] > m) m = logB[(t + 1) * K + k];
    for (let k = 0; k < K; k++) b[k] = Math.exp(logB[(t + 1) * K + k] - m);

    let sum = 0;
    for (let i = 0; i < K; i++) {
      let acc = 0;
      for (let j = 0; j < K; j++) acc += A[i * K + j] * b[j] * beta[(t + 1) * K + j];
      beta[t * K + i] = acc;
      sum += acc;
    }
    // Renormalize each row; gamma/xi are normalized per-t anyway so the
    // dropped constant cancels out.
    if (sum > 0 && Number.isFinite(sum)) {
      for (let i = 0; i < K; i++) beta[t * K + i] /= sum;
    } else {
      for (let i = 0; i < K; i++) beta[t * K + i] = 1 / K;
    }
  }
  return beta;
}

/** Smoothed state probabilities P(z_t = k | x_1..T). Uses the whole series — training only. */
export function posteriors(X: Float64Array, T: number, p: HmmParams): { gamma: Float64Array; logLik: number } {
  const logB = logEmissions(X, T, p);
  const { alpha, logLik } = forward(logB, T, p);
  const beta = backward(logB, T, p);
  const gamma = new Float64Array(T * p.K);
  for (let t = 0; t < T; t++) {
    let sum = 0;
    for (let k = 0; k < p.K; k++) {
      const v = alpha[t * p.K + k] * beta[t * p.K + k];
      gamma[t * p.K + k] = v;
      sum += v;
    }
    for (let k = 0; k < p.K; k++) gamma[t * p.K + k] = sum > 0 ? gamma[t * p.K + k] / sum : 1 / p.K;
  }
  return { gamma, logLik };
}

/**
 * Filtered probabilities P(z_t = k | x_1..t) — the causal ones.
 * This is what a live strategy is allowed to see: no future bars leak in.
 */
export function filter(X: Float64Array, T: number, p: HmmParams): { alpha: Float64Array; logLik: number } {
  const logB = logEmissions(X, T, p);
  const { alpha, logLik } = forward(logB, T, p);
  return { alpha, logLik };
}

/** One-step-ahead state forecast P(z_{t+1} | x_1..t) from a filtered distribution. */
export function predictNext(filtered: Float64Array, offset: number, p: HmmParams): Float64Array {
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
export function viterbi(X: Float64Array, T: number, p: HmmParams): Int32Array {
  const { K, pi, A } = p;
  const logB = logEmissions(X, T, p);
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

/** k-means++ seeding, then a few Lloyd iterations, to initialize the state means. */
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
    // Sample the next center proportional to squared distance.
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
        // Empty cluster: re-seed it on a random point rather than let it die.
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

function initParams(X: Float64Array, T: number, D: number, K: number, rng: () => number, varFloor: number): HmmParams {
  const { centers, assign } = kmeansInit(X, T, D, K, rng);
  const vari = new Float64Array(K * D);
  const counts = new Float64Array(K);
  for (let t = 0; t < T; t++) {
    counts[assign[t]]++;
    for (let d = 0; d < D; d++) {
      const diff = X[t * D + d] - centers[assign[t] * D + d];
      vari[assign[t] * D + d] += diff * diff;
    }
  }
  for (let k = 0; k < K; k++) {
    for (let d = 0; d < D; d++) {
      vari[k * D + d] = counts[k] > 1 ? Math.max(vari[k * D + d] / counts[k], varFloor) : 1;
    }
  }

  // Persistent chain: regimes should last, so bias the diagonal heavily.
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

  return { K, D, pi, A, mu: centers, vari };
}

/** One Baum-Welch (EM) sweep. Returns the log-likelihood of the *current* params. */
function emStep(X: Float64Array, T: number, p: HmmParams, opts: Required<FitOptions>): number {
  const { K, D } = p;
  const logB = logEmissions(X, T, p);
  const { alpha, logLik } = forward(logB, T, p);
  const beta = backward(logB, T, p);

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

  // Expected transition counts, accumulated with per-t normalized xi.
  const xiSum = new Float64Array(K * K);
  const b = new Float64Array(K);
  for (let t = 0; t < T - 1; t++) {
    let m = -Infinity;
    for (let k = 0; k < K; k++) if (logB[(t + 1) * K + k] > m) m = logB[(t + 1) * K + k];
    for (let k = 0; k < K; k++) b[k] = Math.exp(logB[(t + 1) * K + k] - m);

    let sum = 0;
    const tmp = new Float64Array(K * K);
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

  // --- M step ---
  for (let k = 0; k < K; k++) p.pi[k] = gamma[k];

  for (let i = 0; i < K; i++) {
    let rowSum = 0;
    for (let j = 0; j < K; j++) rowSum += xiSum[i * K + j] + opts.transitionPrior;
    for (let j = 0; j < K; j++) {
      p.A[i * K + j] = rowSum > 0 ? (xiSum[i * K + j] + opts.transitionPrior) / rowSum : 1 / K;
    }
  }

  const wsum = new Float64Array(K);
  const muNew = new Float64Array(K * D);
  for (let t = 0; t < T; t++) {
    for (let k = 0; k < K; k++) {
      const w = gamma[t * K + k];
      wsum[k] += w;
      for (let d = 0; d < D; d++) muNew[k * D + d] += w * X[t * D + d];
    }
  }
  for (let k = 0; k < K; k++) {
    if (wsum[k] > 1e-8) for (let d = 0; d < D; d++) muNew[k * D + d] /= wsum[k];
    else for (let d = 0; d < D; d++) muNew[k * D + d] = p.mu[k * D + d];
  }

  const varNew = new Float64Array(K * D);
  for (let t = 0; t < T; t++) {
    for (let k = 0; k < K; k++) {
      const w = gamma[t * K + k];
      for (let d = 0; d < D; d++) {
        const diff = X[t * D + d] - muNew[k * D + d];
        varNew[k * D + d] += w * diff * diff;
      }
    }
  }
  for (let k = 0; k < K; k++) {
    for (let d = 0; d < D; d++) {
      const v = wsum[k] > 1e-8 ? varNew[k * D + d] / wsum[k] : p.vari[k * D + d];
      p.vari[k * D + d] = Math.max(v, opts.varFloor);
    }
  }
  p.mu.set(muNew);

  return logLik;
}

export function fit(X: Float64Array, T: number, D: number, options: FitOptions = {}): FitResult {
  const opts: Required<FitOptions> = {
    states: options.states ?? 3,
    maxIter: options.maxIter ?? 200,
    tol: options.tol ?? 1e-6,
    restarts: options.restarts ?? 5,
    varFloor: options.varFloor ?? 1e-8,
    transitionPrior: options.transitionPrior ?? 0.1,
    seed: options.seed ?? 42,
    verbose: options.verbose ?? false,
  };
  const K = opts.states;
  if (T < K * 10) throw new Error(`need at least ${K * 10} observations for ${K} states, got ${T}`);

  let best: FitResult | null = null;
  for (let r = 0; r < opts.restarts; r++) {
    const rng = makeRng(opts.seed + r * 7919);
    const p = initParams(X, T, D, K, rng, opts.varFloor);
    let prev = -Infinity;
    let iter = 0;
    let converged = false;
    let logLik = -Infinity;

    for (; iter < opts.maxIter; iter++) {
      logLik = emStep(X, T, p, opts);
      // EM is monotone in the likelihood, so compare per-observation improvement.
      if (Math.abs(logLik - prev) / T < opts.tol) { converged = true; break; }
      prev = logLik;
    }
    if (opts.verbose) {
      console.error(`  restart ${r}: logLik/T = ${(logLik / T).toFixed(6)} after ${iter} iters`);
    }
    if (!best || logLik > best.logLik) {
      best = { params: p, logLik, iterations: iter, converged };
    }
  }
  return sortStates(best!);
}

/**
 * Relabel states in ascending order of mean of feature 0 (the log return),
 * so state 0 is always the most bearish and state K-1 the most bullish.
 * Without this, state indices are arbitrary across restarts and refits.
 */
export function sortStates(res: FitResult): FitResult {
  const p = res.params;
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
  return { ...res, params: { K, D, pi, A, mu, vari } };
}

/** Stationary distribution of the chain, by power iteration. Useful sanity check. */
export function stationary(p: HmmParams): Float64Array {
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
export function expectedDuration(p: HmmParams, k: number): number {
  const stay = p.A[k * p.K + k];
  return stay >= 1 ? Infinity : 1 / (1 - stay);
}

export function serialize(p: HmmParams): string {
  return JSON.stringify({
    K: p.K, D: p.D,
    pi: Array.from(p.pi), A: Array.from(p.A),
    mu: Array.from(p.mu), vari: Array.from(p.vari),
  }, null, 2);
}

export function deserialize(json: string): HmmParams {
  const o = JSON.parse(json);
  return {
    K: o.K, D: o.D,
    pi: new Float64Array(o.pi), A: new Float64Array(o.A),
    mu: new Float64Array(o.mu), vari: new Float64Array(o.vari),
  };
}
