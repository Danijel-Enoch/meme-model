/**
 * Statistical jump model: regimes by penalised clustering, no Markov chain.
 *
 * The premise, from Bemporad, Breschi, Piga & Boyd (2018) and applied to market
 * regimes by Nystrup, Kolm & Lindström and by Shu, Yu & Mulvey (2024): fit K
 * centroids in feature space and a state path at the same time, paying a fixed
 * price lambda every time the path switches.
 *
 *     minimise   sum_t  ||z_t - theta_{s_t}||^2  +  lambda * #{t : s_t != s_{t-1}}
 *
 * That is the whole model. What it does NOT have is the interesting part:
 *
 *   no transition matrix   — persistence is one number, not K(K-1) of them
 *   no emission density    — no Gaussian, no covariance, nothing to misspecify
 *   no duration pmf        — the HSMM's 87 parameters collapse into lambda
 *
 * A 3-state model over 3 features costs 9 parameters plus lambda, against 26
 * for the Gaussian HMM and 113 for the HSMM. Given that this repo's own power
 * analysis found the 45-day window running at 19 observations per parameter,
 * that difference is not cosmetic.
 *
 * Three reasons to expect it to behave better here specifically:
 *
 * TURNOVER IS THE BINDING CONSTRAINT. The measured problem was never forecast
 * quality — it was that the strategy trades ~94 round trips a month, 187x the
 * turnover at which Novy-Marx & Velikov find equity anomalies keep their
 * significance after costs. lambda is a direct dial on that. An HMM can only
 * reduce switching by inflating its transition diagonal, which fights the
 * likelihood; here it costs nothing but a larger lambda.
 *
 * OUTLIERS STILL CAPTURE A STATE — a claim worth writing down because it is the
 * one I expected to hold and it does not. The intuition is that a lone spike
 * must pay 2*lambda to be visited and left again, so a large enough penalty
 * should absorb it. Measured, it never does: the spike's SQUARED distance grows
 * as the square of its magnitude while the penalty grows linearly, so by the
 * time lambda is large enough to absorb a 25-sigma bar it has already collapsed
 * the real regimes. With one 40-sigma bar in 400, state occupancy runs
 * 201/198/1 at lambda = 0 and still 200/199/1 at lambda = 800; at 1600 the
 * whole path becomes a single state. There is no window in between. jump.test.ts
 * pins this.
 *
 * So the division of labour is sharper than "jump models beat HMMs": Student-t
 * emissions are what fix outlier capture (this repo measured the bullish mean
 * falling 4.3 -> 1.1 bps/bar when they were applied), and the jump penalty is
 * what fixes turnover. Different failures, different tools. If the features fed
 * in here are unbounded, winsorise them first — which is part of why the
 * literature feeds jump models bounded downside-risk features rather than raw
 * returns.
 *
 * THE ONLINE RULE IS NATIVELY CAUSAL. `classifyOnline` assigns the current bar
 * using only the current bar and the previous state. There is no smoothed
 * variant to accidentally trade, which is the mistake this repo warns about
 * everywhere it mentions Viterbi.
 *
 * What it gives up: no likelihood, so no ELBO, no credible intervals, and no
 * model selection by evidence. It is a point estimate of a partition, and the
 * only honest way to choose K and lambda is out-of-sample.
 */

import { makeRng } from "./hmm";

export interface JumpParams {
  K: number;
  D: number;
  /** K*D cluster centres, in the scaled feature space. */
  centroids: Float64Array;
  /** Price of one regime switch, in units of squared feature distance. */
  lambda: number;
}

export interface JumpFitOptions {
  states?: number;
  lambda?: number;
  maxIter?: number;
  restarts?: number;
  seed?: number;
}

export interface JumpFitResult {
  params: JumpParams;
  /** The fitted (smoothed) path. For trading, use `classifyOnline` instead. */
  path: Int32Array;
  objective: number;
  iterations: number;
  converged: boolean;
  /** Regime switches in the fitted path — the quantity lambda buys down. */
  switches: number;
}

const sqDist = (X: Float64Array, t: number, c: Float64Array, k: number, D: number) => {
  let acc = 0;
  for (let d = 0; d < D; d++) {
    const diff = X[t * D + d] - c[k * D + d];
    acc += diff * diff;
  }
  return acc;
};

/** T*K matrix of squared distances from each observation to each centroid. */
export function lossMatrix(X: Float64Array, T: number, p: JumpParams): Float64Array {
  const L = new Float64Array(T * p.K);
  for (let t = 0; t < T; t++) {
    for (let k = 0; k < p.K; k++) L[t * p.K + k] = sqDist(X, t, p.centroids, k, p.D);
  }
  return L;
}

/**
 * The exact minimiser of the objective over paths, given centroids.
 *
 * Viterbi's structure with a constant penalty in place of log transition
 * probabilities. The inner minimisation is the one trick worth noting: because
 * the penalty is the SAME for every switch,
 *
 *     min_j ( V[t-1][j] + lambda*[j != k] )  =  min( V[t-1][k], lambda + min_j V[t-1][j] )
 *
 * so the running best over all j is computed once per step rather than per
 * (j, k) pair. That takes the recursion from O(T*K^2) to O(T*K), which matters
 * when lambda is being swept over a grid.
 */
export function jumpPath(L: Float64Array, T: number, K: number, lambda: number): Int32Array {
  if (T === 0) return new Int32Array(0);
  const V = new Float64Array(T * K);
  const back = new Int32Array(T * K).fill(-1);
  for (let k = 0; k < K; k++) V[k] = L[k];

  for (let t = 1; t < T; t++) {
    let best = Infinity, bestJ = 0;
    for (let j = 0; j < K; j++) {
      if (V[(t - 1) * K + j] < best) { best = V[(t - 1) * K + j]; bestJ = j; }
    }
    for (let k = 0; k < K; k++) {
      const stay = V[(t - 1) * K + k];
      const jump = best + lambda;
      // Ties go to staying put: with equal cost, the less active path is the
      // one that does not pay a spread in the real world.
      if (stay <= jump) { V[t * K + k] = L[t * K + k] + stay; back[t * K + k] = k; }
      else { V[t * K + k] = L[t * K + k] + jump; back[t * K + k] = bestJ; }
    }
  }

  const path = new Int32Array(T);
  let end = 0;
  for (let k = 1; k < K; k++) if (V[(T - 1) * K + k] < V[(T - 1) * K + end]) end = k;
  path[T - 1] = end;
  for (let t = T - 1; t > 0; t--) path[t - 1] = back[t * K + path[t]];
  return path;
}

export function countSwitches(path: ArrayLike<number>): number {
  let n = 0;
  for (let t = 1; t < path.length; t++) if (path[t] !== path[t - 1]) n++;
  return n;
}

export function jumpObjective(
  L: Float64Array, T: number, K: number, lambda: number, path: ArrayLike<number>,
): number {
  let acc = 0;
  for (let t = 0; t < T; t++) acc += L[t * K + path[t]];
  return acc + lambda * countSwitches(path);
}

/** Centroids are the mean of the observations assigned to them. */
function updateCentroids(
  X: Float64Array, T: number, D: number, K: number, path: ArrayLike<number>,
  previous: Float64Array,
): Float64Array {
  const sums = new Float64Array(K * D);
  const counts = new Int32Array(K);
  for (let t = 0; t < T; t++) {
    const k = path[t];
    counts[k]++;
    for (let d = 0; d < D; d++) sums[k * D + d] += X[t * D + d];
  }
  const out = new Float64Array(K * D);
  for (let k = 0; k < K; k++) {
    if (counts[k] === 0) {
      // An emptied cluster keeps its previous position rather than collapsing
      // to the origin, which in scaled features is the densest region and would
      // make the empty state immediately steal points from its neighbours.
      for (let d = 0; d < D; d++) out[k * D + d] = previous[k * D + d];
      continue;
    }
    for (let d = 0; d < D; d++) out[k * D + d] = sums[k * D + d] / counts[k];
  }
  return out;
}

/** k-means++ seeding: spread the initial centroids out by squared distance. */
function seedCentroids(
  X: Float64Array, T: number, D: number, K: number, rng: () => number,
): Float64Array {
  const c = new Float64Array(K * D);
  const first = Math.floor(rng() * T);
  for (let d = 0; d < D; d++) c[d] = X[first * D + d];
  const closest = new Float64Array(T).fill(Infinity);
  for (let k = 1; k < K; k++) {
    let total = 0;
    for (let t = 0; t < T; t++) {
      const dist = sqDist(X, t, c, k - 1, D);
      if (dist < closest[t]) closest[t] = dist;
      total += closest[t];
    }
    let target = rng() * total, pick = T - 1;
    for (let t = 0; t < T; t++) {
      target -= closest[t];
      if (target <= 0) { pick = t; break; }
    }
    for (let d = 0; d < D; d++) c[k * D + d] = X[pick * D + d];
  }
  return c;
}

/** Relabel states by ascending centroid on feature 0, the repo's convention. */
function sortStates(p: JumpParams, path: Int32Array): { params: JumpParams; path: Int32Array } {
  const { K, D } = p;
  const order = Array.from({ length: K }, (_, k) => k)
    .sort((a, b) => p.centroids[a * D] - p.centroids[b * D]);
  const rank = new Int32Array(K);
  order.forEach((old, neu) => { rank[old] = neu; });
  const centroids = new Float64Array(K * D);
  for (let neu = 0; neu < K; neu++) {
    for (let d = 0; d < D; d++) centroids[neu * D + d] = p.centroids[order[neu] * D + d];
  }
  const out = new Int32Array(path.length);
  for (let t = 0; t < path.length; t++) out[t] = rank[path[t]];
  return { params: { ...p, centroids }, path: out };
}

/**
 * Coordinate descent: alternate the exact path given centroids with the exact
 * centroids given the path.
 *
 * Each half is a global minimisation of one block, so the objective is
 * non-increasing and, because the path lives in a finite set, the iteration
 * terminates. It terminates at a local minimum, not the global one — hence
 * restarts, which is also why `objective` is returned for comparison.
 */
export function fitJump(
  X: Float64Array, T: number, D: number, options: JumpFitOptions = {},
): JumpFitResult {
  const K = options.states ?? 3;
  const lambda = options.lambda ?? 0;
  const maxIter = options.maxIter ?? 100;
  const restarts = Math.max(1, options.restarts ?? 8);
  const rng = makeRng(options.seed ?? 42);
  if (T < K) throw new Error(`need at least ${K} observations for ${K} states, got ${T}`);

  let best: JumpFitResult | null = null;
  for (let r = 0; r < restarts; r++) {
    let centroids = seedCentroids(X, T, D, K, rng);
    let params: JumpParams = { K, D, centroids, lambda };
    let path = jumpPath(lossMatrix(X, T, params), T, K, lambda);
    let objective = Infinity;
    let iter = 0;
    let converged = false;

    for (; iter < maxIter; iter++) {
      centroids = updateCentroids(X, T, D, K, path, centroids);
      params = { K, D, centroids, lambda };
      const L = lossMatrix(X, T, params);
      const next = jumpPath(L, T, K, lambda);
      const value = jumpObjective(L, T, K, lambda, next);

      let same = true;
      for (let t = 0; t < T; t++) if (next[t] !== path[t]) { same = false; break; }
      path = next;
      objective = value;
      if (same) { converged = true; iter++; break; }
    }

    if (!best || objective < best.objective) {
      const sorted = sortStates(params, path);
      best = {
        params: sorted.params, path: sorted.path,
        objective, iterations: iter, converged,
        switches: countSwitches(sorted.path),
      };
    }
  }
  return best!;
}

/**
 * Causal assignment, one bar at a time — the rule a live system runs.
 *
 * NOT nearest-centroid-plus-a-penalty-on-the-last-label. That is the naive
 * version, and Shu, Yu & Mulvey (2024) reject it explicitly: "A basic k-means
 * style online inference would involve directly assigning features x_t to the
 * nearest centroid, i.e. argmin_k l(x_t, theta_k), which ignores temporal
 * information." It ignores it because the previous label is a hard decision that
 * has already thrown away how close the contest was.
 *
 * The right rule carries the ARRIVAL COST — the running value function of the
 * same dynamic program, which is exactly Nystrup, Kolm & Lindström's (2020)
 * Exhibit 3:
 *
 *     V[0][k] = L[0][k]
 *     V[t][k] = L[t][k] + min_j ( V[t-1][j] + lambda*[j != k] )
 *     s_t     = argmin_k V[t][k]
 *
 * This is the forward pass of `jumpPath` read off row by row instead of
 * backtracked, so it is causal by construction: V[t] depends on nothing after t.
 * Shu et al.'s "run the DP over a lookback window and take the last state" is
 * the same computation seeded further back, which is why the two papers describe
 * the online rule differently and mean the same thing.
 *
 * It still cannot beat the batch path, which is allowed to know that a jump pays
 * off two bars later. jump.test.ts asserts that inequality rather than hoping
 * for it.
 */
export function classifyOnline(
  X: Float64Array, T: number, p: JumpParams, initial = -1,
): Int32Array {
  const { K, lambda } = p;
  const out = new Int32Array(T);
  const V = new Float64Array(K);
  const next = new Float64Array(K);

  for (let t = 0; t < T; t++) {
    if (t === 0) {
      for (let k = 0; k < K; k++) {
        // A caller who knows the previous state pays to leave it, exactly as
        // any other bar would.
        V[k] = sqDist(X, 0, p.centroids, k, p.D)
          + (initial >= 0 && k !== initial ? lambda : 0);
      }
    } else {
      let best = Infinity;
      for (let j = 0; j < K; j++) if (V[j] < best) best = V[j];
      for (let k = 0; k < K; k++) {
        next[k] = sqDist(X, t, p.centroids, k, p.D) + Math.min(V[k], best + lambda);
      }
      V.set(next);
    }

    let bestK = 0;
    for (let k = 1; k < K; k++) if (V[k] < V[bestK]) bestK = k;
    out[t] = bestK;

    // The value function accumulates loss without bound over a long series and
    // only differences matter, so rebase each step. Without this, V overflows
    // to Infinity on a multi-year run and every state ties.
    const floor = V[bestK];
    for (let k = 0; k < K; k++) V[k] -= floor;
  }
  return out;
}

/**
 * Clip every feature to +/- `sigma` standard deviations, computed on THIS
 * window only.
 *
 * Not in the papers; it is in the authors' reference implementation, ahead of
 * standardisation, and this repo's own measurement says why it has to be. With
 * squared loss a lone spike is worth its magnitude SQUARED to capture while the
 * penalty costs lambda, so no penalty absorbs a 25-sigma bar before it has
 * flattened the real regimes. Clipping changes the arithmetic rather than
 * fighting it — see the outlier tests in jump.test.ts.
 *
 * Fit this on the training window and apply the SAME bounds to everything that
 * follows it; recomputing them live leaks, and it also silently rescales what
 * lambda means.
 */
export function clipBounds(X: Float64Array, T: number, D: number, sigma = 3): {
  lo: Float64Array; hi: Float64Array;
} {
  const lo = new Float64Array(D);
  const hi = new Float64Array(D);
  for (let d = 0; d < D; d++) {
    let mean = 0;
    for (let t = 0; t < T; t++) mean += X[t * D + d];
    mean /= Math.max(T, 1);
    let varr = 0;
    for (let t = 0; t < T; t++) varr += (X[t * D + d] - mean) ** 2;
    const sd = Math.sqrt(varr / Math.max(T, 1));
    lo[d] = mean - sigma * sd;
    hi[d] = mean + sigma * sd;
  }
  return { lo, hi };
}

export function applyClip(
  X: Float64Array, T: number, D: number, b: { lo: Float64Array; hi: Float64Array },
): Float64Array {
  const out = new Float64Array(T * D);
  for (let t = 0; t < T; t++) {
    for (let d = 0; d < D; d++) {
      out[t * D + d] = Math.max(b.lo[d], Math.min(b.hi[d], X[t * D + d]));
    }
  }
  return out;
}

/**
 * The jump model wearing an HmmParams face.
 *
 * `mu` IS the centroid matrix — under squared loss the cluster centre in the
 * scaled feature space is exactly the state's mean, so `stateMeanReturns` and
 * `stateVols` read it correctly with no special case. `pi` and `A` are the
 * empirical frequencies and transitions of the fitted path, recorded so dwell
 * times and diagnostics print the same way they do for the other models.
 *
 * `A` is NOT used to predict. The jump model has no transition law, and
 * inventing one here would quietly turn it back into the Markov model it exists
 * to avoid. Bemporad et al. (2018) Corollary 1 gives the map in the other
 * direction — a flat penalty lambda corresponds to an implied off-diagonal
 * probability of exp(-lambda)/(1 + (K-1)exp(-lambda)) — which is worth knowing
 * and is still not a forecast.
 */
export interface JumpModelParams {
  K: number;
  D: number;
  pi: Float64Array;
  A: Float64Array;
  mu: Float64Array;
  vari: Float64Array;
  /** Alias of `mu`, under the name the jump literature uses. */
  centroids: Float64Array;
  lambda: number;
}

export function jumpToParams(
  X: Float64Array, T: number, D: number, res: JumpFitResult,
): JumpModelParams {
  const { K } = res.params;
  const counts = new Int32Array(K);
  const A = new Float64Array(K * K);
  for (let t = 0; t < T; t++) {
    counts[res.path[t]]++;
    if (t > 0) A[res.path[t - 1] * K + res.path[t]]++;
  }
  for (let i = 0; i < K; i++) {
    let row = 0;
    for (let j = 0; j < K; j++) row += A[i * K + j];
    for (let j = 0; j < K; j++) A[i * K + j] = row > 0 ? A[i * K + j] / row : (i === j ? 1 : 0);
  }

  const vari = new Float64Array(K * D);
  for (let t = 0; t < T; t++) {
    const k = res.path[t];
    for (let d = 0; d < D; d++) {
      const diff = X[t * D + d] - res.params.centroids[k * D + d];
      vari[k * D + d] += diff * diff;
    }
  }
  for (let k = 0; k < K; k++) {
    for (let d = 0; d < D; d++) {
      vari[k * D + d] = counts[k] > 1 ? Math.max(vari[k * D + d] / counts[k], 1e-8) : 1e-8;
    }
  }

  const pi = new Float64Array(K);
  for (let k = 0; k < K; k++) pi[k] = T > 0 ? counts[k] / T : 1 / K;

  return {
    K, D, pi, A,
    mu: res.params.centroids,
    vari,
    centroids: res.params.centroids,
    lambda: res.params.lambda,
  };
}

/** Mean run length of each state in a fitted path — the jump model's dwell. */
export function pathDurations(path: ArrayLike<number>, K: number): number[] {
  const runs = new Array(K).fill(0);
  const bars = new Array(K).fill(0);
  let prev = -1;
  for (let t = 0; t < path.length; t++) {
    bars[path[t]]++;
    if (path[t] !== prev) runs[path[t]]++;
    prev = path[t];
  }
  return runs.map((n, k) => (n > 0 ? bars[k] / n : 1));
}
