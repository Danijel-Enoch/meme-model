import { expect, test, describe } from "bun:test";
import {
  fitJump, jumpPath, lossMatrix, jumpObjective, countSwitches, classifyOnline,
  type JumpParams,
} from "./jump";
import { makeRng, randn } from "./hmm";

function params(centroids: number[], D: number, lambda: number): JumpParams {
  return { K: centroids.length / D, D, centroids: new Float64Array(centroids), lambda };
}

/** Every path in {0..K-1}^T, scored directly. The definition the DP must match. */
function bruteForcePath(L: Float64Array, T: number, K: number, lambda: number) {
  let best = Infinity, bestPath: number[] = [];
  const walk = (t: number, path: number[]) => {
    if (t === T) {
      const v = jumpObjective(L, T, K, lambda, path);
      if (v < best - 1e-12) { best = v; bestPath = [...path]; }
      return;
    }
    for (let k = 0; k < K; k++) walk(t + 1, [...path, k]);
  };
  walk(0, []);
  return { objective: best, path: bestPath };
}

describe("the dynamic program", () => {
  test("matches exhaustive search over every path", () => {
    const rng = makeRng(3);
    for (const lambda of [0, 0.5, 2, 10]) {
      const T = 8, K = 3, D = 1;
      const X = new Float64Array(Array.from({ length: T }, () => randn(rng)));
      const p = params([-1, 0, 1], D, lambda);
      const L = lossMatrix(X, T, p);
      const bf = bruteForcePath(L, T, K, lambda);
      const dp = jumpPath(L, T, K, lambda);
      expect(jumpObjective(L, T, K, lambda, dp)).toBeCloseTo(bf.objective, 10);
    }
  });

  test("lambda = 0 reduces to nearest-centroid assignment", () => {
    const X = new Float64Array([-0.9, 1.1, -1.2, 0.2, 0.8]);
    const p = params([-1, 1], 1, 0);
    const L = lossMatrix(X, 5, p);
    expect(Array.from(jumpPath(L, 5, 2, 0))).toEqual([0, 1, 0, 1, 1]);
  });

  test("a large lambda collapses the path to one state", () => {
    const X = new Float64Array([-1, 1, -1, 1, -1, 1]);
    const p = params([-1, 1], 1, 1000);
    const path = jumpPath(lossMatrix(X, 6, p), 6, 2, 1000);
    expect(countSwitches(path)).toBe(0);
  });

  test("switches fall monotonically as lambda rises", () => {
    const rng = makeRng(11);
    const T = 400;
    const X = new Float64Array(Array.from({ length: T }, (_, t) =>
      (Math.floor(t / 25) % 2 === 0 ? -1 : 1) + 0.8 * randn(rng)));
    // 76 switches at lambda 0, down to 1 at 64 and 0 at 128. The plateau at
    // 15 between lambda 4 and 32 is the true 25-bar run structure: once the
    // spurious switches are gone, the next ones cost real fit to remove.
    let prev = Infinity;
    for (const lambda of [0, 1, 2, 4, 8, 16, 32, 64, 128]) {
      const p = params([-1, 1], 1, lambda);
      const s = countSwitches(jumpPath(lossMatrix(X, T, p), T, 2, lambda));
      expect(s).toBeLessThanOrEqual(prev);
      prev = s;
    }
    expect(prev).toBe(0);
  });

  test("a single bar is empty, and a single observation does not crash", () => {
    const p = params([-1, 1], 1, 1);
    expect(jumpPath(new Float64Array(0), 0, 2, 1).length).toBe(0);
    const one = jumpPath(lossMatrix(new Float64Array([0.9]), 1, p), 1, 2, 1);
    expect(one.length).toBe(1);
    expect(one[0]).toBe(1);
  });
});

describe("fitting", () => {
  /** Two regimes that alternate in long runs — the shape a market is claimed to have. */
  function regimes(T: number, runLength: number, seed: number, noise = 0.6) {
    const rng = makeRng(seed);
    const X = new Float64Array(T);
    const truth = new Int32Array(T);
    for (let t = 0; t < T; t++) {
      const k = Math.floor(t / runLength) % 2;
      truth[t] = k;
      X[t] = (k === 0 ? -1.5 : 1.5) + noise * randn(rng);
    }
    return { X, truth };
  }

  test("recovers regimes it was shown", () => {
    const { X, truth } = regimes(600, 30, 5);
    const res = fitJump(X, 600, 1, { states: 2, lambda: 2, restarts: 4, seed: 1 });
    let hit = 0;
    for (let t = 0; t < 600; t++) if (res.path[t] === truth[t]) hit++;
    expect(hit / 600).toBeGreaterThan(0.95);
    // States come out ordered by centroid, matching the repo's convention.
    expect(res.params.centroids[0]).toBeLessThan(res.params.centroids[1]);
  });

  test("the objective never increases across coordinate descent", () => {
    const { X } = regimes(300, 20, 7);
    let prev = Infinity;
    for (let iters = 1; iters <= 8; iters++) {
      const r = fitJump(X, 300, 1, { states: 2, lambda: 1, maxIter: iters, restarts: 1, seed: 2 });
      expect(r.objective).toBeLessThanOrEqual(prev + 1e-9);
      prev = r.objective;
    }
  });

  test("more restarts never find a worse optimum", () => {
    const { X } = regimes(400, 25, 9);
    const few = fitJump(X, 400, 1, { states: 3, lambda: 1.5, restarts: 1, seed: 4 });
    const many = fitJump(X, 400, 1, { states: 3, lambda: 1.5, restarts: 12, seed: 4 });
    expect(many.objective).toBeLessThanOrEqual(few.objective + 1e-9);
  });

  test("an emptied cluster holds its position instead of collapsing", () => {
    // Three states asked for, two present, and a huge lambda so the third can
    // never be entered. It must not drift to the origin and steal points.
    const { X } = regimes(200, 50, 13);
    const res = fitJump(X, 200, 1, { states: 3, lambda: 500, restarts: 2, seed: 6 });
    expect(res.params.centroids.every((c) => Number.isFinite(c))).toBe(true);
    expect(res.switches).toBe(0);
  });

  test("THE OUTLIER TEST: the jump penalty does NOT prevent outlier capture", () => {
    // I expected the opposite, and the measurement says otherwise. Rydén,
    // Terasvirta & Asbrink (1998) noted that a Gaussian HMM lets "very
    // exceptional observations" take a regime to themselves, and it looked as
    // though paying 2*lambda to visit and leave a lone spike would stop that.
    //
    // It does not, and the reason is dimensional: squared loss makes the spike
    // worth ~magnitude^2 to capture while the penalty costs ~lambda. A
    // 40-magnitude bar is worth ~1600, so no lambda absorbs it before lambda
    // gets large enough to flatten the genuine regimes too.
    const { X } = regimes(400, 40, 17);
    X[200] = 40; // a ~25-sigma bar

    const occupancy = (path: Int32Array) => {
      const c = [0, 0, 0];
      for (const k of path) c[k]++;
      return c;
    };

    for (const lambda of [0, 30, 400, 800]) {
      const r = fitJump(X, 400, 1, { states: 3, lambda, restarts: 6, seed: 8 });
      // Some state is holding essentially that one bar, at every penalty that
      // still leaves more than one regime standing.
      expect(Math.min(...occupancy(r.path))).toBeLessThanOrEqual(1);
    }
    // And the penalty that finally does absorb it has destroyed the structure.
    const flattened = fitJump(X, 400, 1, { states: 3, lambda: 1600, restarts: 6, seed: 8 });
    expect(flattened.switches).toBe(0);
  });

  test("which is why the features have to be bounded first", () => {
    // Winsorise at 4 sigma and the arithmetic changes completely. The spike is
    // now 4.0 rather than 40, so capturing it saves (4.0 - 1.5)^2 = 6.25 rather
    // than ~1600, and a penalty of 4 is already more than enough to refuse.
    // The rule of thumb: lambda has to exceed about half the squared distance
    // from the worst observation to its nearest centroid — which is a statement
    // about the features, not about the model.
    const { X } = regimes(400, 40, 17);
    X[200] = 40;
    const clipped = Float64Array.from(X, (x) => Math.max(-4, Math.min(4, x)));

    // Too small a penalty and the clipped spike still takes a state.
    const weak = fitJump(clipped, 400, 1, { states: 3, lambda: 2, restarts: 6, seed: 8 });
    const weakOcc = [0, 0, 0];
    for (const k of weak.path) weakOcc[k]++;
    expect(Math.min(...weakOcc)).toBeLessThanOrEqual(2);

    // Enough penalty, and all three states hold real mass.
    const r = fitJump(clipped, 400, 1, { states: 3, lambda: 4, restarts: 6, seed: 8 });
    const occ = [0, 0, 0];
    for (const k of r.path) occ[k]++;
    expect(Math.min(...occ)).toBeGreaterThan(2);
    expect(r.switches).toBeLessThan(weak.switches);
  });
});

describe("online classification", () => {
  const p = params([-1.5, 0, 1.5], 1, 3);

  test("is causal: later bars cannot change an earlier label", () => {
    const rng = makeRng(21);
    const X = new Float64Array(Array.from({ length: 300 }, () => 1.5 * randn(rng)));
    const full = classifyOnline(X, 300, p);
    const partial = classifyOnline(X.slice(0, 150), 150, p);
    for (let t = 0; t < 150; t++) expect(full[t]).toBe(partial[t]);
  });

  test("never beats the batch path, because it cannot see ahead", () => {
    const rng = makeRng(23);
    const T = 500;
    const X = new Float64Array(Array.from({ length: T }, (_, t) =>
      (Math.floor(t / 30) % 2 === 0 ? -1.5 : 1.5) + 0.9 * randn(rng)));
    const L = lossMatrix(X, T, p);
    const batch = jumpObjective(L, T, p.K, p.lambda, jumpPath(L, T, p.K, p.lambda));
    const online = jumpObjective(L, T, p.K, p.lambda, classifyOnline(X, T, p));
    expect(online).toBeGreaterThanOrEqual(batch - 1e-9);
  });

  test("and it switches less as lambda rises, which is the point", () => {
    const rng = makeRng(29);
    const X = new Float64Array(Array.from({ length: 600 }, () => 1.2 * randn(rng)));
    let prev = Infinity;
    for (const lambda of [0, 1, 2, 5, 20, 100]) {
      const s = countSwitches(classifyOnline(X, 600, { ...p, lambda }));
      expect(s).toBeLessThanOrEqual(prev);
      prev = s;
    }
    // One switch survives any penalty, and it is not a bug: bar 0 is assigned
    // on distance alone, with no prior state to be charged for leaving, so the
    // path can move off that first choice exactly once and then lock.
    expect(prev).toBeLessThanOrEqual(1);
  });

  test("the first bar has no previous state to be charged against", () => {
    const X = new Float64Array([1.4, 1.4]);
    expect(classifyOnline(X, 2, p)[0]).toBe(2);
    // Seeded with a different state, the penalty can hold it there.
    expect(classifyOnline(X, 2, { ...p, lambda: 100 }, 0)[0]).toBe(0);
  });

  test("it carries the arrival cost, and is therefore not the naive rule", () => {
    // The distinction Shu, Yu & Mulvey (2024) insist on: "A basic k-means style
    // online inference would involve directly assigning features x_t to the
    // nearest centroid ... which ignores temporal information."
    //
    // Naive: look only at the previous LABEL, a hard decision that has already
    // discarded how close the contest was. Arrival cost: carry the running
    // value function, so a state that has been cheap for a while stays cheap.
    const naive = (X: Float64Array, T: number, q: typeof p) => {
      const out = new Int32Array(T);
      let prev = -1;
      for (let t = 0; t < T; t++) {
        let best = Infinity, bestK = 0;
        for (let k = 0; k < q.K; k++) {
          const diff = X[t] - q.centroids[k];
          const cost = diff * diff + (prev >= 0 && k !== prev ? q.lambda : 0);
          if (cost < best) { best = cost; bestK = k; }
        }
        out[t] = bestK; prev = bestK;
      }
      return out;
    };

    const rng = makeRng(41);
    const X = new Float64Array(Array.from({ length: 800 }, () => 1.3 * randn(rng)));
    const q = { ...p, lambda: 3 };
    const arrival = classifyOnline(X, 800, q);
    const lazy = naive(X, 800, q);

    let differences = 0;
    for (let t = 0; t < 800; t++) if (arrival[t] !== lazy[t]) differences++;
    expect(differences).toBeGreaterThan(0);
  });

  test("and it is what lets a LARGE penalty stay usable", () => {
    // This is the property that matters, and it is not the one I first assumed.
    // Measured against known regimes, the naive rule is actually the more
    // accurate of the two at moderate penalties (.9285 vs .8660 at lambda 5).
    // It falls apart at large ones:
    //
    //     lambda    arrival-cost   naive
    //          3          .8125    .8355
    //          5          .8660    .9285
    //          8          .8990    .9400
    //         12          .9250    .9230
    //         20          .9075    .7070   <- naive freezes
    //
    // The reason is structural. Under the naive rule a single bar has to
    // overcome lambda on its own, so past some penalty the path simply stops
    // moving. The arrival cost integrates evidence across bars, so a sustained
    // shift can still pay for the jump even when no single bar could.
    //
    // That is exactly why this repo needs the arrival-cost form: the whole
    // reason to reach for a jump model here is turnover control, which means
    // running lambda HIGH, which is precisely where the shortcut fails.
    const rng = makeRng(47);
    const T = 2000, run = 25;
    const X = new Float64Array(T);
    const truth = new Int32Array(T);
    for (let t = 0; t < T; t++) {
      const k = Math.floor(t / run) % 2 === 0 ? 0 : 2;
      truth[t] = k;
      X[t] = (k === 0 ? -1.5 : 1.5) + 1.4 * randn(rng);
    }
    const naive = (q: typeof p) => {
      const out = new Int32Array(T);
      let prev = -1;
      for (let t = 0; t < T; t++) {
        let best = Infinity, bestK = 0;
        for (let k = 0; k < q.K; k++) {
          const diff = X[t] - q.centroids[k];
          const cost = diff * diff + (prev >= 0 && k !== prev ? q.lambda : 0);
          if (cost < best) { best = cost; bestK = k; }
        }
        out[t] = bestK; prev = bestK;
      }
      return out;
    };
    const score = (path: Int32Array) => {
      let hit = 0;
      for (let t = 0; t < T; t++) if (path[t] === truth[t]) hit++;
      return hit / T;
    };

    const big = { ...p, lambda: 20 };
    expect(score(classifyOnline(X, T, big))).toBeGreaterThan(0.85);
    expect(score(naive(big))).toBeLessThan(0.75);
    // And it degrades gracefully rather than falling off a cliff.
    expect(score(classifyOnline(X, T, big)))
      .toBeGreaterThan(score(classifyOnline(X, T, { ...p, lambda: 12 })) - 0.05);
  });

  test("it is exactly the forward DP read row by row", () => {
    // Definitional: V[t][k] = L[t][k] + min_j(V[t-1][j] + lambda[j!=k]), and the
    // online label is argmin_k V[t][k]. Recomputed here independently.
    const rng = makeRng(43);
    const T = 200;
    const X = new Float64Array(Array.from({ length: T }, () => 1.1 * randn(rng)));
    const q = { ...p, lambda: 2.5 };
    const L = lossMatrix(X, T, q);
    const V = new Float64Array(q.K);
    const expected = new Int32Array(T);
    for (let t = 0; t < T; t++) {
      if (t === 0) for (let k = 0; k < q.K; k++) V[k] = L[k];
      else {
        let best = Infinity;
        for (let j = 0; j < q.K; j++) best = Math.min(best, V[j]);
        const next = new Float64Array(q.K);
        for (let k = 0; k < q.K; k++) next[k] = L[t * q.K + k] + Math.min(V[k], best + q.lambda);
        V.set(next);
      }
      let bk = 0;
      for (let k = 1; k < q.K; k++) if (V[k] < V[bk]) bk = k;
      expected[t] = bk;
    }
    expect(Array.from(classifyOnline(X, T, q))).toEqual(Array.from(expected));
  });
});
