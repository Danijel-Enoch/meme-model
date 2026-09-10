/**
 * Hidden semi-Markov model (explicit-duration HMM).
 *
 * A plain HMM forces geometric dwell times: P(stay n more bars) decays at a
 * constant rate, so the hazard of leaving a regime never rises no matter how
 * long you have been in it. That is wrong for market regimes, and it has a
 * concrete cost — the one-step-ahead state distribution can never be more
 * confident than the transition diagonal, so a persistent-but-finite regime
 * like a pump can never be called sharply.
 *
 * Here each state carries its own duration distribution p_j(d), learned
 * non-parametrically over d = 1..maxDuration. Self-transitions are forbidden
 * (the diagonal of A is zero) because dwell time is the duration model's job.
 *
 * Everything runs in log space: segment likelihoods multiply across up to
 * `maxDuration` bars, which underflows immediately in linear space.
 */

import {
  fit as fitHmm, logEmissions, makeRng,
  type HmmParams, type FitOptions,
} from "./hmm";

export interface HsmmParams {
  K: number;
  D: number;
  /** P(first segment is state j). */
  pi: Float64Array;
  /** K*K transition matrix between *different* states; diagonal is zero. */
  A: Float64Array;
  mu: Float64Array;
  vari: Float64Array;
  /** K*maxDuration duration pmf; dur[j*maxDuration + (d-1)] = P(duration d | state j). */
  dur: Float64Array;
  maxDuration: number;
}

const NEG_INF = -Infinity;

function logSumExp(values: number[]): number {
  let max = NEG_INF;
  for (const v of values) if (v > max) max = v;
  if (max === NEG_INF) return NEG_INF;
  let sum = 0;
  for (const v of values) sum += Math.exp(v - max);
  return max + Math.log(sum);
}

/** In-place accumulator version, avoids allocating an array per call. */
class LogAcc {
  private max = NEG_INF;
  private sum = 0;
  add(v: number) {
    if (v === NEG_INF) return;
    if (v > this.max) {
      this.sum = this.sum * Math.exp(this.max - v) + 1;
      this.max = v;
    } else {
      this.sum += Math.exp(v - this.max);
    }
  }
  value(): number {
    return this.max === NEG_INF ? NEG_INF : this.max + Math.log(this.sum);
  }
  reset() { this.max = NEG_INF; this.sum = 0; }
}

const safeLog = (x: number) => (x > 0 ? Math.log(x) : NEG_INF);

/**
 * Prefix sums of log emissions, so the log-likelihood of a whole segment is
 * one subtraction: segLogB(j, s, e) = P[(e+1)*K+j] - P[s*K+j].
 */
function emissionPrefix(logB: Float64Array, T: number, K: number): Float64Array {
  const P = new Float64Array((T + 1) * K);
  for (let t = 0; t < T; t++) {
    for (let j = 0; j < K; j++) P[(t + 1) * K + j] = P[t * K + j] + logB[t * K + j];
  }
  return P;
}

interface ForwardHsmm {
  /** logAlpha[t*K+j] = log P(o_0..t, a segment of state j ENDS at t). */
  logAlpha: Float64Array;
  /** logEntry[s*K+j] = log P(o_0..s-1, a segment of state j STARTS at s). */
  logEntry: Float64Array;
  logLik: number;
}

function forwardHsmm(P: Float64Array, T: number, p: HsmmParams, logA: Float64Array, logDur: Float64Array): ForwardHsmm {
  const { K, maxDuration } = p;
  const logAlpha = new Float64Array(T * K).fill(NEG_INF);
  const logEntry = new Float64Array(T * K).fill(NEG_INF);
  const acc = new LogAcc();

  for (let j = 0; j < K; j++) logEntry[j] = safeLog(p.pi[j]);

  for (let t = 0; t < T; t++) {
    // A segment of state j ending at t could have started anywhere in
    // [t-maxDuration+1, t]; sum over every admissible length.
    for (let j = 0; j < K; j++) {
      acc.reset();
      const dMax = Math.min(t + 1, maxDuration);
      for (let d = 1; d <= dMax; d++) {
        const s = t - d + 1;
        const entry = logEntry[s * K + j];
        if (entry === NEG_INF) continue;
        const seg = P[(t + 1) * K + j] - P[s * K + j];
        acc.add(entry + logDur[j * maxDuration + (d - 1)] + seg);
      }
      logAlpha[t * K + j] = acc.value();
    }
    // Entries for the next index, from segments that just ended at t.
    if (t + 1 < T) {
      for (let j = 0; j < K; j++) {
        acc.reset();
        for (let i = 0; i < K; i++) {
          if (i === j) continue; // self-transitions are the duration model's job
          const a = logA[i * K + j];
          if (a === NEG_INF || logAlpha[t * K + i] === NEG_INF) continue;
          acc.add(logAlpha[t * K + i] + a);
        }
        logEntry[(t + 1) * K + j] = acc.value();
      }
    }
  }

  const finals: number[] = [];
  for (let j = 0; j < K; j++) finals.push(logAlpha[(T - 1) * K + j]);
  return { logAlpha, logEntry, logLik: logSumExp(finals) };
}

/** logBeta[t*K+i] = log P(o_{t+1}..o_{T-1} | a segment of state i ends at t). */
function backwardHsmm(P: Float64Array, T: number, p: HsmmParams, logA: Float64Array, logDur: Float64Array): Float64Array {
  const { K, maxDuration } = p;
  const logBeta = new Float64Array(T * K).fill(NEG_INF);
  for (let i = 0; i < K; i++) logBeta[(T - 1) * K + i] = 0;
  const acc = new LogAcc();
  const inner = new LogAcc();

  for (let t = T - 2; t >= 0; t--) {
    for (let i = 0; i < K; i++) {
      acc.reset();
      for (let j = 0; j < K; j++) {
        if (i === j) continue;
        const a = logA[i * K + j];
        if (a === NEG_INF) continue;
        inner.reset();
        const dMax = Math.min(T - 1 - t, maxDuration);
        for (let d = 1; d <= dMax; d++) {
          const e = t + d;
          const seg = P[(e + 1) * K + j] - P[(t + 1) * K + j];
          const b = logBeta[e * K + j];
          if (b === NEG_INF) continue;
          inner.add(logDur[j * maxDuration + (d - 1)] + seg + b);
        }
        const v = inner.value();
        if (v !== NEG_INF) acc.add(a + v);
      }
      logBeta[t * K + i] = acc.value();
    }
  }
  return logBeta;
}

export interface HsmmFitResult {
  params: HsmmParams;
  logLik: number;
  iterations: number;
  converged: boolean;
}

export interface HsmmFitOptions extends FitOptions {
  maxDuration?: number;
  /** Smoothing added to expected duration counts, keeps the pmf from spiking. */
  durationPrior?: number;
}

/** Seed an HSMM from a fitted HMM: same emissions, geometric durations. */
export function fromHmm(h: HmmParams, maxDuration: number): HsmmParams {
  const { K, D } = h;
  const A = new Float64Array(K * K);
  for (let i = 0; i < K; i++) {
    // Renormalize the off-diagonal, since the HSMM forbids self-transitions.
    let off = 0;
    for (let j = 0; j < K; j++) if (i !== j) off += h.A[i * K + j];
    for (let j = 0; j < K; j++) A[i * K + j] = i === j ? 0 : off > 0 ? h.A[i * K + j] / off : 1 / (K - 1);
  }
  const dur = new Float64Array(K * maxDuration);
  for (let j = 0; j < K; j++) {
    // The HMM's implied geometric dwell distribution is the natural warm start.
    const stay = Math.min(Math.max(h.A[j * K + j], 1e-6), 1 - 1e-6);
    let total = 0;
    for (let d = 1; d <= maxDuration; d++) {
      const pr = Math.pow(stay, d - 1) * (1 - stay);
      dur[j * maxDuration + (d - 1)] = pr;
      total += pr;
    }
    for (let d = 0; d < maxDuration; d++) dur[j * maxDuration + d] /= total;
  }
  return { K, D, pi: Float64Array.from(h.pi), A, mu: Float64Array.from(h.mu), vari: Float64Array.from(h.vari), dur, maxDuration };
}

function logMatrices(p: HsmmParams) {
  const logA = new Float64Array(p.K * p.K);
  for (let i = 0; i < p.K * p.K; i++) logA[i] = safeLog(p.A[i]);
  const logDur = new Float64Array(p.K * p.maxDuration);
  for (let i = 0; i < logDur.length; i++) logDur[i] = safeLog(p.dur[i]);
  return { logA, logDur };
}

/** One EM sweep. Returns the log-likelihood of the parameters on entry. */
function emStepHsmm(X: Float64Array, T: number, p: HsmmParams, varFloor: number, durationPrior: number, transitionPrior: number): number {
  const { K, D, maxDuration } = p;
  const logB = logEmissions(X, T, { K, D, pi: p.pi, A: p.A, mu: p.mu, vari: p.vari });
  const P = emissionPrefix(logB, T, K);
  const { logA, logDur } = logMatrices(p);

  const { logAlpha, logEntry, logLik } = forwardHsmm(P, T, p, logA, logDur);
  if (!Number.isFinite(logLik)) return logLik;
  const logBeta = backwardHsmm(P, T, p, logA, logDur);

  // Occupancy is accumulated with a difference array: a segment contributes its
  // posterior to every bar it covers, and doing that by iterating the bars would
  // cost an extra factor of maxDuration.
  const diff = new Float64Array(K * (T + 1));
  const durCount = new Float64Array(K * maxDuration);
  const piCount = new Float64Array(K);

  for (let e = 0; e < T; e++) {
    for (let j = 0; j < K; j++) {
      if (logBeta[e * K + j] === NEG_INF) continue;
      const dMax = Math.min(e + 1, maxDuration);
      for (let d = 1; d <= dMax; d++) {
        const s = e - d + 1;
        const entry = logEntry[s * K + j];
        if (entry === NEG_INF) continue;
        const seg = P[(e + 1) * K + j] - P[s * K + j];
        const lp = entry + logDur[j * maxDuration + (d - 1)] + seg + logBeta[e * K + j] - logLik;
        const w = Math.exp(lp);
        if (!(w > 0)) continue;
        diff[j * (T + 1) + s] += w;
        diff[j * (T + 1) + e + 1] -= w;
        durCount[j * maxDuration + (d - 1)] += w;
        if (s === 0) piCount[j] += w;
      }
    }
  }

  const gamma = new Float64Array(T * K);
  for (let j = 0; j < K; j++) {
    let run = 0;
    for (let t = 0; t < T; t++) {
      run += diff[j * (T + 1) + t];
      gamma[t * K + j] = run;
    }
  }

  // Expected transitions, counted only at segment boundaries.
  const xi = new Float64Array(K * K);
  const inner = new LogAcc();
  for (let e = 0; e < T - 1; e++) {
    for (let i = 0; i < K; i++) {
      if (logAlpha[e * K + i] === NEG_INF) continue;
      for (let j = 0; j < K; j++) {
        if (i === j || logA[i * K + j] === NEG_INF) continue;
        inner.reset();
        const dMax = Math.min(T - 1 - e, maxDuration);
        for (let d = 1; d <= dMax; d++) {
          const end = e + d;
          const seg = P[(end + 1) * K + j] - P[(e + 1) * K + j];
          if (logBeta[end * K + j] === NEG_INF) continue;
          inner.add(logDur[j * maxDuration + (d - 1)] + seg + logBeta[end * K + j]);
        }
        const v = inner.value();
        if (v === NEG_INF) continue;
        xi[i * K + j] += Math.exp(logAlpha[e * K + i] + logA[i * K + j] + v - logLik);
      }
    }
  }

  // --- M step ---
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

  const wsum = new Float64Array(K);
  const muNew = new Float64Array(K * D);
  for (let t = 0; t < T; t++) {
    for (let j = 0; j < K; j++) {
      const w = gamma[t * K + j];
      wsum[j] += w;
      for (let d = 0; d < D; d++) muNew[j * D + d] += w * X[t * D + d];
    }
  }
  for (let j = 0; j < K; j++) {
    if (wsum[j] > 1e-8) for (let d = 0; d < D; d++) muNew[j * D + d] /= wsum[j];
    else for (let d = 0; d < D; d++) muNew[j * D + d] = p.mu[j * D + d];
  }
  const varNew = new Float64Array(K * D);
  for (let t = 0; t < T; t++) {
    for (let j = 0; j < K; j++) {
      const w = gamma[t * K + j];
      for (let d = 0; d < D; d++) {
        const diffv = X[t * D + d] - muNew[j * D + d];
        varNew[j * D + d] += w * diffv * diffv;
      }
    }
  }
  for (let j = 0; j < K; j++) {
    for (let d = 0; d < D; d++) {
      const v = wsum[j] > 1e-8 ? varNew[j * D + d] / wsum[j] : p.vari[j * D + d];
      p.vari[j * D + d] = Math.max(v, varFloor);
    }
  }
  p.mu.set(muNew);

  return logLik;
}

export function fitHsmm(X: Float64Array, T: number, D: number, options: HsmmFitOptions = {}): HsmmFitResult {
  const K = options.states ?? 3;
  const maxDuration = options.maxDuration ?? 60;
  const maxIter = options.maxIter ?? 100;
  const tol = options.tol ?? 1e-6;
  const varFloor = options.varFloor ?? 1e-8;
  const durationPrior = options.durationPrior ?? 0.05;
  const transitionPrior = options.transitionPrior ?? 0.1;

  // Warm start from a plain HMM. Its emissions are already close, and starting
  // EM from a good segmentation matters more here than for the HMM because the
  // duration pmf has maxDuration free parameters per state to place.
  const hmm = fitHmm(X, T, D, {
    ...options,
    states: K,
    restarts: options.restarts ?? 4,
    maxIter: options.maxIter ?? 200,
  });
  const p = fromHmm(hmm.params, maxDuration);

  let prev = NEG_INF;
  let iter = 0;
  let converged = false;
  let logLik = NEG_INF;
  for (; iter < maxIter; iter++) {
    logLik = emStepHsmm(X, T, p, varFloor, durationPrior, transitionPrior);
    if (!Number.isFinite(logLik)) break;
    if (Math.abs(logLik - prev) / T < tol) { converged = true; break; }
    prev = logLik;
  }
  return sortHsmmStates({ params: p, logLik, iterations: iter, converged });
}

/** Relabel states by ascending mean of feature 0, matching the HMM convention. */
export function sortHsmmStates(res: HsmmFitResult): HsmmFitResult {
  const p = res.params;
  const { K, D, maxDuration } = p;
  const order = Array.from({ length: K }, (_, k) => k).sort((a, b) => p.mu[a * D] - p.mu[b * D]);

  const pi = new Float64Array(K);
  const A = new Float64Array(K * K);
  const mu = new Float64Array(K * D);
  const vari = new Float64Array(K * D);
  const dur = new Float64Array(K * maxDuration);
  for (let ni = 0; ni < K; ni++) {
    const oi = order[ni];
    pi[ni] = p.pi[oi];
    for (let d = 0; d < D; d++) {
      mu[ni * D + d] = p.mu[oi * D + d];
      vari[ni * D + d] = p.vari[oi * D + d];
    }
    for (let d = 0; d < maxDuration; d++) dur[ni * maxDuration + d] = p.dur[oi * maxDuration + d];
    for (let nj = 0; nj < K; nj++) A[ni * K + nj] = p.A[oi * K + order[nj]];
  }
  return { ...res, params: { K, D, pi, A, mu, vari, dur, maxDuration } };
}

/** Mean dwell time implied by each state's learned duration pmf. */
export function expectedDurations(p: HsmmParams): number[] {
  return Array.from({ length: p.K }, (_, j) => {
    let m = 0;
    for (let d = 1; d <= p.maxDuration; d++) m += d * p.dur[j * p.maxDuration + (d - 1)];
    return m;
  });
}

/** Survival S_j(k) = P(duration >= k). Drives the hazard used when predicting. */
function survival(p: HsmmParams): Float64Array {
  const { K, maxDuration } = p;
  const S = new Float64Array(K * (maxDuration + 2));
  for (let j = 0; j < K; j++) {
    let acc = 0;
    for (let d = maxDuration; d >= 1; d--) {
      acc += p.dur[j * maxDuration + (d - 1)];
      S[j * (maxDuration + 2) + d] = acc;
    }
    S[j * (maxDuration + 2) + maxDuration + 1] = 0;
  }
  return S;
}

export interface HsmmFilterResult {
  /** T*K, P(state at t = j | o_0..t). Causal — no future bars used. */
  stateProb: Float64Array;
  /** T*K, P(state at t+1 = j | o_0..t). The one-step-ahead forecast. */
  nextProb: Float64Array;
  logLik: number;
}

/**
 * Causal filtering with run-length tracking.
 *
 * The HSMM's answer to "what state am I in" needs a joint over (state, how long
 * it has already run), because the chance of leaving depends on elapsed time.
 * Carrying that run-length axis is precisely what lets the model say "this
 * regime is 30 bars old and typically lasts 20, so it is about to end" — the
 * call a geometric HMM structurally cannot make.
 */
export function filterHsmm(X: Float64Array, T: number, p: HsmmParams): HsmmFilterResult {
  const { K, D, maxDuration } = p;
  const logB = logEmissions(X, T, { K, D, pi: p.pi, A: p.A, mu: p.mu, vari: p.vari });
  const P = emissionPrefix(logB, T, K);
  const { logA, logDur } = logMatrices(p);
  const { logAlpha, logEntry, logLik } = forwardHsmm(P, T, p, logA, logDur);
  const S = survival(p);
  const sIdx = (j: number, k: number) => j * (maxDuration + 2) + k;

  const stateProb = new Float64Array(T * K);
  const nextProb = new Float64Array(T * K);
  // g[j][k] = P(in state j, run length k so far | o_0..t), unnormalized.
  const g = new Float64Array(K * (maxDuration + 1));

  const logW = new Float64Array(K * (maxDuration + 1));

  for (let t = 0; t < T; t++) {
    g.fill(0);
    logW.fill(NEG_INF);
    // Collect log weights first and rescale by their own maximum. Normalizing
    // against the full-sequence logLik instead would overflow at small t, where
    // the prefix likelihood is many orders of magnitude larger than the total.
    let maxLog = NEG_INF;
    for (let j = 0; j < K; j++) {
      const kMax = Math.min(t + 1, maxDuration);
      for (let k = 1; k <= kMax; k++) {
        const s = t - k + 1;
        const entry = logEntry[s * K + j];
        if (entry === NEG_INF) continue;
        const surv = S[sIdx(j, k)];
        if (!(surv > 0)) continue;
        const seg = P[(t + 1) * K + j] - P[s * K + j];
        const lw = entry + seg + Math.log(surv);
        if (!Number.isFinite(lw)) continue;
        logW[j * (maxDuration + 1) + k] = lw;
        if (lw > maxLog) maxLog = lw;
      }
    }
    let total = 0;
    if (maxLog !== NEG_INF) {
      for (let j = 0; j < K; j++) {
        for (let k = 1; k <= maxDuration; k++) {
          const lw = logW[j * (maxDuration + 1) + k];
          if (lw === NEG_INF) continue;
          const w = Math.exp(lw - maxLog);
          g[j * (maxDuration + 1) + k] = w;
          total += w;
        }
      }
    }
    if (!(total > 0)) {
      for (let j = 0; j < K; j++) { stateProb[t * K + j] = 1 / K; nextProb[t * K + j] = 1 / K; }
      continue;
    }
    for (let j = 0; j < K; j++) {
      let sum = 0;
      for (let k = 1; k <= maxDuration; k++) sum += g[j * (maxDuration + 1) + k];
      stateProb[t * K + j] = sum / total;
    }

    // One step ahead: each (state, run length) either survives one more bar or
    // ends and hands off through A. The hazard is S(k+1)/S(k), not a constant.
    for (let j = 0; j < K; j++) {
      let stay = 0, leave = 0;
      for (let k = 1; k <= maxDuration; k++) {
        const w = g[j * (maxDuration + 1) + k];
        if (w === 0) continue;
        const sk = S[sIdx(j, k)];
        const sk1 = S[sIdx(j, k + 1)];
        const cont = sk > 0 ? sk1 / sk : 0;
        stay += w * cont;
        leave += w * (1 - cont);
      }
      nextProb[t * K + j] += stay / total;
      for (let m = 0; m < K; m++) {
        if (m === j) continue;
        nextProb[t * K + m] += (leave / total) * p.A[j * K + m];
      }
    }
  }
  return { stateProb, nextProb, logLik };
}

/** Segmental Viterbi: the single most likely (state, duration) segmentation. */
export function viterbiHsmm(X: Float64Array, T: number, p: HsmmParams): Int32Array {
  const { K, D, maxDuration } = p;
  const logB = logEmissions(X, T, { K, D, pi: p.pi, A: p.A, mu: p.mu, vari: p.vari });
  const P = emissionPrefix(logB, T, K);
  const { logA, logDur } = logMatrices(p);

  const delta = new Float64Array(T * K).fill(NEG_INF);
  const backState = new Int32Array(T * K).fill(-1);
  const backDur = new Int32Array(T * K).fill(0);

  for (let e = 0; e < T; e++) {
    for (let j = 0; j < K; j++) {
      let best = NEG_INF, bi = -1, bd = 0;
      const dMax = Math.min(e + 1, maxDuration);
      for (let d = 1; d <= dMax; d++) {
        const s = e - d + 1;
        const seg = P[(e + 1) * K + j] - P[s * K + j];
        const dl = logDur[j * maxDuration + (d - 1)];
        if (dl === NEG_INF) continue;
        if (s === 0) {
          const v = safeLog(p.pi[j]) + dl + seg;
          if (v > best) { best = v; bi = -1; bd = d; }
        } else {
          for (let i = 0; i < K; i++) {
            if (i === j || logA[i * K + j] === NEG_INF || delta[(s - 1) * K + i] === NEG_INF) continue;
            const v = delta[(s - 1) * K + i] + logA[i * K + j] + dl + seg;
            if (v > best) { best = v; bi = i; bd = d; }
          }
        }
      }
      delta[e * K + j] = best;
      backState[e * K + j] = bi;
      backDur[e * K + j] = bd;
    }
  }

  const path = new Int32Array(T);
  let e = T - 1;
  let best = NEG_INF, j = 0;
  for (let k = 0; k < K; k++) if (delta[e * K + k] > best) { best = delta[e * K + k]; j = k; }
  while (e >= 0 && j >= 0) {
    const d = backDur[e * K + j] || 1;
    for (let t = e; t > e - d && t >= 0; t--) path[t] = j;
    const prev = backState[e * K + j];
    e = e - d;
    j = prev;
    if (j < 0) break;
  }
  return path;
}

export function serializeHsmm(p: HsmmParams): string {
  return JSON.stringify({
    K: p.K, D: p.D, maxDuration: p.maxDuration,
    pi: Array.from(p.pi), A: Array.from(p.A),
    mu: Array.from(p.mu), vari: Array.from(p.vari), dur: Array.from(p.dur),
  });
}

export function deserializeHsmm(json: string): HsmmParams {
  const o = JSON.parse(json);
  return {
    K: o.K, D: o.D, maxDuration: o.maxDuration,
    pi: new Float64Array(o.pi), A: new Float64Array(o.A),
    mu: new Float64Array(o.mu), vari: new Float64Array(o.vari), dur: new Float64Array(o.dur),
  };
}
