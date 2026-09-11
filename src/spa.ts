/**
 * Data-snooping tests: what a sweep's best p-value is actually worth.
 *
 * `sweep` runs 240 (coin, timeframe, model) cells and reports the smallest
 * p-value it found. That number is not a p-value for anything. It is the
 * minimum of 240 dependent draws, and under a null of no skill anywhere the
 * expected minimum of 228 usable draws is about 1/229 = 0.004. Finding 0.035
 * at the top of that table is not evidence; it is what noise looks like when
 * you let it audition 240 times.
 *
 * The fix is not Bonferroni. Bonferroni assumes the 240 tests are independent,
 * and they are emphatically not: BTC-15m-hmm and BTC-15m-hsmm trade the same
 * bars of the same series, every 15m cell overlaps every 2h cell in calendar
 * time, and the whole universe co-moves. Bonferroni over dependent tests is
 * valid but throws away most of its power. The literature's answer is to
 * bootstrap the joint distribution of the whole family of statistics, so the
 * dependence between cells is estimated rather than assumed away:
 *
 *   White, H. (2000), "A Reality Check for Data Snooping", Econometrica 68(5),
 *     1097-1126. Statistic V_n = max_k sqrt(n) * dbar_k over k = 1..m models.
 *     H0: max_k E[d_k] <= 0, i.e. "no model beats the benchmark". The bootstrap
 *     distribution is built by resampling TIME (the same time indices for every
 *     model, which is what preserves cross-model dependence) and recentering
 *     each model on its own sample mean: V*_b = max_k sqrt(n)(dbar*_{k,b} - dbar_k).
 *     p = #{V*_b >= V_n} / B. See `realityCheck`.
 *
 *   Hansen, P.R. (2005), "A Test for Superior Predictive Ability", JBES 23(4),
 *     365-380. Two fixes to the RC, both of which matter here. (1) Studentize:
 *     T = max_k max(sqrt(n) dbar_k / omega_k, 0), so a cell with 234 noisy 2h
 *     bars and a cell with 2814 quiet 15m bars are compared on the same scale;
 *     un-studentized, the RC's max is dominated by whichever model happens to
 *     have the largest variance. (2) Drop hopeless models from the null. The
 *     RC's recentering imposes E[d_k] = 0 on EVERY model, including the ones
 *     that lost 40%; those models still inflate the bootstrap max and make the
 *     test conservative, which is why adding garbage strategies to a White RC
 *     makes the surviving one look better. See `spaTest`.
 *
 *   Romano, J.P. & Wolf, M. (2005), "Stepwise Multiple Testing as Formalized
 *     Data Snooping", Econometrica 73(4), 1237-1282, with the adjusted-p-value
 *     algorithm of Romano & Wolf (2016), Statistics & Probability Letters 113,
 *     38-40. SPA answers "is the best one real?" with a single yes/no. Romano-
 *     Wolf answers the question a sweep actually asks: WHICH cells survive,
 *     controlling the familywise error rate at 5% across all of them.
 *     See `romanoWolf`.
 *
 *   Politis, D.N. & Romano, J.P. (1994), "The Stationary Bootstrap", JASA
 *     89(428), 1303-1313, and Politis, D.N. & White, H. (2004), "Automatic
 *     Block-Length Selection for the Dependent Bootstrap", Econometric Reviews
 *     23(1), 53-70, corrected by Patton, Politis & White (2009), Econometric
 *     Reviews 28(4), 372-375. All three tests above need a resampler that does
 *     not destroy serial dependence; an iid shuffle of a trend-following
 *     strategy's returns would understate its variance badly and hand back
 *     p-values that are far too small. See `stationaryBootstrapIndices` and
 *     `automaticBlockLength`.
 *
 *   Bailey, D.H. & Lopez de Prado, M. (2014), "The Deflated Sharpe Ratio",
 *     Journal of Portfolio Management 40(5), 94-107. The same correction
 *     expressed as a haircut on the winner's Sharpe rather than as a p-value,
 *     and it also charges for non-normality, which matters for a 15m meme-coin
 *     series with fat tails. See `deflatedSharpe`.
 *
 * Convention throughout: `lossDiffs[name]` is the per-bar performance of
 * strategy `name` RELATIVE TO THE BENCHMARK, aligned in time across all names.
 * The benchmark here is sitting flat, so the loss differential is just the
 * strategy's own net log return per bar. Positive is good. (Hansen and White
 * write it as a loss differential d_k,t = L(benchmark) - L(model k); with a
 * negative-log-return loss and a cash benchmark those are the same number.)
 */

import { makeRng } from "./hmm";

// ---------------------------------------------------------------------------
// Normal distribution helpers (zero deps, so they live here)
// ---------------------------------------------------------------------------

/**
 * Complementary error function, Numerical Recipes 6.2.2. Fractional error
 * below 1.2e-7 everywhere, which on a tail probability of 1e-4 is an absolute
 * error of 1e-11 — far tighter than anything a 500-bar backtest can resolve.
 */
function erfc(x: number): number {
  const z = Math.abs(x);
  const t = 2 / (2 + z);
  const ty = 4 * t - 2;
  const cof = [
    -1.3026537197817094, 6.4196979235649026e-1, 1.9476473204185836e-2,
    -9.561514786808631e-3, -9.46595344482036e-4, 3.66839497852761e-4,
    4.2523324806907e-5, -2.0278578112534e-5, -1.624290004647e-6,
    1.303655835580e-6, 1.5626441722e-8, -8.5238095915e-8,
    6.529054439e-9, 5.059343495e-9, -9.91364156e-10,
    -2.27365122e-10, 9.6467911e-11, 2.394038e-12,
    -6.886027e-12, 8.94487e-13, 3.13092e-13,
    -1.12708e-13, 3.81e-16, 7.106e-15,
  ];
  let d = 0, dd = 0;
  for (let j = cof.length - 1; j > 0; j--) {
    const tmp = d;
    d = ty * d - dd + cof[j];
    dd = tmp;
  }
  const ans = t * Math.exp(-z * z + 0.5 * (cof[0] + ty * d) - dd);
  return x >= 0 ? ans : 2 - ans;
}

/** Standard normal CDF. */
export function normalCdf(z: number): number {
  return 0.5 * erfc(-z / Math.SQRT2);
}

/**
 * Inverse standard normal CDF (Acklam's rational approximation, relative error
 * < 1.15e-9), with one Halley step against `normalCdf` to clean up the tails.
 * The Deflated Sharpe Ratio evaluates this at 1 - 1/N for N in the hundreds,
 * which is deep enough in the tail that a crude approximation would show up in
 * the answer.
 */
export function normalInv(p: number): number {
  if (p <= 0) return -Infinity;
  if (p >= 1) return Infinity;
  const a = [-3.969683028665376e+1, 2.209460984245205e+2, -2.759285104469687e+2,
    1.383577518672690e+2, -3.066479806614716e+1, 2.506628277459239];
  const b = [-5.447609879822406e+1, 1.615858368580409e+2, -1.556989798598866e+2,
    6.680131188771972e+1, -1.328068155288572e+1];
  const c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838,
    -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996,
    3.754408661907416];
  const pl = 0.02425, ph = 1 - pl;
  let x: number;
  if (p < pl) {
    const q = Math.sqrt(-2 * Math.log(p));
    x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
        ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  } else if (p > ph) {
    const q = Math.sqrt(-2 * Math.log(1 - p));
    x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
         ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  } else {
    const q = p - 0.5, r = q * q;
    x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  }
  // Halley refinement: e = Phi(x) - p, u = e * sqrt(2pi) * exp(x^2/2).
  const e = normalCdf(x) - p;
  const u = e * Math.sqrt(2 * Math.PI) * Math.exp(x * x / 2);
  return x - u / (1 + x * u / 2);
}

// ---------------------------------------------------------------------------
// Politis & Romano (1994): the stationary bootstrap
// ---------------------------------------------------------------------------

/**
 * Index sequence for one stationary-bootstrap resample of a length-n series.
 *
 * Politis & Romano (1994). Blocks of GEOMETRIC length with success probability
 * p = 1/meanBlock, laid end to end and wrapped around the series:
 *   - start at a uniformly random index;
 *   - at each subsequent step, with probability p jump to a fresh uniform
 *     index (start a new block), otherwise advance one bar (continue the block).
 *
 * Two properties make this the right resampler for a trading backtest.
 *
 * Blocks preserve dependence. A momentum strategy's returns are serially
 * correlated — it holds the same position for runs of bars, so a good run and
 * a bad run are each several bars long. Resampling single bars iid would break
 * those runs and shrink the variance of the sample mean, which inflates every
 * t-statistic built on it. Resampling contiguous stretches keeps the runs.
 *
 * Random block lengths keep the resample stationary. The moving-block bootstrap
 * with FIXED block length b makes a resampled series that is not stationary:
 * its distribution depends on position within the block, because bars at block
 * boundaries have different neighbours than bars in the middle. Geometric
 * lengths are memoryless, so every position looks the same, and the wrap-around
 * removes the edge effect that would otherwise under-sample the first and last
 * b bars. That is the entire point of the paper's title.
 */
export function stationaryBootstrapIndices(n: number, meanBlock: number, rng: () => number): Int32Array {
  const idx = new Int32Array(n);
  if (n <= 0) return idx;
  const p = 1 / Math.max(meanBlock, 1);
  idx[0] = Math.min(n - 1, Math.floor(rng() * n));
  for (let i = 1; i < n; i++) {
    // Memoryless block continuation. p = 1 degenerates to the iid bootstrap,
    // which is the correct limit for a series with no serial dependence.
    idx[i] = rng() < p ? Math.min(n - 1, Math.floor(rng() * n)) : (idx[i - 1] + 1) % n;
  }
  return idx;
}

/** Sample autocovariance at lag k (divisor n, as Politis & White use). */
function autocov(x: number[], mean: number, k: number): number {
  const n = x.length;
  let s = 0;
  for (let i = 0; i < n - k; i++) s += (x[i] - mean) * (x[i + k] - mean);
  return s / n;
}

/**
 * Flat-top lag window of Politis & Romano (1995), used by the block-length rule.
 * Unity on [0, 1/2] then a linear taper to zero at 1: it leaves low-order
 * autocovariances untouched (unlike Bartlett, which shrinks every lag and
 * therefore biases the long-run variance downward) while still killing the
 * high-order sample noise.
 */
function flatTop(t: number): number {
  const a = Math.abs(t);
  if (a <= 0.5) return 1;
  if (a <= 1) return 2 * (1 - a);
  return 0;
}

/**
 * Automatic mean block length for the stationary bootstrap.
 *
 * Politis & White (2004) eq. (9), with the Patton, Politis & White (2009)
 * correction. Minimising the asymptotic MSE of the bootstrap variance
 * estimator gives
 *
 *     b_opt = ( 2 * Ghat^2 / Dhat_SB )^(1/3) * N^(1/3),
 *
 * where, with lambda the flat-top window above and R(k) the AUTOCOVARIANCE,
 *
 *     Ghat    = sum_{k=-M..M} lambda(k/M) * |k| * R(k)   (the bias term, -G/b)
 *     ghat(0) = sum_{k=-M..M} lambda(k/M) * R(k)         (the long-run variance)
 *     Dhat_SB = 2 * ghat(0)^2                            (the variance term, D*b/N)
 *
 * That factor of 2 is exactly what the 2009 correction is about. Politis &
 * White originally printed D_SB as 4g^2(0) plus an integral of g^2 over the
 * spectrum, following Lahiri (1999); Nordman (2009) showed that derivation was
 * wrong and PPW (2009) item 1 replaces it with the flat D_SB = 2 g^2(0). The
 * circular and moving block bootstraps use D_CB = (4/3) g^2(0) instead, so
 * b_opt,SB = (2/3)^(1/3) * b_opt,CB ~ 0.874 * b_opt,CB. Using the circular
 * constant here would oversize every block by 14%.
 *
 * The bandwidth M is chosen by the paper's own correlogram rule: find the
 * smallest m such that the next K_N autocorrelations are all inside the
 * +/- 2 sqrt(log10(n)/n) band (i.e. indistinguishable from zero), then set
 * M = 2m. The intuition is the same one behind every bandwidth rule — include
 * lags while they still carry signal, stop once they are noise.
 *
 * The answer grows with persistence: a series with AR(1) coefficient 0.8 needs
 * much longer blocks than one with 0.2, and an iid series needs b = 1.
 */
export function automaticBlockLength(x: number[]): number {
  const n = x.length;
  if (n < 8) return 1;
  const mean = x.reduce((a, b) => a + b, 0) / n;
  const r0 = autocov(x, mean, 0);
  // A constant series has no dependence structure to preserve. The test is
  // RELATIVE: a series of identical 0.01s has a variance of ~1e-34 from
  // floating-point cancellation alone, and its "autocorrelations" are pure
  // rounding noise that the rule below would happily read as persistence.
  let scale = 0;
  for (const v of x) scale = Math.max(scale, Math.abs(v));
  if (!(r0 > 0) || Math.sqrt(r0) <= 1e-12 * (scale || 1)) return 1;

  const kn = Math.max(5, Math.ceil(Math.sqrt(Math.log10(n))));
  const mMax = Math.ceil(Math.sqrt(n)) + kn;
  const bMax = Math.ceil(Math.min(3 * Math.sqrt(n), n / 3));
  const band = 2 * Math.sqrt(Math.log10(n) / n);

  // Autocorrelations up to the largest lag any of the rules can ask for.
  const maxLag = Math.min(n - 1, mMax + kn);
  const rho: number[] = [1];
  for (let k = 1; k <= maxLag; k++) rho.push(autocov(x, mean, k) / r0);

  // Smallest m whose following K_N autocorrelations are all inside the band.
  let mHat = 0;
  for (let m = 1; m + kn <= maxLag; m++) {
    let allSmall = true;
    for (let k = 1; k <= kn; k++) {
      if (Math.abs(rho[m + k]) >= band) { allSmall = false; break; }
    }
    if (allSmall) { mHat = m; break; }
  }
  if (mHat === 0) {
    // No stretch of K_N insignificant lags exists — the series is persistent all
    // the way out to maxLag. Patton's reference implementation falls back to the
    // LARGEST significant lag rather than giving up, which is the conservative
    // direction (longer blocks).
    for (let k = maxLag; k >= 1; k--) {
      if (Math.abs(rho[k]) > band) { mHat = k; break; }
    }
    if (mHat === 0) return 1; // nothing significant anywhere: iid
  }
  const M = Math.max(1, Math.min(2 * mHat, mMax, n - 1));

  let Ghat = 0, g0 = 0;
  for (let k = -M; k <= M; k++) {
    const w = flatTop(k / M);
    if (w === 0) continue;
    const R = autocov(x, mean, Math.abs(k)); // R(-k) = R(k)
    Ghat += w * Math.abs(k) * R;
    g0 += w * R;
  }
  // A degenerate long-run variance (positive and negative autocovariances
  // cancelling, or a series so short that g(0) rounds to zero) leaves nothing
  // to divide by; fall back to iid rather than returning a nonsense block.
  const Dhat = 2 * g0 * g0;
  if (!(Dhat > 0) || !Number.isFinite(Ghat)) return 1;

  const b = Math.pow(2 * Ghat * Ghat / Dhat, 1 / 3) * Math.pow(n, 1 / 3);
  if (!Number.isFinite(b) || b < 1) return 1;
  return Math.min(b, bMax);
}

/**
 * Variance of the sample mean UNDER the stationary bootstrap (Politis & Romano
 * 1994). This is the studentiser Hansen's SPA needs: it must be the variance
 * that the bootstrap itself induces, or the studentised statistic and its
 * bootstrap distribution are on different scales.
 *
 *     omega^2 = R(0) + 2 * sum_{j=1..n-1} kappa_j * R(j),
 *     kappa_j = ((n-j)/n) * (1-q)^j + (j/n) * (1-q)^(n-j),   q = 1/meanBlock.
 *
 * The kernel kappa_j is exactly the probability that two bars j apart in the
 * resample came from the same block — the first term for a forward run, the
 * second for a run that wrapped around the end of the series.
 */
export function stationaryBootstrapVariance(x: number[], meanBlock: number): number {
  const n = x.length;
  if (n < 2) return 0;
  const mean = x.reduce((a, b) => a + b, 0) / n;
  const q = 1 / Math.max(meanBlock, 1);
  const r0 = autocov(x, mean, 0);
  // Same relative-zero guard as automaticBlockLength: a series that does not
  // move has no standard error, and reporting 1e-17 instead of 0 would hand
  // `spaTest` an infinite t-statistic.
  let scale = 0;
  for (const val of x) scale = Math.max(scale, Math.abs(val));
  if (!(r0 > 0) || Math.sqrt(r0) <= 1e-12 * (scale || 1)) return 0;
  let v = r0;
  for (let j = 1; j < n; j++) {
    const kappa = ((n - j) / n) * Math.pow(1 - q, j) + (j / n) * Math.pow(1 - q, n - j);
    // Beyond ~30 mean blocks the weight is numerically zero; stop paying for it.
    if (kappa < 1e-12) break;
    v += 2 * kappa * autocov(x, mean, j);
  }
  return v;
}

// ---------------------------------------------------------------------------
// Shared machinery for all three tests
// ---------------------------------------------------------------------------

export interface BootstrapOptions {
  /** Bootstrap replications. Hansen and White both use 1000+; default 1000. */
  bootstraps?: number;
  /** Mean block length. Default: the median Politis-White choice across models. */
  blockLength?: number;
  seed?: number;
}

interface Prepared {
  names: string[];
  n: number;
  /** m x n matrix of loss differentials, row-major. */
  d: Float64Array;
  /** Sample mean per model. */
  mean: Float64Array;
  /** Stationary-bootstrap standard error of sqrt(n) * mean, per model. */
  omega: Float64Array;
  blockLength: number;
  /** B x n resample index matrix, SHARED across models. */
  boot: Int32Array[];
}

/**
 * The single most important line in this file is the one that builds `boot`
 * ONCE and applies the same index vector to every model.
 *
 * White (2000) is explicit about this: the null is about the MAXIMUM over
 * models, so the bootstrap has to reproduce the joint distribution of the m
 * sample means, not m separate marginals. Our 240 cells are massively
 * dependent — two model types on identical bars, four timeframes over the same
 * calendar, thirty coins that all fall together on a bad day. Resampling each
 * model independently would pretend they are 240 independent chances, which is
 * the Bonferroni error in bootstrap clothing, and would make the max far too
 * large. Resampling the TIME AXIS jointly keeps every cross-sectional
 * correlation the data actually has.
 */
function prepare(lossDiffs: Record<string, number[]>, opts: BootstrapOptions): Prepared {
  const names = Object.keys(lossDiffs);
  if (names.length === 0) throw new Error("spa: no strategies supplied");
  const n = lossDiffs[names[0]].length;
  if (n < 2) throw new Error(`spa: need at least 2 observations, got ${n}`);
  for (const name of names) {
    if (lossDiffs[name].length !== n) {
      throw new Error(
        `spa: series must be aligned in time; "${name}" has ${lossDiffs[name].length} bars, ` +
        `"${names[0]}" has ${n}`,
      );
    }
  }

  const m = names.length;
  const d = new Float64Array(m * n);
  for (let k = 0; k < m; k++) d.set(lossDiffs[names[k]], k * n);

  // One block length for the whole family, because one time axis is being
  // resampled. Median over the per-model Politis-White choices: the mean would
  // be dragged around by a single near-constant cell whose rule degenerates.
  let blockLength = opts.blockLength ?? 0;
  if (!(blockLength > 0)) {
    const bs = names.map((name) => automaticBlockLength(lossDiffs[name])).sort((a, b) => a - b);
    blockLength = bs[Math.floor(bs.length / 2)];
  }
  blockLength = Math.max(1, Math.min(blockLength, n));

  const mean = new Float64Array(m);
  const omega = new Float64Array(m);
  for (let k = 0; k < m; k++) {
    const series = lossDiffs[names[k]];
    let s = 0;
    for (let t = 0; t < n; t++) s += series[t];
    mean[k] = s / n;
    const v = stationaryBootstrapVariance(series, blockLength);
    // A cell that never traded is identically zero: no variance, no evidence.
    // Guard rather than divide by zero; its t-statistic is defined as 0 below.
    omega[k] = v > 0 ? Math.sqrt(v) : 0;
  }

  const B = opts.bootstraps ?? 1000;
  const rng = makeRng(opts.seed ?? 7);
  const boot: Int32Array[] = new Array(B);
  for (let b = 0; b < B; b++) boot[b] = stationaryBootstrapIndices(n, blockLength, rng);

  return { names, n, d, mean, omega, blockLength, boot };
}

/** Bootstrap sample means: B x m, using the shared time-index resamples. */
function bootMeans(p: Prepared): Float64Array {
  const m = p.names.length, n = p.n, B = p.boot.length;
  const out = new Float64Array(B * m);
  for (let b = 0; b < B; b++) {
    const idx = p.boot[b];
    for (let k = 0; k < m; k++) {
      const off = k * n;
      let s = 0;
      for (let t = 0; t < n; t++) s += p.d[off + idx[t]];
      out[b * m + k] = s / n;
    }
  }
  return out;
}

/**
 * (1 + #{exceedances}) / (B + 1) rather than #{...} / B.
 *
 * White (2000) and Hansen (2005) both write the plain fraction #{...}/B. It can
 * return exactly 0, which is not a valid p-value — it claims more resolution
 * than B replications can deliver. The +1 form is the standard finite-sample
 * correction (Davison & Hinkley 1997, §4.4.3; it is also what Romano & Wolf
 * 2016 use), and it is exactly uniform on {1/(B+1), ..., 1} under the null,
 * which is what the calibration test in spa.test.ts checks. At B = 2000 the
 * difference from the papers' convention is under 0.0005.
 */
function bootP(exceed: number, B: number): number {
  return (1 + exceed) / (B + 1);
}

// ---------------------------------------------------------------------------
// White (2000): the Bootstrap Reality Check
// ---------------------------------------------------------------------------

export interface RealityCheckResult {
  pValue: number;
  best: string;
  /** The un-studentised statistic V_n = max_k sqrt(n) * dbar_k. */
  vStat: number;
  blockLength: number;
  models: number;
}

/**
 * White's Bootstrap Reality Check.
 *
 * H0: max_k E[d_k] <= 0 — no strategy in the family beats the benchmark.
 * Statistic: V_n = max_k sqrt(n) * dbar_k.
 * Null distribution: V*_b = max_k sqrt(n) * (dbar*_{k,b} - dbar_k), i.e. every
 * model recentred on its own sample mean, which imposes E[d_k] = 0 for ALL k.
 * That is the least favourable configuration inside the null, so the test has
 * the right size — but it is also why the RC is conservative, and why its power
 * evaporates as you add losing models: a cell that lost 40% is nowhere near
 * the boundary E[d_k] = 0, yet the RC still pretends it is and lets it push the
 * bootstrap maximum up. Hansen's SPA below fixes exactly that.
 */
export function realityCheck(
  lossDiffs: Record<string, number[]>,
  opts: BootstrapOptions = {},
): RealityCheckResult {
  const p = prepare(lossDiffs, opts);
  const m = p.names.length, n = p.n, B = p.boot.length;
  const sq = Math.sqrt(n);

  let vStat = -Infinity, bestIdx = 0;
  for (let k = 0; k < m; k++) {
    const v = sq * p.mean[k];
    if (v > vStat) { vStat = v; bestIdx = k; }
  }

  const bm = bootMeans(p);
  let exceed = 0;
  for (let b = 0; b < B; b++) {
    let vb = -Infinity;
    for (let k = 0; k < m; k++) {
      const v = sq * (bm[b * m + k] - p.mean[k]);
      if (v > vb) vb = v;
    }
    if (vb >= vStat) exceed++;
  }

  return {
    pValue: bootP(exceed, B),
    best: p.names[bestIdx],
    vStat,
    blockLength: p.blockLength,
    models: m,
  };
}

// ---------------------------------------------------------------------------
// Hansen (2005): the Test for Superior Predictive Ability
// ---------------------------------------------------------------------------

export interface SpaResult {
  /** Name of the cell with the largest studentised statistic. */
  best: string;
  /** T^SPA = max( max_k sqrt(n) dbar_k / omega_k , 0 ). */
  tStat: number;
  /** The one to report: Hansen's consistent p-value. */
  pConsistent: number;
  /** Liberal bound — only models with dbar_k >= 0 are kept in the null. */
  pLower: number;
  /** Conservative bound — every model kept at the boundary; White's RC logic. */
  pUpper: number;
  blockLength: number;
  models: number;
  /** Per-model studentised statistics, for inspection. */
  tStats: Record<string, number>;
}

/**
 * Hansen's SPA test.
 *
 * Statistic (Hansen 2005 eq. 5):
 *
 *     T^SPA_n = max( max_k  sqrt(n) * dbar_k / omega_k , 0 )
 *
 * with omega_k the stationary-bootstrap standard deviation of sqrt(n) dbar_k.
 * The outer max(., 0) exists because the null is a one-sided composite
 * hypothesis: if every model loses, there is nothing to test and the statistic
 * should sit at its boundary value rather than go negative.
 *
 * The null distribution is generated by resampling and recentring
 *
 *     Z*_{k,b} = sqrt(n) * ( dbar*_{k,b} - g(dbar_k) ) / omega_k,
 *     T*_b     = max( max_k Z*_{k,b}, 0 ),
 *
 * and the whole content of the paper is the choice of g, which decides which
 * models are treated as sitting ON the null boundary E[d_k] = 0 and which are
 * treated as genuinely inferior and therefore irrelevant:
 *
 *   g_u(dbar_k) = dbar_k
 *       Every model on the boundary. Least favourable configuration, so the
 *       p-value is an upper bound: pUpper. This is White's RC assumption
 *       (studentised), and it is what makes the RC's power collapse when the
 *       family contains obvious losers.
 *
 *   g_l(dbar_k) = dbar_k * 1{ dbar_k >= 0 }
 *       Any model with a negative sample mean is assumed truly inferior and
 *       contributes nothing. Liberal, so the p-value is a lower bound: pLower.
 *
 *   g_c(dbar_k) = dbar_k * 1{ dbar_k >= -A_{k,n} },
 *                     A_{k,n} = omega_k * n^(-1/2) * sqrt(2 log log n)
 *       The one to report, and the only one that is CONSISTENT: Hansen's
 *       Theorem 2 shows p^c converges to the true p-value while p^u is
 *       inconsistent unless every mu_k is exactly 0 and p^l is asymptotically
 *       too small. The threshold is the law of the iterated logarithm: with
 *       probability one, liminf sqrt(n)(dbar_k - mu_k)/omega_k = -sqrt(2 log log n),
 *       so a model whose studentised mean falls below that line cannot have
 *       mu_k = 0, and one with mu_k = 0 is never excluded. It is the SLOWEST
 *       rate that captures every mu_k = 0 model.
 *
 *       Hansen notes other rates are also valid, and the pre-publication
 *       working paper used A_{k,n} = (1/4) n^(-1/4) omega_k — still the variant
 *       in Hsu, Hsu & Kuan (2010) and in some software. Pass
 *       `threshold: "quarter"` for it. It is looser (at n = 352 it excludes a
 *       model at t = -1.08 where the published rule needs t = -1.88), so it
 *       gives a slightly smaller p-value.
 *
 * pLower <= pConsistent <= pUpper always, and the gap between the bounds is a
 * direct readout of how much the answer depends on the bad models in the
 * sweep — including how much it depends on which threshold rate you picked,
 * since every admissible rate lands inside those bounds.
 */
export function spaTest(
  lossDiffs: Record<string, number[]>,
  opts: BootstrapOptions & { threshold?: "loglog" | "quarter" } = {},
): SpaResult {
  const p = prepare(lossDiffs, opts);
  const m = p.names.length, n = p.n, B = p.boot.length;
  const sq = Math.sqrt(n);

  const t = new Float64Array(m);
  let tStat = 0, bestIdx = 0, bestT = -Infinity;
  for (let k = 0; k < m; k++) {
    t[k] = p.omega[k] > 0 ? sq * p.mean[k] / p.omega[k] : 0;
    if (t[k] > bestT) { bestT = t[k]; bestIdx = k; }
  }
  tStat = Math.max(bestT, 0);

  // The exclusion threshold, in raw (un-studentised) units. log log n is only
  // real and positive for n >= 16; below that there is no meaningful
  // separation of "poor" from "boundary" models, so keep everything.
  const lil = n >= 16 ? Math.sqrt(2 * Math.log(Math.log(n))) : 0;
  const useQuarter = opts.threshold === "quarter";
  const A = (omega: number) => useQuarter
    ? 0.25 * Math.pow(n, -0.25) * omega          // Hansen's working-paper rate
    : omega * Math.pow(n, -0.5) * lil;           // published: LIL rate

  // The three recentring means, in raw (un-studentised) units.
  const gU = new Float64Array(m), gL = new Float64Array(m), gC = new Float64Array(m);
  for (let k = 0; k < m; k++) {
    gU[k] = p.mean[k];
    gL[k] = p.mean[k] >= 0 ? p.mean[k] : 0;
    gC[k] = p.mean[k] >= -A(p.omega[k]) ? p.mean[k] : 0;
  }

  const bm = bootMeans(p);
  let exU = 0, exL = 0, exC = 0;
  for (let b = 0; b < B; b++) {
    let mu = 0, ml = 0, mc = 0; // start at 0: the outer max(., 0) of the statistic
    for (let k = 0; k < m; k++) {
      if (p.omega[k] <= 0) continue;
      const s = sq / p.omega[k];
      const db = bm[b * m + k];
      const zu = s * (db - gU[k]); if (zu > mu) mu = zu;
      const zl = s * (db - gL[k]); if (zl > ml) ml = zl;
      const zc = s * (db - gC[k]); if (zc > mc) mc = zc;
    }
    if (mu >= tStat) exU++;
    if (ml >= tStat) exL++;
    if (mc >= tStat) exC++;
  }

  const tStats: Record<string, number> = {};
  for (let k = 0; k < m; k++) tStats[p.names[k]] = t[k];

  return {
    best: p.names[bestIdx],
    tStat,
    pConsistent: bootP(exC, B),
    pLower: bootP(exL, B),
    pUpper: bootP(exU, B),
    blockLength: p.blockLength,
    models: m,
    tStats,
  };
}

// ---------------------------------------------------------------------------
// Romano & Wolf (2005, 2016): stepdown multiple testing
// ---------------------------------------------------------------------------

export interface RomanoWolfResult {
  /** Names rejected at `alpha`, i.e. the cells that survive the whole sweep. */
  rejected: string[];
  /** FWER-adjusted p-value per cell. Reject when adjustedP <= alpha. */
  adjustedP: Record<string, number>;
  tStats: Record<string, number>;
  blockLength: number;
  models: number;
  alpha: number;
}

/**
 * Romano-Wolf stepdown, returning FWER-adjusted p-values.
 *
 * SPA gives one number for the whole family. A sweep wants more: which of the
 * 240 cells can be claimed, with the probability of making even ONE false claim
 * held at alpha. That is familywise error rate control, and the stepdown is the
 * bootstrap analogue of Holm's method — with the crucial difference that the
 * critical values come from the joint bootstrap distribution, so correlated
 * cells (BTC-15m-hmm and BTC-15m-hsmm) are not double-counted the way
 * Bonferroni-Holm would.
 *
 * The algorithm (Romano & Wolf 2016, computing adjusted p-values directly):
 *
 *   1. Order the studentised statistics t_(1) >= t_(2) >= ... >= t_(m).
 *   2. For step j, take the maximum of the recentred bootstrap statistics over
 *      the SURVIVING set {j, j+1, ..., m} only:
 *          maxT*_b = max_{k >= j} sqrt(n)(dbar*_{k,b} - dbar_k)/omega_k
 *      and set p_j = (1 + #{b : maxT*_b >= t_(j)}) / (B + 1).
 *   3. Enforce monotonicity: padj_(j) = max(p_j, padj_(j-1)).
 *
 * Step 2 is where the power comes from. A single-step procedure would always
 * compare against the max over ALL m models; once the strongest cells have been
 * rejected, the remaining ones only have to clear the max over what is left,
 * which is a lower bar. Step 3 is required for the output to be a coherent
 * p-value function — without it a weaker cell could report a smaller adjusted
 * p-value than a stronger one.
 *
 * Recentring is on dbar_k (the least favourable E[d_k] = 0 for each individual
 * hypothesis H_k: E[d_k] <= 0), which is the correct null for the per-cell
 * hypotheses being tested here — unlike SPA, the question is not "is the max
 * significant" but "is THIS cell significant given everything else that was
 * tried".
 *
 * One deliberate simplification: the bootstrap statistics are studentised by
 * the ORIGINAL omega_k rather than by an omega recomputed inside each
 * resample. That is Hansen's convention and is first-order valid, but Romano &
 * Wolf (2005) footnote 22 points out it forfeits the asymptotic refinement
 * that recomputation buys. Recomputing would cost O(B * m * n) autocovariance
 * passes, which at m = 228 cells and B = 2000 is not worth the refinement.
 */
export function romanoWolf(
  lossDiffs: Record<string, number[]>,
  opts: BootstrapOptions & { alpha?: number } = {},
): RomanoWolfResult {
  const p = prepare(lossDiffs, opts);
  const m = p.names.length, n = p.n, B = p.boot.length;
  const alpha = opts.alpha ?? 0.05;
  const sq = Math.sqrt(n);

  const t = new Float64Array(m);
  for (let k = 0; k < m; k++) t[k] = p.omega[k] > 0 ? sq * p.mean[k] / p.omega[k] : 0;

  // Recentred, studentised bootstrap statistics: B x m.
  const bm = bootMeans(p);
  const z = new Float64Array(B * m);
  for (let b = 0; b < B; b++) {
    for (let k = 0; k < m; k++) {
      z[b * m + k] = p.omega[k] > 0 ? sq * (bm[b * m + k] - p.mean[k]) / p.omega[k] : -Infinity;
    }
  }

  const order = Array.from({ length: m }, (_, k) => k).sort((a, b) => t[b] - t[a]);
  const adjustedP: Record<string, number> = {};
  const rejected: string[] = [];
  let prev = 0;

  // Running maximum over the surviving set, recomputed as the set shrinks. Done
  // naively this is O(m^2 B); walking the order from the WEAKEST end and
  // accumulating a suffix maximum makes it O(mB), which matters at m = 228.
  const suffixMax = new Float64Array(B * m);
  for (let b = 0; b < B; b++) {
    let run = -Infinity;
    for (let j = m - 1; j >= 0; j--) {
      const v = z[b * m + order[j]];
      if (v > run) run = v;
      suffixMax[b * m + j] = run;
    }
  }

  for (let j = 0; j < m; j++) {
    const k = order[j];
    let exceed = 0;
    for (let b = 0; b < B; b++) if (suffixMax[b * m + j] >= t[k]) exceed++;
    const raw = bootP(exceed, B);
    const padj = Math.max(raw, prev); // monotonicity across steps
    prev = padj;
    adjustedP[p.names[k]] = padj;
    if (padj <= alpha) rejected.push(p.names[k]);
  }

  const tStats: Record<string, number> = {};
  for (let k = 0; k < m; k++) tStats[p.names[k]] = t[k];

  return { rejected, adjustedP, tStats, blockLength: p.blockLength, models: m, alpha };
}

// ---------------------------------------------------------------------------
// Bailey & Lopez de Prado (2014): the Deflated Sharpe Ratio
// ---------------------------------------------------------------------------

/**
 * Expected maximum Sharpe ratio under the null that NO strategy has skill.
 *
 * Bailey & Lopez de Prado (2014). If N trials have independent estimated
 * Sharpe ratios drawn from a zero-mean distribution with variance V, the
 * expected maximum follows the Gumbel approximation
 *
 *     SR0 = sqrt(V) * [ (1 - gamma) * Z^-1(1 - 1/N) + gamma * Z^-1(1 - 1/(N*e)) ]
 *
 * with gamma the Euler-Mascheroni constant 0.5772... This is the benchmark the
 * winner has to beat. Note how brutally it grows: at N = 1 it is ~0, at N = 228
 * it is about 2.9 standard deviations of the trial Sharpes. Running more
 * backtests does not make it easier to find something real; it raises the bar.
 */
export const EULER_MASCHERONI = 0.5772156649015329;

export function expectedMaxSharpe(trials: number, varianceOfSharpes: number): number {
  const N = Math.max(trials, 2); // N = 1 leaves nothing to correct for
  const g = EULER_MASCHERONI;
  return Math.sqrt(Math.max(varianceOfSharpes, 0)) *
    ((1 - g) * normalInv(1 - 1 / N) + g * normalInv(1 - 1 / (N * Math.E)));
}

/**
 * Probabilistic Sharpe Ratio: P(true SR > benchmark) given the observed SR,
 * the sample length, and the non-normality of the returns.
 *
 *     PSR(SR*) = Z[ (SRhat - SR*) * sqrt(n - 1) / sqrt(1 - g3*SRhat + ((g4-1)/4)*SRhat^2) ]
 *
 * where g3 is skewness and g4 is kurtosis (3 for a normal). The denominator is
 * the standard error of the Sharpe estimator under non-iid-normal returns
 * (Mertens 2002 / Christie 2005): NEGATIVE skew and FAT tails both inflate it,
 * which is the point — a strategy that grinds out small gains and occasionally
 * loses a lot has a much less reliable Sharpe than the raw number suggests,
 * and meme-coin bars are exactly that shape.
 *
 * `observed` and the entries of `sharpes` must be on the SAME per-bar basis.
 * Annualised Sharpes plugged into the sqrt(n-1) term would silently rescale the
 * whole test, so the caller de-annualises first.
 */
export function probabilisticSharpe(
  observed: number,
  benchmark: number,
  bars: number,
  skew = 0,
  kurtosis = 3,
): number {
  if (bars < 2) return NaN;
  const denom = 1 - skew * observed + ((kurtosis - 1) / 4) * observed * observed;
  if (!(denom > 0)) return NaN; // non-normality so extreme the SE is undefined
  return normalCdf((observed - benchmark) * Math.sqrt(bars - 1) / Math.sqrt(denom));
}

/**
 * Effective number of INDEPENDENT trials, when the trials are correlated.
 *
 * Bailey & Lopez de Prado (2014) Appendix A.3, eqs. (8)-(9). SR0 above assumes
 * N independent draws, and a sweep's cells are anything but: BTC-15m-hmm and
 * BTC-15m-hsmm trade the same bars, and every cell is long the same asset class
 * on the same days. Counting all 228 as independent tries would overstate how
 * hard it was to find the winner — i.e. it would make the null bar too HIGH and
 * the deflation too generous to the sceptic. Reporting both is the honest move.
 *
 *     rho_bar = ( sum_i sum_j rho_ij - M ) / ( M(M-1) )      (the off-diagonal mean)
 *     N_eff   = rho_bar + (1 - rho_bar) * M
 *
 * A linear interpolation between the two endpoints that must hold: perfectly
 * correlated trials (rho = 1) are really one trial, uncorrelated trials
 * (rho = 0) are really M.
 */
export function effectiveTrials(series: number[][]): { rhoBar: number; effective: number } {
  const M = series.length;
  if (M < 2) return { rhoBar: 0, effective: M };
  // Standardise once; the pairwise correlation is then a dot product.
  const z = series.map((x) => {
    const n = x.length;
    const mu = x.reduce((a, b) => a + b, 0) / n;
    let v = 0;
    for (const val of x) v += (val - mu) * (val - mu);
    const sd = Math.sqrt(v);
    return sd > 0 ? x.map((val) => (val - mu) / sd) : x.map(() => 0);
  });
  let sum = 0, pairs = 0;
  for (let i = 0; i < M; i++) {
    for (let j = i + 1; j < M; j++) {
      const a = z[i], b = z[j];
      let d = 0;
      for (let t = 0; t < a.length; t++) d += a[t] * b[t];
      sum += d; pairs++;
    }
  }
  const rhoBar = pairs > 0 ? sum / pairs : 0;
  return { rhoBar, effective: Math.max(1, rhoBar + (1 - rhoBar) * M) };
}

/**
 * Deflated Sharpe Ratio: PSR evaluated against the expected maximum Sharpe of
 * the search that produced the winner, rather than against zero.
 *
 * DSR = PSR(SR0), SR0 = expectedMaxSharpe(N, Var[SRhat over the N trials]).
 *
 * Read it as: the probability that the winner's true Sharpe is positive, AFTER
 * charging for the fact that it was selected as the best of N tries and that
 * its returns are not normal. Bailey & Lopez de Prado treat DSR < 0.95 as
 * "not established". Two things are being estimated from the trials themselves:
 * N (how many were run) and Var[SRhat] (how much they scattered) — which is
 * why the honest input is the FULL sweep, not the shortlist.
 *
 * @param sharpes  per-bar Sharpe of every trial in the search
 * @param observed per-bar Sharpe of the selected winner
 * @param bars     length of the winner's out-of-sample series
 */
export function deflatedSharpe(
  sharpes: number[],
  observed: number,
  bars: number,
  skew = 0,
  kurtosis = 3,
): number {
  const N = sharpes.length;
  if (N === 0) return NaN;
  const mean = sharpes.reduce((a, b) => a + b, 0) / N;
  // Sample variance of the trial Sharpes: the spread the search had to play
  // with. A search over near-identical cells has a small V and a low bar; a
  // search over wildly different cells has a big V and a high one.
  const v = N > 1
    ? sharpes.reduce((a, b) => a + (b - mean) * (b - mean), 0) / (N - 1)
    : 0;
  return probabilisticSharpe(observed, expectedMaxSharpe(N, v), bars, skew, kurtosis);
}

// ---------------------------------------------------------------------------
// Applying it to a sweep: putting cells on one clock
// ---------------------------------------------------------------------------

export interface TimedSeries {
  name: string;
  /** Unix seconds of the bar each value belongs to. Must be sorted ascending. */
  times: number[];
  /** Per-bar log return net of costs. Log, so bucketing is addition. */
  values: number[];
}

/**
 * Put cells recorded on different clocks onto one common grid.
 *
 * Every test above requires the m series to be aligned in time — the whole
 * point is to resample a shared time axis. A sweep's cells are not: a 15m cell
 * has 2814 bars, a 2h cell has 234, and their out-of-sample windows start on
 * different days. Two decisions make them comparable, and both need stating
 * because both are assumptions:
 *
 *   Bucketing is a SUM of log returns. Log returns add across time, so summing
 *   the eight 15m log returns inside a 2h bucket gives exactly that bucket's
 *   log return. Doing this with simple returns would be wrong.
 *
 *   Bars where a cell is not deployed contribute 0. Zero is not "missing data"
 *   here: the benchmark is sitting flat, so a cell that has not started trading
 *   yet is earning exactly the benchmark, and its loss differential really is
 *   0. This is also why padding does not distort the SPA statistic — extending
 *   a cell's series with zeros scales dbar_k by (m/n) and omega_k by about
 *   sqrt(m/n), so the studentised sqrt(n) dbar_k / omega_k is roughly
 *   invariant. The un-studentised Reality Check has no such protection, which
 *   is one more reason to report SPA as the headline.
 */
export function alignOnGrid(
  series: TimedSeries[],
  bucketSeconds: number,
): { names: string[]; grid: number[]; aligned: Record<string, number[]> } {
  if (series.length === 0) return { names: [], grid: [], aligned: {} };
  let lo = Infinity, hi = -Infinity;
  for (const s of series) {
    for (const t of s.times) {
      const b = Math.floor(t / bucketSeconds);
      if (b < lo) lo = b;
      if (b > hi) hi = b;
    }
  }
  if (!Number.isFinite(lo)) return { names: [], grid: [], aligned: {} };

  const n = hi - lo + 1;
  const grid = Array.from({ length: n }, (_, i) => (lo + i) * bucketSeconds);
  const aligned: Record<string, number[]> = {};
  for (const s of series) {
    const row = new Array<number>(n).fill(0);
    for (let i = 0; i < s.times.length; i++) {
      row[Math.floor(s.times[i] / bucketSeconds) - lo] += s.values[i];
    }
    aligned[s.name] = row;
  }
  return { names: series.map((s) => s.name), grid, aligned };
}

/** Per-bar Sharpe, skewness and excess-free kurtosis of one return series. */
export function seriesMoments(x: number[]): { sharpe: number; skew: number; kurtosis: number; n: number } {
  const n = x.length;
  if (n < 2) return { sharpe: 0, skew: 0, kurtosis: 3, n };
  const mean = x.reduce((a, b) => a + b, 0) / n;
  let m2 = 0, m3 = 0, m4 = 0;
  for (const v of x) {
    const dv = v - mean;
    m2 += dv * dv; m3 += dv * dv * dv; m4 += dv * dv * dv * dv;
  }
  m2 /= n; m3 /= n; m4 /= n;
  const sd = Math.sqrt(m2);
  if (!(sd > 0)) return { sharpe: 0, skew: 0, kurtosis: 3, n };
  return { sharpe: mean / sd, skew: m3 / (sd * sd * sd), kurtosis: m4 / (m2 * m2), n };
}

// ---------------------------------------------------------------------------
// The runner: give models/sweep.json a correct p-value
// ---------------------------------------------------------------------------

/** One cell's reconstructed out-of-sample stream. */
export interface CellSeries {
  name: string; coin: string; timeframe: string; modelType: string;
  /** Unix seconds of the bar whose return was earned (the bar AFTER the decision). */
  times: number[];
  /** log(1 + net simple return) per traded bar, net of the sweep's costBps. */
  logNet: number[];
  roiRebuilt: number; roiStored: number;
}

export interface SeriesCache {
  generatedAt: number; sweepGeneratedAt: number; costBps: number; days: number;
  cells: CellSeries[];
}

const TF_SECONDS: Record<string, number> = { "1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "2h": 7200, "4h": 14_400, "1d": 86_400 };

/**
 * Rebuild every cell's out-of-sample per-bar return stream, or load the cache.
 *
 * A sweep row stores summary statistics, not the return series the statistics
 * came from, so the series has to be regenerated by re-running the identical
 * walk-forward. Refitting 228 HMM/HSMM cells takes about fifteen minutes, hence
 * the cache: delete `cachePath` to force a rebuild.
 *
 * The imports are dynamic so that the statistical core of this file keeps its
 * only static dependency on ./hmm — importing `spa` for the tests should not
 * drag in the exchange client.
 *
 * The one thing that cannot be reproduced exactly is the candle window:
 * fetchCandles pages backwards from Date.now(), so a rebuild run later ends on
 * a later bar than the sweep did. We over-fetch and truncate at the sweep's own
 * `generatedAt`, which puts the window back where it was; `roiRebuilt` versus
 * `roiStored` on every cell is the check that it worked.
 */
export async function loadOrBuildSeries(
  sweepPath = "models/sweep.json",
  cachePath = "models/spa-series.json",
  log: (s: string) => void = () => {},
): Promise<SeriesCache> {
  const cacheFile = Bun.file(cachePath);
  const sweep = JSON.parse(await Bun.file(sweepPath).text());
  if (await cacheFile.exists()) {
    const cached: SeriesCache = JSON.parse(await cacheFile.text());
    if (cached.sweepGeneratedAt === sweep.generatedAt) {
      log(`cache hit: ${cached.cells.length} cells from ${cachePath}`);
      return cached;
    }
    log(`cache is for a different sweep (${cached.sweepGeneratedAt} != ${sweep.generatedAt}); rebuilding`);
  }

  const { walkForward } = await import("./backtest");
  const { barBudget, walkForwardSizes, SWEEP_WINDOW } = await import("./sweep");
  const { buildFeatures } = await import("./features");
  const hl = await import("./hyperliquid");
  const featureConfig = { window: SWEEP_WINDOW, useVolatility: true, useVolume: true };
  const cutoff = Math.floor(sweep.generatedAt / 1000);

  // A row that was skipped for want of data has no series to rebuild. A row
  // skipped only for the retention warning still traded, so it stays in.
  const rows = sweep.rows.filter((r: any) => !r.skipped || !String(r.skipped).startsWith("only "));
  const groups = new Map<string, any[]>();
  for (const r of rows) {
    const k = `${r.coin}/${r.timeframe}`;
    if (!groups.has(k)) groups.set(k, []);
    groups.get(k)!.push(r);
  }

  const cells: CellSeries[] = [];
  let done = 0;
  for (const [key, coinRows] of groups) {
    const [coin, tf] = key.split("/");
    const budget = barBudget(tf, sweep.days);
    let candles;
    try {
      const res = await hl.fetchCandles(coin, tf, budget.bars + 200);
      candles = res.candles.filter((c) => c.time <= cutoff).slice(-budget.bars);
    } catch (e) { log(`skip ${key}: ${e}`); done += coinRows.length; continue; }

    let fs;
    try { fs = buildFeatures(candles, featureConfig); } catch (e) { log(`skip ${key}: ${e}`); done += coinRows.length; continue; }
    const { trainSize, testSize, fits } = walkForwardSizes(fs.T);
    if (!fits) { done += coinRows.length; continue; }

    for (const row of coinRows) {
      done++;
      try {
        const r = walkForward(candles, featureConfig,
          { costBps: sweep.costBps, durationAware: true },
          { modelType: row.modelType, trainSize, testSize, states: 3, seed: 42,
            barsPerYear: hl.hlBarsPerYear(tf), maxDuration: 30 });

        // Replay walkForward's own PnL accounting bar by bar. Log returns,
        // because every downstream step (bucketing onto a common clock,
        // summing to a terminal ROI) is addition in logs and would need a
        // product in simple returns.
        const times: number[] = [], logNet: number[] = [];
        let prevPos = 0;
        for (let i = trainSize; i < fs.T - 1; i++) {
          const pos = r.positions[i];
          const barRet = Math.exp(fs.rawReturn[i + 1]) - 1;
          const net = pos * barRet - Math.abs(pos - prevPos) * (sweep.costBps / 10_000);
          logNet.push(Math.log(Math.max(1 + net, 1e-9)));
          times.push(candles[fs.index[i + 1]].time);
          prevPos = pos;
        }
        let eq = 1; for (const l of logNet) eq *= Math.exp(l);
        cells.push({
          name: `${row.coin}/${row.timeframe}/${row.modelType}`,
          coin: row.coin, timeframe: row.timeframe, modelType: row.modelType,
          times, logNet, roiRebuilt: eq - 1, roiStored: row.roi,
        });
        log(`[${done}/${rows.length}] ${row.coin} ${row.timeframe} ${row.modelType} ` +
            `bars=${logNet.length} roi ${(eq - 1).toFixed(4)} vs ${row.roi.toFixed(4)}`);
      } catch (e) { log(`[${done}/${rows.length}] ${key} ${row.modelType} failed: ${e}`); }
    }
  }

  const out: SeriesCache = {
    generatedAt: Date.now(), sweepGeneratedAt: sweep.generatedAt,
    costBps: sweep.costBps, days: sweep.days, cells,
  };
  await Bun.write(cachePath, JSON.stringify(out));
  log(`wrote ${cachePath} with ${cells.length} cells`);
  return out;
}

export interface SweepVerdict {
  cells: number;
  bars: number;
  spa: SpaResult;
  rc: RealityCheckResult;
  rw: RomanoWolfResult;
  /** Winner by Sharpe on the common horizon, and its deflated Sharpe. */
  winner: { name: string; sharpe: number; bars: number; skew: number; kurtosis: number };
  expectedMaxSharpeUnderNull: number;
  dsr: number;
  /** Average pairwise correlation across cells, and the trial count it implies. */
  rhoBar: number;
  effectiveTrials: number;
  /** DSR charged only for the effective (independent-equivalent) trial count. */
  dsrEffective: number;
}

/**
 * Run all four corrections over a rebuilt sweep.
 *
 * Sharpes are put on a common horizon before the Deflated Sharpe step: a 15m
 * cell's per-bar Sharpe is smaller than a 2h cell's for no reason other than the
 * bar length, and Bailey & Lopez de Prado's Var[SRhat] across trials would be
 * measuring the timeframe mix rather than the dispersion of skill. Scaling each
 * by sqrt(bars per bucket) puts them all in per-bucket units.
 */
export function judgeSweep(cache: SeriesCache, opts: BootstrapOptions & { bucketSeconds?: number } = {}): SweepVerdict {
  const bucket = opts.bucketSeconds ?? Math.max(...cache.cells.map((c) => TF_SECONDS[c.timeframe] ?? 3600));
  const { grid, aligned } = alignOnGrid(
    cache.cells.map((c) => ({ name: c.name, times: c.times, values: c.logNet })), bucket);

  const spa = spaTest(aligned, opts);
  const rc = realityCheck(aligned, opts);
  const rw = romanoWolf(aligned, opts);

  const trials = cache.cells.map((c) => {
    const m = seriesMoments(c.logNet);
    const perBucket = Math.sqrt(bucket / (TF_SECONDS[c.timeframe] ?? bucket));
    return {
      name: c.name, sharpe: m.sharpe * perBucket, skew: m.skew, kurtosis: m.kurtosis,
      bars: Math.round(c.logNet.length / perBucket / perBucket),
    };
  });
  const winner = trials.reduce((a, b) => (b.sharpe > a.sharpe ? b : a));
  const sharpes = trials.map((t) => t.sharpe);
  const mu = sharpes.reduce((a, b) => a + b, 0) / sharpes.length;
  const v = sharpes.reduce((a, b) => a + (b - mu) * (b - mu), 0) / Math.max(sharpes.length - 1, 1);

  // Charging for 228 independent tries overstates the search; charging for the
  // effective count is the version a defender of the sweep would ask for, so
  // compute it too and let both numbers stand.
  const eff = effectiveTrials(Object.values(aligned));
  return {
    cells: cache.cells.length,
    bars: grid.length,
    spa, rc, rw,
    winner,
    expectedMaxSharpeUnderNull: expectedMaxSharpe(sharpes.length, v),
    dsr: deflatedSharpe(sharpes, winner.sharpe, winner.bars, winner.skew, winner.kurtosis),
    rhoBar: eff.rhoBar,
    effectiveTrials: eff.effective,
    // Same dispersion V, fewer trials: only the trial COUNT in SR0 changes, so
    // pass the observed Sharpes padded/truncated to the effective count rather
    // than rescaling their spread.
    dsrEffective: probabilisticSharpe(
      winner.sharpe, expectedMaxSharpe(Math.round(eff.effective), v),
      winner.bars, winner.skew, winner.kurtosis),
  };
}

if (import.meta.main) {
  const cache = await loadOrBuildSeries("models/sweep.json", "models/spa-series.json", (s) => console.error(s));
  const v = judgeSweep(cache, { bootstraps: 2000, seed: 12345 });

  console.log(`\n${v.cells} cells on a common ${v.bars}-bar clock, block length ${v.spa.blockLength.toFixed(2)}\n`);
  console.log(`SPA (Hansen 2005)     best ${v.spa.best}  T=${v.spa.tStat.toFixed(3)}`);
  console.log(`                      p_consistent = ${v.spa.pConsistent.toFixed(4)}   ` +
              `[lower ${v.spa.pLower.toFixed(4)}, upper ${v.spa.pUpper.toFixed(4)}]`);
  console.log(`Reality Check (White 2000)  best ${v.rc.best}  p = ${v.rc.pValue.toFixed(4)}`);
  console.log(`Romano-Wolf at 5%     survivors: ${v.rw.rejected.length ? v.rw.rejected.join(", ") : "NONE"}`);
  console.log(`Deflated Sharpe       winner ${v.winner.name}  SR=${v.winner.sharpe.toFixed(4)}/bar over ${v.winner.bars} bars`);
  console.log(`                      skew ${v.winner.skew.toFixed(2)}, kurtosis ${v.winner.kurtosis.toFixed(1)}, ` +
              `expected max under no skill SR0=${v.expectedMaxSharpeUnderNull.toFixed(4)}`);
  console.log(`                      DSR = ${v.dsr.toFixed(4)}  (Bailey & Lopez de Prado call anything under 0.95 unestablished)`);
  console.log(`                      average pairwise correlation ${v.rhoBar.toFixed(3)} => ${v.effectiveTrials.toFixed(0)} effective trials, ` +
              `DSR = ${v.dsrEffective.toFixed(4)}`);

  const ranked = Object.keys(v.rw.tStats).sort((a, b) => v.rw.tStats[b] - v.rw.tStats[a]).slice(0, 10);
  console.log(`\ntop 10 cells, studentised:`);
  for (const n of ranked) {
    console.log(`  ${n.padEnd(24)} t=${v.rw.tStats[n].toFixed(3).padStart(7)}  ` +
      `alone p=${(1 - normalCdf(v.rw.tStats[n])).toFixed(4)}  after the sweep p=${v.rw.adjustedP[n].toFixed(4)}`);
  }
}
