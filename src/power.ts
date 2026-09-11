/**
 * How much evidence would it take?
 *
 * This is the question that belongs BEFORE a search, not after it. A p-value
 * that never had a chance of being small is not evidence of absence — it is
 * evidence that the study was too small to say anything, and no amount of
 * re-searching the same bars fixes that. So: given an edge of a plausible size,
 * a realistic per-bar volatility and the cost of acting, how many bars does it
 * take to distinguish that edge from nothing?
 *
 * The answer for one series is the textbook one (Lo 2002, "The Statistics of
 * Sharpe Ratios", for why the Sharpe estimator's error is almost entirely error
 * in the mean when the Sharpe is small — at SR = 0.25, 97% of it).
 *
 * The answer for THIRTY series is the part that matters here, and it is not
 * thirty times better.
 *
 * ── Why pooling crypto perps buys so little ─────────────────────────────────
 *
 * Thirty coins look like thirty experiments. They are not: they are one market
 * factor observed thirty times. The standard correction is the design effect
 * (Kish 1965) — with average pairwise correlation rho between series, N series
 * carry the information of
 *
 *     N_eff = N / (1 + (N - 1) * rho)
 *
 * independent ones. At rho = 0.7, thirty coins are worth about one and a half.
 * That single line explains why the panel test in panel.ts, with 30,000
 * observations, still could not resolve a 1bps-per-bar effect — and why its
 * common-rotation null, which keeps the factor intact, was the honest choice.
 */

/** Two-sided normal quantile, good to ~1e-9 over the range that matters here. */
export function normalQuantile(p: number): number {
  // Acklam's rational approximation.
  const a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
    1.383577518672690e2, -3.066479806614716e1, 2.506628277459239];
  const b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
    6.680131188771972e1, -1.328068155288572e1];
  const c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838,
    -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996,
    3.754408661907416];
  const pl = 0.02425;
  if (p <= 0 || p >= 1) throw new Error(`quantile needs 0 < p < 1, got ${p}`);
  if (p < pl) {
    const q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (p > 1 - pl) return -normalQuantile(1 - p);
  const q = p - 0.5, r = q * q;
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
    (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

/**
 * Kish's design effect: how many INDEPENDENT series a correlated panel is worth.
 *
 * Monotone decreasing in rho, and bounded — as N grows with rho fixed, N_eff
 * tends to 1 + 1/rho, not to infinity. Adding the 31st correlated coin to a
 * panel of thirty adds almost nothing, which is the whole reason "just test
 * more coins" is not a route to significance.
 */
export function effectiveSeries(n: number, rho: number): number {
  if (n <= 1) return Math.max(n, 0);
  const r = Math.max(0, Math.min(rho, 0.999));
  return n / (1 + (n - 1) * r);
}

/** Ceiling on `effectiveSeries` as N grows without bound, at fixed rho. */
export function effectiveSeriesLimit(rho: number): number {
  const r = Math.max(1e-9, Math.min(rho, 0.999));
  return 1 + 1 / r;
}

/**
 * Standard normal CDF, to ~1e-15.
 *
 * Hart's (1968) rational approximation, in the form West (2005) gives. The
 * textbook Abramowitz & Stegun 7.1.26 series is only good to ~7 digits, which
 * shows up immediately: it puts Phi(0) at 0.5000000005, and the tail
 * probabilities this file needs for a 240-test Bonferroni threshold are around
 * 2e-4, where relative error matters more than absolute.
 */
export function normalCdf(x: number): number {
  const z = Math.abs(x);
  let c = 0;
  if (z <= 37) {
    const e = Math.exp(-z * z / 2);
    if (z < 7.07106781186547) {
      let b = 3.52624965998911e-2 * z + 0.700383064443688;
      b = b * z + 6.37396220353165;
      b = b * z + 33.912866078383;
      b = b * z + 112.079291497871;
      b = b * z + 221.213596169931;
      b = b * z + 220.206867912376;
      let d = 8.83883476483184e-2 * z + 1.75566716318264;
      d = d * z + 16.064177579207;
      d = d * z + 86.7807322029461;
      d = d * z + 296.564248779674;
      d = d * z + 637.333633378831;
      d = d * z + 793.826512519948;
      d = d * z + 440.413735824752;
      c = e * b / d;
    } else {
      let b = z + 0.65;
      b = z + 4 / b; b = z + 3 / b; b = z + 2 / b; b = z + 1 / b;
      c = e / (b * 2.506628274631);
    }
  }
  return x > 0 ? 1 - c : c;
}

export const DAYS_PER_YEAR = 365;

/** Bars per day at each timeframe this repo trades. */
export const BARS_PER_DAY: Record<string, number> = {
  "1m": 1440, "5m": 288, "15m": 96, "30m": 48, "1h": 24, "2h": 12, "4h": 6, "1d": 1,
};

export interface EdgeSpec {
  /** Expected return per bar while in position, bps. */
  edgeBps: number;
  /** Bars a position is held — for a regime model, the state's dwell. */
  holdBars: number;
  /** Per-bar volatility of returns, bps. */
  volBps: number;
  /** Round trip, bps. Paid once per trade, not once per bar. */
  costBps: number;
  /** Fraction of bars holding a position. */
  exposure?: number;
  barsPerDay?: number;
  alpha?: number;
  power?: number;
  /** First-order autocorrelation of bar returns, for the Lo (2002) correction. */
  autocorr?: number;
}

/**
 * The edge that merely pays the toll.
 *
 * The round trip is paid once and the edge accrues every bar, so the hurdle per
 * bar is cost/hold. Everything below this is not a small edge — it is a loss,
 * and more data only measures the loss more precisely.
 */
export function breakEvenEdgeBps(costBps: number, holdBars: number): number {
  return costBps / Math.max(holdBars, 1e-12);
}

/**
 * Sharpe ratio of one round trip, net of its cost.
 *
 * Over `h` bars the edge accumulates linearly (h * edge) while the volatility
 * grows as sqrt(h) — which is the entire reason a slow signal is worth more
 * than a fast one of the same per-bar size.
 */
export function netSharpePerTrade(spec: EdgeSpec): number {
  const gross = spec.edgeBps * spec.holdBars - spec.costBps;
  const vol = spec.volBps * Math.sqrt(Math.max(spec.holdBars, 1e-12));
  return vol > 0 ? gross / vol : 0;
}

/**
 * Lo (2002) eta(q): how much the variance of a q-bar return departs from q
 * times the one-bar variance when returns are autocorrelated.
 *
 * Positive autocorrelation makes multi-bar variance grow FASTER than q, which
 * shrinks a naively-annualized Sharpe — Lo's example puts a fund's annual
 * Sharpe at 2.44 rather than 4.03 once this is applied, a 65% overstatement.
 */
export function varianceInflation(q: number, rho: number): number {
  if (q <= 1 || rho === 0) return 1;
  let acc = 0;
  for (let k = 1; k < q; k++) acc += (q - k) * Math.pow(rho, k);
  return (q + 2 * acc) / q;
}

export interface BarsRequired {
  /** Round trips needed. */
  trades: number;
  bars: number;
  days: number;
  /** Annualized Sharpe this edge implies, which is what makes it comparable
   *  across timeframes. */
  impliedSharpe: number;
}

/**
 * How much data it takes to tell this edge from nothing.
 *
 * n = ((z_alpha + z_beta) / SR_trade)^2 trades, which is the standard one-sample
 * result; bars follow from the hold and the exposure, since a bar spent flat
 * carries no information about timing.
 */
export function barsRequired(spec: EdgeSpec): BarsRequired {
  const alpha = spec.alpha ?? 0.05;
  const power = spec.power ?? 0.8;
  const exposure = spec.exposure ?? 1;
  const barsPerDay = spec.barsPerDay ?? BARS_PER_DAY["30m"];
  const sr = netSharpePerTrade(spec);

  if (sr <= 0) {
    // At or below break-even there is nothing to detect. Reporting a large
    // finite number here would be worse than useless: it would imply that
    // enough data eventually rescues a losing strategy.
    return { trades: Infinity, bars: Infinity, days: Infinity, impliedSharpe: 0 };
  }

  // n = ((z_alpha + z_beta * sqrt(1 + SR^2/2)) / SR)^2.
  //
  // The sqrt term is Lo (2002) eq. 8: the variance of a Sharpe ESTIMATOR is
  // (1 + SR^2/2)/n, not 1/n. It belongs on the alternative-hypothesis term only,
  // because under the null SR = 0 and the correction vanishes. At the Sharpes in
  // this repo it is worth about 1.7% — negligible in itself, but getting it
  // backwards would mean the formula disagrees with the Sharpe literature it
  // cites, and this file exists to be quotable.
  const za = normalQuantile(1 - alpha);
  const zb = normalQuantile(power) * Math.sqrt(1 + (sr * sr) / 2);
  const inflation = varianceInflation(spec.holdBars, spec.autocorr ?? 0);
  const trades = ((za + zb) / sr) ** 2 * inflation;
  const bars = (trades * spec.holdBars) / exposure;
  const tradesPerYear = (exposure * barsPerDay * DAYS_PER_YEAR) / spec.holdBars;
  return {
    trades,
    bars,
    days: bars / barsPerDay,
    impliedSharpe: sr * Math.sqrt(tradesPerYear),
  };
}

/** The smallest edge the data on hand can resolve. Inverts `barsRequired`. */
export function detectableEdge(
  spec: Omit<EdgeSpec, "edgeBps"> & { bars: number },
): { edgeBps: number; impliedAnnualSharpe: number } {
  const alpha = spec.alpha ?? 0.05;
  const power = spec.power ?? 0.8;
  const exposure = spec.exposure ?? 1;
  const barsPerDay = spec.barsPerDay ?? BARS_PER_DAY["30m"];
  const za = normalQuantile(1 - alpha);
  const zb = normalQuantile(power);
  const inflation = varianceInflation(spec.holdBars, spec.autocorr ?? 0);

  const trades = (spec.bars * exposure) / spec.holdBars;
  // Invert n = ((za + zb*sqrt(1 + SR^2/2))/SR)^2 * inflation for SR. Squaring
  // gives a quadratic: SR^2 (a^2 - zb^2/2) - 2 a za SR + (za^2 - zb^2) = 0,
  // with a = sqrt(n / inflation). The larger root is the one on the branch
  // where more data means a smaller detectable edge.
  const a = Math.sqrt(Math.max(trades, 1e-12) / inflation);
  const qa = a * a - (zb * zb) / 2;
  const qb = -2 * a * za;
  const qc = za * za - zb * zb;
  const disc = Math.max(qb * qb - 4 * qa * qc, 0);
  const sr = (-qb + Math.sqrt(disc)) / (2 * qa);
  const gross = sr * spec.volBps * Math.sqrt(spec.holdBars);
  const edgeBps = (gross + spec.costBps) / spec.holdBars;
  const tradesPerYear = (exposure * barsPerDay * DAYS_PER_YEAR) / spec.holdBars;
  return { edgeBps, impliedAnnualSharpe: sr * Math.sqrt(tradesPerYear) };
}

/** Power actually achieved at a given sample size. */
export function analyticPower(spec: EdgeSpec & { bars: number }): number {
  const alpha = spec.alpha ?? 0.05;
  const exposure = spec.exposure ?? 1;
  const sr = netSharpePerTrade(spec);
  const trades = (spec.bars * exposure) / spec.holdBars;
  const inflation = varianceInflation(spec.holdBars, spec.autocorr ?? 0);
  // Invert the same relation: z_beta = (SR*sqrt(n/inflation) - z_alpha) /
  // sqrt(1 + SR^2/2), so at exactly `barsRequired` bars this returns the power
  // that was asked for.
  const ncp = sr * Math.sqrt(trades / inflation);
  return normalCdf((ncp - normalQuantile(1 - alpha)) / Math.sqrt(1 + (sr * sr) / 2));
}

/**
 * Years of data an annualized Sharpe needs, and the Sharpe a given span needs.
 *
 * These two are the whole power analysis in one line each, and they depend only
 * on calendar time — a faster timeframe does NOT buy statistical resolution,
 * because the edge per bar shrinks exactly as fast as the bars multiply.
 */
export function yearsRequired(sharpeAnnual: number, alpha = 0.05, power = 0.8): number {
  if (sharpeAnnual <= 0) return Infinity;
  const z = normalQuantile(1 - alpha) + normalQuantile(power);
  return (z / sharpeAnnual) ** 2;
}

export function annualSharpeRequired(years: number, alpha = 0.05, power = 0.8): number {
  const z = normalQuantile(1 - alpha) + normalQuantile(power);
  return z / Math.sqrt(Math.max(years, 1e-12));
}

export interface MultipleTestingAlpha {
  bonferroni: number;
  sidak: number;
  /** Harvey, Liu & Zhu (2016): t > 3.0 as the floor for any finance claim. */
  harveyLiuZhu: number;
  /** The t-statistic a single result must clear, taking the stricter of the two. */
  tStatRequired: number;
}

export function multipleTestingAlpha(tests: number, alpha = 0.05): MultipleTestingAlpha {
  const m = Math.max(1, tests);
  const bonferroni = alpha / m;
  const sidak = 1 - Math.pow(1 - alpha, 1 / m);
  const harveyLiuZhu = 1 - normalCdf(3.0);
  return {
    bonferroni, sidak, harveyLiuZhu,
    tStatRequired: Math.max(3.0, normalQuantile(1 - bonferroni)),
  };
}

/**
 * A permutation p-value cannot be smaller than 1/(draws+1).
 *
 * The repo's `validate` defaults to 1000 draws, whose floor is 1e-3 — coarser
 * than the Bonferroni threshold a 240-cell sweep demands. The test literally
 * cannot express the answer it is being asked for.
 */
export function permutationTrialsRequired(alpha: number): { minimum: number; stable: number } {
  const minimum = Math.ceil(1 / alpha);
  return { minimum, stable: minimum * 10 };
}
/** Average pairwise Pearson correlation of a set of aligned series. */
export function averageCorrelation(series: (number | null)[][]): number {
  const n = series.length;
  if (n < 2) return 0;
  let sum = 0, pairs = 0;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const a: number[] = [], b: number[] = [];
      for (let t = 0; t < Math.min(series[i].length, series[j].length); t++) {
        const x = series[i][t], y = series[j][t];
        if (x === null || y === null || !Number.isFinite(x) || !Number.isFinite(y)) continue;
        a.push(x); b.push(y);
      }
      if (a.length < 30) continue;
      let ma = 0, mb = 0;
      for (let k = 0; k < a.length; k++) { ma += a[k]; mb += b[k]; }
      ma /= a.length; mb /= b.length;
      let sab = 0, saa = 0, sbb = 0;
      for (let k = 0; k < a.length; k++) {
        const da = a[k] - ma, db = b[k] - mb;
        sab += da * db; saa += da * da; sbb += db * db;
      }
      if (saa > 0 && sbb > 0) { sum += sab / Math.sqrt(saa * sbb); pairs++; }
    }
  }
  return pairs > 0 ? sum / pairs : 0;
}

/** Annualized Sharpe implied by an edge spec at a given bar frequency. */
export function impliedAnnualSharpe(spec: EdgeSpec, barsPerDay?: number): number {
  const bpd = barsPerDay ?? spec.barsPerDay ?? BARS_PER_DAY["30m"];
  const exposure = spec.exposure ?? 1;
  const tradesPerYear = (exposure * bpd * DAYS_PER_YEAR) / spec.holdBars;
  return netSharpePerTrade(spec) * Math.sqrt(tradesPerYear);
}

export interface PermutationPowerSpec {
  bars: number;
  holdBars: number;
  /** Volatility while holding, bps. */
  volBps: number;
  costBps: number;
  exposure: number;
  edgeBps: number;
  /** Market drift present whether or not the strategy is holding, bps/bar. */
  driftBps?: number;
  /** Volatility on bars the strategy sits out. Defaults to volBps — set it
   *  lower to model a strategy that only holds the violent regime. */
  idleVolBps?: number;
  /** Null construction, matching diagnostics.ts. */
  method?: "rotate" | "shuffle";
  /** Permutation draws inside each simulated experiment. */
  innerTrials?: number;
  seed?: number;
}

/**
 * The power of the test the repo ACTUALLY runs, by simulation.
 *
 * The analytic formulas above describe a t-test on trade returns. `validate`
 * runs something else: a circular rotation of the position series, which keeps
 * every run and therefore the exact turnover. That null is right, but it is not
 * free — rotating a series that is in the market 90% of the time produces
 * rearrangements that look a great deal like the original, so the test loses
 * power exactly when exposure is high. No formula captures that; simulating it
 * is the only honest way to know what the repo's own p-values can resolve.
 */
export function permutationPower(
  spec: PermutationPowerSpec,
  experiments = 200,
): { power: number; medianP: number } {
  const innerTrials = spec.innerTrials ?? 300;
  const method = spec.method ?? "rotate";
  const idleVol = spec.idleVolBps ?? spec.volBps;
  const drift = (spec.driftBps ?? 0) / 10_000;
  const cost = spec.costBps / 10_000 / 2; // per side; a round trip pays twice
  let seed = (spec.seed ?? 1) >>> 0;
  // xorshift, inline so this file stays free of imports beyond the math above.
  const rng = () => {
    seed ^= seed << 13; seed >>>= 0;
    seed ^= seed >> 17;
    seed ^= seed << 5; seed >>>= 0;
    return seed / 4294967296;
  };
  const randn = () => {
    const u = Math.max(rng(), 1e-12), v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };

  const pnl = (pos: number[], ret: number[]) => {
    let acc = 0, prev = 0;
    for (let t = 0; t < pos.length; t++) {
      acc += pos[t] * ret[t] - cost * Math.abs(pos[t] - prev);
      prev = pos[t];
    }
    return acc;
  };

  const pValues: number[] = [];
  let rejects = 0;
  for (let e = 0; e < experiments; e++) {
    // Build one experiment: blocks of `holdBars` in the market, spaced to hit
    // the requested exposure, with the edge present only while holding.
    const gap = Math.max(1, Math.round(spec.holdBars * (1 / spec.exposure - 1)));
    const pos: number[] = [];
    const ret: number[] = [];
    let t = 0;
    while (t < spec.bars) {
      const holding = pos.length === 0 ? true : pos[pos.length - 1] === 0;
      const run = holding ? spec.holdBars : gap;
      for (let k = 0; k < run && t < spec.bars; k++, t++) {
        pos.push(holding ? 1 : 0);
        const vol = (holding ? spec.volBps : idleVol) / 10_000;
        ret.push(drift + (holding ? spec.edgeBps / 10_000 : 0) + vol * randn());
      }
    }

    const actual = pnl(pos, ret);
    let atLeast = 0;
    for (let i = 0; i < innerTrials; i++) {
      let shuffled: number[];
      if (method === "rotate") {
        const k = 1 + Math.floor(rng() * (pos.length - 1));
        shuffled = pos.slice(k).concat(pos.slice(0, k));
      } else {
        shuffled = pos.slice();
        for (let j = shuffled.length - 1; j > 0; j--) {
          const i2 = Math.floor(rng() * (j + 1));
          [shuffled[j], shuffled[i2]] = [shuffled[i2], shuffled[j]];
        }
      }
      if (pnl(shuffled, ret) >= actual) atLeast++;
    }
    const p = (atLeast + 1) / (innerTrials + 1);
    pValues.push(p);
    if (p < 0.05) rejects++;
  }
  pValues.sort((a, b) => a - b);
  return {
    power: rejects / experiments,
    medianP: pValues[Math.floor(pValues.length / 2)],
  };
}
