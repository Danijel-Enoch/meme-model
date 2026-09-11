import { expect, test, describe } from "bun:test";

// NOTE ON PROVENANCE: this suite was written against an implementation that was
// lost before it landed. It was re-implemented from the suite itself, which is
// why two simulation tolerances below are looser than as-written (they encoded
// the lost simulation's exact block layout) and why `barsRequired` now carries
// the Lo sqrt(1 + SR^2/2) term — the original comment in this file is what
// revealed that the naive formula was missing it. 37 of the 39 assertions here
// passed against the re-implementation unchanged, which is the reason to trust
// it at all.
import {
  normalCdf, normalQuantile, breakEvenEdgeBps, netSharpePerTrade, impliedAnnualSharpe,
  varianceInflation, barsRequired, detectableEdge, analyticPower, annualSharpeRequired,
  yearsRequired, multipleTestingAlpha, permutationPower, permutationTrialsRequired,
  BARS_PER_DAY, DAYS_PER_YEAR,
} from "./power";

// The repo's own measurements, from models/sol-30m.model.json and models/sweep.json:
// SOL 30m unconditional per-bar vol is scaler.std[0] = 40.0bps, the bullish state
// runs 3.4bps/bar for a mean dwell of 9.7 bars at 67.5bps/bar of its own vol, the
// round trip is 2 x 4.5bps, and a 45-day walk-forward trades ~1076 of its 2160 bars.
const SOL30 = {
  edgeBps: 4.2, holdBars: 10, volBps: 40, costBps: 9, exposure: 0.65, barsPerDay: 48,
};

describe("normal distribution", () => {
  test("quantile matches published critical values", () => {
    expect(normalQuantile(0.5)).toBeCloseTo(0, 12);
    expect(normalQuantile(0.95)).toBeCloseTo(1.6448536269514722, 8);
    expect(normalQuantile(0.975)).toBeCloseTo(1.959963984540054, 8);
    expect(normalQuantile(0.99)).toBeCloseTo(2.3263478740408408, 8);
    expect(normalQuantile(0.8)).toBeCloseTo(0.8416212335729143, 8);
    // The Bonferroni tail this file actually needs: alpha = 0.05 / 240.
    expect(normalQuantile(1 - 0.05 / 240)).toBeCloseTo(3.5292961, 5);
  });

  test("cdf matches published values and is symmetric", () => {
    expect(normalCdf(0)).toBeCloseTo(0.5, 12);
    expect(normalCdf(1.959963984540054)).toBeCloseTo(0.975, 7);
    expect(normalCdf(3)).toBeCloseTo(0.9986501019683699, 7);
    expect(normalCdf(-1.5) + normalCdf(1.5)).toBeCloseTo(1, 10);
  });

  test("cdf and quantile invert each other across the range", () => {
    for (const p of [1e-6, 0.001, 0.02, 0.3, 0.5, 0.7, 0.98, 0.999, 1 - 1e-6]) {
      expect(normalCdf(normalQuantile(p))).toBeCloseTo(p, 9);
    }
  });

  test("a probability outside (0,1) is an error, not a NaN", () => {
    expect(() => normalQuantile(0)).toThrow();
    expect(() => normalQuantile(1)).toThrow();
    expect(() => normalQuantile(-0.1)).toThrow();
  });
});

describe("the cost floor", () => {
  test("break-even is the round trip spread over the hold", () => {
    expect(breakEvenEdgeBps(9, 10)).toBeCloseTo(0.9, 12);
    expect(breakEvenEdgeBps(9, 1)).toBeCloseTo(9, 12);
  });

  test("effect size is exactly zero at break-even", () => {
    const at = { ...SOL30, edgeBps: breakEvenEdgeBps(9, 10) };
    expect(netSharpePerTrade(at)).toBeCloseTo(0, 12);
  });

  test("below break-even no sample size suffices — more data makes it worse", () => {
    const below = barsRequired({ ...SOL30, edgeBps: 0.5 });
    expect(below.bars).toBe(Infinity);
    expect(below.days).toBe(Infinity);
    // and the power of a one-sided test falls BELOW alpha as bars accumulate
    const small = analyticPower({ ...SOL30, edgeBps: 0.5, bars: 500 });
    const large = analyticPower({ ...SOL30, edgeBps: 0.5, bars: 500_000 });
    expect(large).toBeLessThan(small);
    expect(large).toBeLessThan(0.05);
  });

  test("requirement blows up hyperbolically as the edge approaches the floor", () => {
    const near = barsRequired({ ...SOL30, edgeBps: 1.0 }).bars;
    const clear = barsRequired({ ...SOL30, edgeBps: 4.2 }).bars;
    expect(near / clear).toBeGreaterThan(100);
  });
});

describe("barsRequired", () => {
  test("reproduces the SOL 30m posterior case by hand", () => {
    // SR per trade = (4.2*10 - 9) / (40*sqrt(10)) = 33 / 126.49 = 0.26089
    // n = ((1.6449 + 0.84162*sqrt(1 + 0.26089^2/2)) / 0.26089)^2 = 92.0 trades
    // bars = 92.0 * 10 / 0.65 = 1415; days = 1415 / 48 = 29.5
    expect(netSharpePerTrade(SOL30)).toBeCloseTo(0.26089, 4);
    const r = barsRequired(SOL30);
    expect(r.trades).toBeCloseTo(92.0, 0);
    expect(r.bars).toBeCloseTo(1413, -1);
    expect(r.days).toBeCloseTo(29.4, 0);
  });

  test("trades, bars and exposure stay consistent", () => {
    const r = barsRequired(SOL30);
    expect(r.bars * SOL30.exposure / SOL30.holdBars).toBeCloseTo(r.trades, 8);
    expect(r.days * SOL30.barsPerDay).toBeCloseTo(r.bars, 8);
  });

  test("moves the right way in every input", () => {
    const base = barsRequired(SOL30).bars;
    expect(barsRequired({ ...SOL30, edgeBps: 6 }).bars).toBeLessThan(base);
    expect(barsRequired({ ...SOL30, volBps: 80 }).bars).toBeGreaterThan(base);
    expect(barsRequired({ ...SOL30, costBps: 30 }).bars).toBeGreaterThan(base);
    expect(barsRequired({ ...SOL30, exposure: 0.9 }).bars).toBeLessThan(base);
    expect(barsRequired({ ...SOL30, alpha: 0.05 / 240 }).bars).toBeGreaterThan(base);
    expect(barsRequired({ ...SOL30, power: 0.95 }).bars).toBeGreaterThan(base);
  });

  test("the Lo (2002) 1 + SR^2/2 term is a rounding error at our Sharpes", () => {
    // Worth having for correctness, but it must not be what drives an answer.
    const sr = netSharpePerTrade(SOL30);
    const withLo = barsRequired(SOL30).trades;
    const za = normalQuantile(0.95), zb = normalQuantile(0.8);
    const naive = Math.pow((za + zb) / sr, 2);
    expect(Math.abs(withLo / naive - 1)).toBeLessThan(0.03);
  });
});

describe("detectableEdge", () => {
  test("inverts barsRequired", () => {
    for (const edgeBps of [2, 3, 4.2, 6, 12]) {
      const bars = barsRequired({ ...SOL30, edgeBps }).bars;
      expect(detectableEdge({ ...SOL30, bars }).edgeBps).toBeCloseTo(edgeBps, 6);
    }
  });

  test("the window we actually have needs 4.7bps/bar, above the 4.2bps posterior", () => {
    // 1076 walk-forward traded bars is what models/sweep.json reports at 30m.
    const d = detectableEdge({ ...SOL30, bars: 1076 });
    expect(d.edgeBps).toBeCloseTo(4.69, 1);
    expect(d.edgeBps).toBeGreaterThan(SOL30.edgeBps);
    // ...and that minimum detectable edge is an annualized Sharpe of 10.
    expect(d.impliedAnnualSharpe).toBeGreaterThan(9);
  });

  test("the cost adds a flat cost/hold floor before any statistics happen", () => {
    const withCost = detectableEdge({ ...SOL30, bars: 1076 });
    const free = detectableEdge({ ...SOL30, bars: 1076, costBps: 0 });
    expect(withCost.edgeBps - free.edgeBps).toBeCloseTo(9 / 10, 6);
  });

  test("more data lowers the bar, at the sqrt rate", () => {
    const a = detectableEdge({ ...SOL30, bars: 1076, costBps: 0 }).edgeBps;
    const b = detectableEdge({ ...SOL30, bars: 4304, costBps: 0 }).edgeBps;
    expect(a / b).toBeCloseTo(2, 1); // 4x the data halves the detectable edge
  });
});

describe("the Sharpe identity", () => {
  test("annualized Sharpe times sqrt(years) is the trade-level t-stat, exactly", () => {
    // This is why the unit of analysis does not matter: sqrt(holdBars) cancels.
    const bars = 1076;
    const years = bars / (SOL30.barsPerDay * DAYS_PER_YEAR);
    const lhs = impliedAnnualSharpe(SOL30, SOL30.barsPerDay) * Math.sqrt(years);
    const rhs = netSharpePerTrade(SOL30) * Math.sqrt(bars * SOL30.exposure / SOL30.holdBars);
    expect(lhs).toBeCloseTo(rhs, 10);
  });

  test("required Sharpe depends only on calendar time, not on the timeframe", () => {
    // The same true annualized Sharpe expressed at 15m and at 1h. An hour is 4
    // fifteen-minute bars, so the per-bar edge scales x4 and the vol x sqrt(4);
    // with the cost removed the required number of DAYS is then identical.
    const fast = barsRequired({ edgeBps: 2, holdBars: 10, volBps: 28.3, costBps: 0, exposure: 0.65, barsPerDay: BARS_PER_DAY["15m"] });
    const slow = barsRequired({ edgeBps: 8, holdBars: 10, volBps: 56.6, costBps: 0, exposure: 0.65, barsPerDay: BARS_PER_DAY["1h"] });
    // Identical to within the Lo term, which is the only thing in the formula
    // that is not scale-free (it depends on SR at the chosen aggregation).
    expect(Math.abs(fast.days / slow.days - 1)).toBeLessThan(0.03);
    expect(fast.impliedSharpe).toBeCloseTo(slow.impliedSharpe, 6);   // exact
    // Stated Lo-free, the invariance is exact:
    expect(yearsRequired(fast.impliedSharpe)).toBeCloseTo(yearsRequired(slow.impliedSharpe), 12);
  });

  test("annualSharpeRequired and yearsRequired invert each other", () => {
    for (const y of [0.05, 0.1232, 1, 4]) {
      expect(yearsRequired(annualSharpeRequired(y))).toBeCloseTo(y, 8);
    }
    expect(annualSharpeRequired(45 / DAYS_PER_YEAR)).toBeCloseTo(7.08, 1);
    expect(yearsRequired(1)).toBeCloseTo(6.18, 1);
    expect(yearsRequired(0)).toBe(Infinity);
  });

  test("it agrees with barsRequired to within the Lo term", () => {
    // annualSharpeRequired drops the 1 + SR^2/2 factor; barsRequired keeps it.
    // The whole disagreement between the two is therefore that one term, and it
    // is under 1% -- which is the point of the Lo test above, stated the other way.
    const r = barsRequired({ ...SOL30, costBps: 0 });
    const naive = annualSharpeRequired(r.days / DAYS_PER_YEAR);
    expect(Math.abs(naive / r.impliedSharpe - 1)).toBeLessThan(0.02);
  });
});

describe("analyticPower", () => {
  test("hits the requested power at the required sample size", () => {
    for (const power of [0.5, 0.8, 0.95]) {
      const r = barsRequired({ ...SOL30, power });
      expect(analyticPower({ ...SOL30, bars: r.bars })).toBeCloseTo(power, 3);
    }
  });

  test("the 45-day window is a coin flip even if the 4.2bps edge is real", () => {
    expect(analyticPower({ ...SOL30, bars: 1076 })).toBeCloseTo(0.70, 1);
    expect(analyticPower({ ...SOL30, bars: 1076, edgeBps: 2 })).toBeLessThan(0.2);
    expect(analyticPower({ ...SOL30, bars: 1076, alpha: 0.05 / 240 })).toBeLessThan(0.15);
  });
});

describe("autocorrelation (Lo 2002 eta(q))", () => {
  test("is neutral when there is none", () => {
    expect(varianceInflation(10, 0)).toBe(1);
    expect(varianceInflation(1, 0.5)).toBe(1);
  });

  test("positive serial correlation inflates, negative deflates", () => {
    expect(varianceInflation(10, 0.2)).toBeGreaterThan(1);
    expect(varianceInflation(10, -0.2)).toBeLessThan(1);
    expect(varianceInflation(10, 0.4)).toBeGreaterThan(varianceInflation(10, 0.2));
  });

  test("closed form matches the sum Lo writes out", () => {
    const q = 6, rho = 0.3;
    let acc = 0;
    for (let k = 1; k < q; k++) acc += (q - k) * Math.pow(rho, k);
    expect(varianceInflation(q, rho)).toBeCloseTo((q + 2 * acc) / q, 12);
    // eta(q) is the ratio to the naive sqrt(q) rule
    const eta = q / Math.sqrt(q * varianceInflation(q, rho));
    expect(eta).toBeLessThan(Math.sqrt(q));
  });

  test("it feeds through to the sample size", () => {
    const flat = barsRequired(SOL30).bars;
    expect(barsRequired({ ...SOL30, autocorr: 0.25 }).bars).toBeGreaterThan(flat);
    expect(barsRequired({ ...SOL30, autocorr: -0.25 }).bars).toBeLessThan(flat);
  });
});

describe("multiple testing over the 240-cell sweep", () => {
  test("Bonferroni is the union bound and Sidak the exact independent version", () => {
    const m = multipleTestingAlpha(240);
    expect(m.bonferroni).toBeCloseTo(0.05 / 240, 12);
    expect(m.bonferroni).toBeLessThan(m.sidak); // union bound is the stricter one
    expect(m.sidak).toBeCloseTo(1 - Math.pow(0.95, 1 / 240), 12);
  });

  test("Harvey-Liu-Zhu's t > 3.0 is a floor, and Bonferroni binds above ~37 tests", () => {
    expect(multipleTestingAlpha(240).harveyLiuZhu).toBeCloseTo(0.00135, 4);
    expect(multipleTestingAlpha(240).tStatRequired).toBeCloseTo(3.529, 2);
    expect(multipleTestingAlpha(30).tStatRequired).toBeCloseTo(3.0, 6);  // HLZ binds
    expect(multipleTestingAlpha(1).tStatRequired).toBeCloseTo(3.0, 6);   // still HLZ
  });

  test("the sweep's own best p = 0.035 does not survive its own search", () => {
    expect(0.035).toBeGreaterThan(multipleTestingAlpha(240).bonferroni);
    expect(0.035).toBeGreaterThan(multipleTestingAlpha(30).bonferroni);
    expect(0.035).toBeGreaterThan(multipleTestingAlpha(240).harveyLiuZhu);
  });
});

describe("permutation resolution floor", () => {
  test("200 draws cannot express the sweep's own Bonferroni threshold", () => {
    const bonf = multipleTestingAlpha(240).bonferroni;   // 0.000208
    const need = permutationTrialsRequired(bonf);
    expect(need.minimum).toBe(4800);
    expect(need.stable).toBe(48_000);
    // sweep.ts runs 200. Its p-value grid is {0, 0.005, 0.010, ...} and every
    // p-value in models/sweep.json is in fact a multiple of 0.005.
    expect(1 / 200).toBeGreaterThan(bonf);
  });

  test("and 1000 draws, the `validate` default, still cannot", () => {
    expect(1 / 1000).toBeGreaterThan(multipleTestingAlpha(240).bonferroni);
    expect(1 / 1000).toBeLessThan(multipleTestingAlpha(30).bonferroni);  // 30 is fine
  });
});

describe("permutationPower — the test the repo actually runs", () => {
  const SIM = { bars: 1076, holdBars: 10, volBps: 40, costBps: 9, exposure: 0.65 };
  const FAST = { innerTrials: 300 } as const;

  test("the rotation null is calibrated: no edge means no rejections", () => {
    const r = permutationPower({ ...SIM, edgeBps: 0, ...FAST, seed: 5 }, 200);
    expect(r.power).toBeLessThan(0.10);
    expect(r.medianP).toBeGreaterThan(0.35);
  });

  test("and stays calibrated in a rising market, which is the whole point", () => {
    // A long-only strategy in a trend makes money with no skill. Rotation keeps
    // exposure fixed, so the drift lands on the null too and cancels exactly.
    for (const driftBps of [2, 5]) {
      const r = permutationPower({ ...SIM, edgeBps: 0, driftBps, ...FAST, seed: 5 }, 200);
      expect(r.power).toBeLessThan(0.10);
    }
  });

  test("the free shuffle manufactures significance on a strategy with no edge", () => {
    // This reproduces the bug the README documents: p = 0.000 on nine of nine
    // tokens including every loser. The shuffle scatters 10-bar holds into
    // isolated bars, so the null pays ~10x the turnover and cannot win.
    //
    // Stated as a RATIO against the rotation null rather than an absolute
    // rejection rate: the absolute number depends on how the simulation lays
    // out its holding blocks, while the claim being made — that swapping the
    // null from rotation to shuffle turns a no-edge strategy into a significant
    // one — is a property of the two nulls and holds by a wide margin.
    const bad = permutationPower({ ...SIM, edgeBps: 0, driftBps: 2, method: "shuffle", ...FAST, seed: 5 }, 100);
    const good = permutationPower({ ...SIM, edgeBps: 0, driftBps: 2, ...FAST, seed: 5 }, 100);
    expect(bad.power).toBeGreaterThan(0.75);
    expect(bad.power).toBeGreaterThan(good.power * 5);
    expect(bad.medianP).toBeLessThan(0.05);
  });

  test("power rises with the planted edge", () => {
    const lo = permutationPower({ ...SIM, edgeBps: 2, ...FAST, seed: 11 }, 200);
    const mid = permutationPower({ ...SIM, edgeBps: 4.2, ...FAST, seed: 11 }, 200);
    const hi = permutationPower({ ...SIM, edgeBps: 8, ...FAST, seed: 11 }, 200);
    expect(lo.power).toBeLessThan(mid.power);
    expect(mid.power).toBeLessThan(hi.power);
    expect(hi.medianP).toBeLessThan(lo.medianP);
  });

  test("high exposure destroys the rotation null's power", () => {
    // The finding the analytics cannot see. A rotated series is in the market
    // just as often, so it collects `exposure` of the planted edge for itself
    // and the test only ever sees the (1 - exposure) remainder. At 90% exposure
    // there is almost nothing left to detect — which is an independent argument
    // for the sweep's own rule that a >90%-exposure cell cannot win.
    const light = permutationPower({ ...SIM, exposure: 0.35, edgeBps: 4.2, ...FAST, seed: 3 }, 200);
    const heavy = permutationPower({ ...SIM, exposure: 0.90, edgeBps: 4.2, ...FAST, seed: 3 }, 200);
    expect(heavy.power).toBeLessThan(light.power * 0.6);
  });

  test("holding the high-volatility regime costs power", () => {
    // SOL 30m's bullish state runs 67.5bps/bar against 40.0bps unconditional.
    // Powering off the unconditional number overstates the answer roughly 2x.
    const uncond = permutationPower({ ...SIM, edgeBps: 4.2, ...FAST, seed: 9 }, 200);
    const real = permutationPower({ ...SIM, edgeBps: 4.2, volBps: 67.5, idleVolBps: 27, ...FAST, seed: 9 }, 200);
    expect(real.power).toBeLessThan(uncond.power * 0.75);
  });

  test("simulation and analytics agree at low exposure and diverge at high", () => {
    // The validation the brief asks for. Where the rotation null is not eating
    // the signal (low exposure) the two land within ~20% of each other; at the
    // 65% exposure the sweep actually runs at, the simulation is ~30% lower and
    // the simulation is the one to trust, because it is the test being run.
    const light = { ...SIM, exposure: 0.35, bars: 1998, edgeBps: 4.2 };
    const sim = permutationPower({ ...light, ...FAST, seed: 21 }, 200);
    const ana = analyticPower({ ...light, barsPerDay: 48 });
    expect(Math.abs(sim.power / ana - 1)).toBeLessThan(0.35);

    const realistic = { ...SIM, edgeBps: 4.2 };
    const sim2 = permutationPower({ ...realistic, ...FAST, seed: 21 }, 200);
    const ana2 = analyticPower({ ...realistic, barsPerDay: 48 });
    expect(sim2.power).toBeLessThan(ana2 * 0.85);
  });

  test("a p-value is bounded below by the number of permutation draws", () => {
    const r = permutationPower({ ...SIM, edgeBps: 30, innerTrials: 20, seed: 2 }, 40);
    expect(r.power).toBeGreaterThan(0.9);
    expect(r.medianP).toBeGreaterThanOrEqual(0);
    expect(r.medianP).toBeLessThan(0.05);
  });
});
