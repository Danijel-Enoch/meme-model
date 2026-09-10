import { expect, test, describe } from "bun:test";
import {
  barBudget, barsPerDay, walkForwardSizes, pickBest, bestByCoin, rejectionReason,
  buildSavedModel, modelFileName, SWEEP_BAR_CAP, SWEEP_WINDOW,
  type SweepRow,
} from "./sweep";
import { fit } from "./hmm";
import { buildFeatures, fitScaler, applyScaler } from "./features";
import { generateSynthetic } from "./data";

/** A row with sane defaults; each test overrides only the field it is about. */
function row(over: Partial<SweepRow>): SweepRow {
  return {
    coin: "BTC", timeframe: "1h", modelType: "hmm",
    bars: 1000, trainSize: 500, testSize: 167, refits: 3, barsTraded: 499,
    roi: 0, buyHold: 0, sharpe: 0, maxDD: 0, exposure: 0.4, trades: 10,
    medianRandom: 0, excessRoi: 0, pValue: 0.5, ceilingHold20: 1,
    ...over,
  };
}

describe("bar budgeting", () => {
  test("bars per day matches the interval length", () => {
    expect(barsPerDay("1h")).toBe(24);
    expect(barsPerDay("15m")).toBe(96);
    expect(barsPerDay("2h")).toBe(12);
  });

  test("45 days fits inside the retention cap from 15m up", () => {
    for (const tf of ["15m", "30m", "1h", "2h"]) {
      const b = barBudget(tf, 45);
      expect(b.truncated).toBe(false);
      expect(b.bars).toBe(Math.ceil(45 * barsPerDay(tf)));
      expect(b.daysCovered).toBeCloseTo(45, 6);
    }
  });

  test("5m cannot reach three weeks, which is why it is not a default", () => {
    const b = barBudget("5m", 45);
    expect(b.truncated).toBe(true);
    expect(b.bars).toBe(SWEEP_BAR_CAP);
    expect(b.wanted).toBe(45 * 288);
    // ~17 days, well short of 21.
    expect(b.daysCovered).toBeLessThan(21);
  });

  test("the cap never inflates a short request", () => {
    const b = barBudget("1h", 5);
    expect(b.bars).toBe(120);
    expect(b.truncated).toBe(false);
  });
});

describe("walk-forward sizing", () => {
  test("half to train, a third of that to trade", () => {
    const s = walkForwardSizes(2000);
    expect(s.trainSize).toBe(1000);
    expect(s.testSize).toBe(333);
    expect(s.needed).toBe(1000 + 333 + 2);
    expect(s.fits).toBe(true);
  });

  test("the training clamp binds at both ends", () => {
    expect(walkForwardSizes(100).trainSize).toBe(300);    // floor
    expect(walkForwardSizes(20_000).trainSize).toBe(1500); // ceiling
  });

  test("the test clamp binds at both ends", () => {
    expect(walkForwardSizes(400).testSize).toBe(100);      // train clamped to 300 -> 100
    expect(walkForwardSizes(20_000).testSize).toBe(500);   // train 1500 -> 500
  });

  test("a series too short for its own clamped sizes does not fit", () => {
    // 300 train + 100 test + 2 = 402 rows demanded, and the floor cannot go lower.
    expect(walkForwardSizes(401).fits).toBe(false);
    expect(walkForwardSizes(402).fits).toBe(true);
  });

  test("a 45-day 2h series clears the floor", () => {
    const bars = barBudget("2h", 45).bars - SWEEP_WINDOW;
    expect(walkForwardSizes(bars).fits).toBe(true);
  });
});

describe("best-row selection", () => {
  test("a 95%-exposure monster loses to a modest 40%-exposure row", () => {
    const hold = row({ timeframe: "1h", exposure: 0.95, roi: 3.0, excessRoi: 2.5, trades: 8 });
    const real = row({ timeframe: "30m", exposure: 0.40, roi: 0.15, excessRoi: 0.12, trades: 8 });
    expect(pickBest([hold, real])?.timeframe).toBe("30m");
  });

  test("exactly 90% exposure is still allowed", () => {
    const edge = row({ timeframe: "2h", exposure: 0.90, excessRoi: 0.5 });
    expect(pickBest([edge])?.timeframe).toBe("2h");
  });

  test("two trades is not a measurement", () => {
    const thin = row({ timeframe: "15m", trades: 2, excessRoi: 5.0 });
    const thick = row({ timeframe: "1h", trades: 3, excessRoi: 0.01 });
    expect(pickBest([thin, thick])?.timeframe).toBe("1h");
  });

  test("skipped rows never win, however good they look", () => {
    const skipped = row({ timeframe: "5m", excessRoi: 9.9, skipped: "only reaches 17 days" });
    const clean = row({ timeframe: "1h", excessRoi: 0.02 });
    expect(pickBest([skipped, clean])?.timeframe).toBe("1h");
    expect(pickBest([skipped])).toBeUndefined();
  });

  test("ranking is on excess ROI, not ROI", () => {
    // The bigger raw return is entirely explained by the null at that exposure.
    const lucky = row({ timeframe: "15m", roi: 1.2, medianRandom: 1.19, excessRoi: 0.01 });
    const skilled = row({ timeframe: "2h", roi: 0.3, medianRandom: 0.05, excessRoi: 0.25 });
    expect(pickBest([lucky, skilled])?.timeframe).toBe("2h");
  });

  test("ties break on the lower p-value", () => {
    const a = row({ timeframe: "15m", excessRoi: 0.2, pValue: 0.30 });
    const b = row({ timeframe: "1h", excessRoi: 0.2, pValue: 0.01 });
    expect(pickBest([a, b])?.timeframe).toBe("1h");
    expect(pickBest([b, a])?.timeframe).toBe("1h");
  });

  test("no eligible row means no winner at all", () => {
    expect(pickBest([])).toBeUndefined();
    expect(pickBest([row({ trades: 0 })])).toBeUndefined();
    expect(pickBest([row({ exposure: 0.99 })])).toBeUndefined();
  });

  test("bestByCoin keys by coin and omits coins that qualified nowhere", () => {
    const best = bestByCoin([
      row({ coin: "BTC", timeframe: "1h", excessRoi: 0.1 }),
      row({ coin: "BTC", timeframe: "2h", excessRoi: 0.4 }),
      row({ coin: "DOGE", timeframe: "1h", trades: 1, excessRoi: 8 }),
    ]);
    expect(Object.keys(best)).toEqual(["BTC"]);
    expect(best.BTC.timeframe).toBe("2h");
  });

  test("rejection reasons name the filter that bit", () => {
    expect(rejectionReason([row({ trades: 1 }), row({ trades: 2 })])).toContain("traded");
    expect(rejectionReason([row({ exposure: 0.97 })])).toContain("exposure");
    expect(rejectionReason([row({ skipped: "fetch failed: HTTP 500" })])).toContain("HTTP 500");
  });
});

describe("saved model shape", () => {
  // Fit something real and small: the assertion is about the envelope
  // `predict --model` reads, so the params inside must be genuine.
  const { candles } = generateSynthetic({ bars: 400, seed: 3 });
  const fs = buildFeatures(candles, { window: SWEEP_WINDOW });
  const scaler = fitScaler(fs.X, fs.T, fs.D);
  const params = fit(applyScaler(fs.X, fs.T, fs.D, scaler), fs.T, fs.D, { states: 3, restarts: 1, maxIter: 20 }).params;
  const r = row({ coin: "SOL", timeframe: "30m", modelType: "hmm", bars: fs.T });
  const saved = buildSavedModel(r, params, scaler, fs.names, fs.T);

  test("carries exactly the fields predict reads", () => {
    expect(Object.keys(saved).sort()).toEqual(
      ["label", "modelType", "names", "params", "scaler", "source", "trainedBars", "window"],
    );
  });

  test("params round-trip through JSON with the HMM's own shape", () => {
    const p = JSON.parse(JSON.stringify(saved.params));
    expect(p.K).toBe(3);
    expect(p.D).toBe(fs.D);
    expect(p.pi.length).toBe(3);
    expect(p.A.length).toBe(9);
    expect(p.mu.length).toBe(3 * fs.D);
    expect(p.vari.length).toBe(3 * fs.D);
  });

  test("the scaler survives as plain arrays, not typed ones", () => {
    expect(Array.isArray(saved.scaler.mean)).toBe(true);
    expect(Array.isArray(saved.scaler.std)).toBe(true);
    expect(saved.scaler.mean.length).toBe(fs.D);
    expect(saved.scaler.std.length).toBe(fs.D);
  });

  test("the source lets predict reload the market without repeating flags", () => {
    expect(saved.source).toEqual({ coin: "SOL", timeframe: "30m" });
    expect(saved.window).toBe(SWEEP_WINDOW);
    expect(saved.names).toEqual(fs.names);
    // predict rebuilds features from these two booleans.
    expect(saved.names.includes("realizedVol")).toBe(true);
    expect(saved.names.includes("volumeSurge")).toBe(true);
  });

  test("the whole envelope is JSON-serializable and reloads identically", () => {
    const reloaded = JSON.parse(JSON.stringify(saved));
    expect(reloaded.modelType).toBe("hmm");
    expect(reloaded.trainedBars).toBe(fs.T);
    expect(reloaded.label).toBe("SOL-PERP 30m");
  });

  test("file names are unique per coin, timeframe and model", () => {
    expect(modelFileName(r)).toBe("sol-30m-hmm.model.json");
    expect(modelFileName(row({ coin: "ETH", timeframe: "2h", modelType: "hsmm" })))
      .toBe("eth-2h-hsmm.model.json");
  });
});
