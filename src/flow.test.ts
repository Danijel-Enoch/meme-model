import { expect, test, describe } from "bun:test";
import { buildFeatures, type Candle } from "./features";
import {
  alignStep,
  augmentFeatures,
  buildFlowColumns,
  fetchFundingRange,
  firstValidRow,
  flowFeatures,
  fundingChunks,
  FUNDING_PAGE_HOURS,
  icStats,
  measureIc,
  nextBarReturn,
  pairValid,
  pearson,
  projectToRows,
  ranks,
  spearman,
  trimLeadingInvalid,
  type FlowCandle,
  type FundingBar,
  type StepPoint,
} from "./flow";

const HOUR = 3600_000;

/** A 30m candle grid starting at a round hour, with deterministic contents. */
function grid(n: number, startSec = 1_700_000_000, secs = 1800): FlowCandle[] {
  const out: FlowCandle[] = [];
  let px = 100;
  for (let i = 0; i < n; i++) {
    // A cheap deterministic wiggle; nothing here depends on its distribution.
    px *= 1 + 0.004 * Math.sin(i * 1.7) + 0.001 * Math.cos(i * 0.31);
    out.push({
      time: startSec + i * secs,
      open: px,
      high: px * 1.002,
      low: px * 0.998,
      close: px,
      volume: 1000 + 300 * Math.sin(i * 0.9),
      trades: Math.round(50 + 20 * Math.sin(i * 0.5) + (i % 7)),
    });
  }
  return out;
}

/** Hourly funding rows stamped a few ms past the hour, exactly like the API. */
function funding(fromSec: number, hours: number, skewMs = 26): FundingBar[] {
  const out: FundingBar[] = [];
  const firstHour = Math.ceil(fromSec / 3600) * 3600;
  for (let h = 0; h < hours; h++) {
    out.push({
      timeMs: (firstHour + h * 3600) * 1000 + skewMs,
      rate: 1e-5 * Math.sin(h * 0.4),
      premium: 1e-4 * Math.cos(h * 0.23),
    });
  }
  return out;
}

const CFG = { window: 5, useVolatility: true, useVolume: true };

// ---------------------------------------------------------------------------

describe("step alignment onto the candle grid", () => {
  test("a bar sees only prints stamped strictly before its close instant", () => {
    const candles = grid(4); // closes at +1800, +3600, +5400, +7200 seconds
    const t0 = candles[0].time;
    const pts: StepPoint[] = [
      { timeMs: (t0 + 1800) * 1000, value: 10 },       // exactly bar 0's close
      { timeMs: (t0 + 1800) * 1000 + 26, value: 20 },  // 26ms after it
      { timeMs: (t0 + 5400) * 1000 - 1, value: 30 },   // 1ms before bar 2's close
    ];
    const a = alignStep(pts, candles, { intervalSeconds: 1800, maxStaleMs: 10 * HOUR });

    // Bar 0 closes at exactly t0+1800; the print stamped at that instant is
    // NOT available to it. Nothing earlier exists, so bar 0 is invalid.
    expect(a.valid[0]).toBe(0);
    expect(Number.isNaN(a.value[0])).toBe(true);
    // Bar 1 sees both of the first two; the newest wins.
    expect(a.value[1]).toBe(20);
    expect(a.value[2]).toBe(30);
    // Bar 3 forward-fills the last one.
    expect(a.value[3]).toBe(30);
  });

  test("the aligned value never depends on a print that arrives later", () => {
    const candles = grid(20);
    const pts = funding(candles[0].time, 12).map((f) => ({ timeMs: f.timeMs, value: f.rate }));
    const early = alignStep(pts.slice(0, 4), candles, { intervalSeconds: 1800, maxStaleMs: 200 * HOUR });
    const late = alignStep(pts, candles, { intervalSeconds: 1800, maxStaleMs: 200 * HOUR });
    // Up to the bar where the 4th print is still the newest, the two agree.
    const cutoffMs = pts[4].timeMs;
    for (let i = 0; i < candles.length; i++) {
      if ((candles[i].time + 1800) * 1000 <= cutoffMs) {
        expect(late.valid[i]).toBe(early.valid[i]);
        if (early.valid[i]) expect(late.value[i]).toBe(early.value[i]);
      }
    }
  });

  test("forward-fill stops at the staleness cap and never fills backwards", () => {
    const candles = grid(12);
    const t0 = candles[0].time;
    const pts: StepPoint[] = [{ timeMs: (t0 + 1801) * 1000, value: 7 }];
    // Cap of one hour plus the bar: the print survives ~3 bars, then goes stale.
    const a = alignStep(pts, candles, { intervalSeconds: 1800, maxStaleMs: HOUR + 1800_000 });

    expect(a.valid[0]).toBe(0);            // before the print: no backward fill
    expect(Number.isNaN(a.value[0])).toBe(true);
    expect(a.value[2]).toBe(7);            // fresh
    const lastValid = [...a.valid].lastIndexOf(1);
    expect(lastValid).toBeGreaterThan(1);
    expect(lastValid).toBeLessThan(candles.length - 1);
    for (let i = lastValid + 1; i < candles.length; i++) {
      expect(a.valid[i]).toBe(0);
      expect(Number.isNaN(a.value[i])).toBe(true);
    }
  });

  test("a gap in the hourly series goes stale rather than being interpolated", () => {
    const candles = grid(24);
    const t0 = candles[0].time;
    const all = funding(t0, 12);
    // Remove hours 3..7 entirely.
    const holed = all.filter((_, i) => i < 3 || i >= 8).map((f) => ({ timeMs: f.timeMs, value: f.rate }));
    const a = alignStep(holed, candles, { intervalSeconds: 1800, maxStaleMs: 3 * HOUR + 1800_000 });
    const bad = [...a.valid].filter((v) => !v).length;
    expect(bad).toBeGreaterThan(0);
    // Every valid value must equal one of the observed points — no averages.
    const seen = new Set(holed.map((p) => p.value));
    for (let i = 0; i < candles.length; i++) if (a.valid[i]) expect(seen.has(a.value[i])).toBe(true);
  });

  test("no points at all yields no valid rows", () => {
    const candles = grid(6);
    const a = alignStep([], candles, { intervalSeconds: 1800, maxStaleMs: HOUR });
    expect([...a.valid].every((v) => v === 0)).toBe(true);
  });
});

// ---------------------------------------------------------------------------

describe("alignment against a hand-built grid", () => {
  test("each 30m bar is assigned the funding hour that closed before it", () => {
    // Bars open at 12:00, 12:30, 13:00, 13:30 UTC. Funding stamped on the hour
    // plus 26ms, matching the real API.
    const t12 = 1_700_000_000 - (1_700_000_000 % 3600); // a round hour
    const candles = grid(4, t12, 1800);
    const rows: FundingBar[] = [
      { timeMs: t12 * 1000 + 26, rate: 1e-4, premium: 5e-4 },          // 12:00
      { timeMs: (t12 + 3600) * 1000 + 26, rate: 2e-4, premium: 6e-4 }, // 13:00
    ];
    const cols = buildFlowColumns(candles, rows, "30m", {
      window: 2, fundingWindowHours: 1, carryHours: 1, maxStaleHours: 3,
    });
    const f = cols.cols[cols.names.indexOf("fundingBps")];
    const fv = cols.colValid[cols.names.indexOf("fundingBps")];

    // Bar 0 [12:00,12:30) closes at 12:30 -> sees the 12:00 print.
    expect(fv[0]).toBe(1);
    expect(f[0]).toBeCloseTo(1, 10);
    // Bar 1 [12:30,13:00) closes at exactly 13:00:00.000, before the 13:00
    // print's 13:00:00.026 stamp -> still the 12:00 rate.
    expect(f[1]).toBeCloseTo(1, 10);
    // Bar 2 [13:00,13:30) closes at 13:30 -> the 13:00 print.
    expect(f[2]).toBeCloseTo(2, 10);
    expect(f[3]).toBeCloseTo(2, 10);

    const p = cols.cols[cols.names.indexOf("premiumBps")];
    expect(p[0]).toBeCloseTo(5, 10);
    expect(p[3]).toBeCloseTo(6, 10);
  });

  test("trade-count surge matches a hand-computed log ratio", () => {
    const candles = grid(6);
    candles[0].trades = 10; candles[1].trades = 10; candles[2].trades = 10;
    candles[3].trades = 40; candles[4].trades = 10; candles[5].trades = 10;
    const cols = buildFlowColumns(candles, [], "30m", { window: 3, useFunding: false, usePremium: false });
    const d = cols.names.indexOf("tradeCountSurge");
    // Bar 3: baseline over bars 1..3 = (10+10+40)/3 = 20, so log(40/20) = ln 2.
    expect(cols.cols[d][3]).toBeCloseTo(Math.log(2), 6);
    // Bar 2: baseline = 10, so log(10/10) = 0.
    expect(cols.cols[d][2]).toBeCloseTo(0, 6);
    // Rows before the window fills are invalid, not zero.
    expect(cols.colValid[d][0]).toBe(0);
    expect(cols.colValid[d][1]).toBe(0);
    expect(cols.colValid[d][2]).toBe(1);
  });

  test("projection onto FeatureSet rows preserves the candle mapping", () => {
    const candles = grid(40);
    const fs = buildFeatures(candles, CFG);
    const raw = buildFlowColumns(candles, funding(candles[0].time, 24), "30m", { window: CFG.window });
    const proj = projectToRows(raw, fs.index);

    expect(proj.cols[0].length).toBe(fs.T);
    for (let r = 0; r < fs.T; r++) {
      const i = fs.index[r];
      for (let c = 0; c < raw.cols.length; c++) {
        if (raw.colValid[c][i]) expect(proj.cols[c][r]).toBe(raw.cols[c][i]);
        expect(proj.colValid[c][r]).toBe(raw.colValid[c][i]);
      }
    }
    // Row r must correspond to candle r + window, the same contract
    // buildFeatures publishes.
    for (let r = 0; r < fs.T; r++) expect(fs.index[r]).toBe(r + CFG.window);
  });
});

// ---------------------------------------------------------------------------

describe("causality: future bars cannot change a past feature row", () => {
  test("appending candles and funding leaves every earlier flow row identical", () => {
    const long = grid(160);
    const longFunding = funding(long[0].time, 90);

    const shortCandles = long.slice(0, 100);
    const cutoffMs = (shortCandles[shortCandles.length - 1].time + 1800) * 1000;
    const shortFunding = longFunding.filter((f) => f.timeMs < cutoffMs);

    const fsLong = buildFeatures(long, CFG);
    const fsShort = buildFeatures(shortCandles, CFG);
    const cLong = flowFeatures(fsLong, long, longFunding, "30m", { window: CFG.window });
    const cShort = flowFeatures(fsShort, shortCandles, shortFunding, "30m", { window: CFG.window });

    expect(cShort.names).toEqual(cLong.names);
    for (let c = 0; c < cLong.names.length; c++) {
      for (let r = 0; r < fsShort.T; r++) {
        expect(cShort.colValid[c][r]).toBe(cLong.colValid[c][r]);
        if (cShort.colValid[c][r]) {
          expect(cShort.cols[c][r]).toBeCloseTo(cLong.cols[c][r], 12);
        }
      }
    }
  });

  test("corrupting future candles and future funding does not move earlier rows", () => {
    const candles = grid(160);
    const f = funding(candles[0].time, 90);
    const clean = buildFlowColumns(candles, f, "30m", { window: 5 });

    const cutIdx = 100;
    const cutMs = (candles[cutIdx].time + 1800) * 1000;
    const wrecked = candles.map((c, i) =>
      i >= cutIdx
        ? { ...c, close: c.close * 9, volume: c.volume * 41, trades: c.trades * 17 }
        : c);
    const wreckedFunding = f.map((p) => (p.timeMs >= cutMs ? { ...p, rate: p.rate * 55, premium: p.premium * -13 } : p));
    const dirty = buildFlowColumns(wrecked, wreckedFunding, "30m", { window: 5 });

    for (let c = 0; c < clean.names.length; c++) {
      for (let i = 0; i < cutIdx; i++) {
        expect(dirty.colValid[c][i]).toBe(clean.colValid[c][i]);
        if (clean.colValid[c][i]) expect(dirty.cols[c][i]).toBeCloseTo(clean.cols[c][i], 12);
      }
    }
  });

  test("the next-bar target is the only thing here that looks forward", () => {
    const candles = grid(60);
    const fs = buildFeatures(candles, CFG);
    const t = nextBarReturn(fs);
    for (let r = 0; r + 1 < fs.T; r++) expect(t.value[r]).toBe(fs.rawReturn[r + 1]);
    expect(t.valid[fs.T - 1]).toBe(0); // the last row has no next bar
  });

  test("a gap in the candle index invalidates the next-bar target there", () => {
    const candles = grid(40);
    // Delete one candle to open a hole in the series.
    const holed = [...candles.slice(0, 20), ...candles.slice(21)];
    const fs = buildFeatures(holed, CFG);
    const t = nextBarReturn(fs, { candles: holed, intervalSeconds: 1800 });
    // Every valid target row must sit next to a genuinely adjacent candle.
    for (let r = 0; r + 1 < fs.T; r++) {
      const adjacent = holed[r + 1 + CFG.window].time - holed[r + CFG.window].time === 1800;
      expect(!!t.valid[r]).toBe(adjacent);
    }
  });
});

// ---------------------------------------------------------------------------

describe("augmentFeatures", () => {
  const candles = grid(60);
  const fs = buildFeatures(candles, CFG);

  test("appends columns and leaves the original untouched", () => {
    const before = { X: Float64Array.from(fs.X), D: fs.D, names: [...fs.names] };
    const a = new Float64Array(fs.T).fill(1.5);
    const b = new Float64Array(fs.T).map((_, i) => i);
    const out = augmentFeatures(fs, { names: ["alpha", "beta"], cols: [a, b] });

    expect(out.D).toBe(fs.D + 2);
    expect(out.T).toBe(fs.T);
    expect(out.names).toEqual([...before.names, "alpha", "beta"]);

    // Original is byte-for-byte what it was.
    expect(fs.D).toBe(before.D);
    expect(fs.names).toEqual(before.names);
    for (let i = 0; i < fs.X.length; i++) expect(fs.X[i]).toBe(before.X[i]);

    // Every original value is preserved at its new stride, and the new columns
    // land in order after them.
    for (let r = 0; r < fs.T; r++) {
      for (let d = 0; d < fs.D; d++) expect(out.X[r * out.D + d]).toBe(fs.X[r * fs.D + d]);
      expect(out.X[r * out.D + fs.D]).toBe(1.5);
      expect(out.X[r * out.D + fs.D + 1]).toBe(r);
    }
  });

  test("mutating the result cannot reach back into the input", () => {
    const out = augmentFeatures(fs, { names: ["z"], cols: [new Float64Array(fs.T)] });
    out.index[0] = -999;
    out.rawReturn[0] = -999;
    expect(fs.index[0]).toBe(CFG.window);
    expect(fs.rawReturn[0]).not.toBe(-999);
  });

  test("refuses NaN, wrong lengths and duplicate names", () => {
    const bad = new Float64Array(fs.T).fill(1);
    bad[3] = NaN;
    expect(() => augmentFeatures(fs, { names: ["x"], cols: [bad] })).toThrow(/not finite at row 3/);
    expect(() => augmentFeatures(fs, { names: ["x"], cols: [new Float64Array(fs.T - 1)] })).toThrow(/rows, expected/);
    expect(() => augmentFeatures(fs, { names: ["logReturn"], cols: [new Float64Array(fs.T)] })).toThrow(/duplicate/);
    expect(() => augmentFeatures(fs, { names: ["a", "b"], cols: [new Float64Array(fs.T)] })).toThrow(/names but/);
  });

  test("end to end: flow columns append cleanly once the leading run is trimmed", () => {
    const f = funding(candles[0].time, 40);
    const cols = flowFeatures(fs, candles, f, "30m", { window: CFG.window, fundingWindowHours: 6, carryHours: 3 });
    // The z-scores need a warm-up, so early rows are invalid by construction.
    expect(firstValidRow(cols.valid)).toBeGreaterThan(0);
    const trimmed = trimLeadingInvalid(fs, cols);
    const aug = augmentFeatures(trimmed.fs, trimmed.cols);
    expect(aug.D).toBe(fs.D + cols.names.length);
    expect(aug.T).toBe(fs.T - trimmed.dropped);
    expect([...aug.X].every(Number.isFinite)).toBe(true);
    // Trimming keeps the rows contiguous — a Markov chain has no way to see a hole.
    for (let r = 1; r < aug.T; r++) expect(aug.index[r]).toBe(aug.index[r - 1] + 1);
  });

  test("trimming refuses a hole that is not at the front", () => {
    const cols = flowFeatures(fs, candles, funding(candles[0].time, 40), "30m", { window: CFG.window, fundingWindowHours: 2, carryHours: 2 });
    const start = firstValidRow(cols.valid);
    cols.valid[start + 3] = 0;
    expect(() => trimLeadingInvalid(fs, cols)).toThrow(/hole/);
  });
});

// ---------------------------------------------------------------------------

describe("the fundingHistory paging trap", () => {
  /**
   * A stand-in for the real endpoint's documented-nowhere behaviour: it serves
   * the FIRST 500 hourly rows at or after startTime and stops, honouring
   * endTime as an upper bound. Verified against the live API on SOL:
   * startTime = now-45d with no endTime returned 2026-07-28..2026-08-17.
   */
  function fakeServer(firstHourMs: number, hours: number) {
    let calls = 0;
    const all: number[] = [];
    for (let h = 0; h < hours; h++) all.push(firstHourMs + h * HOUR + 26);
    const transport = async (body: any) => {
      calls++;
      if (body.type !== "fundingHistory") throw new Error("unexpected call");
      const start = Number(body.startTime);
      const end = body.endTime === undefined ? Infinity : Number(body.endTime);
      const rows = all.filter((t) => t >= start && t <= end).slice(0, FUNDING_PAGE_HOURS);
      return rows.map((t) => ({ coin: body.coin, time: t, fundingRate: String((t / HOUR) % 7 * 1e-6), premium: String(-1e-5) }));
    };
    return { transport, calls: () => calls, all };
  }

  const now = 1_800_000_000_000;
  const firstHour = now - 200 * 24 * HOUR;

  test("one naive request for 45 days back returns the OLDEST 500 hours", () => {
    const s = fakeServer(firstHour, 200 * 24);
    const start = now - 45 * 24 * HOUR;
    // What the trap looks like: ask once, get days 45..24 and nothing since.
    return s.transport({ type: "fundingHistory", coin: "SOL", startTime: start }).then((rows: any[]) => {
      expect(rows.length).toBe(FUNDING_PAGE_HOURS);
      expect(rows[rows.length - 1].time).toBeLessThan(now - 20 * 24 * HOUR);
    });
  });

  test("chunked fetching covers the window right up to the present", async () => {
    const s = fakeServer(firstHour, 200 * 24);
    const start = now - 45 * 24 * HOUR;
    const got = await fetchFundingRange("SOL", start, now, { transport: s.transport });

    expect(got.length).toBe(45 * 24);
    expect(got[0].timeMs).toBeGreaterThanOrEqual(start);
    // The newest row is within an hour of "now", which is exactly what the
    // naive single request fails to deliver.
    expect(now - got[got.length - 1].timeMs).toBeLessThan(HOUR);
    // Strictly increasing, no duplicates across the chunk seams.
    for (let i = 1; i < got.length; i++) expect(got[i].timeMs).toBeGreaterThan(got[i - 1].timeMs);
    // 45 days is 1080 hours, so three requests at 480 hours each.
    expect(s.calls()).toBe(3);
  });

  test("chunks are under the page limit and tile the range exactly", () => {
    const chunks = fundingChunks(0, 1080 * HOUR);
    expect(chunks.length).toBe(3);
    for (const c of chunks) expect((c.endTime - c.startTime) / HOUR).toBeLessThan(FUNDING_PAGE_HOURS);
    expect(chunks[0].startTime).toBe(0);
    expect(chunks[chunks.length - 1].endTime).toBe(1080 * HOUR);
    for (let i = 1; i < chunks.length; i++) expect(chunks[i].startTime).toBe(chunks[i - 1].endTime);
    expect(fundingChunks(5, 5)).toEqual([]);
    expect(fundingChunks(10, 5)).toEqual([]);
  });

  test("a short history is not padded and does not loop forever", async () => {
    const s = fakeServer(now - 10 * HOUR, 10);
    const got = await fetchFundingRange("SOL", now - 45 * 24 * HOUR, now, { transport: s.transport });
    expect(got.length).toBe(10);
  });
});

// ---------------------------------------------------------------------------

describe("statistics", () => {
  test("pearson matches a hand case and is scale invariant", () => {
    const x = [1, 2, 3, 4, 5];
    const y = [2, 4, 6, 8, 10];
    expect(pearson(x, y)).toBeCloseTo(1, 12);
    expect(pearson(x, y.map((v) => -v))).toBeCloseTo(-1, 12);
    expect(pearson(x, [1, 1, 1, 1, 1])).toBeNaN();
  });

  test("spearman sees a monotone relation that pearson understates", () => {
    const x = [1, 2, 3, 4, 5, 6];
    const y = x.map((v) => Math.exp(v));
    expect(spearman(x, y)).toBeCloseTo(1, 12);
    expect(pearson(x, y)).toBeLessThan(0.95);
  });

  test("ranks average over ties", () => {
    expect([...ranks([10, 20, 20, 30])]).toEqual([1, 2.5, 2.5, 4]);
    expect([...ranks([5, 5, 5])]).toEqual([2, 2, 2]);
  });

  test("icStats reports n, and t scales with it", () => {
    const n = 400;
    const x = new Float64Array(n).map((_, i) => Math.sin(i));
    const y = new Float64Array(n).map((_, i) => Math.sin(i) * 0.3 + Math.cos(i * 3.1) * 0.9);
    const r = icStats(x, y);
    expect(r.n).toBe(n);
    expect(Math.abs(r.t)).toBeGreaterThan(Math.abs(r.ic));
  });

  test("pairValid keeps only rows where both sides are usable", () => {
    const x = Float64Array.from([1, 2, NaN, 4]);
    const xv = Uint8Array.from([1, 1, 1, 1]);
    const y = Float64Array.from([1, 2, 3, 4]);
    const yv = Uint8Array.from([1, 0, 1, 1]);
    const p = pairValid(x, xv, y, yv);
    expect([...p.x]).toEqual([1, 4]);
    expect([...p.y]).toEqual([1, 4]);
  });

  test("measureIc reports the flow columns and any named price feature", () => {
    const candles = grid(300);
    const fs = buildFeatures(candles, CFG);
    const cols = flowFeatures(fs, candles, funding(candles[0].time, 200), "30m", {
      window: CFG.window, fundingWindowHours: 12, carryHours: 6,
    });
    const rows = measureIc(fs, cols, ["logReturn"]);
    expect(rows[0].name).toBe("logReturn");
    expect(rows.map((r) => r.name)).toEqual(["logReturn", ...cols.names]);
    for (const r of rows) {
      expect(r.full.n).toBeGreaterThan(0);
      expect(r.firstHalf.n + r.secondHalf.n).toBeLessThanOrEqual(r.full.n);
    }
  });
});
