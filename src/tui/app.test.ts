import { expect, test, describe } from "bun:test";
import { createTestRenderer } from "@opentui/core/testing";
import { createApp } from "./app";
import { demoState, renderFrame } from "./preview";
import {
  initialState, keyToAction, reduce, TABS, TIMEFRAMES, currentRow,
  type AppState, type CoinRow,
} from "./model";

function rows(n = 3): CoinRow[] {
  return Array.from({ length: n }, (_, i) => ({
    coin: ["BTC", "ETH", "SOL"][i % 3] + (i > 2 ? i : ""),
    timeframe: "30m", modelType: "hmm" as const,
    volume24h: 1e9, markPrice: 100,
    excessRoi: 0.05, roi: 0.1, buyHold: 0.2, exposure: 0.5, trades: 9, pValue: 0.4,
  }));
}

describe("view model", () => {
  test("the cursor wraps and clears everything derived from the old coin", () => {
    let s: AppState = { ...initialState(rows()), fit: {} as any, backtest: {} as any, chart: {} as any };
    s = reduce(s, { type: "cursor", delta: 1 });
    expect(s.cursor).toBe(1);
    // Showing BTC's fit under ETH's name is the quiet lie this guards against.
    expect(s.fit).toBeNull();
    expect(s.backtest).toBeNull();
    expect(s.chart).toBeNull();

    s = reduce(s, { type: "cursor", delta: -2 });
    expect(s.cursor).toBe(2);
  });

  test("paper results survive moving the cursor", () => {
    // The paper account is not per-coin — it is the account. Clearing it on a
    // cursor move would look like the position vanished.
    let s: AppState = { ...initialState(rows()), paper: { equity: 1000 } as any };
    s = reduce(s, { type: "cursor", delta: 1 });
    expect(s.paper).not.toBeNull();
  });

  test("timeframe and model type cycle on the selected row only", () => {
    let s = initialState(rows());
    s = reduce(s, { type: "cursor", delta: 1 });
    s = reduce(s, { type: "timeframe", delta: 1 });
    const i = TIMEFRAMES.indexOf("30m");
    expect(currentRow(s)!.timeframe).toBe(TIMEFRAMES[(i + 1) % TIMEFRAMES.length]);
    expect(s.rows[0].timeframe).toBe("30m");

    s = reduce(s, { type: "modelType" });
    expect(currentRow(s)!.modelType).toBe("hsmm");
    s = reduce(s, { type: "modelType" });
    expect(currentRow(s)!.modelType).toBe("hmm");
  });

  test("tabs cycle in both directions and jump by number", () => {
    let s = initialState(rows());
    expect(s.tab).toBe("fit");
    s = reduce(s, { type: "tab", delta: -1 });
    expect(s.tab).toBe(TABS[TABS.length - 1]);
    s = reduce(s, { type: "tabTo", tab: "paper" });
    expect(s.tab).toBe("paper");
  });

  test("an empty universe never produces an out-of-range cursor", () => {
    let s = initialState([]);
    s = reduce(s, { type: "cursor", delta: 1 });
    expect(s.cursor).toBe(0);
    expect(currentRow(s)).toBeNull();
    s = reduce(s, { type: "rows", rows: rows(2) });
    s = reduce(s, { type: "cursor", delta: 5 });
    expect(s.cursor).toBeLessThan(2);
  });

  test("key map covers the documented keys and claims nothing else", () => {
    expect(keyToAction("j")).toEqual({ type: "cursor", delta: 1 });
    expect(keyToAction("tab", true)).toEqual({ type: "tab", delta: -1 });
    expect(keyToAction("f")).toBe("fit");
    expect(keyToAction("q")).toBe("quit");
    // Unclaimed keys must fall through so a focused widget can have them.
    expect(keyToAction("z")).toBeNull();
    expect(keyToAction("F1")).toBeNull();
  });
});

describe("rendering", () => {
  test("a full frame draws every pane and stays inside the terminal", async () => {
    const frame = await renderFrame(demoState(), 120, 34);
    const lines = frame.split("\n").filter((l) => l.length > 0);
    expect(lines.length).toBeLessThanOrEqual(34);
    for (const l of lines) expect([...l].length).toBeLessThanOrEqual(120);

    expect(frame).toContain("universe");
    expect(frame).toContain("SOL 30m");
    expect(frame).toContain("drift-up");
    expect(frame).toContain("regime");
    expect(frame).toContain("q quit");
  });

  test("an 80x24 terminal still renders without overflowing", async () => {
    // The floor this UI has to survive: the default size of a non-TTY and of
    // most SSH sessions nobody has resized.
    const frame = await renderFrame(demoState(), 80, 24);
    const lines = frame.split("\n").filter((l) => l.length > 0);
    expect(lines.length).toBeLessThanOrEqual(24);
    for (const l of lines) expect([...l].length).toBeLessThanOrEqual(80);
    expect(frame).toContain("universe");
  });

  test("with no data the chart says what to press rather than drawing nothing", async () => {
    const frame = await renderFrame(initialState(rows()), 100, 28);
    expect(frame).toContain("no data");
    expect(frame).toContain("press f to fit");
  });

  test("keys drive the app through the same path the terminal uses", async () => {
    const { renderer, renderOnce } = await createTestRenderer({ width: 100, height: 30 });
    let quit = 0, fits = 0;
    const app = createApp(renderer, initialState(rows()), {
      fit: () => fits++, backtest() {}, replay() {}, paperToggle() {}, sweep() {}, quit: () => quit++,
    });
    app.handleKey("down");
    expect(app.state().cursor).toBe(1);
    app.handleKey("3");
    expect(app.state().tab).toBe("paper");
    app.handleKey("f");
    app.handleKey("q");
    expect(fits).toBe(1);
    expect(quit).toBe(1);
    await renderOnce();
    app.destroy();
    renderer.destroy();
  });

  test("the help overlay replaces the tab body", async () => {
    const s = reduce(demoState(), { type: "help" });
    const frame = await renderFrame(s, 110, 30);
    expect(frame).toContain("is skill");
    expect(frame).not.toContain("log-lik/bar");
  });
});

describe("layout discipline", () => {
  test("a long universe list never paints outside its panel", async () => {
    // The bug this pins: Yoga does not clip children, so 30 rows in a box that
    // fits 12 used to spill past the footer and over the tab pane.
    const many = Array.from({ length: 30 }, (_, i) => ({
      coin: `C${i}`, timeframe: "30m", modelType: "hmm" as const,
      volume24h: 1e6, markPrice: 1, excessRoi: 0.01, roi: 0, buyHold: 0,
      exposure: 0.5, trades: 5, pValue: 0.5,
    }));
    const frame = await renderFrame({ ...demoState(), rows: many, cursor: 0 }, 100, 26);
    const lines = frame.split("\n");
    // Every row of the universe column must sit inside the panel's borders.
    const panel = lines.filter((l) => l.includes("│"));
    expect(panel.length).toBeGreaterThan(5);
    const footer = lines.findIndex((l) => l.includes("q quit"));
    expect(footer).toBeGreaterThan(0);
    for (const l of lines.slice(footer + 1)) expect(l.trim()).toBe("");
    // And the last coin in the list cannot appear, because it does not fit.
    expect(frame).not.toContain("C29");
  });

  test("the cursor stays visible when it moves past the fold", async () => {
    const many = Array.from({ length: 30 }, (_, i) => ({
      coin: `C${i}`, timeframe: "30m", modelType: "hmm" as const,
      volume24h: 1e6, markPrice: 1, excessRoi: 0.01, roi: 0, buyHold: 0,
      exposure: 0.5, trades: 5, pValue: 0.5,
    }));
    const frame = await renderFrame({ ...demoState(), rows: many, cursor: 25 }, 100, 26);
    expect(frame).toContain("C25");
  });
});
