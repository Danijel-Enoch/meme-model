/**
 * Render one frame of the TUI to stdout with synthetic data.
 *
 * Useful on its own (`bun run src/tui/preview.ts`) and the same path the
 * snapshot test drives — the headless renderer needs no terminal, so what you
 * see here is exactly what the tests assert on.
 */
import { createTestRenderer } from "@opentui/core/testing";
import { createApp } from "./app";
import { initialState, type AppState, type CoinRow } from "./model";
import type { Candle } from "../features";

export function demoState(): AppState {
  const coins = ["BTC", "ETH", "HYPE", "ZEC", "SOL", "PUMP", "PONS", "XRP", "NEAR", "LIT", "DOGE", "AVAX"];
  const rows: CoinRow[] = coins.map((c, i) => ({
    coin: c, timeframe: ["15m", "30m", "1h", "2h"][i % 4], modelType: i % 3 === 0 ? "hsmm" : "hmm",
    volume24h: 1e9 / (i + 1), markPrice: 100 * (i + 1),
    excessRoi: i === 3 ? null : Math.sin(i) * 0.08,
    roi: 0.12, buyHold: 0.2, exposure: 0.4, trades: 12,
    pValue: i === 3 ? null : 0.2 + 0.05 * i,
  }));

  const candles: Candle[] = [];
  let p = 100;
  for (let i = 0; i < 400; i++) {
    const drift = i > 250 && i < 300 ? 0.004 : -0.0004;
    p *= 1 + drift + 0.004 * Math.sin(i / 7);
    candles.push({
      time: 1750000000 + i * 1800,
      open: p * (1 - 0.001), high: p * 1.002, low: p * 0.998, close: p, volume: 1000 + i,
    });
  }
  return {
    ...initialState(rows),
    cursor: 4,
    chart: {
      coin: "SOL", timeframe: "30m", candles, K: 3,
      states: candles.map((_, i) => (i > 250 && i < 300 ? 2 : i % 40 < 20 ? 0 : 1)),
      positions: candles.map((_, i) => (i > 255 && i < 305 ? 1 : 0)),
    },
    fit: {
      coin: "SOL", timeframe: "30m", modelType: "hsmm", bars: 2875,
      logLikPerBar: -3.4498, converged: true,
      states: [
        { label: "chop", meanRetBps: -0.5, volPct: 0.23, freq: 0.324, durationBars: 8.4 },
        { label: "chop", meanRetBps: 1.0, volPct: 0.10, freq: 0.303, durationBars: 7.9 },
        { label: "drift-up", meanRetBps: 3.4, volPct: 0.51, freq: 0.372, durationBars: 9.7 },
      ],
      transitions: [0, 0.58, 0.42, 0.87, 0, 0.13, 0.9, 0.1, 0],
      durationModes: [
        [{ state: 0, d: 5, p: 0.16 }, { state: 0, d: 6, p: 0.11 }, { state: 0, d: 1, p: 0.1 }],
        [{ state: 1, d: 2, p: 0.19 }, { state: 1, d: 3, p: 0.13 }, { state: 1, d: 11, p: 0.12 }],
        [{ state: 2, d: 5, p: 0.52 }, { state: 2, d: 9, p: 0.12 }],
      ],
    },
    message: "fitted in 1.8s",
  };
}

export async function renderFrame(state: AppState, width = 120, height = 34): Promise<string> {
  const { renderer, renderOnce, captureCharFrame } = await createTestRenderer({ width, height });
  const app = createApp(renderer, state);
  await renderOnce();
  // A second pass: the first lays the boxes out, and only then do the chart
  // functions know how many columns they have to draw into.
  app.render();
  await renderOnce();
  const frame = captureCharFrame();
  app.destroy();
  renderer.destroy();
  return frame;
}

if (import.meta.main) {
  console.log(await renderFrame(demoState()));
}
