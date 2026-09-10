/**
 * The forward loop: poll for bars that have actually closed, and trade them.
 *
 * The one rule this file exists to enforce is that the bar Hyperliquid returns
 * last is the bar still forming. Its close is whatever the price happens to be
 * this second and it will change before the bar ends, so signalling on it means
 * acting on information that does not exist yet — the live-trading cousin of
 * the lookahead the backtest is so careful to avoid. Everything here works off
 * closed bars only, and the forming bar is used for nothing but the mark price
 * shown on screen.
 */

import type { Candle } from "../features";
import { signalNow, type LiveSignal, type StrategyConfig } from "../backtest";
import { onBar, markToMarket, type PaperState } from "../paper";
import * as hl from "../hyperliquid";
import type { RuntimeModel } from "./jobs";

/** Bars whose interval has fully elapsed. The rest is the forming bar. */
export function closedCandles(candles: Candle[], intervalSecs: number, nowMs = Date.now()): Candle[] {
  const cutoff = Math.floor(nowMs / 1000);
  return candles.filter((c) => c.time + intervalSecs <= cutoff);
}

/** When the bar currently forming will close, in ms — for the countdown. */
export function nextCloseMs(lastClosed: Candle | undefined, intervalSecs: number, nowMs = Date.now()): number | null {
  if (!lastClosed) return null;
  const nextClose = (lastClosed.time + 2 * intervalSecs) * 1000;
  return nextClose > nowMs ? nextClose : nowMs;
}

export interface PollResult {
  /** Closed bars newly applied to the account this poll — usually 0 or 1. */
  applied: number;
  /** The most recent closed bar, whether or not it was new. */
  lastClosed?: Candle;
  /** Mark price from the forming bar, for display only. */
  mark?: number;
  signal?: LiveSignal;
  nextCloseMs: number | null;
}

export interface PollOptions {
  coin: string;
  timeframe: string;
  model: RuntimeModel;
  strategy: StrategyConfig;
  /** How many candles to pull. Needs to cover the model's feature lookback with
   *  room for the filter to warm up, not the whole training window. */
  lookback?: number;
  now?: number;
  /** Injectable for tests; defaults to the real API. */
  fetch?: (coin: string, timeframe: string, bars: number) => Promise<Candle[]>;
}

/**
 * One poll. Applies every closed bar the account has not seen yet, in order —
 * if the terminal was asleep for an hour, the account catches up bar by bar
 * rather than jumping straight to the present and pretending it held through.
 */
export async function pollOnce(state: PaperState, opts: PollOptions): Promise<PollResult> {
  const secs = hl.intervalSeconds(opts.timeframe);
  const lookback = opts.lookback ?? 600;
  const nowMs = opts.now ?? Date.now();
  const fetcher = opts.fetch ?? (async (coin, tf, bars) => (await hl.fetchCandles(coin, tf, bars)).candles);

  const all = await fetcher(opts.coin, opts.timeframe, lookback);
  const closed = closedCandles(all, secs, nowMs);
  const forming = all.length > closed.length ? all[all.length - 1] : undefined;
  const last = closed[closed.length - 1];
  const result: PollResult = {
    applied: 0,
    lastClosed: last,
    mark: forming?.close ?? last?.close,
    nextCloseMs: nextCloseMs(last, secs, nowMs),
  };
  if (!last) return result;

  const featureConfig = {
    window: opts.model.window,
    useVolatility: opts.model.names.includes("realizedVol"),
    useVolume: opts.model.names.includes("volumeSurge"),
  };
  const seen = state.lastBar[opts.coin] ?? 0;
  const fresh = closed.filter((c) => c.time > seen);

  for (const bar of fresh) {
    const upTo = closed.slice(0, closed.findIndex((c) => c.time === bar.time) + 1);
    if (upTo.length < featureConfig.window + 3) continue; // not enough history to signal
    const prev = state.positions[opts.coin]?.position ?? 0;
    const sig = signalNow(
      opts.model.params as never, opts.model.modelType, opts.model.scaler, opts.model.names,
      upTo, featureConfig, opts.strategy, prev,
    );
    onBar(state, opts.coin, bar, sig.target, "live");
    result.applied++;
    result.signal = sig;
  }

  // Mark the open position to the forming bar so the screen ticks between
  // closes. This moves equity for display; it never trades.
  if (result.mark !== undefined) {
    markToMarket(state, { [opts.coin]: result.mark }, Math.floor(nowMs / 1000));
  }
  return result;
}
