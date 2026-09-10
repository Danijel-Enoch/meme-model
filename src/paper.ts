/**
 * Paper trading engine.
 *
 * The point of paper trading a regime model is to find out whether the backtest
 * was telling the truth, so the only thing that matters here is that the
 * accounting is the SAME accounting. Every rule below mirrors `walkForward`'s
 * PnL loop or `simulatePerp`'s account model, and `paper.test.ts` pins the
 * equivalence to 1e-9 rather than trusting the resemblance.
 *
 * Four rules carry all of it:
 *
 *   Decide on bar i, realize on bar i+1. `onBar` marks the position carried
 *   INTO this bar before it looks at the new target, so a signal can never earn
 *   the return of the bar that produced it. That is the failure mode this repo
 *   cares most about and it is the first thing the tests check.
 *
 *   Fees land on notional traded — |dPosition| x allotted capital x leverage.
 *   A flip from +1 to -1 moves two units and pays for both, because it is two
 *   orders.
 *
 *   Liquidation is path-dependent. The worst tick inside the bar decides it,
 *   so the low is used for longs and the high for shorts, and the maintenance
 *   fraction is Hyperliquid's own rather than a number invented here.
 *
 *   Idempotence. A live loop polls the exchange on a timer and will re-serve
 *   the same closed bar several times before the next one exists. Applying a
 *   bar twice would double its return and its fees, so the bar timestamp is the
 *   key and anything at or before the last one applied is dropped.
 */

import { signalNow, type ModelType, type StrategyConfig } from "./backtest";
import { buildFeatures, type Candle, type FeatureConfig, type Scaler } from "./features";
import type { HmmParams } from "./hmm";
import type { HsmmParams } from "./hsmm";
import { maintenanceMarginFraction } from "./hyperliquid";

export interface PaperFill {
  time: number; coin: string; side: "buy" | "sell";
  price: number; sizeUsd: number; feeUsd: number;
  /** "entry" | "exit" | "flip" | "liquidation" — why the fill happened, with
   *  the caller's note appended after a space if one was given. "exit" means the
   *  coin is flat afterwards; a same-sign resize is an "entry". */
  reason: string;
}

export interface PaperPosition {
  coin: string;
  /** Signed fraction of allotted capital, -1..1, matching backtest positions. */
  position: number;
  entryPrice: number; entryTime: number; notionalUsd: number;
  barsHeld: number; unrealizedUsd: number;
}

export interface PaperConfig {
  startingEquity: number;   // default 1000
  leverage: number;         // default 1
  costBps: number;          // default 4.5 per side, charged on notional traded
  fundingPerHour?: number;  // optional, applied to open notional
  /**
   * How many coins the equity is split across: each gets equity / maxCoins to
   * size against. Defaults to 1, so a single-coin run is sized off the whole
   * account and reproduces walkForward exactly. Set it to the number of coins
   * the driver intends to trade BEFORE the first bar — leaving it at 1 while
   * feeding five coins allots the full account to each of them.
   */
  maxCoins?: number;
  /**
   * The VENUE's max leverage for the coin, not the account's. Hyperliquid sets
   * the maintenance margin at half the initial margin required at that figure,
   * so this is the input `maintenanceMarginFraction` wants. 10 is the CLI's own
   * fallback when the market list has not been fetched.
   */
  maxLeverage?: number;
}

export interface PaperState {
  id: string; createdAt: number; updatedAt: number;
  config: PaperConfig;
  equity: number; cash: number; realizedUsd: number; feesUsd: number; fundingUsd: number;
  positions: Record<string, PaperPosition>;
  fills: PaperFill[];
  equityCurve: { time: number; equity: number; buyHold: number }[];
  /** Last bar timestamp applied per coin — the idempotency key. */
  lastBar: Record<string, number>;
  liquidated: boolean;
  /** Close of the last bar applied per coin. The mark the next bar's return and
   *  the next liquidation check are measured from. */
  lastPrice: Record<string, number>;
  /** Per-coin buy-and-hold sleeve in dollars: the coin's allotment, bought at
   *  the first bar and never touched. Unleveraged on purpose — "did it beat
   *  doing nothing" is a question about the asset, not about the margin. */
  buyHold: Record<string, number>;
}

/** All timestamps in a PaperState are unix SECONDS, matching Candle.time. */
const now = () => Math.floor(Date.now() / 1000);

const DEFAULTS: Required<Omit<PaperConfig, "fundingPerHour">> & { fundingPerHour: number } = {
  startingEquity: 1000,
  leverage: 1,
  costBps: 4.5,
  fundingPerHour: 0,
  maxCoins: 1,
  maxLeverage: 10,
};

function resolved(c: PaperConfig) {
  return {
    startingEquity: c.startingEquity ?? DEFAULTS.startingEquity,
    leverage: c.leverage ?? DEFAULTS.leverage,
    costBps: c.costBps ?? DEFAULTS.costBps,
    fundingPerHour: c.fundingPerHour ?? DEFAULTS.fundingPerHour,
    maxCoins: Math.max(c.maxCoins ?? DEFAULTS.maxCoins, 1),
    maxLeverage: c.maxLeverage ?? DEFAULTS.maxLeverage,
  };
}

export function newPaperState(config: Partial<PaperConfig> = {}, id?: string): PaperState {
  const cfg: PaperConfig = { ...DEFAULTS, ...config };
  const t = now();
  return {
    id: id ?? `paper-${t.toString(36)}-${Math.random().toString(36).slice(2, 8)}`,
    createdAt: t,
    updatedAt: t,
    config: cfg,
    equity: cfg.startingEquity,
    cash: cfg.startingEquity,
    realizedUsd: 0,
    feesUsd: 0,
    fundingUsd: 0,
    positions: {},
    fills: [],
    equityCurve: [],
    lastBar: {},
    liquidated: false,
    lastPrice: {},
    buyHold: {},
  };
}

/** Sum of the buy-and-hold sleeves plus the capital never allotted to a coin. */
function buyHoldTotal(state: PaperState): number {
  const cfg = resolved(state.config);
  const coins = Object.keys(state.buyHold);
  let held = 0;
  for (const c of coins) held += state.buyHold[c];
  const unallotted = cfg.startingEquity * Math.max(cfg.maxCoins - coins.length, 0) / cfg.maxCoins;
  return held + unallotted;
}

/** Equity = settled cash plus whatever the open positions have moved since the
 *  last bar close. Right after `onBar` the unrealized leg is zero by
 *  construction, because each bar's move is realized into cash at its mark. */
function remark(state: PaperState): void {
  let unreal = 0;
  for (const c of Object.keys(state.positions)) unreal += state.positions[c].unrealizedUsd;
  state.equity = state.cash + unreal;
}

/**
 * Apply one closed bar for one coin: mark the existing position to this bar,
 * then move to `target`.
 */
export function onBar(state: PaperState, coin: string, bar: Candle, target: number, note?: string): void {
  if (state.liquidated) return;                      // a wiped account does not trade
  const lastTime = state.lastBar[coin];
  if (lastTime !== undefined && bar.time <= lastTime) return;   // already applied

  const cfg = resolved(state.config);
  const pos = state.positions[coin];
  const prevPrice = state.lastPrice[coin];
  const prevPosition = pos ? pos.position : 0;

  // --- 1. Mark the position carried INTO this bar ---
  if (pos && prevPrice !== undefined && prevPrice > 0 && lastTime !== undefined) {
    const sign = pos.position > 0 ? 1 : -1;
    const notional = pos.notionalUsd;

    // Funding accrues on open notional for every hour of bar time. Longs pay a
    // positive rate, which is why the sign is carried through.
    const hours = (bar.time - lastTime) / 3600;
    const funding = cfg.fundingPerHour !== 0 ? sign * notional * cfg.fundingPerHour * hours : 0;

    // Liquidation before the close: what kills a leveraged account is the worst
    // tick inside the bar, not where it settles. Marking the close first would
    // trade straight through a wipeout.
    const adverse = sign > 0 ? bar.low : bar.high;
    const worstPnl = sign * notional * (adverse / prevPrice - 1);
    const maintenance = notional * maintenanceMarginFraction(cfg.maxLeverage);
    if (state.cash + worstPnl - funding <= maintenance) {
      state.realizedUsd += worstPnl;
      state.fundingUsd += funding;
      state.cash += worstPnl - funding;
      const fee = notional * (cfg.costBps / 10_000);
      state.feesUsd += fee;
      state.cash -= fee;
      state.fills.push({
        time: bar.time, coin, side: sign > 0 ? "sell" : "buy",
        price: adverse, sizeUsd: notional, feeUsd: fee, reason: "liquidation",
      });
      // The margin is gone — same terminal state `simulatePerp` records.
      delete state.positions[coin];
      state.cash = 0;
      state.equity = 0;
      state.liquidated = true;
      state.lastBar[coin] = bar.time;
      state.lastPrice[coin] = bar.close;
      if (state.buyHold[coin] !== undefined) state.buyHold[coin] *= bar.close / prevPrice;
      state.equityCurve.push({ time: bar.time, equity: 0, buyHold: buyHoldTotal(state) });
      state.updatedAt = now();
      return;
    }

    const pnl = sign * notional * (bar.close / prevPrice - 1);
    state.realizedUsd += pnl;
    state.fundingUsd += funding;
    state.cash += pnl - funding;
    pos.barsHeld++;
    pos.unrealizedUsd = 0;
  }

  // Buy and hold marks on the same bar, so the comparison is bar for bar.
  if (state.buyHold[coin] === undefined) state.buyHold[coin] = cfg.startingEquity / cfg.maxCoins;
  else if (prevPrice !== undefined && prevPrice > 0) state.buyHold[coin] *= bar.close / prevPrice;

  state.lastBar[coin] = bar.time;
  state.lastPrice[coin] = bar.close;
  remark(state);

  // --- 2. Record the bar ---
  // Recorded after the mark and before the order, so the point is the value of
  // what was held THROUGH this bar. That makes the series directly comparable,
  // bar for bar, with walkForward's equity curve, whose cost for the decision at
  // bar i is charged into the step that produces bar i+1. The fee just below
  // lands on the next point; `state.equity` is always the live, post-fee figure.
  state.equityCurve.push({ time: bar.time, equity: state.equity, buyHold: buyHoldTotal(state) });

  // --- 3. Move to the target ---
  const cap = Math.max(-1, Math.min(1, target));
  const delta = Math.abs(cap - prevPosition);
  // Capital allotted to this coin, sized off equity marked to this bar. This is
  // what makes a held position track equity: walkForward's PnL is a FRACTION of
  // current equity, so the notional is re-based every bar. That re-basing is
  // free here, exactly as it is there — a real account would pay to rebalance.
  const allotted = state.equity / cfg.maxCoins;

  if (delta > 1e-12) {
    const tradedUsd = delta * allotted * cfg.leverage;
    const fee = tradedUsd * (cfg.costBps / 10_000);
    state.feesUsd += fee;
    state.cash -= fee;
    // The structural reason comes first and a caller's note is appended, never
    // substituted: `paperStats` rebuilds round trips by reading whether a fill
    // left the coin flat, so the vocabulary has to survive annotation.
    const why = cap === 0 ? "exit"
      : prevPosition === 0 ? "entry"
      : prevPosition * cap < 0 ? "flip"
      : "entry";
    state.fills.push({
      time: bar.time, coin,
      side: cap > prevPosition ? "buy" : "sell",
      price: bar.close,
      sizeUsd: tradedUsd,
      feeUsd: fee,
      reason: note ? `${why} ${note}` : why,
    });
  }

  if (cap === 0) {
    delete state.positions[coin];
  } else {
    const sameSide = pos && pos.position * cap > 0;
    state.positions[coin] = {
      coin,
      position: cap,
      entryPrice: sameSide ? pos!.entryPrice : bar.close,
      entryTime: sameSide ? pos!.entryTime : bar.time,
      notionalUsd: Math.abs(cap) * allotted * cfg.leverage,
      barsHeld: sameSide ? pos!.barsHeld : 0,
      unrealizedUsd: 0,
    };
  }

  remark(state);
  state.updatedAt = now();
}

/**
 * Mark open positions without trading, for the ticking display between bars.
 *
 * Unrealized PnL is measured from the last BAR close, not from the entry, since
 * every bar's move is already realized into cash at its mark. Nothing is pushed
 * onto the equity curve: the curve is bar-indexed and a tick is not a bar.
 */
export function markToMarket(state: PaperState, prices: Record<string, number>, time: number): void {
  if (state.liquidated) return;
  for (const coin of Object.keys(state.positions)) {
    const pos = state.positions[coin];
    const price = prices[coin];
    const mark = state.lastPrice[coin];
    if (!price || !mark || price <= 0 || mark <= 0) continue;
    const sign = pos.position > 0 ? 1 : -1;
    pos.unrealizedUsd = sign * pos.notionalUsd * (price / mark - 1);
  }
  remark(state);
  state.updatedAt = time;
}

/**
 * Drive the engine over historical candles with a fitted model.
 *
 * The signal for each bar is computed from candles up to and including that bar
 * only. That is O(T^2) filtering rather than one pass, and it is worth it: it
 * makes the replay structurally incapable of seeing a bar it has not reached,
 * so the equivalence with `walkForward` is evidence rather than a coincidence
 * of how the loop was written.
 *
 * `warmupBars` is the number of leading feature rows to walk the filter over
 * before trading — pass the walk-forward training size to reproduce a block
 * exactly, since the harness also warms its belief on the training rows.
 */
export function replay(
  state: PaperState, coin: string, candles: Candle[],
  model: { params: HmmParams | HsmmParams; modelType: ModelType; scaler: Scaler; names: string[]; window: number },
  strategy: StrategyConfig,
  warmupBars = 0,
): void {
  const featureConfig: FeatureConfig = {
    window: model.window,
    useVolatility: model.names.includes("realizedVol"),
    useVolume: model.names.includes("volumeSurge"),
  };
  const fs = buildFeatures(candles, featureConfig);

  let prev = 0;
  // Row 0 cannot be signalled: its candle prefix is one bar short of the
  // lookback `buildFeatures` insists on, so trading opens at row 1 at the
  // earliest even when no warm-up was asked for.
  for (let i = Math.max(warmupBars, 1); i < fs.T; i++) {
    if (state.liquidated) break;
    const at = fs.index[i];
    const sig = signalNow(
      model.params, model.modelType, model.scaler, model.names,
      candles.slice(0, at + 1), featureConfig, strategy, prev,
    );
    onBar(state, coin, candles[at], sig.target);
    // Carry the position actually held, not the target: after a liquidation or
    // a skipped bar the two differ, and the strategy's hysteresis keys off the
    // real one.
    prev = state.positions[coin] ? state.positions[coin].position : 0;
  }
}

/** Write atomically: a half-written state file read by the TUI on the next
 *  start would be worse than no state file at all. */
export async function savePaper(path: string, state: PaperState): Promise<void> {
  state.updatedAt = now();
  const tmp = `${path}.${process.pid.toString(36)}.tmp`;
  await Bun.write(tmp, JSON.stringify(state));
  const { rename } = await import("node:fs/promises");
  await rename(tmp, path);
}

export async function loadPaper(path: string): Promise<PaperState> {
  const raw = await Bun.file(path).text();
  const s = JSON.parse(raw) as PaperState;
  // Tolerate a file written before a field existed rather than crashing a
  // running session on it.
  s.positions ??= {};
  s.fills ??= [];
  s.equityCurve ??= [];
  s.lastBar ??= {};
  s.lastPrice ??= {};
  s.buyHold ??= {};
  s.liquidated ??= false;
  return s;
}

export function paperStats(state: PaperState): {
  roi: number; sharpe: number; maxDD: number; winRate: number;
  trades: number; exposure: number; feeDrag: number; vsBuyHold: number;
} {
  const cfg = resolved(state.config);
  const curve = state.equityCurve;
  const start = cfg.startingEquity;
  const roi = start > 0 ? state.equity / start - 1 : 0;

  // Log returns for the risk stats, for walkForward's reason: arithmetic means
  // can be positive while the curve compounds downward.
  const rets: number[] = [];
  let peak = curve.length ? curve[0].equity : start;
  let maxDD = 0;
  for (let i = 0; i < curve.length; i++) {
    if (i > 0) {
      const prev = curve[i - 1].equity;
      if (prev > 0) rets.push(Math.log(Math.max(curve[i].equity / prev, 1e-9)));
    }
    peak = Math.max(peak, curve[i].equity);
    if (peak > 0) maxDD = Math.max(maxDD, 1 - curve[i].equity / peak);
  }

  // Annualize against the cadence the curve was actually sampled at, so a 5m
  // paper run and a 1h one are not reported on the same scale.
  let barSecs = 0;
  if (curve.length > 2) {
    const gaps: number[] = [];
    for (let i = 1; i < curve.length; i++) gaps.push(curve[i].time - curve[i - 1].time);
    gaps.sort((a, b) => a - b);
    barSecs = gaps[Math.floor(gaps.length / 2)];
  }
  const barsPerYear = barSecs > 0 ? (365.25 * 24 * 3600) / barSecs : 0;
  const n = rets.length || 1;
  const mean = rets.reduce((a, b) => a + b, 0) / n;
  const variance = rets.reduce((a, b) => a + (b - mean) * (b - mean), 0) / Math.max(n - 1, 1);
  const sd = Math.sqrt(variance);
  const sharpe = sd > 0 && barsPerYear > 0 ? (mean / sd) * Math.sqrt(barsPerYear) : 0;

  // Round trips, rebuilt from the fill log. A fill whose reason leaves the coin
  // flat closes the trip; anything else opens or resizes one. The trip is scored
  // on the equity change across it, which is the only per-trade figure the
  // engine keeps — so it includes the fees the trip paid, as it should.
  const openAt: Record<string, number> = {};
  const equityAt = (t: number): number => {
    let v = curve.length ? curve[0].equity : start;
    for (const p of curve) { if (p.time > t) break; v = p.equity; }
    return v;
  };
  let trades = 0, wins = 0;
  const spans: { from: number; to: number }[] = [];
  for (const f of state.fills) {
    const why = f.reason.split(" ")[0];
    const flat = why === "exit" || why === "liquidation";
    if (flat) {
      const from = openAt[f.coin];
      if (from !== undefined) {
        trades++;
        if (equityAt(f.time) > equityAt(from)) wins++;
        spans.push({ from, to: f.time });
        delete openAt[f.coin];
      }
    } else if (openAt[f.coin] === undefined) {
      openAt[f.coin] = f.time;
    }
  }
  for (const coin of Object.keys(openAt)) spans.push({ from: openAt[coin], to: Infinity });

  let exposed = 0;
  for (const p of curve) {
    if (spans.some((s) => p.time >= s.from && p.time < s.to)) exposed++;
  }

  const bh = curve.length ? curve[curve.length - 1].buyHold : start;
  return {
    roi,
    sharpe,
    maxDD,
    winRate: trades > 0 ? wins / trades : 0,
    trades,
    exposure: curve.length > 0 ? exposed / curve.length : 0,
    feeDrag: start > 0 ? state.feesUsd / start : 0,
    vsBuyHold: roi - (start > 0 ? bh / start - 1 : 0),
  };
}
