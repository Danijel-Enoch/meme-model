/**
 * Leveraged perpetual account simulation.
 *
 * A backtest that reports percentages hides three things that decide whether a
 * small leveraged account survives, so all three are modelled explicitly here:
 *
 *   Fees are charged on NOTIONAL, not on equity. At 2x, a $50 account trades
 *   $100, so a 4.5bps taker fee costs 9bps of equity per side, not 4.5.
 *
 *   Funding accrues every hour against the notional too, in the same doubled
 *   proportion. It is invisible in a spot backtest and it never stops while a
 *   position is open.
 *
 *   Liquidation is path-dependent. What matters is the worst point INSIDE the
 *   trade, not the close, so the bar low is used for longs and the high for
 *   shorts. A close-only simulation will happily trade through a wipeout.
 */

import { buildFeatures, type Candle, type FeatureConfig } from "./features";
import type { FundingPoint } from "./hyperliquid";

export interface PerpConfig {
  startingEquity: number;
  leverage: number;
  /** Per side, in basis points, charged on notional. */
  takerBps: number;
  /** Equity below notional x this is a liquidation. */
  maintenanceMarginFraction: number;
  /** Exchange minimum order value; smaller intended trades are skipped. */
  minOrderUsd?: number;
}

export interface PerpTrade {
  entryTime: number;
  exitTime: number;
  direction: 1 | -1;
  barsHeld: number;
  entryPrice: number;
  exitPrice: number;
  notional: number;
  grossPnl: number;
  fees: number;
  funding: number;
  netPnl: number;
  equityAfter: number;
  win: boolean;
  liquidated: boolean;
}

export interface PerpResult {
  trades: PerpTrade[];
  startingEquity: number;
  finalEquity: number;
  peakEquity: number;
  maxDrawdown: number;
  totalFees: number;
  totalFunding: number;
  liquidated: boolean;
  liquidationTime: number | null;
  equityCurve: { time: number; equity: number }[];
  barsInMarket: number;
  bars: number;
}

/** Sum the hourly funding charged between two timestamps. */
function fundingBetween(funding: FundingPoint[], fromTime: number, toTime: number): number {
  let total = 0;
  for (const f of funding) {
    if (f.time > fromTime && f.time <= toTime) total += f.rate;
  }
  return total;
}

/**
 * Run a position series through a leveraged account.
 *
 * `positions` is indexed by feature row, matching walkForward. The position
 * held at row i earns the move into row i+1, and the notional is recomputed at
 * each entry from current equity, which is what fixed-leverage sizing means.
 */
export function simulatePerp(
  candles: Candle[],
  featureConfig: FeatureConfig,
  positions: number[],
  funding: FundingPoint[],
  cfg: PerpConfig,
  fromRow = 0,
): PerpResult {
  const fs = buildFeatures(candles, featureConfig);
  const fee = cfg.takerBps / 10_000;
  const minOrder = cfg.minOrderUsd ?? 10;

  let equity = cfg.startingEquity;
  let peak = equity;
  let maxDd = 0;
  let totalFees = 0;
  let totalFunding = 0;
  let barsInMarket = 0;
  let liquidated = false;
  let liquidationTime: number | null = null;

  const trades: PerpTrade[] = [];
  const equityCurve: { time: number; equity: number }[] = [];

  let open: {
    row: number; dir: 1 | -1; entryPrice: number; notional: number;
    size: number; fees: number; funding: number; entryTime: number;
  } | null = null;

  const closeAt = (row: number, price: number, time: number, forced: boolean) => {
    if (!open) return;
    const gross = open.dir * (price - open.entryPrice) * open.size;
    const exitFee = Math.abs(open.size * price) * fee;
    const fees = open.fees + exitFee;
    const net = gross - fees - open.funding;
    equity = Math.max(equity + net, 0);
    totalFees += exitFee;
    trades.push({
      entryTime: open.entryTime,
      exitTime: time,
      direction: open.dir,
      barsHeld: row - open.row,
      entryPrice: open.entryPrice,
      exitPrice: price,
      notional: open.notional,
      grossPnl: gross,
      fees,
      funding: open.funding,
      netPnl: net,
      equityAfter: equity,
      win: net > 0,
      liquidated: forced,
    });
    open = null;
  };

  for (let i = fromRow; i < fs.T - 1 && !liquidated; i++) {
    const candleIdx = fs.index[i];
    const bar = candles[candleIdx];
    const nextBar = candles[fs.index[i + 1]];
    const pos = positions[i] ?? 0;
    const dir: 1 | -1 | 0 = pos > 1e-9 ? 1 : pos < -1e-9 ? -1 : 0;

    if (open && dir !== open.dir) closeAt(i, bar.close, bar.time, false);

    if (!open && dir !== 0 && equity > 0) {
      const notional = equity * cfg.leverage;
      if (notional >= minOrder) {
        const entryFee = notional * fee;
        totalFees += entryFee;
        open = {
          row: i, dir, entryPrice: bar.close, notional,
          size: notional / bar.close,
          fees: entryFee, funding: 0, entryTime: bar.time,
        };
      }
    }

    if (open) {
      barsInMarket++;
      // Funding accrues on notional for the hours spanned by this bar.
      const f = fundingBetween(funding, bar.time, nextBar.time);
      const charge = open.dir * f * open.notional; // longs pay a positive rate
      open.funding += charge;
      totalFunding += charge;

      // Liquidation check uses the worst point inside the next bar, not its close.
      const adverse = open.dir > 0 ? nextBar.low : nextBar.high;
      const worstPnl = open.dir * (adverse - open.entryPrice) * open.size;
      const worstEquity = equity + worstPnl - open.fees - open.funding;
      const maintenance = open.notional * cfg.maintenanceMarginFraction;
      if (worstEquity <= maintenance) {
        // Liquidated: the position is closed at the liquidation price and the
        // margin is gone.
        closeAt(i + 1, adverse, nextBar.time, true);
        equity = 0;
        liquidated = true;
        liquidationTime = nextBar.time;
        equityCurve.push({ time: nextBar.time, equity: 0 });
        break;
      }
    }

    // Mark to market for the curve.
    const markPnl = open ? open.dir * (nextBar.close - open.entryPrice) * open.size - open.fees - open.funding : 0;
    const marked = equity + markPnl;
    peak = Math.max(peak, marked);
    maxDd = Math.max(maxDd, peak > 0 ? 1 - marked / peak : 0);
    equityCurve.push({ time: nextBar.time, equity: marked });
  }

  if (open && !liquidated) {
    const lastRow = fs.T - 1;
    closeAt(lastRow, candles[fs.index[lastRow]].close, candles[fs.index[lastRow]].time, false);
  }

  return {
    trades,
    startingEquity: cfg.startingEquity,
    finalEquity: equity,
    peakEquity: peak,
    maxDrawdown: maxDd,
    totalFees,
    totalFunding,
    liquidated,
    liquidationTime,
    equityCurve,
    barsInMarket,
    bars: Math.max(fs.T - 1 - fromRow, 0),
  };
}
