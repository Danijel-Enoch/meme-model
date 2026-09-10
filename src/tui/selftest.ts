/**
 * End-to-end smoke of the TUI's machinery without a terminal: worker boot,
 * live fetch, fit, walk-forward, and a replay through the paper account.
 *
 *   bun run src/tui/selftest.ts --coin SOL --timeframe 30m
 *
 * Hits the network on purpose — it exists to catch the failures unit tests
 * cannot: a Worker that will not start under Bun, a model that will not cross
 * structured clone, an API shape that changed.
 */
import { JobClient, type BacktestResultMsg, type FitResult } from "./jobs";
import { barBudget } from "../sweep";
import { newPaperState, onBar, paperStats } from "../paper";

const arg = (n: string, d: string) => {
  const i = process.argv.indexOf(`--${n}`);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : d;
};
const coin = arg("coin", "SOL");
const timeframe = arg("timeframe", "30m");
const days = Number(arg("days", "45"));
const { bars } = barBudget(timeframe, days);

const jobs = new JobClient();
const t0 = Date.now();

const fitRes = await jobs.run<FitResult>({
  kind: "fit", coin, timeframe, bars,
  states: 3, modelType: "hsmm", maxDuration: 30, seed: 42, restarts: 8,
}).promise;
console.log(`fit      ${coin} ${timeframe}: ${fitRes.fit.bars} bars, log-lik/bar ` +
  `${fitRes.fit.logLikPerBar.toFixed(4)}, ${((Date.now() - t0) / 1000).toFixed(1)}s`);
for (const s of fitRes.fit.states) {
  console.log(`         ${s.label.padEnd(9)} ${s.meanRetBps.toFixed(1).padStart(6)}bps  ` +
    `dwell ${s.durationBars.toFixed(1)}b  freq ${(s.freq * 100).toFixed(1)}%`);
}

const t1 = Date.now();
const bt = await jobs.run<BacktestResultMsg>({
  kind: "backtest", coin, timeframe, bars,
  states: 3, modelType: "hsmm", maxDuration: 30, costBps: 4.5, trials: 200, seed: 42, restarts: 4,
}).promise;
const b = bt.backtest;
console.log(`backtest ${b.barsTraded} bars, ${b.refits} refits, ${b.trades} trades, ` +
  `${((Date.now() - t1) / 1000).toFixed(1)}s`);
console.log(`         roi ${(b.roi * 100).toFixed(2)}%  b&h ${(b.buyHold * 100).toFixed(2)}%  ` +
  `expos ${(b.exposure * 100).toFixed(0)}%  p ${b.pValue?.toFixed(3)}`);

// Replay the walk-forward's own positions through the account.
const paper = newPaperState({ startingEquity: 1000, leverage: 1, costBps: 4.5, maxCoins: 1 });
for (let i = 0; i < bt.chart.candles.length; i++) {
  onBar(paper, coin, bt.chart.candles[i], bt.chart.positions[i], "selftest");
  if (paper.liquidated) break;
}
const st = paperStats(paper);
console.log(`paper    equity $${paper.equity.toFixed(2)}  roi ${(st.roi * 100).toFixed(2)}%  ` +
  `fills ${paper.fills.length}  fees $${paper.feesUsd.toFixed(2)}  maxDD ${(st.maxDD * 100).toFixed(1)}%`);
console.log(`         backtest roi ${(b.roi * 100).toFixed(2)}% vs paper ${(st.roi * 100).toFixed(2)}%` +
  `  (gap ${((st.roi - b.roi) * 100).toFixed(2)}pp)`);

jobs.terminate();
