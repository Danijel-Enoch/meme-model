#!/usr/bin/env bun
/**
 * CLI: bun run src/cli.ts <demo|train|backtest|predict> [flags]
 */

import { fit, posteriors, viterbi, filter, predictNext, stationary, expectedDuration, serialize, deserialize, type HmmParams } from "./hmm";
import { buildFeatures, fitScaler, applyScaler, unscale, type Candle, type Scaler, type FeatureConfig } from "./features";
import { loadCsv, generateSynthetic, toCsv } from "./data";
import {
  searchPairs, trendingPools, topPools, fetchOhlcv, fillGaps, parseTimeframe, barsPerYear,
  resolveNetwork, cachePath, readCache, writeCache,
} from "./sources";
import { walkForward, stateMeanReturns, extractTrades, summarizeTrades, type StrategyConfig, type WalkForwardConfig, type ModelType, type Trade } from "./backtest";
import { fitHsmm, filterHsmm, viterbiHsmm, expectedDurations, serializeHsmm, deserializeHsmm, type HsmmParams } from "./hsmm";
import { permutationTest, ceilingAnalysis, tradedSeries } from "./diagnostics";
import { walkForwardConfluence, DEFAULT_STACK, type TimeframeSpec } from "./confluence";
import * as hl from "./hyperliquid";
import { simulatePerp } from "./perp";
import { runMcmc } from "./mcmc";
import { fitVb, selectStates } from "./vb";
import {
  posteriorStateMeans, posteriorDurations, posteriorSignal, DEFAULT_PRIORS,
  type PosteriorDraws,
} from "./posterior";

type Flags = Record<string, string | boolean>;

function parseArgs(argv: string[]): { cmd: string; flags: Flags } {
  const cmd = argv[0] ?? "demo";
  const flags: Flags = {};
  for (let i = 1; i < argv.length; i++) {
    const a = argv[i];
    if (!a.startsWith("--")) continue;
    const key = a.slice(2);
    const next = argv[i + 1];
    if (next && !next.startsWith("--")) { flags[key] = next; i++; }
    else flags[key] = true;
  }
  return { cmd, flags };
}

const num = (f: Flags, k: string, d: number) => (f[k] !== undefined ? Number(f[k]) : d);
const bool = (f: Flags, k: string, d = false) => (f[k] !== undefined ? f[k] !== "false" : d);
const str = (f: Flags, k: string, d?: string) => (f[k] !== undefined ? String(f[k]) : d);

/**
 * Lookback for the rolling features.
 *
 * Was 20. A window longer than a regime cannot see that regime, and the
 * regimes here are short: on synthetic data with 4.5-bar pumps, dropping
 * 20 -> 3 lifted Viterbi regime recovery from 53% to 73%, and 5 is the
 * best compromise across the timeframes tested.
 */
const DEFAULT_WINDOW = 5;

const pct = (x: number) => `${(x * 100).toFixed(2)}%`;
const bps = (x: number) => `${(x * 10_000).toFixed(2)}bps`;

/** A label for each state, derived from its mean return rather than assumed. */
function labelStates(muRet: number[]): string[] {
  return muRet.map((r) => {
    const b = r * 10_000;
    if (b > 20) return "PUMP";
    if (b > 3) return "drift-up";
    if (b < -20) return "DUMP";
    if (b < -3) return "bleed";
    return "chop";
  });
}

function sparkline(values: number[], width = 60): string {
  if (values.length === 0) return "";
  const chars = "▁▂▃▄▅▆▇█";
  const step = Math.max(1, Math.floor(values.length / width));
  const sampled: number[] = [];
  for (let i = 0; i < values.length; i += step) sampled.push(values[i]);
  const lo = Math.min(...sampled), hi = Math.max(...sampled);
  const range = hi - lo || 1;
  return sampled.map((v) => chars[Math.min(7, Math.floor(((v - lo) / range) * 7.999))]).join("");
}

function printStateTable(p: HmmParams | HsmmParams, scaler: Scaler, names: string[]) {
  const isHsmm = "dur" in p;
  const muRet = Array.from({ length: p.K }, (_, k) => unscale(p.mu[k * p.D], 0, scaler));
  const labels = labelStates(muRet);
  const durations = isHsmm
    ? expectedDurations(p as HsmmParams)
    : Array.from({ length: p.K }, (_, k) => expectedDuration(p as HmmParams, k));
  // An HSMM has no self-transitions, so its chain has no stationary occupancy
  // in the HMM sense; time share comes from durations instead.
  const share = isHsmm
    ? (() => {
        const total = durations.reduce((a, b) => a + b, 0);
        return durations.map((d) => d / total);
      })()
    : Array.from(stationary(p as HmmParams));

  console.log(`\nStates (${isHsmm ? "HSMM" : "HMM"}, sorted by mean return, most bearish first)`);
  console.log("  #  label      mean ret   bar vol    freq   avg duration");
  for (let k = 0; k < p.K; k++) {
    const volIdx = names.indexOf("realizedVol");
    const v = volIdx >= 0 ? Math.exp(unscale(p.mu[k * p.D + volIdx], volIdx, scaler)) : NaN;
    console.log(
      `  ${k}  ${labels[k].padEnd(9)} ${(muRet[k] * 10_000).toFixed(1).padStart(8)}bps` +
      `  ${Number.isFinite(v) ? pct(v).padStart(7) : "     n/a"}` +
      `  ${pct(share[k]).padStart(6)}  ${durations[k].toFixed(1).padStart(6)} bars`,
    );
  }

  if (isHsmm) {
    const h = p as HsmmParams;
    console.log("\nDuration distribution  P(regime lasts d bars)");
    for (let k = 0; k < h.K; k++) {
      // Show where the mass actually sits, not all maxDuration buckets.
      const top = Array.from({ length: h.maxDuration }, (_, i) => [i + 1, h.dur[k * h.maxDuration + i]] as [number, number])
        .sort((a, b) => b[1] - a[1]).slice(0, 5).sort((a, b) => a[0] - b[0]);
      console.log(`  ${k} (${labels[k]}): ` + top.map(([d, pr]) => `${d}bar=${(pr * 100).toFixed(0)}%`).join("  "));
    }
    console.log("\nTransition matrix  P(next | current), self-transitions excluded by design");
  } else {
    console.log("\nTransition matrix  P(next | current)");
  }
  console.log(`       ${Array.from({ length: p.K }, (_, j) => `->${j}`.padStart(7)).join("")}`);
  for (let i = 0; i < p.K; i++) {
    const row = Array.from({ length: p.K }, (_, j) => p.A[i * p.K + j].toFixed(3).padStart(7)).join("");
    console.log(`  ${i}: ${row}   (${labels[i]})`);
  }
}

function printBacktest(title: string, m: ReturnType<typeof walkForward>["metrics"], equity: number[], bh: number[]) {
  console.log(`\n${title}`);
  console.log(`  bars traded      ${m.bars}`);
  console.log(`  strategy return  ${pct(m.totalReturn)}`);
  console.log(`  buy & hold       ${pct(m.buyHoldReturn)}`);
  console.log(`  sharpe (ann.)    ${m.sharpe.toFixed(2)}`);
  console.log(`  max drawdown     ${pct(m.maxDrawdown)}`);
  console.log(`  hit rate         ${pct(m.hitRate)}`);
  console.log(`  trades           ${m.trades}  (turnover ${m.turnover.toFixed(1)}x)`);
  console.log(`  time in market   ${pct(m.exposure)}`);
  console.log(`  cost drag        ${pct(m.costDrag)} of notional`);
  console.log(`\n  strategy  ${sparkline(equity)}`);
  console.log(`  hold      ${sparkline(bh)}`);
}

function strategyFrom(f: Flags): StrategyConfig {
  const cost = num(f, "cost", 30);
  return {
    // Leave entry undefined when unflagged so the backtest's cost-aware
    // default (2x per-side cost) applies instead of a hardcoded number.
    entryBps: f["entry"] !== undefined ? num(f, "entry", 0) : undefined,
    exitBps: f["exit"] !== undefined ? num(f, "exit", 0) : undefined,
    allowShort: bool(f, "short"),
    costBps: cost,
    volTarget: num(f, "vol-target", 0),
    maxPosition: num(f, "max-pos", 1),
    confidence: num(f, "confidence", 0),
    durationAware: bool(f, "duration-aware"),
  };
}

/**
 * Accuracy of a recovered state path against known truth, maximized over all
 * relabelings. State indices from EM are arbitrary, so a raw element-wise
 * comparison mostly measures which permutation the fit happened to land on.
 */
function bestPermutationAccuracy(path: Int32Array, truth: Int32Array, K: number): { acc: number; perm: number[] } {
  const idx = Array.from({ length: K }, (_, i) => i);
  const perms: number[][] = [];
  const permute = (cur: number[], rest: number[]) => {
    if (rest.length === 0) { perms.push(cur); return; }
    rest.forEach((v, i) => permute([...cur, v], [...rest.slice(0, i), ...rest.slice(i + 1)]));
  };
  permute([], idx);

  let best = 0, bestPerm = idx;
  for (const perm of perms) {
    let hit = 0;
    for (let t = 0; t < path.length; t++) if (perm[path[t]] === truth[t]) hit++;
    if (hit > best) { best = hit; bestPerm = perm; }
  }
  return { acc: best / path.length, perm: bestPerm };
}

function modelType(f: Flags): ModelType {
  const m = str(f, "model-type", bool(f, "hsmm") ? "hsmm" : "hmm")!;
  if (m !== "hmm" && m !== "hsmm") throw new Error(`--model-type must be hmm or hsmm, got "${m}"`);
  return m;
}

function wfFrom(f: Flags, defaultBarsPerYear = 105_120): WalkForwardConfig {
  return {
    modelType: modelType(f),
    maxDuration: num(f, "max-duration", 30),
    trainSize: num(f, "train", 1500),
    testSize: num(f, "test", 500),
    states: num(f, "states", 3),
    seed: num(f, "seed", 42),
    restarts: num(f, "restarts", 4),
    // Derived from the data's own bar spacing unless overridden.
    barsPerYear: num(f, "bars-per-year", defaultBarsPerYear),
    verbose: bool(f, "verbose"),
  };
}

/** Meme coin prices span 1e-9 to 1e3; fixed decimals blow the column apart. */
const price = (x: number) =>
  !Number.isFinite(x) || x === 0 ? "?" :
  x >= 0.01 ? x.toFixed(4) :
  x.toExponential(2);

const usd = (x: number) =>
  x >= 1e9 ? `$${(x / 1e9).toFixed(1)}B` :
  x >= 1e6 ? `$${(x / 1e6).toFixed(1)}M` :
  x >= 1e3 ? `$${(x / 1e3).toFixed(0)}k` : `$${x.toFixed(0)}`;

interface DataSet {
  candles: Candle[];
  label: string;
  /** Derived from the actual bar spacing, so Sharpe annualizes correctly. */
  barsPerYear: number;
  /** Present when the data came from an API — recorded into saved models. */
  source?: { network: string; pool: string; timeframe: string };
}

/**
 * Infer bar spacing from the timestamps themselves rather than trusting a flag.
 * Uses the median gap, which shrugs off the missing bars that riddle thin pools.
 */
function inferBarsPerYear(candles: Candle[], fallback = 105_120): number {
  if (candles.length < 10) return fallback;
  const deltas: number[] = [];
  for (let i = 1; i < candles.length; i++) deltas.push(candles[i].time - candles[i - 1].time);
  deltas.sort((a, b) => a - b);
  const median = deltas[Math.floor(deltas.length / 2)];
  // Timestamps in seconds land in a sane range; anything else is a row index.
  if (!Number.isFinite(median) || median <= 0 || median > 86_400 * 31) return fallback;
  return (365.25 * 24 * 3600) / median;
}

/** Resolve a pool address from --pool, or from --token via DexScreener search. */
async function resolvePool(f: Flags): Promise<{ network: string; pool: string; label: string }> {
  const network = resolveNetwork(str(f, "network", "solana")!);
  const pool = str(f, "pool");
  if (pool) return { network, pool, label: `${network}:${pool.slice(0, 8)}` };

  const token = str(f, "token")!;
  console.log(`searching DexScreener for "${token}" on ${network}...`);
  const pairs = await searchPairs(token, network, num(f, "min-liquidity", 10_000), num(f, "min-volume", 1_000));
  if (pairs.length === 0) {
    throw new Error(
      `no pools for "${token}" on ${network} with >${usd(num(f, "min-liquidity", 10_000))} liquidity ` +
      `and >${usd(num(f, "min-volume", 1_000))} 24h volume.\n` +
      `  Try --network <chain>, lower --min-liquidity / --min-volume, or pass --pool <address>.`,
    );
  }
  const best = pairs[0];
  console.log(
    `  using ${best.baseSymbol}/${best.quoteSymbol} on ${best.dex} — ` +
    `${usd(best.liquidityUsd)} liquidity, ${usd(best.volume24h)} 24h volume`,
  );
  if (pairs.length > 1) {
    console.log(`  (${pairs.length} pools matched; picked the deepest. Run \`search\` to see the rest.)`);
  }
  return { network: best.network, pool: best.pairAddress, label: `${best.baseSymbol}/${best.quoteSymbol}` };
}

/** Fetch candles from GeckoTerminal, with caching and gap reporting. */
async function fetchDataSet(f: Flags): Promise<DataSet> {
  const { network, pool, label } = await resolvePool(f);
  const tfLabel = str(f, "timeframe", str(f, "tf", "5m"))!;
  const tf = parseTimeframe(tfLabel);
  const bars = num(f, "bars", 3000);

  const cache = cachePath(network, pool, tfLabel, bars);
  const maxAge = num(f, "cache-minutes", 10) * 60_000;
  let res = bool(f, "no-cache") ? null : await readCache(cache, maxAge);
  if (res) {
    console.log(`  using cached data (${res.candles.length} bars, < ${num(f, "cache-minutes", 10)}min old)`);
  } else {
    console.log(`fetching ${bars} ${tfLabel} bars from GeckoTerminal...`);
    res = await fetchOhlcv(network, pool, tfLabel, bars, { verbose: bool(f, "verbose") });
    if (!bool(f, "no-cache")) await writeCache(cache, res);
  }

  let candles = res.candles;
  if (candles.length === 0) throw new Error(`no candles returned for ${network}:${pool}`);

  const first = new Date(candles[0].time * 1000).toISOString().slice(0, 16).replace("T", " ");
  const last = new Date(candles[candles.length - 1].time * 1000).toISOString().slice(0, 16).replace("T", " ");
  console.log(`  ${candles.length} bars of ${res.baseSymbol}/${res.quoteSymbol}, ${first} -> ${last} UTC` +
    (res.requests ? ` (${res.requests} requests)` : ""));
  if (res.duplicatesDropped > 0) {
    console.log(`  dropped ${res.duplicatesDropped} duplicate bars from page overlaps`);
  }

  if (res.gaps > 0) {
    const share = res.gaps / (candles.length + res.gaps);
    console.log(`  ${res.gaps} missing bars (${pct(share)} of the window, longest run ${res.largestGap})` +
      ` — intervals with no trades at all`);
    if (bool(f, "fill")) {
      const filledRes = fillGaps(candles, tf.seconds, num(f, "max-fill", 12));
      candles = filledRes.candles;
      console.log(`  --fill: inserted ${filledRes.filled} flat bars` +
        (filledRes.skipped > 0 ? `, left ${filledRes.skipped} in runs too long to fill` : ""));
    } else if (share > 0.05) {
      console.log(`  warning: the bar spacing is not constant, which breaks the model's`);
      console.log(`  fixed-time-step assumption. Consider --fill, a longer --timeframe,`);
      console.log(`  or a more liquid pool.`);
    }
  }

  return {
    candles,
    label: `${res.baseSymbol}/${res.quoteSymbol} ${tfLabel}`,
    barsPerYear: barsPerYear(tf),
    source: { network, pool, timeframe: tfLabel },
  };
}

/** Hyperliquid perps: continuous order book, so no gap handling is needed. */
async function hyperliquidDataSet(f: Flags): Promise<DataSet> {
  const coin = str(f, "coin")!.toUpperCase();
  const interval = str(f, "timeframe", str(f, "tf", "5m"))!;
  const bars = num(f, "bars", 5000);

  console.log(`fetching ${bars} ${interval} candles for ${coin} from Hyperliquid...`);
  const res = await hl.fetchCandles(coin, interval, bars, { verbose: bool(f, "verbose") });
  if (res.candles.length === 0) throw new Error(`no candles for ${coin} — check the symbol with \`top --source hyperliquid\``);

  const first = new Date(res.candles[0].time * 1000).toISOString().slice(0, 16).replace("T", " ");
  const lastT = new Date(res.candles[res.candles.length - 1].time * 1000).toISOString().slice(0, 16).replace("T", " ");
  console.log(`  ${res.candles.length} bars, ${first} -> ${lastT} UTC (${res.requests} requests)`);
  console.log(res.gaps === 0
    ? "  no missing bars"
    : `  ${res.gaps} missing bars — unusual for a perp, check the symbol`);

  return {
    candles: res.candles,
    label: `${coin}-PERP ${interval}`,
    barsPerYear: hl.hlBarsPerYear(interval),
  };
}

async function getDataSet(f: Flags): Promise<DataSet> {
  const csv = str(f, "csv");
  if (csv) {
    const candles = await loadCsv(csv);
    return { candles, label: csv, barsPerYear: inferBarsPerYear(candles) };
  }
  if (f["coin"]) return hyperliquidDataSet(f);
  if (f["pool"] || f["token"]) return fetchDataSet(f);

  const { candles } = generateSynthetic({ bars: num(f, "bars", 6000), seed: num(f, "seed", 7) });
  console.log("(no --csv / --token / --pool given, using synthetic regime-switching data)");
  return { candles, label: "synthetic", barsPerYear: 105_120 };
}

async function getCandles(f: Flags): Promise<Candle[]> {
  return (await getDataSet(f)).candles;
}

async function cmdTrain(f: Flags) {
  const ds = await getDataSet(f);
  const candles = ds.candles;
  const window = num(f, "window", DEFAULT_WINDOW);
  const fs = buildFeatures(candles, { window, useVolatility: !bool(f, "no-vol"), useVolume: !bool(f, "no-volume") });
  const scaler = fitScaler(fs.X, fs.T, fs.D);
  const Z = applyScaler(fs.X, fs.T, fs.D, scaler);

  const mt = modelType(f);
  console.log(`\nFitting ${num(f, "states", 3)}-state ${mt.toUpperCase()} on ${fs.T} bars of ${ds.label}`);
  console.log(`features: ${fs.names.join(", ")}  window: ${window}`);

  let params: HmmParams | HsmmParams;
  let logLik: number, iters: number, converged: boolean;
  if (mt === "hsmm") {
    const res = fitHsmm(Z, fs.T, fs.D, {
      states: num(f, "states", 3),
      restarts: num(f, "restarts", 4),
      seed: num(f, "seed", 42),
      maxDuration: num(f, "max-duration", 30),
    });
    params = res.params; logLik = res.logLik; iters = res.iterations; converged = res.converged;
  } else {
    const res = fit(Z, fs.T, fs.D, {
      states: num(f, "states", 3),
      restarts: num(f, "restarts", 8),
      seed: num(f, "seed", 42),
      maxIter: num(f, "max-iter", 300),
      verbose: bool(f, "verbose"),
    });
    params = res.params; logLik = res.logLik; iters = res.iterations; converged = res.converged;
  }
  console.log(`log-likelihood/bar: ${(logLik / fs.T).toFixed(4)}  (${iters} iters, converged: ${converged})`);

  printStateTable(params, scaler, fs.names);

  const path = mt === "hsmm"
    ? viterbiHsmm(Z, fs.T, params as HsmmParams)
    : viterbi(Z, fs.T, params as HmmParams);
  const counts = new Array(params.K).fill(0);
  for (let t = 0; t < fs.T; t++) counts[path[t]]++;
  console.log(`\nViterbi path occupancy: ${counts.map((c, k) => `${k}=${pct(c / fs.T)}`).join("  ")}`);

  if (mt === "hsmm") {
    const fl = filterHsmm(Z, fs.T, params as HsmmParams);
    const cur = Array.from({ length: params.K }, (_, k) => fl.stateProb[(fs.T - 1) * params.K + k]);
    const nx = Array.from({ length: params.K }, (_, k) => fl.nextProb[(fs.T - 1) * params.K + k]);
    const lbl = labelStates(Array.from({ length: params.K }, (_, k) => unscale(params.mu[k * params.D], 0, scaler)));
    const top = cur.indexOf(Math.max(...cur));
    console.log(`current state (filtered): ${lbl[top]} (state ${top}, p=${pct(cur[top])})`);
    console.log(`next bar forecast: ${nx.map((v, k) => `${lbl[k]} ${pct(v)}`).join("  ")}`);
  } else {
    console.log(`current state (filtered): ${describeNow(Z, fs.T, fs.D, params as HmmParams, scaler)}`);
  }

  const out = str(f, "out");
  if (out) {
    await Bun.write(out, JSON.stringify({
      modelType: mt,
      params: JSON.parse(mt === "hsmm" ? serializeHsmm(params as HsmmParams) : serialize(params as HmmParams)),
      scaler: { mean: Array.from(scaler.mean), std: Array.from(scaler.std) },
      names: fs.names,
      window,
      trainedBars: fs.T,
      // Recording the source lets `predict` reload the same pool without
      // the caller having to repeat every flag — and makes a stale model
      // obvious when the pool no longer matches.
      source: ds.source ?? null,
      label: ds.label,
    }, null, 2));
    console.log(`\nsaved model -> ${out}`);
  }
}

function describeNow(Z: Float64Array, T: number, D: number, p: HmmParams, scaler: Scaler): string {
  const { alpha } = filter(Z, T, p);
  const cur = Array.from({ length: p.K }, (_, k) => alpha[(T - 1) * p.K + k]);
  const muRet = stateMeanReturns(p, scaler);
  const labels = labelStates(muRet);
  const top = cur.indexOf(Math.max(...cur));
  return `${labels[top]} (state ${top}, p=${pct(cur[top])})`;
}

async function cmdBacktest(f: Flags) {
  const ds = await getDataSet(f);
  const candles = ds.candles;
  console.log(`\nWalk-forward backtest on ${candles.length} candles of ${ds.label}`);
  console.log(`  train ${num(f, "train", 1500)} / test ${num(f, "test", 500)} bars, ` +
    `${num(f, "states", 3)} states, cost ${num(f, "cost", 30)}bps per side, ` +
    `entry ${num(f, "entry", num(f, "cost", 30) * 2)}bps${bool(f, "short") ? ", shorts on" : ""}`);

  const featureCfg = {
    window: num(f, "window", DEFAULT_WINDOW),
    useVolatility: !bool(f, "no-vol"),
    useVolume: !bool(f, "no-volume"),
  };
  // Derive the feature names from the same config the model was built with,
  // so the state table stays correct when features are switched off.
  const names = buildFeatures(candles.slice(0, featureCfg.window + 5), featureCfg).names;

  const r = walkForward(candles, featureCfg, strategyFrom(f), wfFrom(f, ds.barsPerYear));
  console.log(`  ${r.refits} model refits (each block traded with parameters fit only on prior bars)`);
  printBacktest("Out-of-sample results", r.metrics, r.equity, r.buyHoldEquity);

  if (r.lastModel) printStateTable(r.lastModel.params, r.lastModel.scaler, names);

  // A zero-trade result at realistic costs says nothing on its own, so always
  // show where the edge actually dies unless the user opts out.
  if (!bool(f, "no-sweep")) {
    console.log("\nCost sensitivity");
    costSweep(candles, featureCfg, strategyFrom(f),
      { ...wfFrom(f, ds.barsPerYear), restarts: 3 });
  }
}

async function cmdPredict(f: Flags) {
  const modelPath = str(f, "model");
  if (!modelPath) throw new Error("--model <path> required (produce one with `train --out model.json`)");
  const saved = JSON.parse(await Bun.file(modelPath).text());
  const savedType: ModelType = saved.modelType === "hsmm" ? "hsmm" : "hmm";
  const p = savedType === "hsmm"
    ? deserializeHsmm(JSON.stringify(saved.params))
    : deserialize(JSON.stringify(saved.params));
  const scaler: Scaler = { mean: new Float64Array(saved.scaler.mean), std: new Float64Array(saved.scaler.std) };

  // Reuse the pool the model was trained on unless the caller names another.
  const flags: Flags = { ...f };
  if (!flags["csv"] && !flags["pool"] && !flags["token"] && saved.source) {
    flags["pool"] = saved.source.pool;
    flags["network"] = saved.source.network;
    if (!flags["timeframe"] && !flags["tf"]) flags["timeframe"] = saved.source.timeframe;
    console.log(`using the model's source: ${saved.label ?? saved.source.pool} on ${saved.source.network}`);
  }

  const candles = await getCandles(flags);
  const fs = buildFeatures(candles, {
    window: saved.window,
    useVolatility: saved.names.includes("realizedVol"),
    useVolume: saved.names.includes("volumeSurge"),
  });
  const Z = applyScaler(fs.X, fs.T, fs.D, scaler);

  const muRet = Array.from({ length: p.K }, (_, k) => unscale(p.mu[k * p.D], 0, scaler));
  const labels = labelStates(muRet);
  let now: number[], next: number[];
  if (savedType === "hsmm") {
    const fl = filterHsmm(Z, fs.T, p as HsmmParams);
    now = Array.from({ length: p.K }, (_, k) => fl.stateProb[(fs.T - 1) * p.K + k]);
    next = Array.from({ length: p.K }, (_, k) => fl.nextProb[(fs.T - 1) * p.K + k]);
  } else {
    const { alpha } = filter(Z, fs.T, p as HmmParams);
    now = Array.from({ length: p.K }, (_, k) => alpha[(fs.T - 1) * p.K + k]);
    next = Array.from(predictNext(alpha, (fs.T - 1) * p.K, p as HmmParams));
  }

  let expR = 0;
  for (let k = 0; k < p.K; k++) expR += next[k] * muRet[k];

  console.log(`\nLast bar: close ${candles[candles.length - 1].close}   (${savedType.toUpperCase()})`);
  console.log("\n  state   label        P(now)   P(next bar)   mean ret");
  for (let k = 0; k < p.K; k++) {
    console.log(`    ${k}     ${labels[k].padEnd(10)} ${pct(now[k]).padStart(7)}  ${pct(next[k]).padStart(11)}   ${(muRet[k] * 10_000).toFixed(1).padStart(7)}bps`);
  }
  const cost = num(f, "cost", 30) / 10_000;
  const entry = num(f, "entry", num(f, "cost", 30) * 2) / 10_000;
  console.log(`\n  expected next-bar return: ${bps(expR)}`);
  console.log(`  round-trip cost:          ${bps(cost * 2)}`);
  const signal = expR > Math.max(entry, cost * 2) ? "LONG" : expR < -Math.max(entry, cost * 2) && bool(f, "short") ? "SHORT" : "FLAT";
  console.log(`  signal:                   ${signal}`);
  if (signal !== "FLAT" && Math.abs(expR) < cost * 2) {
    console.log("  note: edge is smaller than the round trip — not worth taking.");
  }
}

/**
 * Sweep the cost assumption and report where the edge dies.
 *
 * Any regime model looks profitable at zero fees. The only question that
 * decides whether it is tradeable is how much friction the signal survives,
 * so the entry threshold tracks cost at 2x per side across the sweep.
 */
function costSweep(
  candles: Candle[],
  featureCfg: FeatureConfig,
  base: StrategyConfig,
  wf: WalkForwardConfig,
  levels = [0, 5, 10, 20, 30, 50],
): { breakeven: number; at10: ReturnType<typeof walkForward> | null } {
  console.log("\n  cost/side   entry    trades   exposure    return   sharpe   maxDD");
  let breakeven = -1;
  let at10: ReturnType<typeof walkForward> | null = null;

  for (const cost of levels) {
    const r = walkForward(candles, featureCfg, { ...base, costBps: cost, entryBps: 2 * cost }, wf);
    console.log(
      `  ${String(cost).padStart(6)}bps  ${String(2 * cost).padStart(4)}bps  ` +
      `${String(r.metrics.trades).padStart(7)}  ${pct(r.metrics.exposure).padStart(8)}  ` +
      `${pct(r.metrics.totalReturn).padStart(9)}  ${r.metrics.sharpe.toFixed(2).padStart(6)}  ` +
      `${pct(r.metrics.maxDrawdown).padStart(6)}`,
    );
    if (r.metrics.totalReturn > 0) breakeven = cost;
    if (cost === 10) at10 = r;
  }

  console.log(
    breakeven < 0
      ? "\n  No cost level produced a positive return — no edge in this signal here."
      : breakeven === 0
        ? "\n  Profitable only at zero cost. The signal is real but smaller than any" +
          "\n  realistic fee, so as specified it is not tradeable."
        : `\n  Edge survives to roughly ${breakeven}bps per side, and dies above that.`,
  );
  console.log("  Meme coin execution (DEX fee + priority fee + slippage on thin books)");
  console.log("  routinely costs more than that. A regime model is not by itself a strategy.");
  return { breakeven, at10 };
}

/**
 * Walk-forward, then ask whether the timing actually carried information.
 * A long-only strategy in a rising series makes money with no skill at all;
 * this is the check that separates the two.
 */
async function cmdValidate(f: Flags) {
  const ds = await getDataSet(f);
  const featureCfg = {
    window: num(f, "window", DEFAULT_WINDOW),
    useVolatility: !bool(f, "no-vol"),
    useVolume: !bool(f, "no-volume"),
  };
  const wf = wfFrom(f, ds.barsPerYear);
  const costs = str(f, "costs", "0,10,30")!.split(",").map(Number);

  console.log(`\nValidation on ${ds.label} — ${wf.modelType!.toUpperCase()}, ${ds.candles.length} candles`);
  console.log(`  null hypothesis: only time-in-market matters, timing is worthless`);
  console.log(`  ${num(f, "trials", 1000)} shuffles preserve exposure exactly and destroy only the order\n`);
  console.log("  cost      ROI    trades  expos   median random   5th..95th        p-value");

  for (const cost of costs) {
    const r = walkForward(ds.candles, featureCfg,
      cost === 0 ? { ...strategyFrom(f), costBps: 0, entryBps: 0 } : { ...strategyFrom(f), costBps: cost }, wf);
    const ts = tradedSeries(ds.candles, featureCfg, r.positions, wf.trainSize!);
    const perm = permutationTest(ts.positions, ts.returns,
      { trials: num(f, "trials", 1000), seed: num(f, "seed", 99), costBps: cost });
    // Beating random timing and making money are different claims. A strategy
    // can lose less than chance would and still be worthless.
    const flag = perm.pValue >= 0.05
      ? "  not distinguishable from luck"
      : perm.actualReturn > 0
        ? "  REAL TIMING SKILL"
        : "  beats random timing, still loses money";
    console.log(
      `  ${String(cost).padStart(3)}bps ${pct(perm.actualReturn).padStart(9)} ${String(r.metrics.trades).padStart(6)}  ` +
      `${pct(perm.exposure).padStart(5)}  ${pct(perm.medianRandom).padStart(13)}   ` +
      `${pct(perm.p05).padStart(8)}..${pct(perm.p95).padStart(8)}  ${perm.pValue.toFixed(3).padStart(6)}${flag}`,
    );
  }
  console.log("\n  A p-value above 0.05 means the return is exposure, not skill — however good");
  console.log("  it looks. This is the test that a rising market will otherwise pass for you.");
}

/**
 * What perfect foresight would earn on this series. If even the oracle cannot
 * clear your cost assumption, the series holds no tradeable regime signal at
 * that cost and no model improvement will change it.
 */
async function cmdCeiling(f: Flags) {
  const ds = await getDataSet(f);
  const featureCfg = {
    window: num(f, "window", DEFAULT_WINDOW),
    useVolatility: !bool(f, "no-vol"),
    useVolume: !bool(f, "no-volume"),
  };
  const costs = str(f, "costs", "0,10,30,50")!.split(",").map(Number);
  const holds = str(f, "holds", "1,5,20")!.split(",").map(Number);
  const skip = num(f, "train", 1500);

  const res = ceilingAnalysis(ds.candles, featureCfg, { costs, holds, skip });
  console.log(`\nCeiling analysis on ${ds.label} — ${res.bars} bars, buy & hold ${pct(res.buyHold)}`);
  console.log("  These use future information. They are not strategies; they are upper bounds.\n");
  console.log("  oracle                trades  expos" + costs.map((c) => `${c}bps`.padStart(12)).join(""));

  const byName = new Map<string, typeof res.rows>();
  for (const r of res.rows) {
    if (!byName.has(r.name)) byName.set(r.name, []);
    byName.get(r.name)!.push(r);
  }
  for (const [name, rows] of byName) {
    const first = rows[0];
    console.log(
      `  ${name.padEnd(20)} ${String(first.trades).padStart(6)}  ${pct(first.exposure).padStart(5)}` +
      rows.map((r) => pct(r.roi).padStart(12)).join(""),
    );
  }
  console.log("\n  'next bar' bounds any one-step-ahead model, and trades every bar, so costs");
  console.log("  hit it hardest. 'hold k bars' is the honest ceiling for a regime model, which");
  console.log("  is meant to make few durable calls. If that row is negative at your cost,");
  console.log("  stop modelling — the money is not there.");
}

/**
 * Top-down multi-timeframe confluence: bias / setup / trigger.
 *
 * Its value is turnover, not prediction. A single 5m model fires hundreds of
 * trades and cannot survive a 30bps round trip; demanding that three
 * timeframes agree cuts that by roughly 40x, which is what makes trading at
 * realistic cost possible at all. Whether the trades are any GOOD is a separate
 * question — check `vs hold` and the p-value before believing the ROI.
 */
async function cmdConfluence(f: Flags) {
  const ds = await getDataSet(f);
  const featureCfg = {
    window: num(f, "window", DEFAULT_WINDOW),
    useVolatility: !bool(f, "no-vol"),
    useVolume: !bool(f, "no-volume"),
  };
  const cost = num(f, "cost", 30);
  const trainSize = num(f, "train", 2000);
  const testSize = num(f, "test", 500);

  // Factors are in base bars: with 5m data, 12 = 1h and 3 = 15m.
  const factors = str(f, "factors", "12,3,1")!.split(",").map(Number);
  if (factors.length !== 3) throw new Error("--factors needs exactly three values, slow to fast (e.g. 12,3,1)");
  const stack: TimeframeSpec[] = [
    { label: `${factors[0]}x`, factor: factors[0], role: "bias" },
    { label: `${factors[1]}x`, factor: factors[1], role: "setup" },
    { label: `${factors[2]}x`, factor: factors[2], role: "trigger" },
  ];

  console.log(`\nConfluence on ${ds.label}`);
  console.log(`  bias ${factors[0]}x  ->  setup ${factors[1]}x  ->  trigger ${factors[2]}x base bars`);
  console.log(`  ${modelType(f).toUpperCase()}, ${num(f, "states", 3)} states, ${cost}bps/side, train ${trainSize} / test ${testSize}`);

  const conf = walkForwardConfluence(ds.candles, stack, {
    featureConfig: featureCfg,
    states: num(f, "states", 3),
    modelType: modelType(f),
    maxDuration: num(f, "max-duration", 30),
    seed: num(f, "seed", 42),
    restarts: num(f, "restarts", 2),
    trainSize, testSize,
  }, {
    gateBps: num(f, "gate", 0),
    // Deliberately not the cost: a confluence trade is held for many bars, so
    // the edge accumulates. Gating the trigger on the full round trip stops
    // every trade before it starts.
    triggerBps: num(f, "trigger", 0),
    biasConfidence: num(f, "bias-confidence", 0),
    exitOnBiasFlip: !bool(f, "no-flip-exit"),
    allowShort: bool(f, "short"),
  });

  const trades = extractTrades(ds.candles, featureCfg, conf.positions, cost, conf.firstTradableRow);
  const st = summarizeTrades(trades);
  const ts = tradedSeries(ds.candles, featureCfg, conf.positions, conf.firstTradableRow);
  const exposure = ts.positions.filter((p) => Math.abs(p) > 1e-9).length / ts.positions.length;
  let bh = 1;
  for (const r of ts.returns) bh *= 1 + r;

  console.log(`\n  trades          ${st.trades}`);
  console.log(`  win rate        ${st.trades ? pct(st.winRate) : "n/a"}`);
  console.log(`  ROI             ${pct(st.roi)}`);
  console.log(`  buy & hold      ${pct(bh - 1)}`);
  console.log(`  vs hold         ${pct(st.roi - (bh - 1))}`);
  console.log(`  time in market  ${pct(exposure)}`);
  console.log(`  avg hold        ${st.trades ? st.avgBarsHeld.toFixed(0) + " bars" : "n/a"}`);
  console.log(`  profit factor   ${st.trades ? (st.profitFactor === Infinity ? "inf" : st.profitFactor.toFixed(2)) : "n/a"}`);

  if (st.trades > 0 && !bool(f, "no-validate")) {
    const perm = permutationTest(ts.positions, ts.returns,
      { trials: num(f, "trials", 1000), seed: num(f, "seed", 99), costBps: cost });
    console.log(`\n  timing test (${perm.method} null, turnover preserved): p = ${perm.pValue.toFixed(3)}`);
    console.log(`  median random timing at the same exposure: ${pct(perm.medianRandom)}`);
    console.log(perm.pValue < 0.05
      ? "  Significant at 0.05 — but if you are scanning many tokens, divide that\n  threshold by how many you tried before believing it."
      : "  Not distinguishable from luck. The ROI is exposure, not timing.");
  }

  const showN = num(f, "show", 10);
  if (trades.length > 0) {
    console.log(`\n  last ${Math.min(showN, trades.length)} trades`);
    console.log("    dir   entry (UTC)      exit (UTC)       held    gross      net");
    for (const t of trades.slice(-showN)) {
      const fmt = (x: number) => new Date(x * 1000).toISOString().slice(5, 16).replace("T", " ");
      console.log(`    ${(t.direction > 0 ? "LONG" : "SHORT").padEnd(5)} ${fmt(t.entryTime)}    ${fmt(t.exitTime)}  ` +
        `${(t.barsHeld + "b").padStart(6)}  ${pct(t.grossReturn).padStart(7)}  ${pct(t.netReturn).padStart(7)}${t.win ? "" : "  L"}`);
    }
  }
}

/**
 * Dollar P&L for a real leveraged account, rather than a percentage.
 *
 * Three things only show up once the account is modelled: fees land on notional
 * (at 2x a 4.5bps taker costs 9bps of equity per side), funding accrues hourly
 * on that same notional, and liquidation depends on the worst tick inside the
 * trade rather than the close.
 */
async function cmdAccount(f: Flags) {
  const coin = str(f, "coin");
  if (!coin) throw new Error("--coin <SYMBOL> required (Hyperliquid perp)");
  const ds = await getDataSet(f);
  const featureCfg = {
    window: num(f, "window", DEFAULT_WINDOW),
    useVolatility: !bool(f, "no-vol"),
    useVolume: !bool(f, "no-volume"),
  };
  const equity = num(f, "equity", 50);
  const leverage = num(f, "leverage", 2);
  const takerBps = num(f, "cost", hl.HL_TAKER_BPS);
  const trainSize = num(f, "train", 3000);
  const testSize = num(f, "test", 400);
  const days = num(f, "days", 14);

  const markets = await hl.topMarkets(200);
  const market = markets.find((m) => m.coin === coin.toUpperCase());
  const mmf = hl.maintenanceMarginFraction(market?.maxLeverage ?? 10);

  const lastTime = ds.candles[ds.candles.length - 1].time;
  const funding = await hl.fetchFunding(coin.toUpperCase(), (lastTime - (days + 7) * 86_400) * 1000);

  const positions = bool(f, "single")
    ? walkForward(ds.candles, featureCfg, { costBps: takerBps, durationAware: true },
        { ...wfFrom(f, ds.barsPerYear), trainSize, testSize }).positions
    : walkForwardConfluence(ds.candles, DEFAULT_STACK,
        { featureConfig: featureCfg, states: num(f, "states", 3), modelType: modelType(f),
          seed: num(f, "seed", 42), restarts: num(f, "restarts", 2), trainSize, testSize },
        { gateBps: 0, triggerBps: 0, exitOnBiasFlip: true }).positions;

  const fs = buildFeatures(ds.candles, featureCfg);
  const cutoff = lastTime - days * 86_400;
  let from = trainSize;
  for (let i = 0; i < fs.T; i++) {
    if (ds.candles[fs.index[i]].time >= cutoff) { from = Math.max(i, trainSize); break; }
  }

  const perpCfg = { startingEquity: equity, leverage, takerBps,
                    maintenanceMarginFraction: mmf, minOrderUsd: num(f, "min-order", 10) };
  const strat = simulatePerp(ds.candles, featureCfg, positions, funding, perpCfg, from);
  const hold = simulatePerp(ds.candles, featureCfg, new Array(fs.T).fill(1), funding, perpCfg, from);

  const money = (x: number) => `$${x.toFixed(2)}`;
  console.log(`\n${coin.toUpperCase()}-PERP — last ${days} days, $${equity} account at ${leverage}x`);
  console.log(`  ${takerBps}bps taker on notional, hourly funding, liquidation at ${(mmf * 100).toFixed(2)}% maintenance`);
  console.log(`  strategy: ${bool(f, "single") ? "single timeframe" : "3TF confluence"}\n`);

  console.log(`  trades          ${strat.trades.length}` +
    (strat.trades.length ? `  (${pct(strat.trades.filter((t) => t.win).length / strat.trades.length)} win rate)` : ""));
  console.log(`  starting equity ${money(equity)}`);
  console.log(`  final equity    ${money(strat.finalEquity)}   ${strat.finalEquity >= equity ? "+" : ""}${money(strat.finalEquity - equity)}  (${pct(strat.finalEquity / equity - 1)})`);
  console.log(`  fees paid       ${money(strat.totalFees)}`);
  console.log(`  funding paid    ${money(strat.totalFunding)}`);
  console.log(`  max drawdown    ${pct(strat.maxDrawdown)}`);
  console.log(`  liquidated      ${strat.liquidated ? `YES at ${new Date(strat.liquidationTime! * 1000).toISOString().slice(0, 16)}` : "no"}`);
  console.log(`\n  ${leverage}x buy & hold  ${money(hold.finalEquity)}   ${hold.finalEquity >= equity ? "+" : ""}${money(hold.finalEquity - equity)}  (${pct(hold.finalEquity / equity - 1)})${hold.liquidated ? "  LIQUIDATED" : ""}`);
  console.log(`  vs hold         ${strat.finalEquity >= hold.finalEquity ? "+" : ""}${money(strat.finalEquity - hold.finalEquity)}`);

  if (strat.trades.length > 0) {
    console.log("\n    dir   entry (UTC)      exit (UTC)       held   notional     P&L   equity");
    for (const t of strat.trades.slice(-num(f, "show", 10))) {
      const fmt = (x: number) => new Date(x * 1000).toISOString().slice(5, 16).replace("T", " ");
      console.log(`    ${(t.direction > 0 ? "LONG" : "SHORT").padEnd(5)} ${fmt(t.entryTime)}    ${fmt(t.exitTime)}  ` +
        `${(t.barsHeld + "b").padStart(5)}  ${money(t.notional).padStart(8)}  ${(t.netPnl >= 0 ? "+" : "") + money(t.netPnl).padStart(6)}  ` +
        `${money(t.equityAfter).padStart(7)}${t.liquidated ? "  LIQ" : ""}`);
    }
  }
}

/**
 * Bayesian posterior over the model.
 *
 * Every other command plugs EM's point estimates into the signal as if they
 * were known. This puts credible intervals on them and, more usefully, on the
 * signal itself — so the question becomes "what is the probability the expected
 * move clears the round trip", which is the one a trader actually has.
 *
 * Two engines answer it. Variational EM (default) optimizes a bound on the
 * evidence and reports i.i.d. draws from the fitted approximation; `--gibbs`
 * runs the sampler instead, which is slower and asymptotically exact. Run both
 * when a number is about to matter: mean-field variational intervals are known
 * to come out too narrow, and narrow is the direction that flatters a signal.
 */
async function cmdPosterior(f: Flags) {
  const ds = await getDataSet(f);
  const featureCfg = {
    window: num(f, "window", DEFAULT_WINDOW),
    useVolatility: !bool(f, "no-vol"),
    useVolume: !bool(f, "no-volume"),
  };
  const costBps = num(f, "cost", hl.HL_TAKER_BPS);
  const roundTrip = (2 * costBps) / 10_000;
  const useGibbs = bool(f, "gibbs");
  const kappa0 = num(f, "kappa0", DEFAULT_PRIORS.kappa0);
  const draws = num(f, "draws", 4000);

  const fs = buildFeatures(ds.candles, featureCfg);
  const scaler = fitScaler(fs.X, fs.T, fs.D);
  const Z = applyScaler(fs.X, fs.T, fs.D, scaler);

  // Let the ELBO pick K when asked. Only variational EM can do this — the
  // sampler has no comparable quantity, so --select-states forces the engine.
  let states = num(f, "states", 3);
  let selection: ReturnType<typeof selectStates> | null = null;
  const selectSpec = str(f, "select-states");
  if (selectSpec !== undefined) {
    const candidates = String(selectSpec).split(",").map((x) => Number(x.trim())).filter((x) => x >= 2);
    console.log(`\nSelecting the number of regimes by ELBO over K = ${candidates.join(", ")}`);
    selection = selectStates(Z, fs.T, fs.D, candidates, {
      seed: num(f, "seed", 11), priors: { kappa0 }, draws: 0,
    });
    console.log("    K    ELBO/bar    occupied states");
    for (const row of selection.scores) {
      const mark = row.states === selection.best.K ? "  <-" : "";
      console.log(`   ${String(row.states).padStart(2)}  ${row.elboPerBar.toFixed(5).padStart(10)}  ${String(row.occupied).padStart(10)}${mark}`);
    }
    states = selection.best.K;
  }

  // Warm-start from EM. The sampler mixes far better from a sensible
  // segmentation; variational EM lands in the same place either way but gets
  // there in a handful of iterations instead of a few dozen.
  const em = fit(Z, fs.T, fs.D, { states, restarts: num(f, "restarts", 4), seed: num(f, "seed", 42) });

  console.log(`\nPosterior for ${ds.label} — ${states} states, ${fs.T} bars`);
  let res: PosteriorDraws;
  if (useGibbs) {
    const iterations = num(f, "iterations", 1200);
    console.log(`  Gibbs: ${iterations} sweeps, ${num(f, "burn-in", Math.floor(iterations / 2))} burn-in, thin ${num(f, "thin", 3)}`);
    console.log(`  prior kappa0 = ${kappa0} (shrinkage toward zero drift)\n`);
    res = runMcmc(Z, fs.T, fs.D, {
      states, iterations,
      burnIn: num(f, "burn-in", Math.floor(iterations / 2)),
      thin: num(f, "thin", 3),
      seed: num(f, "seed", 11),
      init: em.params,
      priors: { kappa0 },
    });
  } else {
    const vb = fitVb(Z, fs.T, fs.D, {
      states, init: em.params, draws,
      maxIter: num(f, "iterations", 300),
      seed: num(f, "seed", 11),
      priors: { kappa0 },
    });
    const pruned = Array.from(vb.occupancy).filter((n) => n <= 0.01 * fs.T).length;
    console.log(`  variational EM: ELBO/bar ${(vb.elbo / fs.T).toFixed(5)} after ${vb.iterations} iterations` +
      ` (${vb.converged ? "converged" : "hit the iteration cap"}), ${draws} independent draws`);
    if (pruned > 0) {
      console.log(`  ${pruned} of ${states} states carry almost no occupancy — the data does not support them.`);
    }
    console.log(`  prior kappa0 = ${kappa0} (shrinkage toward zero drift)\n`);
    res = vb;
  }

  const un = (v: number) => unscale(v, 0, scaler);
  const ints = posteriorStateMeans(res, 0, un, 0.9);
  const durs = posteriorDurations(res, 0.9);
  const b = (x: number) => (x * 10_000).toFixed(1);

  console.log("  state   EM point   posterior mean     90% credible interval    P(mu>0)   dwell");
  for (let k = 0; k < states; k++) {
    console.log(
      `    ${k}    ${b(un(em.params.mu[k * fs.D])).padStart(8)}bps ${b(ints[k].mean).padStart(12)}bps  ` +
      `[${b(ints[k].lower).padStart(8)}, ${b(ints[k].upper).padStart(7)}]bps  ` +
      `${(ints[k].pPositive * 100).toFixed(0).padStart(5)}%  ${durs[k].median.toFixed(0).padStart(5)}b`);
  }

  const top = states - 1;
  const width = ints[top].upper - ints[top].lower;
  console.log(`\n  bullish-state interval is ${b(width)}bps wide, against a ${(roundTrip * 10_000).toFixed(1)}bps round trip` +
    `  (${(width / roundTrip).toFixed(1)}x)`);
  if (ints[top].upper < roundTrip) {
    console.log("  even the optimistic end of that interval does not clear the round trip.");
  }

  // What the model would actually trade right now.
  const { alpha } = filter(Z, fs.T, em.params);
  const nx = predictNext(alpha, (fs.T - 1) * states, em.params);
  const sig = posteriorSignal(res, Array.from(nx), un, roundTrip, 0.9);
  console.log(`\n  Signal at the last bar (blended over the state belief)`);
  console.log(`    per bar        median ${b(sig.perBar.median)}bps   90% CI [${b(sig.perBar.lower)}, ${b(sig.perBar.upper)}]bps`);
  console.log(`    x ${sig.medianHold.toFixed(1)} bars held  ->  ${b(sig.median)}bps   90% CI [${b(sig.lower)}, ${b(sig.upper)}]bps`);
  console.log(`    P(edge > 0)                    ${(sig.pPositive * 100).toFixed(1)}%`);
  console.log(`    P(edge x hold > round trip)    ${(sig.pAboveCost * 100).toFixed(1)}%`);
  console.log(sig.pAboveCost < 0.5
    ? "    => not worth taking: the posterior does not favour clearing costs."
    : "    => the posterior favours clearing costs.");

  if (!useGibbs) {
    console.log("\n  These intervals come from a mean-field approximation, which is biased");
    console.log("  toward being too narrow. Re-run with --gibbs before acting on a marginal");
    console.log("  call; if the two disagree, believe the sampler.");
  }

  // Where the uncertainty lives: knowing the state would be worth a lot more.
  const dwell = durs[top].median;
  console.log(`\n  If the state were known with certainty, holding the bullish state for its`);
  console.log(`  median ${dwell.toFixed(0)} bars would be worth ~${b(ints[top].mean * dwell)}bps against ${(roundTrip * 10_000).toFixed(1)}bps of cost.`);
  console.log(`  The gap between that and the blended signal above is state uncertainty,`);
  console.log(`  not parameter uncertainty.`);
}

const WINDOWS: [string, number][] = [
  ["24h", 1], ["5d", 5], ["1w", 7], ["2w", 14],
];

/**
 * The trade ledger, plus what each lookback window would have returned.
 *
 * Windows are measured back from the LAST BAR IN THE DATA, not wall clock, so
 * the answer does not drift with when you happen to run it. A trade belongs to
 * a window if it was ENTERED inside it — the question is "what if I started
 * taking these signals N days ago".
 */
async function cmdTrades(f: Flags) {
  const ds = await getDataSet(f);
  const featureCfg = {
    window: num(f, "window", DEFAULT_WINDOW),
    useVolatility: !bool(f, "no-vol"),
    useVolume: !bool(f, "no-volume"),
  };
  const cost = num(f, "cost", 30);
  const wf = wfFrom(f, ds.barsPerYear);
  const r = walkForward(ds.candles, featureCfg, strategyFrom(f), wf);
  const trades = extractTrades(ds.candles, featureCfg, r.positions, cost, wf.trainSize!);

  const last = ds.candles[ds.candles.length - 1].time;
  const priceAt = (t: number) => {
    // Closest candle at or before t.
    let best = ds.candles[0];
    for (const c of ds.candles) { if (c.time <= t) best = c; else break; }
    return best.close;
  };

  console.log(`\n${"=".repeat(92)}`);
  console.log(`${ds.label}  —  ${wf.modelType!.toUpperCase()}, ${cost}bps/side, entry ${strategyFrom(f).entryBps ?? 2 * cost}bps`);
  console.log(`data ends ${new Date(last * 1000).toISOString().slice(0, 16).replace("T", " ")} UTC`);
  console.log("=".repeat(92));

  console.log("\n  window   trades   wins   win rate       ROI    avg hold    best     worst   buy&hold");
  for (const [label, days] of WINDOWS) {
    const cutoff = last - days * 86_400;
    const inWin = trades.filter((t) => t.entryTime >= cutoff);
    const st = summarizeTrades(inWin);
    const bh = priceAt(last) / priceAt(cutoff) - 1;
    console.log(
      `  ${label.padEnd(7)} ${String(st.trades).padStart(6)} ${String(st.wins).padStart(6)}  ` +
      `${(st.trades ? pct(st.winRate) : "  n/a").padStart(8)}  ${pct(st.roi).padStart(9)}  ` +
      `${(st.trades ? st.avgBarsHeld.toFixed(0) + " bars" : "-").padStart(9)}  ` +
      `${(st.trades ? pct(st.bestTrade) : "-").padStart(7)}  ${(st.trades ? pct(st.worstTrade) : "-").padStart(8)}  ${pct(bh).padStart(9)}`,
    );
  }

  const all = summarizeTrades(trades);
  console.log(`\n  full out-of-sample period: ${all.trades} trades, ${pct(all.winRate)} win rate, ` +
    `${pct(all.roi)} ROI, profit factor ${all.profitFactor === Infinity ? "inf" : all.profitFactor.toFixed(2)}`);

  const showN = num(f, "show", 15);
  const recent = trades.slice(-showN);
  if (recent.length > 0) {
    console.log(`\n  last ${recent.length} trades`);
    console.log("    dir   entry (UTC)        exit (UTC)         held    entry px     exit px    gross     net");
    for (const t of recent) {
      const fmt = (x: number) => new Date(x * 1000).toISOString().slice(5, 16).replace("T", " ");
      console.log(
        `    ${(t.direction > 0 ? "LONG" : "SHORT").padEnd(5)} ${fmt(t.entryTime)}      ${fmt(t.exitTime)}   ` +
        `${(t.barsHeld + "b").padStart(5)}  ${price(t.entryPrice).padStart(10)}  ${price(t.exitPrice).padStart(10)}  ` +
        `${pct(t.grossReturn).padStart(7)}  ${pct(t.netReturn).padStart(7)}${t.win ? "" : "  L"}`,
      );
    }
  } else {
    console.log("\n  no trades were taken in the out-of-sample period.");
  }
}

/**
 * Top tokens on a network, screened for whether they can actually be modelled.
 *
 * Ranked by the geometric mean of liquidity and 24h volume, deduped to one pool
 * per base token. Three filters do the real work:
 *   - a liquidity floor, because volume without depth is not a market
 *   - a turnover ceiling, because volume many multiples of liquidity is wash
 *     trading (the worst offender seen was $106M of volume on $0.0000016 of it)
 *   - an age floor, because a pool minutes old has no history to fit
 */
async function cmdTop(f: Flags) {
  if (str(f, "source") === "hyperliquid" || bool(f, "hl")) {
    const limit = num(f, "limit", 10);
    const markets = await hl.topMarkets(limit);
    console.log(`\n${"=".repeat(88)}`);
    console.log(`TOP ${limit} HYPERLIQUID PERPS by 24h notional volume`);
    console.log("=".repeat(88));
    console.log("   #  coin       mark price      24h volume    open interest   24h%   funding/8h   maxLev");
    markets.forEach((m, i) => {
      console.log(
        `  ${String(i + 1).padStart(2)}  ${m.coin.padEnd(9)} ${price(m.markPrice).padStart(11)}  ` +
        `${usd(m.dayNotionalVolume).padStart(12)}  ${usd(m.openInterestUsd).padStart(13)}  ` +
        `${(m.change24h * 100).toFixed(1).padStart(5)}  ${(m.funding * 8 * 100).toFixed(4).padStart(9)}%  ${String(m.maxLeverage).padStart(5)}x`,
      );
    });
    console.log(`\n  base fees: ${hl.HL_TAKER_BPS}bps taker / ${hl.HL_MAKER_BPS}bps maker per side` +
      ` — roughly a tenth of a DEX meme coin round trip`);
    console.log(`\n  next:  bun run src/cli.ts ceiling --coin ${markets[0]?.coin ?? "BTC"} --cost ${hl.HL_TAKER_BPS}`);
    return;
  }
  const networks = str(f, "networks", str(f, "network", "solana"))!.split(",").map((n) => n.trim());
  const limit = num(f, "limit", 10);
  const minLiq = num(f, "min-liquidity", 250_000);
  const minVol = num(f, "min-volume", 100_000);
  const maxTurnover = num(f, "max-turnover", 50);
  const minAge = num(f, "min-age-days", 7);
  const showAll = bool(f, "no-filter");

  for (const net of networks) {
    const resolved = resolveNetwork(net);
    const all = await topPools(resolved, num(f, "pages", 3));

    const rejected: string[] = [];
    const kept: typeof all = [];
    const seenToken = new Set<string>();
    for (const p of all) {
      if (!showAll) {
        if (p.liquidityUsd < minLiq) { rejected.push(`${p.baseSymbol}: thin (${usd(p.liquidityUsd)} liq)`); continue; }
        if (p.volume24h < minVol) { rejected.push(`${p.baseSymbol}: quiet (${usd(p.volume24h)} vol)`); continue; }
        if (p.turnover > maxTurnover) { rejected.push(`${p.baseSymbol}: ${p.turnover.toFixed(0)}x turnover`); continue; }
        if (p.ageDays !== null && p.ageDays < minAge) { rejected.push(`${p.baseSymbol}: ${p.ageDays.toFixed(1)}d old`); continue; }
      }
      // One pool per token — the deepest, since the list is already sorted.
      const key = p.baseSymbol.toLowerCase();
      if (seenToken.has(key)) continue;
      seenToken.add(key);
      kept.push(p);
      if (kept.length >= limit) break;
    }

    console.log(`\n${"=".repeat(96)}`);
    console.log(`TOP ${limit} on ${resolved}` +
      (showAll ? "  (unfiltered)" : `  (>${usd(minLiq)} liq, >${usd(minVol)} vol, <${maxTurnover}x turnover, >${minAge}d old)`));
    console.log("=".repeat(96));
    if (kept.length === 0) {
      console.log("  nothing cleared the filters. Loosen them or try --no-filter.");
      continue;
    }
    console.log("   #  token        pair                liquidity   24h vol   turnover    age   pool address");
    kept.forEach((p, i) => {
      console.log(
        `  ${String(i + 1).padStart(2)}  ${p.baseSymbol.padEnd(12)} ${(p.baseSymbol + "/" + p.quoteSymbol).padEnd(19)} ` +
        `${usd(p.liquidityUsd).padStart(9)} ${usd(p.volume24h).padStart(9)}  ` +
        `${(p.turnover.toFixed(1) + "x").padStart(8)}  ${(p.ageDays !== null ? p.ageDays.toFixed(0) + "d" : "?").padStart(5)}   ${p.address}`,
      );
    });
    if (rejected.length > 0 && bool(f, "verbose")) {
      console.log(`\n  screened out (${rejected.length}): ${rejected.slice(0, 12).join(", ")}${rejected.length > 12 ? " ..." : ""}`);
    } else if (rejected.length > 0) {
      console.log(`\n  ${rejected.length} pools screened out (--verbose to see why)`);
    }
    console.log(`\n  next:  bun run src/cli.ts ceiling --network ${resolved} --pool ${kept[0].address} --timeframe 1h --bars 3000`);
  }
}

/** List candidate pools for a token, deepest liquidity first. */
async function cmdSearch(f: Flags) {
  const query = str(f, "token", str(f, "q"));
  if (!query) throw new Error("--token <symbol or address> required");
  const network = f["network"] ? resolveNetwork(str(f, "network")!) : undefined;
  const minLiq = num(f, "min-liquidity", 10_000);

  console.log(`\nDexScreener search: "${query}"${network ? ` on ${network}` : " (all chains)"}, ` +
    `min ${usd(minLiq)} liquidity / ${usd(num(f, "min-volume", 1_000))} 24h volume`);
  const pairs = await searchPairs(query, network, minLiq, num(f, "min-volume", 1_000));
  if (pairs.length === 0) {
    console.log("no pools matched. Lower --min-liquidity / --min-volume, or drop --network.");
    return;
  }

  console.log("\n  chain      pair                 dex          liquidity   24h vol     age   pool address");
  for (const p of pairs.slice(0, num(f, "limit", 15))) {
    const pair = `${p.baseSymbol}/${p.quoteSymbol}`;
    console.log(
      `  ${p.chain.padEnd(10)} ${pair.padEnd(20)} ${p.dex.padEnd(12)} ` +
      `${usd(p.liquidityUsd).padStart(9)}  ${usd(p.volume24h).padStart(8)}  ` +
      `${(p.ageDays !== null ? `${p.ageDays.toFixed(0)}d` : "?").padStart(5)}   ${p.pairAddress}`,
    );
  }
  console.log("\nSymbols are attacker-controlled and impersonation is routine — pick by");
  console.log("liquidity and age, not by name. Then:");
  console.log(`  bun run src/cli.ts backtest --network ${pairs[0].network} --pool ${pairs[0].pairAddress}`);
}

/** GeckoTerminal trending pools — a starting point when you have no ticker in mind. */
async function cmdTrending(f: Flags) {
  const network = resolveNetwork(str(f, "network", "solana")!);
  console.log(`\nTrending pools on ${network} (GeckoTerminal)`);
  const pools = await trendingPools(network, num(f, "limit", 15));
  console.log("\n  pair                                 price    24h vol   liquidity    24h%   pool address");
  for (const p of pools) {
    console.log(
      `  ${p.name.padEnd(32)} ${price(p.priceUsd).padStart(10)}  ` +
      `${usd(p.volume24h).padStart(9)}  ${usd(p.liquidityUsd).padStart(9)}  ` +
      `${p.change24h.toFixed(1).padStart(6)}   ${p.address}`,
    );
  }
  console.log("\nTrending means high recent volume, not tradeable. Most of these are");
  console.log("minutes old with no history to fit — check the bar count before trusting a fit.");
}

/** Fetch candles and write them to CSV. */
async function cmdFetch(f: Flags) {
  if (!f["pool"] && !f["token"]) throw new Error("--pool <address> or --token <symbol> required");
  const ds = await fetchDataSet(f);
  const out = str(f, "out", "candles.csv")!;
  await Bun.write(out, toCsv(ds.candles));
  console.log(`\nwrote ${ds.candles.length} candles -> ${out}`);
  console.log(`  bun run src/cli.ts backtest --csv ${out}`);
}

/** Dump synthetic candles to CSV — doubles as a template for the expected format. */
async function cmdExport(f: Flags) {
  const out = str(f, "out", "sample.csv")!;
  const { candles } = generateSynthetic({ bars: num(f, "bars", 6000), seed: num(f, "seed", 7) });
  await Bun.write(out, toCsv(candles));
  console.log(`wrote ${candles.length} synthetic candles -> ${out}`);
  console.log("columns: time,open,high,low,close,volume (only close is strictly required)");
}

async function cmdDemo(f: Flags) {
  console.log("=".repeat(72));
  console.log("Meme coin HMM — demo on synthetic 3-regime data");
  console.log("=".repeat(72));

  const { candles, trueStates } = generateSynthetic({ bars: num(f, "bars", 6000), seed: num(f, "seed", 7) });
  const window = num(f, "window", DEFAULT_WINDOW);
  const fs = buildFeatures(candles, { window });
  const scaler = fitScaler(fs.X, fs.T, fs.D);
  const Z = applyScaler(fs.X, fs.T, fs.D, scaler);

  console.log(`\n${candles.length} synthetic 5m candles, ${fs.T} feature rows, features: ${fs.names.join(", ")}`);
  const res = fit(Z, fs.T, fs.D, { states: 3, restarts: 6, seed: 42, verbose: bool(f, "verbose") });
  console.log(`log-likelihood/bar: ${(res.logLik / fs.T).toFixed(4)} (${res.iterations} EM iterations)`);

  printStateTable(res.params, scaler, fs.names);

  // The synthetic generator knows the true regimes, so we can score recovery.
  const path = viterbi(Z, fs.T, res.params);
  const truth = new Int32Array(fs.T);
  for (let t = 0; t < fs.T; t++) truth[t] = trueStates[fs.index[t]];
  const { acc, perm } = bestPermutationAccuracy(path, truth, res.params.K);
  const genLabels = ["dump", "chop", "pump"];
  console.log(`\nViterbi recovers the true hidden regime on ${pct(acc)} of bars (chance = ${pct(1 / res.params.K)}).`);
  console.log(`  fitted state -> generator regime: ${perm.map((g, k) => `${k}->${genLabels[g] ?? g}`).join("  ")}`);
  console.log("  (EM state indices are arbitrary, so this is scored over the best relabeling.)");

  console.log("\n" + "-".repeat(72));
  console.log("Walk-forward backtest — this is the number that matters");
  console.log("-".repeat(72));

  const wfBase = { ...wfFrom(f), restarts: 3, verbose: bool(f, "verbose") };
  const { at10: detailed } = costSweep(candles, { window }, strategyFrom(f), wfBase, [0, 5, 10, 20, 30]);

  const r = detailed ?? walkForward(candles, { window }, strategyFrom(f), wfBase);
  console.log(`\n${r.refits} refits, each trading only on parameters fit to earlier bars`);
  printBacktest("Out-of-sample detail @ 10bps/side", r.metrics, r.equity, r.buyHoldEquity);

  const smoothed = posteriors(Z, fs.T, res.params);
  console.log(`\nIn-sample smoothed log-lik/bar ${(smoothed.logLik / fs.T).toFixed(4)} — quoted for model fit only.`);
  console.log("Never trade smoothed states: they use future bars to label the present.");
  console.log("\nNext: run against real candles with  bun run src/cli.ts backtest --csv your.csv");
}

function usage() {
  console.log(`
meme-hmm — a hidden Markov model for meme coin regimes

  bun run src/cli.ts top --networks solana,base       top tokens, screened
  bun run src/cli.ts trending --network solana        what is moving right now
  bun run src/cli.ts search   --token WIF             find the deepest pool
  bun run src/cli.ts fetch    --token WIF --out w.csv pull candles to CSV
  bun run src/cli.ts backtest --token WIF             walk-forward out-of-sample
  bun run src/cli.ts train    --token WIF --out m.json  fit and inspect regimes
  bun run src/cli.ts predict  --token WIF --model m.json  current state + signal
  bun run src/cli.ts confluence --token WIF            3-timeframe scalping stack
  bun run src/cli.ts account --coin SOL --equity 50 --leverage 2   dollar P&L
  bun run src/cli.ts trades   --token WIF              trade ledger + ROI by window
  bun run src/cli.ts posterior  --coin SOL             credible intervals on the edge
  bun run src/cli.ts validate --token WIF              is it skill or exposure?
  bun run src/cli.ts ceiling  --token WIF              is there money to find?
  bun run src/cli.ts demo                             synthetic walkthrough

Model
  --model-type hmm    hmm (geometric dwell) or hsmm (learned durations)
  --hsmm              shorthand for --model-type hsmm
  --max-duration 30   hsmm: longest dwell the duration pmf can represent

Confluence (top-down: bias -> setup -> trigger)
  --factors 12,3,1    bar multiples, slow to fast (5m base: 12=1h, 3=15m)
  --gate 0            bps the bias/setup must clear
  --trigger 0         bps the entry bar must clear (NOT the cost: a held
                      trade accumulates the edge over many bars)
  --bias-confidence 0 extra demand on P(bull state) at the bias timeframe
  --no-flip-exit      hold until the trigger reverses instead of the bias

Diagnostics
  validate            walk-forward, then shuffle the positions to random times
                      at the same exposure. p > 0.05 = the return is exposure,
                      not timing skill.
  ceiling             what perfect foresight earns here. If the oracle cannot
                      clear your cost, no model can, and you should stop.
  --costs 0,10,30     cost levels to report
  --holds 1,5,20      ceiling: bars an oracle commits for
  --trials 1000       validate: number of shuffles

Posterior (Bayesian HMM; variational EM by default)
  --select-states 2,3,4,5   pick the number of regimes by ELBO
  --draws 4000        independent draws from the variational posterior
  --iterations 300    cap on variational EM iterations
  --kappa0 0.5        prior strength pulling state means toward zero drift
  --gibbs             use the sampler instead. Slower, asymptotically exact,
                      and the thing to check a marginal call against — mean-field
                      credible intervals come out too narrow.
  --iterations 1200   --gibbs: sweeps; half are burn-in by default
  --burn-in <n>       --gibbs: override the burn-in
  --thin 3            --gibbs: keep every nth post-burn-in draw

Account simulation (perps only)
  --equity 50         starting margin in USD
  --leverage 2        position notional = equity x this
  --days 14           lookback window to simulate
  --single            use the single-timeframe model instead of confluence
  Fees hit notional, funding accrues hourly, liquidation uses intrabar extremes.

Hyperliquid perps (public API, no key)
  --coin BTC          perp symbol; implies the Hyperliquid source
  --timeframe 5m      1m 3m 5m 15m 30m 1h 2h 4h 8h 12h 1d
  --bars 5000         ~5000 per request, paged backwards
  top --source hyperliquid   top perps by 24h notional volume
  Base fees are 4.5bps taker / 1.5bps maker per side — pass --cost 4.5

Live data (GeckoTerminal + DexScreener, free, no key)
  --token <symbol>    resolve to the deepest pool via DexScreener search
  --pool <address>    exact pool, skips search
  --network solana    solana | eth | base | bsc | arbitrum | polygon | ...
  --timeframe 5m      1m 5m 15m 1h 4h 12h 1d
  --bars 3000         how many to pull (1000 per request, paged)
  --fill              insert flat bars for untraded intervals
  --max-fill 12       longest gap run worth filling
  --min-liquidity     liquidity floor for search (default $10k)
  top: --networks a,b  --limit 10  --min-age-days 7  --max-turnover 50
       --pages 3  --no-filter  --verbose (show what was screened out)
  --min-volume        24h volume floor for search (default $1k)
  --no-cache          bypass the 10-minute disk cache

Data
  --csv <path>        OHLCV csv (needs a close column; volume optional)
  --window 5          lookback for realized vol and the volume baseline
                      (a window longer than a regime cannot see that regime)
  --no-vol            drop the volatility feature
  --no-volume         drop the volume-surge feature

Model
  --states 3          hidden states (3 = dump / chop / pump)
  --duration-aware    test edge x expected hold against the round trip, rather
                      than demanding one bar clear the whole cost
  --confidence 0      position on P(top state) > this instead of expected
                      return; needs --hsmm to be reachable
  --restarts 8        random restarts, best likelihood wins
  --seed 42           reproducibility
  --out model.json    save fitted model (train only)

Strategy
  --entry <bps>       go long above this expected next-bar return
                      (default: 2x --cost, i.e. must clear the round trip)
  --exit 0            exit below this, bps (hysteresis)
  --short             allow short positions
  --cost 30           fee + slippage per side, bps
  --vol-target 0      scale size to a target per-bar vol (e.g. 0.02)
  --no-sweep          skip the cost-sensitivity table in backtest
  --max-pos 1         position cap

Walk-forward
  --train 1500        training window, bars
  --test 500          bars traded per refit
  --bars-per-year     annualization for sharpe (inferred from bar spacing)
  --verbose           per-block diagnostics
`);
}

const { cmd, flags } = parseArgs(Bun.argv.slice(2));
try {
  if (flags["help"] || flags["h"] || cmd === "help") usage();
  else if (cmd === "demo") await cmdDemo(flags);
  else if (cmd === "train") await cmdTrain(flags);
  else if (cmd === "backtest") await cmdBacktest(flags);
  else if (cmd === "predict") await cmdPredict(flags);
  else if (cmd === "export") await cmdExport(flags);
  else if (cmd === "search") await cmdSearch(flags);
  else if (cmd === "trending") await cmdTrending(flags);
  else if (cmd === "fetch") await cmdFetch(flags);
  else if (cmd === "validate") await cmdValidate(flags);
  else if (cmd === "ceiling") await cmdCeiling(flags);
  else if (cmd === "top") await cmdTop(flags);
  else if (cmd === "trades") await cmdTrades(flags);
  else if (cmd === "confluence") await cmdConfluence(flags);
  else if (cmd === "account") await cmdAccount(flags);
  else if (cmd === "posterior") await cmdPosterior(flags);
  else usage();
} catch (e) {
  console.error(`\nerror: ${e instanceof Error ? e.message : String(e)}`);
  process.exit(1);
}
