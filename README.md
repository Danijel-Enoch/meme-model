# meme-hmm

Hidden Markov and hidden semi-Markov models over crypto price regimes, in
TypeScript on Bun. Zero runtime dependencies — forward-backward, Baum-Welch,
Viterbi and the segmental EM recursions are written out directly.

The premise: a token is not one process. It is a few very different processes
taking turns — a low-volatility chop that dominates the sample, a slow bleed,
and a rare, violent, short-lived pump. You never observe which one is running.
An HMM treats that regime as a hidden state, infers a distribution over it from
price and volume, and forecasts which regime comes next.

It works on DEX meme coins and on Hyperliquid perps. **It has not produced a
tradeable edge on either**, and the more interesting half of this repo is the
machinery that establishes that rather than hiding it. See [Results](#results).

## Quick start

```bash
bun install
bun run demo      # synthetic end-to-end walkthrough
bun test          # 112 tests, incl. brute-force validation of both models
```

Hyperliquid perps — clean data, 4.5bps taker, no API key:

```bash
bun run src/cli.ts top --source hyperliquid                  # top perps by volume
bun run src/cli.ts ceiling    --coin SOL --cost 4.5          # is money there to find?
bun run src/cli.ts confluence --coin SOL --cost 4.5 --timeframe 1h
bun run src/cli.ts validate   --coin SOL --cost 4.5          # skill, or just exposure?
bun run src/cli.ts account    --coin SOL --equity 50 --leverage 2 --days 14
```

DEX pools via GeckoTerminal + DexScreener, also free and keyless:

```bash
bun run src/cli.ts top      --networks solana,base   # screened for modelability
bun run src/cli.ts search   --token WIF              # find the deepest real pool
bun run src/cli.ts backtest --token WIF              # fetch + walk-forward
bun run src/cli.ts trades   --token WIF              # ledger + ROI by window
bun run src/cli.ts train    --token WIF --out wif.json
bun run src/cli.ts predict  --model wif.json         # reloads its own source
```

Or bring a CSV — only a `close`/`price` column is required, and headers are
matched loosely, so exchange and aggregator exports usually load unedited:

```bash
bun run src/cli.ts fetch --coin BTC --out btc.csv
bun run src/cli.ts backtest --csv btc.csv
```

## Data sources

| source | used for | limits |
| --- | --- | --- |
| [Hyperliquid](https://api.hyperliquid.xyz) | perp candles, funding, market list. Continuous order book, so no gaps | ~5000 bars/request |
| [GeckoTerminal](https://api.geckoterminal.com) | DEX OHLCV history | 1000 bars/request, ~30 req/min |
| [DexScreener](https://api.dexscreener.com) | pool discovery, live liquidity and volume | no candle endpoint |

Responses cache to `.cache/` for 10 minutes (`--no-cache` to bypass) and requests
are throttled with backoff, so re-running a backtest is free.

### Five things about this data that bite

**`before_timestamp` is inclusive** (GeckoTerminal). Paging backwards re-serves
the boundary bar; un-deduped that injects a fake zero return every 1000 bars.

**`fundingHistory` returns the *first* ~500 hours after `startTime`**, not the
most recent 500 (Hyperliquid). Asking for 60 days back silently returns data
from 60–39 days ago and leaves your evaluation window uncovered — which showed
up as exactly `$0.00` funding across every coin. `fetchFunding` pages forward.

**A bare ticker does not find the token.** DexScreener search is a literal text
match and dogwifhat's symbol is `$WIF`, so querying `WIF` returns seven
impersonators and not the real one. Short queries are tried in several spellings.

**Ranking by liquidity picks dead pools.** The deepest `$WIF` pool holds $59M and
trades $0 a day; its candles are almost entirely gaps. Pools rank by the
geometric mean of liquidity and volume, with a floor under both — volume alone
rewards wash-traded shells with no depth. The worst offender seen had $106M of
daily volume against $0.0000016 of liquidity.

**Missing bars are invisible.** Untraded intervals are omitted rather than
returned as zero-volume bars, so a "5m bar" can silently span three hours,
breaking the fixed-time-step assumption the Markov chain rests on. The fetcher
counts gaps and warns; `--fill` inserts flat zero-volume bars for short runs and
deliberately refuses long ones, since hours of synthetic flat bars would
fabricate a low-volatility regime that never existed.

Concretely: `$WIF/SOL`, a $5.5M pool, is **26% missing at 5m and 0.6% at 1h**.
Hyperliquid perps are **0% missing at every interval**. Timeframe and venue
choice are mostly data-quality decisions.

## The models

Two, selected with `--model-type` (or `--hsmm`).

**HMM** (default) — state follows a Markov chain, features are Gaussian given
the state:

```
z_t | z_{t-1} ~ Categorical(A[z_{t-1}])
x_t | z_t     ~ N(mu[z_t], diag(var[z_t]))
```

**HSMM** — hidden semi-Markov. Each state learns its own duration distribution
`p_j(d)`; self-transitions are forbidden because dwell time is the duration
model's job.

This is not a refinement, it fixes a specific failure. A plain HMM forces
*geometric* dwell times, so the hazard of leaving a regime is constant no matter
how long you have been in it — which caps how confident the one-step-ahead
forecast can ever be at roughly the transition diagonal. On regimes that always
last exactly 8 bars, asking each model for P(switch) on the true final bar:

```
HSMM 0.998        HMM 0.129
```

The HSMM can say "this regime is 8 bars old and they last 8, so it ends now."
The HMM structurally cannot.

### Features

| feature | why |
| --- | --- |
| `logReturn` | direction |
| `realizedVol` | log realized vol over the lookback |
| `volumeSurge` | `log(volume / rolling mean volume)` |

`--window` defaults to **5**, not 20. A window longer than a regime cannot see
that regime, and these regimes are short: on synthetic data with 4.5-bar pumps,
20 → 3 lifted Viterbi regime recovery from 53% to 73%. On that same data,
dropping both engineered features and fitting on returns alone reached 86% —
they lag through exactly the transitions that matter. They stay on by default
because that result comes from data built to match the model's assumptions;
measure on your own series with `--no-vol --no-volume` before trusting it.

## Not fooling yourself

Three separate mistakes will each manufacture a beautiful equity curve. All
three are guarded, and each guard has a test that fails loudly if it regresses.

**Filtered states, never smoothed.** `posteriors()` gives `P(z_t | x_1..T)` — it
uses bars after `t` to decide what state you were in at `t`. Right for fitting,
fiction for trading. The backtest only calls `filter()`, and a test asserts that
truncating the series leaves earlier beliefs bit-identical.

**Refit only on the past.** `walkForward()` trains on a trailing block, freezes
the parameters, trades the next block, rolls forward. The feature scaler is fit
on training rows too — standardizing over the full series leaks the future
through the mean and standard deviation. A test tampers with late candles and
asserts early positions do not move.

**Higher timeframes are not complete yet.** At 10:05 the 10:00–11:00 hourly bar
does not exist. Reading it at every 5m step is reading tomorrow's newspaper.
Every multi-timeframe alignment goes through `lastCompletedIndex`, and a test
corrupts *all* future base candles and asserts every earlier signal is identical.

### The entry threshold

Naively, go long when expected next-bar return clears the cost. That is a
category error: a regime model holds for the regime's duration, so the profit of
entering is roughly `E[return per bar] × E[bars held] − round trip`. Testing the
per-bar figure against the full round trip silently produces **zero trades** —
5bps a bar never clears 9bps, but 5bps across 10 bars clears it comfortably.
`--duration-aware` spreads the round trip across the expected hold, using the
state durations the model already estimates.

## Top-down confluence

One model per timeframe, combined into a scalping stack:

```
BIAS    1h   which way is the regime leaning?   gate only
SETUP   15m  does the intermediate agree?       gate only
TRIGGER 5m   enter here                         fires the entry
```

All three must agree to open; only the bias has to break to close, because a
scalp should be abandoned the moment the structure justifying it fails.

**What it is actually for: turnover, not prediction.** A single 5m model fires
600+ trades and cannot survive a 30bps round trip, so at realistic cost it takes
*zero* trades. Three-way agreement cuts that to 7–25, which is what makes
trading at real fees possible at all.

## Two diagnostics

These matter more than the backtest. Both were added after a result that looked
excellent turned out to be nothing.

**`validate`** — reruns the strategy's own positions against a null that keeps
exposure and destroys only timing. If the null does as well, the return was
payment for being in the market.

The null has two forms, and picking the wrong one manufactures significance. A
free **shuffle** scatters long holds into isolated bars, so with costs on the
null pays an entry and exit almost every bar — any low-turnover strategy then
"beats random" for reasons unrelated to timing. That mistake briefly produced
**p = 0.000 on nine of nine tokens, including every loser**. A circular
**rotation** preserves every run and therefore exact turnover, changing only
*when* the series applies. Rotation is the default whenever `--cost` > 0.

**`montecarlo`** — runs the *entire* pipeline against series that provably
contain no timing signal, and counts how often it declares victory. This
calibrates every other number in the repo. See the results below.

**`ceiling`** — what perfect foresight earns here, at each cost. If an oracle
that knows the future cannot clear your fee, no model can and you should stop.
It also says the opposite: on `$WIF/SOL` 1h an oracle committing 20 bars at a
time makes **+271% after 30bps**, so the money is unambiguously there.

### What the Monte Carlo found

Null series keep the marginal return distribution, the fat tails and the
return/volume relationship, and destroy only the temporal order. Running the
full confluence pipeline plus permutation test over 100 of them:

| | iid shuffle | block shuffle (keeps vol clustering) |
| --- | --- | --- |
| single-series false-positive rate | **4.1%** | **3.4%** |
| nominal | 5% | 5% |

**The permutation test is honestly calibrated on one series.** A p-value of
0.04 on a single market means what it says.

Scanning is what destroys it:

```
  scan  5 markets, report the best:   4/19 batches "significant" on pure noise
  scan 10 markets, report the best:   3/9  batches "significant"  (33%)
  scan 20 markets, report the best:   3/4  batches "significant"  (75%)
```

This repo scanned 9–10 markets repeatedly. At that width, a third of scans
produce a p < 0.05 result from noise alone — which is precisely the rate at
which the three "significant" hits (SOL, ETH, XRP) appeared, and exactly why
none of them survived a split-half test.

The best null run returned **+74.5%**. Any single impressive backtest in this
domain is consistent with nothing at all.

## Account simulation

`account` reports dollars for a real leveraged perp account, because three
things only appear once you model the account rather than percentages:

- **Fees land on notional.** At 2x, a 4.5bps taker costs 9bps of equity per side.
- **Funding accrues hourly** on that same notional, and never stops while open.
- **Liquidation is path-dependent** — the worst tick *inside* the trade decides
  it, so bar lows are used for longs and highs for shorts. A close-only
  simulation will happily trade through a wipeout.

## Results

### Synthetic data, where a signal is known to exist

Generated from a known 3-regime process. An oracle knowing the true current
regime makes +1185% at 30bps with 6% time in market:

| model | cost | ROI | trades | exposure | p-value |
| --- | --- | --- | --- | --- | --- |
| HMM | 0bps | +149% | 328 | 70% | 0.40 — luck |
| HMM | 30bps | −33% | 2 | 6% | 0.011 |
| **HSMM** | 0bps | **+1347%** | 1022 | 57% | **0.017** |
| **HSMM** | 30bps | **+164%** | 60 | **6%** | **<0.001** |

The HSMM reaches 6% exposure — the oracle's own figure — and clears realistic
costs with significant timing skill. The HMM's healthy-looking +149% is
exposure, not skill.

### Real DEX data: none of it transfers

| dataset | ceiling @30bps (hold 20) | HSMM @30bps | p-value |
| --- | --- | --- | --- |
| USELESS/SOL 5m | +3167% | 0.0% (no trades) | — |
| $WIF/SOL 1h | +271% | +0.2% | 0.018, 8 trades |
| PEPE/WETH 1h | +93% | −2.5% | 0.14 |

Across 9 live tokens at 30bps: confluence mean **+42.6%** against buy & hold
**+73.0%**, beating hold on 2 of 9 — both in downtrends, where the gain came
from being flat rather than from timing.

### The cost hypothesis, tested and rejected

The working explanation was long that fees were the barrier: a ~60bps DEX round
trip against a ~5bps per-bar signal. Hyperliquid tests it directly — same
models, **zero missing bars**, **4.5bps a side**, 208 days of 1h history on the
top 10 perps.

| | mean ROI | beat buy & hold |
| --- | --- | --- |
| single timeframe (duration-aware) | +23.9% | — |
| 3TF confluence | +34.2% | 2 / 9 |
| **buy & hold** | **+77.1%** | — |

Cheaper fees and clean data did not rescue it. Three results crossed p < 0.05
(SOL 0.005, ETH 0.014, XRP 0.035), but 18 tests ran, so ~1 false positive was
expected. Splitting each in half settles it — every edge lives in one half and
vanishes in the other:

```
  coin   strategy    half 1 ROI    p1     half 2 ROI     p2
  SOL    single           20.4%  0.003         5.3%   0.129
  ETH    conf              0.0%  1.000        32.7%   0.015
  XRP    conf             -2.6%  0.714        37.0%   0.041
```

All five coins examined also did better in half 2 — as did buy & hold, which
rallied 20–51%. The "good" half is exposure to a rising market once again.

A third check confirms it independently. Removing **the first 4 bars** of XRP's
5004-bar series — same end date, same parameters — flips it from 5 trades and
+33.4% to **0 trades and 0.0%**. Shifting the training window alone changes
nothing, so this is not offset sensitivity: dropping 4 leading bars moves every
walk-forward boundary, and with five trades the outcome is one or two coin
flips. Treat any result resting on fewer than a few dozen trades as unmeasured,
whatever its p-value.

### $50 account at 2x, top 10 Hyperliquid perps

15m bars, real taker fees and hourly funding, liquidation checked intrabar.

| | 1 week | 2 weeks |
| --- | --- | --- |
| 3TF confluence | **+5.3%** | +3.0% |
| single timeframe | −3.1% | −2.7% |
| 2x buy & hold | +3.3% | **+8.4%** |

Beat hold over one week, lost to it over two — a sign flip between adjacent
windows is what no edge looks like. Single-coin outcomes over 2 weeks ran from
**$28.07 (PUMP, −44%) to $81.21 (ZEC, +62%)**, with 6 of 9 losing money. No
liquidations: at 2x you need roughly a −49% move and the worst was −22%.

Costs were *not* the problem here — fees $0.28–$0.73 and funding $0.08–$0.53 per
coin over two weeks, under 2.5% of equity combined. The P&L came from direction,
and direction was wrong more often than not.

### What that leaves

Four explanations have been eliminated with data: cost (4.5 vs 30bps), data
quality (0% vs 26–88% gaps), the geometric-duration assumption (HSMM), and
timeframe (1m through 4h). Across two venues, two models, four timeframes and
twenty markets, what keeps failing is the premise — that features derived from
past price identify these regimes *before* they pay.

The ceilings confirm the money exists. Nothing in the price history finds it in
advance. The next thing worth trying is data that is not a function of past
price: order flow, holder concentration, LP changes, liquidations.

## Tests

`bun test` — 96 tests. Both models are validated against brute-force
enumeration: all `K^T` state paths for the HMM, and every (state, duration)
segmentation for the HSMM. Forward log-likelihood, filtered marginals, smoothed
marginals and the Viterbi path must match exactly. Plus EM monotonicity,
parameter recovery from generated data, causality of filtering and of feature
construction, multi-timeframe lookahead guards, trade-ledger accounting against
hand-computed values, and leveraged account accounting including liquidation
paths.

## Layout

```
src/hmm.ts          forward-backward, Baum-Welch, Viterbi, filtering (no deps)
src/hsmm.ts         explicit-duration HSMM: segmental EM, run-length filtering
src/features.ts     candles -> features, train-only standardization
src/confluence.ts   multi-timeframe stack, completed-bar alignment
src/backtest.ts     walk-forward harness, signal, trade ledger, metrics
src/perp.ts         leveraged account: notional fees, funding, liquidation
src/diagnostics.ts  permutation test (shuffle/rotate nulls), ceiling analysis
src/montecarlo.ts   null-series generation, pipeline false-positive rate,
                    trade bootstrap for outcome distributions
src/hyperliquid.ts  perp candles, funding, market list
src/sources.ts      GeckoTerminal + DexScreener, paging, gaps, cache
src/data.ts         CSV parsing, synthetic regime-switching generator
src/cli.ts          all commands
```

Tests sit beside each module as `*.test.ts`.

## Commands

```
top         top markets, screened for modelability (--source hyperliquid for perps)
trending    what is moving right now (GeckoTerminal)
search      find the deepest real pool for a ticker
fetch       pull candles to CSV
train       fit and inspect the regimes, optionally save the model
predict     current state + next-bar forecast + signal
backtest    walk-forward out-of-sample, with a cost-sensitivity sweep
confluence  3-timeframe scalping stack
trades      trade ledger + ROI by 24h / 5d / 1w / 2w window
account     dollar P&L for a leveraged perp account
validate    is the return skill, or exposure?
montecarlo  how often does this pipeline cry wolf? plus outcome bootstrap
ceiling     what would perfect foresight earn here?
demo        synthetic end-to-end walkthrough
```

## Flags

```
Hyperliquid --coin BTC  --timeframe 5m|15m|1h|4h|1d  --bars 5000
DEX         --token <sym>  --pool <addr>  --network solana  --fill  --max-fill 12
            --min-liquidity  --min-volume  --no-cache
Data        --csv <path>  --window 5  --no-vol  --no-volume
Model       --model-type hmm|hsmm  --hsmm  --max-duration 30
            --states 3  --restarts 8  --seed 42  --out model.json
Strategy    --cost 4.5  --entry <bps>  --duration-aware  --confidence 0
            --exit 0  --short  --vol-target 0  --max-pos 1
Confluence  --factors 12,3,1  --gate 0  --trigger 0  --bias-confidence 0
            --no-flip-exit
Account     --equity 50  --leverage 2  --days 14  --single  --min-order 10
Diagnostics --trials 1000  --costs 0,10,30  --holds 1,5,20  --show 15
MonteCarlo  --nulls 100  --block 50  --no-bootstrap
Walkfwd     --train 1500  --test 500  --bars-per-year <n>  --verbose
```

## Limits worth knowing

- Gaussian emissions understate crypto tails. The features are chosen to be as
  close to Gaussian as possible (log vol rather than vol), but a genuine 40%
  candle is far outside what the model thinks can happen.
- `K` is fixed, not selected. Compare log-likelihood per bar across `--states`
  with a BIC-style penalty if you want to choose it properly.
- The HSMM costs roughly 15x the HMM to fit (O(T·K·maxDuration) per EM sweep).
  Lower `--max-duration` if that bites; 30 was as good as 60 in testing.
- Hyperliquid retains ~5000 candles per interval, so 5m gives ~17 days and 1h
  gives ~208. Pick the interval for the history you need.
- One backtest is one draw. Bootstrapping the $50 @ 2x ledgers, a run that
  reported $81.21 sat in a 5th-95th range of **$38.86 to $241.79**. With fewer
  than a few dozen trades the headline number carries almost no information.
- Past backtest ROI does not predict future ROI. Measured across 9 tokens split
  into halves, rank correlation was **0.033**, while correlation between
  strategy ROI and buy & hold was **0.950** — the "high ROI" names are simply
  the ones that went up. Do not select tokens this way.
- Nothing here models MEV, honeypots, or the chart ending at zero. Pool
  liquidity picks which pool to read; it does not model your own price impact.
- Backtest results are not a forecast, and none of this is financial advice.
  Slippage on a thin book is worse than any fixed bps assumption, and worst
  exactly when the model wants to trade.
