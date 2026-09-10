# meme-hmm

A hidden Markov model for meme coin regimes, in TypeScript on Bun. No dependencies —
the forward-backward, Baum-Welch, and Viterbi recursions are written out directly.

The premise: a meme coin is not one process. It is a few very different processes
that take turns — a low-volatility chop that dominates the sample, a slow bleed, and
a rare, violent, short-lived pump. You never observe which one is running. An HMM
treats that regime as a hidden state, infers a probability distribution over it from
price and volume, and forecasts which regime comes next.

## Quick start

```bash
bun install
bun run demo                              # synthetic end-to-end walkthrough
bun test                                  # 83 tests, incl. brute-force validation
```

Against live market data — free, no API key, nothing to sign up for:

```bash
bun run src/cli.ts trending --network solana      # what is moving right now
bun run src/cli.ts search   --token WIF           # find the deepest real pool
bun run src/cli.ts backtest --token WIF           # fetch + walk-forward test
bun run src/cli.ts train    --token WIF --out wif.json
bun run src/cli.ts predict  --model wif.json      # reloads the pool it was trained on
bun run src/cli.ts validate --token WIF           # is it skill, or just exposure?
bun run src/cli.ts ceiling  --token WIF           # is there money here to find?
bun run src/cli.ts fetch    --token WIF --out wif.csv   # or just grab the candles
```

Or bring your own CSV — only a `close`/`price` column is required, header names
are matched loosely, so exchange and aggregator exports usually load unedited:

```bash
bun run src/cli.ts backtest --csv wif.csv
```

## Data sources

| source | used for | limits |
| --- | --- | --- |
| [Hyperliquid](https://api.hyperliquid.xyz) | perp OHLCV + market list. Continuous order book, so no gaps | ~5000 bars/request |
| [GeckoTerminal](https://api.geckoterminal.com) | DEX OHLCV history | 1000 bars/request, ~30 req/min |
| [DexScreener](https://api.dexscreener.com) | pool discovery, live liquidity and volume | no candle endpoint |

Hyperliquid is the cleaner venue by a distance, and it was used to falsify the
main hypothesis this project had been carrying — see Results.

```bash
bun run src/cli.ts top --source hyperliquid          # top perps by 24h volume
bun run src/cli.ts ceiling  --coin BTC --cost 4.5    # 4.5bps = HL base taker
bun run src/cli.ts confluence --coin SOL --cost 4.5 --timeframe 1h
```

Responses are cached to `.cache/` for 10 minutes (`--no-cache` to bypass), and
requests are throttled with backoff, so re-running a backtest costs nothing.

Four things about this data bite if you do not handle them, and all four are
handled in `src/sources.ts`:

**`before_timestamp` is inclusive.** Paging backwards re-serves the boundary bar.
Un-deduped, that injects a fake zero return at every 1000-bar boundary.

**A bare ticker does not find the token.** DexScreener search is a literal text
match, and dogwifhat's symbol is `$WIF` — querying `WIF` returns seven
impersonators and not the real one. Short queries are tried in several spellings
and merged.

**Ranking by liquidity picks dead pools.** The deepest `$WIF` pool holds $59M and
trades $0 a day; its candle series is almost entirely gaps. Pools are ranked by
the geometric mean of liquidity and 24h volume, with a floor under both. Ranking
by volume alone rewards wash-traded shells with no depth — you need both.

**Missing bars are invisible.** Untraded intervals are omitted from the response
rather than returned as zero-volume bars, so a "5m bar" can silently span three
hours. That breaks the fixed-time-step assumption the Markov chain rests on. The
fetcher counts gaps and warns; `--fill` inserts flat zero-volume bars for short
runs, and deliberately refuses to fill long ones — hours of synthetic flat bars
would fabricate a low-volatility regime that never existed.

On live `$WIF/SOL`, a $5.5M pool, 5m bars are 26% missing while 1h bars are 0.6%
missing. Timeframe choice is mostly a data-quality decision.

## The models

Two, selected with `--model-type` (or `--hsmm`).

**HMM** (default) — state `z_t` follows a Markov chain, features are Gaussian
given the state:

```
z_t | z_{t-1} ~ Categorical(A[z_{t-1}])
x_t | z_t     ~ N(mu[z_t], diag(var[z_t]))
```

**HSMM** — a hidden semi-Markov model. Each state carries its own learned
duration distribution `p_j(d)`, and self-transitions are forbidden because dwell
time is the duration model's job.

This is not a refinement, it fixes a specific failure. A plain HMM forces
*geometric* dwell times, so the hazard of leaving a regime is constant no matter
how long you have been in it. That caps how confident the one-step-ahead
forecast can ever be at roughly the transition diagonal. Measured on regimes
that always last exactly 8 bars, asking each model for P(switch) on the true
final bar of a run:

```
HSMM 0.998        HMM 0.129
```

The HSMM can say "this regime is 8 bars old and they last 8 bars, so it ends
now." The HMM structurally cannot. That single difference is worth more than
everything else in this repo — see the results below.

Three features, all known at the close of the bar they describe:

| feature | why |
| --- | --- |
| `logReturn` | direction |
| `realizedVol` | log realized vol over the lookback |
| `volumeSurge` | `log(volume / rolling mean volume)` |

`--window` defaults to **5**, not 20. A window longer than a regime cannot see
that regime, and these regimes are short: on synthetic data with 4.5-bar pumps,
20 → 3 lifted Viterbi regime recovery from 53% to 73%. On that same data,
dropping the two engineered features entirely and fitting on returns alone
reached 86% — they lag through exactly the transitions that matter. They are
still on by default because that result comes from data built to match the
model's assumptions; measure on your own series with `--no-vol --no-volume`
before trusting it.

## Trading it, and the two ways that goes wrong

**Use filtered states, never smoothed ones.** `posteriors()` gives
`P(z_t | x_1..T)` — it uses the whole series, including bars after `t`, to decide
what state you were in at `t`. It is the right quantity for fitting and produces
gorgeous, entirely fictional backtests. `filter()` gives `P(z_t | x_1..t)`, which is
what you would actually have known at the time. The backtest uses `filter()` only,
and a test asserts that truncating the series leaves earlier beliefs bit-identical.

**Refit only on the past.** `walkForward()` trains on a trailing block, freezes the
parameters, trades the next block, then rolls forward. The feature scaler is fit on
training rows too — standardizing over the full series leaks the future through the
mean and standard deviation. A test tampers with late candles and asserts that early
positions do not move.

The signal is the expected next-bar return under the one-step-ahead state forecast:

```
E[r_{t+1} | x_1..t] = sum_k P(z_{t+1} = k | x_1..t) * mu_return[k]
```

Go long when that clears the entry threshold, which **defaults to twice the per-side
cost**. Taking a signal whose expected edge is smaller than the cost of expressing it
is the fastest way to bleed out, and it is the default failure of this entire genre
of model.

## Top-down confluence (`confluence`)

One model per timeframe, combined into a scalping stack:

```
BIAS    1h   which way is the regime leaning?   gate only
SETUP   15m  does the intermediate agree?       gate only
TRIGGER 5m   enter here                         fires the entry
```

All three must agree to open. Only the bias has to break to close — a scalp
should be abandoned the moment the structure justifying it fails.

**The lookahead trap.** At 10:05 the 10:00–11:00 hourly bar does not exist yet.
Backtests that read the "current" higher-timeframe bar at every 5m step are
reading tomorrow's newspaper, and they produce beautiful equity curves. Every
alignment here goes through `lastCompletedIndex`, and a test corrupts all future
base candles and asserts that every earlier signal is bit-identical.

**What it is actually for.** Not prediction — turnover. A single 5m model fires
600+ trades and cannot survive a 30bps round trip, so at realistic cost it takes
*zero* trades. Demanding three-way agreement cuts that to 7–25 trades, which is
what makes trading at real fees possible at all. Measured across 9 live tokens:

| | trades | mean ROI @30bps |
| --- | --- | --- |
| single 5m | 0 | 0.0% |
| 3TF confluence | 7–25 | +42.6% |
| buy & hold | — | **+73.0%** |

It trades, and it underperforms holding. It beat buy & hold on 2 of 9 tokens,
both in downtrends, where the gain came from being flat rather than from timing.

## Two diagnostics that decide whether any of this is real

These matter more than the backtest. Both were added after a result that looked
excellent turned out to be nothing.

**`validate`** — reruns the strategy's own positions against a null that keeps
exposure and destroys only the timing. If the null does as well, the return was
payment for being in the market, not skill.

The null has two forms and picking the wrong one manufactures significance.
A free **shuffle** scatters long holds into isolated bars, so with costs on the
null pays an entry and an exit almost every bar — and any low-turnover strategy
"beats random" for reasons that have nothing to do with timing. That mistake
briefly produced p = 0.000 on nine out of nine tokens here, including the ones
losing money. A circular **rotation** preserves every run and therefore the exact
turnover, changing only *when* the series is applied. Rotation is the default
whenever `--cost` is above zero.

**`ceiling`** — what perfect foresight earns on this series, at each cost. If an
oracle that knows the future cannot clear your fee, no model can. It also tells
you the opposite: on real `$WIF/SOL` 1h data an oracle committing for 20 bars at
a time makes **+271% after 30bps**, so the money is unambiguously there.

## Results

## Results

Synthetic data, generated from a known 3-regime process. The oracle that knows
the true current regime makes +1185% at 30bps, with 6% time in market:

| model | cost | ROI | trades | exposure | p-value |
| --- | --- | --- | --- | --- | --- |
| HMM | 0bps | +149% | 328 | 70% | 0.40 — luck |
| HMM | 30bps | −33% | 2 | 6% | 0.011 |
| **HSMM** | 0bps | **+1347%** | 1022 | 57% | **0.017** |
| **HSMM** | 30bps | **+164%** | 60 | **6%** | **<0.001** |

The HSMM reaches 6% exposure — the oracle's own figure — and clears realistic
costs with real, significant timing skill. The HMM's apparently healthy +149% is
exposure, not skill.

**On real meme coin data, none of this transfers.** Same configuration, same
tests, live pools:

| dataset | ceiling @30bps (hold 20) | HSMM @30bps | p-value |
| --- | --- | --- | --- |
| USELESS/SOL 5m | +3167% | 0.0% (no trades) | — |
| $WIF/SOL 1h | +271% | +0.2% | 0.018, 8 trades |
| PEPE/WETH 1h | +93% | −2.5% | 0.14 |

### The cost hypothesis, tested and rejected

For a long stretch the working explanation was that fees were the barrier: a
~60bps DEX round trip against a ~5bps per-bar signal. Hyperliquid perps test
that directly — the same models, on data with **zero missing bars**, at **4.5bps
a side** instead of 30, over 208 days of 1h history on the top 10 markets.

| | mean ROI | beat buy & hold |
| --- | --- | --- |
| single timeframe (duration-aware entry) | +23.9% | — |
| 3TF confluence | +34.2% | 2 / 9 |
| **buy & hold** | **+77.1%** | — |

Cheaper fees and clean data did not rescue it. Three results crossed p < 0.05
(SOL 0.005, ETH 0.014, XRP 0.035), but 18 tests were run, so ~1 false positive
was expected — and splitting each in half settles it:

```
  coin   strategy    half 1 ROI    p1     half 2 ROI     p2
  SOL    single           20.4%  0.003         5.3%   0.129
  ETH    conf              0.0%  1.000        32.7%   0.015
  XRP    conf             -2.6%  0.714        37.0%   0.041
```

A third check settles it independently. Removing **the first 4 bars** of XRP's
5004-bar series — same end date, same parameters — flips the result from 5
trades and +33.4% to **0 trades and 0.0%**. Shifting the training window alone
changes nothing, so this is not offset sensitivity: dropping 4 leading bars
moves every walk-forward block boundary, and with only five trades in the whole
period the outcome is one or two coin flips. Treat any result resting on fewer
than a few dozen trades as unmeasured, whatever its p-value.

Every apparent edge lives in exactly one half and vanishes in the other. All
five coins examined also did better in half 2 — as did buy & hold, which rallied
20–51% — so the "good" half is exposure to a rising market once again.

Cost was never the binding constraint. Neither was data quality, nor the
geometric-duration assumption, nor the timeframe. What has failed consistently,
across two venues, two models, four timeframes and twenty tokens, is the premise
that features derived from past price identify these regimes *before* they pay.

## Tests

`bun test` validates both models against brute-force enumeration — all `K^T`
state paths for the HMM, and every (state, duration) segmentation for the HSMM — forward log-likelihood, filtered marginals, smoothed
marginals, and the Viterbi path all have to match exactly. Plus EM monotonicity,
parameter recovery from data the model generated, causality of filtering and feature
construction, and backtest accounting invariants.

## Layout

```
src/hmm.ts          forward-backward, Baum-Welch, Viterbi, filtering (no deps)
src/hsmm.ts         explicit-duration HSMM: segmental EM, run-length filtering
src/hyperliquid.ts  perp market list and candles, no gaps, 4.5bps fees
src/confluence.ts   multi-timeframe stack, completed-bar alignment
src/diagnostics.ts  permutation test (shuffle/rotate nulls), ceiling analysis
src/features.ts     candles -> features, train-only standardization
src/sources.ts      GeckoTerminal + DexScreener clients, paging, gaps, cache
src/data.ts         CSV parsing, synthetic regime-switching generator
src/backtest.ts     walk-forward harness, signal, cost accounting, metrics
src/cli.ts          trending / search / fetch / train / backtest / predict / demo
src/hmm.test.ts     brute-force validation and model invariants
src/hsmm.test.ts    brute-force over all segmentations, duration recovery
src/diagnostics.test.ts  permutation and ceiling invariants
src/confluence.test.ts   aggregation, alignment, lookahead guards
src/trades.test.ts       ledger accounting vs hand-computed values
src/sources.test.ts timeframes, network aliases, gap filling
```

## Flags

```
Live     --token <sym>  --pool <addr>  --network solana  --timeframe 5m
         --bars 3000  --fill  --max-fill 12  --min-liquidity  --min-volume
         --no-cache
Data     --csv <path>  --window 20  --no-vol  --no-volume
Model    --model-type hmm|hsmm  --hsmm  --max-duration 30
         --states 3  --restarts 8  --seed 42  --out model.json
Conflu   confluence --factors 12,3,1  --gate 0  --trigger 0  --short
Diag     validate --trials 1000  |  ceiling --holds 1,5,20  --costs 0,10,30
         trades --show 15  (ledger + ROI by 24h/5d/1w/2w window)
Strategy --entry <bps, default 2x cost>  --exit 0  --short  --cost 30
         --vol-target 0  --max-pos 1
Walkfwd  --train 1500  --test 500  --bars-per-year 105120  --verbose
```

## Limits worth knowing

- Gaussian emissions understate meme coin tails. The features are chosen to be as
  close to Gaussian as they can be (log vol rather than vol), but a genuine 40%
  candle is still far outside what the model thinks is possible.
- `K` is fixed, not selected. Compare log-likelihood per bar across `--states` values
  with a BIC-style penalty if you want to choose it properly.
- The HSMM costs roughly 15x the HMM to fit (O(T·K·maxDuration) per EM sweep).
  Lower `--max-duration` if that bites; 30 was as good as 60 in testing.
- Nothing here models MEV, honeypots, or the possibility that the chart ends at
  zero — the risks that actually dominate this asset class. Pool liquidity is used
  to pick which pool to read, not to model the price impact of your own order.
- The newest coins, where the pump regime is most dramatic, have the least
  history. A pool minutes old has nothing to fit. `trending` shows what is moving;
  it is not a list of things you can model.
- Backtest results on your own data are not a forecast. Slippage on a thin book is
  worse than any fixed bps assumption, and worst exactly when the model wants to trade.
