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
bun test          # 146 tests, incl. brute-force validation of both models
```

Hyperliquid perps — clean data, 4.5bps taker, no API key:

```bash
bun run src/cli.ts top --source hyperliquid                  # top perps by volume
bun run src/cli.ts ceiling    --coin SOL --cost 4.5          # is money there to find?
bun run src/cli.ts confluence --coin SOL --cost 4.5 --timeframe 1h
bun run src/cli.ts validate   --coin SOL --cost 4.5          # skill, or just exposure?
bun run src/cli.ts posterior  --coin SOL --hsmm              # intervals, dwell inferred
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

Thirty perps at once, then a terminal to drive them:

```bash
bun run src/cli.ts sweep --limit 30 --days 45 --save   # every coin x timeframe x model
bun run tui                                            # fit, backtest, paper trade
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

Both are fitted by EM, and both have a Bayesian counterpart fitted by
variational EM over the same recursions — see
[Three ways to a posterior](#three-ways-to-a-posterior).

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

**`ceiling`** — what perfect foresight earns here, at each cost. If an oracle
that knows the future cannot clear your fee, no model can and you should stop.
It also says the opposite: on `$WIF/SOL` 1h an oracle committing 20 bars at a
time makes **+271% after 30bps**, so the money is unambiguously there.

## Three ways to a posterior

Everything above plugs EM's point estimates into the signal as if they were
known. `mu_k` is estimated from however many bars happened to land in state k,
and a rare pump state might own 200 of 3000. If its standard error is
comparable to the round trip, the signal is noise wearing a point estimate.
`posterior` puts credible intervals on that, and on the traded quantity itself.

Three engines answer it. Two put the same conjugate priors on the same Markov
model, so they are directly comparable. The third moves those priors onto the
semi-Markov model, where the dwell time the edge gets multiplied by is inferred
rather than implied.

**Variational EM** (default, `src/vb.ts`). Approximate the intractable
posterior by a factorized `q(z) q(pi) q(A) q(mu, lambda)` and maximize a lower
bound on the evidence — the ELBO — by alternating exactly like Baum-Welch:

```
E step   q(z)      forward-backward, but run on E[log pi], E[log A] and
                   E[log N(x | mu, 1/lambda)] rather than on point values
M step   q(theta)  the same conjugate Dirichlet and Normal-Gamma updates the
                   sampler uses, driven by responsibilities instead of a path
```

The difference from Baum-Welch is one term. Where EM plugs `mu_k` into the
Gaussian, VB integrates the density against `q(mu, lambda)`, which adds a
`-1/(2 kappa_k)` penalty for not knowing the mean. That is what stops a state
with forty observations from claiming a razor-sharp emission.

Three things fall out that the sampler cannot give:

- **Convergence is a number.** The ELBO is monotone by construction, so
  "converged" is a tolerance, not a judgement about a trace. It gets there in
  ~30 iterations against 1200 sweeps — about **9x faster** end to end.
- **Draws are independent.** `q` is a product of Dirichlets and Normal-Gammas,
  so the intervals come from i.i.d. draws. No burn-in, no thinning, no
  autocorrelation discounting the effective sample size.
- **K stops being an assertion.** The ELBO bounds `log p(x)` for a given K, so
  ELBOs are comparable across K and `--select-states 2,3,4,5` picks the number
  of regimes. States that explain nothing take their occupancy to zero and
  revert exactly to the prior, which is the same answer read off directly.

**Gibbs sampling** (`--gibbs`, `src/mcmc.ts`). Forward-filter-backward-sample
the state path, then draw parameters from their conjugate conditionals.
Asymptotically exact, and kept for exactly that reason.

**Variational EM over durations** (`--hsmm`, also `src/vb.ts`). The same
alternation, the same bound, the same conjugate emissions — with the chain
replaced:

```
q(A[i])     Dirichlet over the K-1 states that are NOT i, since a semi-Markov
            model has no self-transition to give mass to
q(p_j(.))   Dirichlet over d = 1..maxDuration. Dwell stops being 1/(1 - A_kk)
            and becomes something the data gets to say
E step      hsmm.ts's segmental forward-backward, run on E[log pi], E[log A],
            E[log p_j(d)] and E[log N(x | mu, 1/lambda)]
```

That last line is why `hsmm.ts` now hands its recursions a *chain* of log
parameters rather than an `HsmmParams`: maximum-likelihood EM passes the log of
its point estimates, variational EM passes expected logs under the Dirichlets,
and the segmental forward-backward in between is one implementation, validated
once against brute-force enumeration. The M step is the same conjugate update
in both engines too — prior concentration plus expected counts, and the
identical Normal-Gamma block on the emissions. There is deliberately no
`alphaSelf` here: under a semi-Markov model persistence is not a prior on the
diagonal, it is the duration distribution, and it is learned.

This matters for one number in particular. The comparison the command exists to
make is `edge x hold` against the round trip, and `hold` is the dwell — so under
the Markov engines the quantity being integrated over is geometric by
assumption. Here every draw carries its own duration pmf and its own mean dwell,
and `posteriorDurations` reports that instead.

Whether it is worth the parameters is a question about the data, and on real
perps the answer so far is no:

```
SOL-PERP 5m, 4995 bars, K = 3        ELBO/bar   iters   dwell by state
variational EM (Markov)               -3.579      19     7b  9b  9b
variational EM (semi-Markov)          -3.592      68     7b  9b  8b
```

The explicit model reads back nearly the same dwell and pays 90 duration
parameters for the privilege, so the bound is worse. On 5m SOL the geometric
law is not what is wrong. It wins where it should: on a series whose regimes
last exactly 8 bars, the semi-Markov bound beats the Markov one and the
posterior puts a 90% interval on the dwell that brackets 8 — `vb.test.ts`
asserts both.

One caveat that does not apply to the Markov path: there is no sampler for the
semi-Markov model, so the mean-field narrowing measured below has nothing to
check it against here.

### What the approximation costs

Mean-field assumes `q(z)` and `q(theta)` are independent. They are not, and the
standard consequence is credible intervals that are too **narrow** — confident
in proportion to how wrong the independence assumption is.

That is measurable, so it is measured. Twelve independent 800-bar series, 90%
nominal intervals on all three state means:

```
             coverage    mean width
Gibbs             89%         0.150
variational       83%         0.128
```

The sampler lands on its nominal rate. The approximation gives up 14% of the
width and 6 points of coverage for it. Since the whole question this command
exists to answer is whether an interval clears the round trip, that bias points
the flattering way — which is why `--gibbs` is still here, why the CLI says so
after every variational run, and why `vb.test.ts` checks the two against each
other rather than trusting either alone.

On real data they agree closely. SOL 1h, 1995 bars, bullish-state mean:

```
variational   [ 1.8, 21.1] bps      29 iterations
Gibbs         [ 1.4, 20.8] bps    1200 sweeps
```

### The ELBO does not want three states

Asked to choose on SOL 1h, it picks **six**, and keeps improving until it
prunes on its own:

```
K       2       3       4       5       6       8      10      12
ELBO -3.767  -3.574  -3.428  -3.352  -3.292  -3.299  -3.318  -3.364
occ      2       3       4       5       6       7       7       7
```

Read that as a statement about Gaussian emissions, not about markets. A single
heavy-tailed return distribution is cheaper to approximate with several
Gaussians than with one, so extra states buy likelihood by modelling the tails
rather than by finding regimes. The repo defaults to 3 because three states are
interpretable — chop, bleed, pump — not because the evidence prefers them.

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

## The terminal

`bun run tui` is the whole repo behind one screen: the universe on the left with
whatever the sweep learned about each coin, a candle chart with the model's
regime and position strips under it, and four tabs — fit, backtest, paper,
blotter.

```
 SOL-PERP 30m HSMM  99.9      paper $1043.21 +4.3%      p=0.445 — not distinguishable from luck
╭─ universe ─────────╮╭─ SOL 30m ───────────────────────────────────────────────╮
│ coin      tf excess││  115.02 ████████│                                       │
│ BTC      15m    3% ││  113.42           ██│                        │││││      │
│ ETH      15m   -7% ││  111.82              │█│                  │███     ███│ │
│ SOL      30m   -0% ││  regime ▁▁▁▁▁▅▅▅▅▅▅▅▅▅▅▅▅▁▁▁▁▁▁▁▁▁▁▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅ │
│ ZEC      15m   14% ││position ·······················▲▲▲▲▲▲▲▲▲▲▲············· │
╰────────────────────╯╰─────────────────────────────────────────────────────────╯
```

Keys: `j/k` coin, `h/l` timeframe, `m` model, `f` fit, `b` backtest,
`r` replay, `p` paper on/off, `s` sweep, `?` help, `q` quit.

Three things about it are deliberate.

**Nothing expensive runs on the render thread.** A 3-state HSMM over 3000 bars
with 8 restarts is ~3s of solid arithmetic, and a walk-forward does that once
per block. All of it goes to a Worker (`src/tui/worker.ts`), so the chart keeps
repainting and the keyboard keeps responding while a fit is in flight.

**Paper trading decides with the same code the backtest does.** `signalNow` in
`backtest.ts` is what both call, and `paper.test.ts` pins a replay to
`walkForward`'s out-of-sample equity curve at a worst relative error of 4e-16.
If paper and backtest could drift, paper results would say nothing about the
model.

**Only closed bars are traded.** The last candle Hyperliquid returns is the one
still forming; its close is whatever the price is this second. Signalling on it
is the live-trading cousin of the lookahead the backtest works so hard to
avoid, so `live.ts` filters it out and uses it for nothing but the mark price
on screen.

The account persists to `models/paper.json` and resumes on the next launch — a
paper record that resets when you close the terminal cannot answer the one
question it exists for.

## Sweeping the universe

`sweep` runs every (coin, timeframe, model) cell over one window, walks each
one forward, permutes it, and ranks what is left. Thirty perps x four
timeframes x two models is 240 fits, about fifteen minutes.

Ranking is on **excess ROI over the permutation null**, never on ROI — the coin
that went up the most would otherwise win every sweep — and a cell with more
than 90% exposure cannot win at all, because that is buy-and-hold wearing a
model. The best row per coin is refit on the full window and written to
`models/<coin>-<tf>-<model>.model.json`, which `predict --model` and the TUI
both read.

The result on the top 30 perps, 45 days to 2026-09-10, 4.5bps a side:

| # | coin | tf | model | excess | roi | b&h | expos | trades | p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | CASHCAT | 15m | hsmm | +79.9% | 82.4% | 14.1% | 81% | 206 | 0.060 |
| 2 | NEAR | 30m | hmm | +29.6% | 39.6% | 55.3% | 35% | 54 | **0.035** |
| 3 | CHIP | 30m | hmm | +27.6% | 66.2% | 82.2% | 66% | 114 | 0.290 |
| 4 | FARTCOIN | 1h | hsmm | +21.6% | 17.3% | −5.2% | 81% | 43 | 0.080 |
| 5 | ENA | 15m | hmm | +21.5% | 54.1% | 63.5% | 60% | 162 | 0.240 |

**Read the bottom of that table, not the top.** 228 cells produced a p-value and
exactly **2** came in under 0.05 — fewer than the ~11 you would expect from
noise alone at that many tests. Bonferroni over 240 cells puts the bar at
p = 0.0002; the best row here is 0.035. Twenty-three of the 28 winners lost to
buy & hold outright, and the median winner sat in the market 68% of the time.
This is a selection out of 240 attempts, not a finding, and treating the top of
it as a shortlist is exactly the procedure this repo measured at a rank
correlation of 0.033 across halves.

## Tests

`bun test` — 146 tests. Both models are validated against brute-force
enumeration: all `K^T` state paths for the HMM, and every (state, duration)
segmentation for the HSMM. Forward log-likelihood, filtered marginals, smoothed
marginals and the Viterbi path must match exactly. Plus EM monotonicity,
parameter recovery from generated data, causality of filtering and of feature
construction, multi-timeframe lookahead guards, trade-ledger accounting against
hand-computed values, and leveraged account accounting including liquidation
paths.

The variational fit is held to the same standard. `lgamma` and `digamma` are
checked against closed forms, their recurrences, and each other by finite
difference; the ELBO must never decrease; a cold start must reach the same
optimum as an EM warm start; the ELBO must select the true K on data generated
with a known one; unused states must revert to the prior exactly; the draws
must be independent (lag-1 autocorrelation under 0.06, which a Gibbs chain
cannot claim); and the whole posterior is compared against the sampler on the
same series, including the replication study that measures what the mean-field
narrowing costs in coverage. The semi-Markov engine repeats that bar: monotone
ELBO from a cold start, recovery of a duration law no geometric model can
express, pruning of unsupported states, K selection, proper draws, agreement
with the maximum-likelihood fit it was seeded from, and a higher bound than the
Markov engine on data where the durations are not geometric.

## Layout

```
src/hmm.ts          forward-backward, Baum-Welch, Viterbi, filtering (no deps)
src/hsmm.ts         explicit-duration HSMM: segmental EM, run-length filtering.
                    The recursions take a chain of log parameters, so both EM
                    and variational EM drive the same E step
src/features.ts     candles -> features, train-only standardization
src/confluence.ts   multi-timeframe stack, completed-bar alignment
src/backtest.ts     walk-forward harness, signal, trade ledger, metrics
src/perp.ts         leveraged account: notional fees, funding, liquidation
src/diagnostics.ts  permutation test (shuffle/rotate nulls), ceiling analysis
src/vb.ts           Bayesian HMM and HSMM by variational EM: ELBO, expected-log
                    -parameter forward-backward (Markov and segmental), conjugate
                    M step, K selection, i.i.d. draws
src/mcmc.ts         Bayesian HMM by Gibbs sampling: FFBS, conjugate draws.
                    The exact reference the variational fit is checked against
src/posterior.ts    shared by all three engines: priors, conjugate draws, credible
                    intervals on parameters, on dwell times and on the signal
src/hyperliquid.ts  perp candles, funding, market list
src/sources.ts      GeckoTerminal + DexScreener, paging, gaps, cache
src/data.ts         CSV parsing, synthetic regime-switching generator
src/sweep.ts        every coin x timeframe x model, ranked on excess over the
                    permutation null, with the exposure cap that stops
                    buy-and-hold from winning
src/paper.ts        paper account: fills, fees on notional, funding,
                    liquidation on intrabar extremes, replay and live stepping
src/charts.ts       braille line charts, candlesticks, regime strips, axes
src/tui/            the terminal: model.ts (state + key map), app.ts (layout),
                    worker.ts (all compute, off the render thread),
                    live.ts (closed-bar polling), store.ts (disk), index.ts
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
sweep       every coin x timeframe x model over one window, ranked
posterior   credible intervals on the state means and on the edge
            (variational EM by default, --gibbs for the sampler,
             --hsmm to infer the dwell instead of assuming it geometric)
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
Posterior   --select-states 2,3,4,5  --draws 4000  --kappa0 0.5
            --hsmm  --max-duration 30  --alpha-dur 0.05
            --gibbs  --iterations 1200  --burn-in <n>  --thin 3
Walkfwd     --train 1500  --test 500  --bars-per-year <n>  --verbose
```

## Limits worth knowing

- Gaussian emissions understate crypto tails. The features are chosen to be as
  close to Gaussian as possible (log vol rather than vol), but a genuine 40%
  candle is far outside what the model thinks can happen.
- The signal is a point estimate everywhere except `posterior`. If you build on
  this, gate on `P(edge x hold > round trip)` rather than on the plugged-in
  mean — and remember that clearing the hurdle in expectation says nothing about
  the variance of any individual trade.
- `K` defaults to 3 everywhere except `posterior --select-states`, which picks
  it by ELBO. Read the section above before believing the number it returns:
  on real series it selects for tail-fitting as much as for regimes.
- The HSMM costs roughly 15x the HMM to fit (O(T·K·maxDuration) per EM sweep).
  Lower `--max-duration` if that bites; 30 was as good as 60 in testing.
- Hyperliquid retains ~5000 candles per interval, so 5m gives ~17 days and 1h
  gives ~208. Pick the interval for the history you need.
- One backtest is one draw. With fewer than a few dozen trades the headline
  number carries almost no information — removing four leading bars from one
  5004-bar series flipped a result from +33.4% to 0.0%.
- `validate` is calibrated for ONE test on ONE market. Scanning many markets and
  reporting the best is a different experiment, and its p-values are not the
  ones printed. Divide your alpha by how many you tried.
- Past backtest ROI does not predict future ROI. Measured across 9 tokens split
  into halves, rank correlation was **0.033**, while correlation between
  strategy ROI and buy & hold was **0.950** — the "high ROI" names are simply
  the ones that went up. Do not select tokens this way.
- Nothing here models MEV, honeypots, or the chart ending at zero. Pool
  liquidity picks which pool to read; it does not model your own price impact.
- Backtest results are not a forecast, and none of this is financial advice.
  Slippage on a thin book is worse than any fixed bps assumption, and worst
  exactly when the model wants to trade.
