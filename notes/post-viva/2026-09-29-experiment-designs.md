# Experiment designs for Track B steps 3 to 5

Written 29 September 2026, before any of these runs. This is the Gate 5
plan. It fixes the questions, data, search spaces, selection rules and
expected results in advance, so that the results cannot shape the
design after the fact. The evidence for each choice is in
`2026-09-29-gates-2-4-audit.md`, and the literature is in Themes E, F
and G of `lit-review-right-data.md`.

Steps 3 and 4 of the improvement plan are merged here. The 20-seed run
is the final stage of the fair-tuning experiment, because running 20
seeds of untuned settings would repeat the fault the audit found.

## Experiment 1: fair tuning with 20 seeds

### Question

Given an equal tuning budget each, with the reward scale treated as a
setting to search, how do DQN, REINFORCE and PPO rank on simulated
months? Does any of them differ from exposure-matched buy-and-hold on
real months?

### Data

Training uses the dissertation's 3,000 simulated months, so that the
tuning is the only change. Two new splits are drawn from the same
generator with new seeds. The tuning set has 300 months (100 per
regime, base seed 40,000). The test set has 600 months (200 per regime,
base seed 50,000). The test set is generated at the start, its file
hash is recorded here, and it is scored once at the end.

Real months are used in two ways. The 180 transfer months (2006–2017
and 2023–2025) and the AAPL and QQQ series are re-analyses, because
they have been read before. The clean real hold-out is SPY from April
1993 to December 2005, 153 months that no part of the project has
touched. SPY started trading on 29 January 1993, and the features need
21 trading days of history. February 1993 had too few trading days, so
April 1993 is the first month with a full warm-up. The data is
downloaded once, before any run, and its file hash is recorded below. Yahoo Finance adjusts past prices for dividends
and splits at download time, so the frozen file, not a fresh download,
is the hold-out. These years had different interest rates and a younger
ETF market, so the hold-out tests transfer across time rather than a
draw from the same distribution. It is not opened until the final
policies are fixed.

### Frozen data

`freeze_post_viva_data.py` generated and downloaded this data on 29
September 2026, before any Experiment 1 run. It printed counts and hashes
only, so no result on either set has been seen. The full id lists are in
`data/post_viva_data_meta.json`.

| File | Contents | SHA-256 |
|---|---|---|
| `data/sim_post_viva.npz` | tuning set (300 months) and test set (600 months) | `62626159296cc1bbc475a1c543cf30e2dfcce3636d0e44b4cd01b6a23592abd3` |
| `data/spy_holdout_1993_2005.npz` | 153 hold-out months, 1993-04 to 2005-12 | `620e7d2d553302942819eedc203382e449758473846cf5b10e6b47b54313b89a` |
| `data/spy_holdout_1993_2005_raw.csv` | the raw daily download behind it | `0815ff14a7cbdaad4682a24ba7a7e42fddd5bfa65aef77c1e0be1493f4030754` |

### Search spaces

Each method gets a random search of 24 configurations drawn without
replacement from its own declared grid, with a fixed sampling seed, and
3 tuning seeds (101, 102 and 103) per configuration. That is 72 runs
each. Random search with an equal number of trials is used because the
three grids differ in size, and equal counts of hand-picked cells would
not make them equally good searches. The network stays 64-32 for all
three.

Two settings are shared by all three grids. The learning rate is always
searched. The discount factor is searched over {0.99, 1.0}, because the
reported objective is the undiscounted wealth change over a 20-step
month, and the dissertation's 0.99 was a default rather than a choice.

| Method | Grid the 24 trials are drawn from | Grid size |
|---|---|---|
| DQN | reward scale {1, 1e-4} × learning rate {3e-4, 1e-3, 3e-3} × discount {0.99, 1.0} × target sync {250, 1000} × exploration decay fraction {0.2, 0.5} | 48 |
| REINFORCE | learning rate {1e-4, 3e-4, 1e-3, 3e-3} × discount {0.99, 1.0} × entropy coefficient {0, 1e-4, 1e-3, 1e-2} | 32 |
| PPO | reward scale {1, 1e-4} × learning rate {3e-4, 1e-3, 3e-3} × discount {0.99, 1.0} × entropy coefficient {0, 1e-4, 1e-3, 1e-2} × rollout length {512, 2048} | 96 |

REINFORCE does not search the reward scale, because it standardises
its returns within each month and the scale cancels. A test in
`test_learners.py` checks this: dollar and scaled rewards give identical
trades. The entropy range for the two policy-gradient methods is the one
the review specified, so REINFORCE and PPO share one exploration
control, as Dr Nguyen's grouping suggests.

### Selection rule

For each method, the configuration with the highest mean monthly
wealth change on the tuning set, averaged over the three tuning seeds,
is selected. Ties within $1 go to the smaller spread across seeds. No
other criterion is used, and no configuration is chosen by looking at
the test set or any real month. All 72 tuning scores per method are
reported, together with the selected configuration's rank on each of
the three tuning seeds, so a reader can see whether the choice was
stable or a lucky draw.

### Final runs

Each selected configuration trains on 20 new seeds (1000 to 1019). The
dissertation's default configuration for each method also runs on the
same 20 seeds, so the effect of tuning is measured directly. That makes
six cells of 20 runs. Every run is scored on the new simulated test
set, the 180 transfer months, AAPL, QQQ and the 1993–2005 hold-out.

### Logging

PPO logs policy entropy, approximate KL divergence, clip fraction,
value loss and the gradient norm before clipping, once per rollout.
REINFORCE logs mean policy entropy per month. DQN logs the mean
absolute Q-value and the loss. These logs make a collapse visible while
it happens, which the dissertation could not show.

### Statistics

The analysis reuses `run_stats.py`. Simulated results get Welch t-tests
across seeds and seed-level bootstrap intervals. Following Agarwal et
al. (2021), they also get the interquartile mean across seeds and the
probability that one method beats another on a random seed pair. Real
results get the two-level bootstrap (seeds, then 6-month blocks inside
each window), paired with buy-and-hold and with exposure-matched
buy-and-hold. Real results are also reported in excess of the
three-month Treasury-bill rate (FRED series DTB3), because cash in the
environment earns nothing. The share of seeds that respond to the
market state is reported for every cell.

There is one primary comparison: tuned DQN against tuned PPO on the
mean monthly wealth change over the new simulated test set, two-sided.
Every other comparison is secondary. The secondary family (each tuned
method against buy-and-hold, exposure-matched buy-and-hold and the
Treasury-bill benchmark, on each real data set) is corrected with the
Holm method.

Failing to find a difference is not evidence of no difference. For the
real-market comparisons, equivalence is tested with two one-sided tests
against a margin of $20 per month on $10,000, which is 2.4% a year. A
gap smaller than that would not justify an active strategy over holding
the index. A run that crashes is rerun with the same seed. No run is
dropped for its result.

### Expected results, stated in advance

1. Tuned PPO responds to the market state in at least 15 of 20 seeds.
   The Gate 3 check suggests this, because scaled PPO did so on all
   three validation seeds.
2. No ranking of the three methods on simulated months is predicted.
   The primary comparison is reported with its interval whichever way
   it falls.
3. On real months, no method's gap to exposure-matched buy-and-hold is
   distinguishable from zero after the Holm correction. Gate 2 predicts
   this, because the price features gave no linear forecast of the rest
   of the month in real data. Whether the gap is also inside the $20
   margin is reported but not predicted, because the current interval
   ($43 half-width) is wider than the margin.
4. The same holds on the 1993–2005 hold-out.

If expected result 2 reverses the dissertation's ranking, the revised
report must withdraw the claim that DQN beats PPO in the simulator. The
Track A owner is told through a handoff before the text is finalised.

### Cost

On this machine (Apple M4, 10 cores), one 300,000-step run takes about
220 seconds for DQN, 70 to 130 for PPO and 40 to 60 for REINFORCE. The
216 tuning runs take about 8 hours in series. The 360 final runs take
about 10 hours in series. With four jobs in parallel and one PyTorch
thread each, the whole experiment takes about 5 hours. Each (method,
seed) pair runs as its own `agentctl` job with its own result file, so
an interrupted run loses one unit and resumes from the rest.

## Experiment 2: a simulator with a hidden, persistent state

This experiment starts only after Experiment 1 is finished. Its design
is fixed here so that it is shaped by the Gate 2 measurements, not by
Experiment 1's results.

### Question

If the market state is hidden and persistent, as in real data, does
the ranking of the tuned methods change, and does the transfer to real
months improve?

### Generator changes

The first change makes the path continuous. The generator produces one
long daily path, and months are cut from it. The warm-up history of each
month is then the real preceding days of the same path, as in the
real data, so the features at bar 0 describe the previous state rather
than the current one.

The second change hides the state. A hidden Markov model with two or
three states is fitted to the daily returns of the calibration window
(2018–2022), with the number of states chosen by BIC. The states differ
mainly in volatility, as Ang and Timmermann (2012) and Nystrup et al.
(2015) found. The state persists across months under the fitted
transition matrix, so months are no longer independent. Episode labels
record the state path for analysis, but the agent never sees them.

A third change, volatility clustering within a state (GARCH), is added
only if the acceptance test below fails without it. Carrying holdings
from one month into the next changes the decision problem itself, so it
is left for a later cell once this generator passes.

### Acceptance test, before any training

The new generator is measured with `gate_audit.py` against the 60
calibration months, which are the only real months it may be fitted to.

| Property | Calibration months | Pass band |
|---|---|---|
| Excess kurtosis of daily returns | 11.5 | at least 5 |
| Lag-1 autocorrelation of absolute returns | 0.39 | at least 0.25 |
| Month-to-month autocorrelation of volatility | 0.40 | at least 0.25 |
| Realised-label accuracy from bar-0 features, minus the majority-label rate | −2.3 points on transfer months | at most +5 points |
| Rest-of-month R squared at bar 0 | −0.15 on transfer months (interval −0.41 to +0.01) | at most 0.05 |

The last two rows matter most. They require the simulator to carry
about as little direction signal as the real market does. A simulator
that passes the return statistics but still hands the agent the regime
at bar 0 fails.

### Training and evaluation

The three configurations selected in Experiment 1 train on 15,000
fresh months from the new generator, about 106 decisions per DQN
weight, with 20 seeds each. They are scored on a fresh test set from
the same generator, the transfer months and the 1993–2005 hold-out. The
statistics and logging follow Experiment 1.

### Expected results, stated in advance

On the new simulator, all three methods earn less than on the old one,
because the state must be inferred. Their advantage, if any, moves from
direction to volatility: less exposure in high-volatility states rather
than correct calls on direction. No method is predicted to beat
exposure-matched buy-and-hold on real months.

## What these experiments cannot answer

Neither experiment changes the inputs. Both keep the five price
features, so neither can answer Prof Nikitopoulos's question about the
drivers of value. That is the domain-first study, which starts from a
linear regression of forward returns on the drivers named in Theme A of
the review. These two experiments make the existing comparison fair and
the simulator honest, so that the domain-first study starts from a
correct baseline.
