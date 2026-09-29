# What the statistical tests show

Written 28 September 2026 for Track B, step 1 of the improvement plan. The
final report feedback asked for "statistical rigour around the seed
variation". This note tests the saved results without retraining
anything. The script is `experiments/final_v2/run_stats.py` and the
numbers are in `experiments/final_v2/results/stats.json`.

## How the tests work

Each algorithm has six seeds, and each seed has a profit or loss for each
of the 180 real months. Two things could have come out differently: the
seeds that were drawn and the months that happened. The bootstrap
resamples both, 10,000 times. Months are drawn in blocks of six so that
runs of volatile months such as late 2008 stay together. Blocks stay
inside each of the two windows (2006–2017 and 2023–2025), a correction
made on 29 September that moved no interval by more than a few dollars.
A
one-month-at-a-time version is kept as a check. Buy-and-hold uses the same
months as the agent in every resample, so each comparison is paired.

Three measures are tested. The first is the mean profit per month. The
second is the Sharpe ratio, defined as in the dissertation (per seed, then
averaged). The third is the average of the worst 10% of months, which is
18 months out of 180 and measures how bad the bad months are.

## The simulated test holds up

On the 600 held-out simulated episodes, Deep Q-learning earned +$126.94
per episode with a 95% interval of +$119.25 to +$134.62 across the six
seeds. Buy-and-hold lost $60.29 on the same episodes, and the gap is
significant (p < 0.001). Deep Q-learning also beat REINFORCE (Welch
p = 0.001) and PPO (p < 0.001). The six seeds are the only unit here,
because per-episode results were not saved, so these intervals are
narrow only because the six seeds agreed with each other.

The per-condition intervals tell the same story. In falling markets Deep
Q-learning lost between $80 and $127 per episode, against $710 for
buy-and-hold. The REINFORCE and PPO intervals are wide enough to include
both a profit and a loss in every condition, which matches their seeds
collapsing to different fixed habits.

## The real-market advantage over buy-and-hold is not significant

On the 180 real SPY months, Deep Q-learning's Sharpe ratio was 0.224
against 0.198 for buy-and-hold. The difference of +0.026 has a 95%
interval of −0.111 to +0.164 (p = 0.73). The dissertation's statement that
the agent had a higher Sharpe ratio is true of the numbers, but the test
cannot tell that gap apart from chance. The mean profit was $19.96 per
month lower than buy-and-hold, and that gap is not significant either
(p = 0.36).

The fairer comparison is buy-and-hold scaled down to the agent's own
exposure. Deep Q-learning held stock about 71% of the time, and 78% of a
buy-and-hold position earns $62.65 per month against the agent's $60.28
(p = 0.89). Its worst 10% of months averaged −$469 against −$563 for the
scaled position, which is $94 better on every one of the six seeds.
However, the six seeds share the same 180 months, so they are not six
independent results, and the interval for that $94 runs from −$105 to
+$348 (p = 0.45).

The same pattern holds on the other two assets. Deep Q-learning's Sharpe
ratio was 0.226 against 0.265 on AAPL and 0.240 against 0.249 on QQQ,
with neither difference significant. In the 2008 crisis window
(September 2008 to March 2009) it lost $180 on SPY, $5 on AAPL and $29 on
QQQ, against buy-and-hold losses of $3,046, $3,461 and $2,918. That
behaviour is consistent across assets the agent never saw, but it is one
crisis, so it counts as a single event rather than 540 months of
evidence.

## What this means

The only result that clears a significance test is on simulated data.
There, the balanced conditions give the agent a cause it can read. On
real data the agent behaves in a sensible and repeatable way, most
clearly in 2008, but its advantage over holding less stock is too small
to separate from chance in 180 months. This agrees with the report
feedback that the risk-adjusted result is "genuine but modest", and it
goes further, because at this sample size it cannot be confirmed.

It also agrees with the examiners' main point in the viva. If the inputs
held the cause of price moves, the real-market edge should be large
enough to measure, and it is not. Any future write-up should state the
real-market result as "consistent with lower crash exposure, not
statistically distinguishable from a scaled buy-and-hold". The next steps
in Track B (fair tuning and more seeds) narrow the seed side of the
uncertainty, but only more crisis periods or better inputs can narrow the
month side.
