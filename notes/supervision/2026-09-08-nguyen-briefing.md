# Briefing for Nguyen — what changed since yesterday (8 Sep 2026)

Everything you suggested on Sunday has been implemented, and the results
changed the conclusions of the dissertation. This note goes through the
changes in the order you gave them.

## 1. The data

I built a market generator, calibrated on the 2018--2022 training months
only. It produces monthly episodes under three controlled conditions: up,
down, and flat. The drift and volatility of each condition are estimated
from the real months, so an up episode behaves like a real rising month
and a down episode like a real falling month.

The training set is now 3,000 simulated episodes, balanced equally across
the three conditions. There are 300 more for validation and 600 for
testing. This replaces the 60 real months as training data. The 60 real
months are now used only to calibrate the generator.

## 2. The network

The network was reduced from two layers of 128 units (about 18,000
parameters) to a pyramid of 64 and 32 units (about 2,800 parameters).
With 3,000 training episodes, the data now outnumbers the parameters
instead of the other way round.

## 3. The experiments

I did not just add one new experiment. The whole suite was rerun on the
simulated data with the new network: the trade-size calibration, the
drawdown feature test, the risk-penalty sweep, and the fee sensitivity.
All decisions were made on the simulated validation episodes, and the
test sets were touched once.

## 4. The results

The ranking inverted. Deep Q-learning, which earned the least on real
data, now earns +$126.94 per episode on the balanced test set, on all
six seeds, and its actions depend on the state on all six seeds. The
test set is balanced, so no fixed pattern is profitable there; the agent
has to respond to the market to earn anything. REINFORCE and PPO
collapse to fixed corner policies (never trade, or always buy) and earn
nothing. This confirms what you said: the data was the bottleneck, not
the architecture.

Two supporting results also changed. The risk-aware reward, which only
suppressed trading in the first study, now works as intended: it cuts
falling-market losses by 73% for about 1.5% of mean return. And the
profit survives a tenfold fee increase, from 5 to 50 basis points.

## 5. The transfer test

I extended the real test data. The trained policies were run, unchanged,
on every real SPY month outside the calibration window: 2006--2017 plus
2023--2025, which is 180 months and includes the 2008 crisis. Deep
Q-learning was profitable on every seed, held zero exposure through
October 2008, and finished with a higher Sharpe ratio than buy-and-hold
(0.224 against 0.198). Its raw return is lower (+$60.28 per month
against +$80.24), so the claim is a better risk-adjusted return, not a
higher absolute one.

## 6. The document

The dissertation now tells this as two studies. The first study is the
real-data work, compressed to one short section, and its negative results
motivate the main study. Chapter 5 was rewritten around the simulated
experiments and the transfer test. The main body is now exactly 80 pages,
as the handbook asks, with the appendices reduced as well.

## Questions I want to ask him

- Is the two-study framing acceptable, or should the first study be
  compressed further?
- Is the transfer claim worded correctly: higher Sharpe, lower raw
  return, with the 2008 crisis avoided?
- Anything he wants added before submission (bootstrap confidence
  intervals are the main candidate)?
