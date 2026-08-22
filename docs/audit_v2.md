# Code audit before the final experimental run

Written 22 Aug 2026 on branch `final-v2-state-architecture`. Every magnitude
quoted here is reproducible with:

```
./venv/bin/python experiments/final_v2/data.py
./venv/bin/python experiments/final_v2/verify_fixes.py
```

which writes `experiments/final_v2/results/verify_fixes.json`.

Twelve issues were found in the code that produced the provisional results. Two
would have invalidated the headline comparison on their own. The old code is left
untouched on `main` for provenance, since Chapter 3 cites it as design history;
all corrections live in the new `experiments/final_v2/` package.

---

## 1. Rolling market features were truncated at the episode boundary

**Severity: high. Affected every learned result.**

`env_full.py` computed momentum, volatility and the trend gap *inside* each
episode with a shortened window:

```python
def _momentum(self) -> float:
    k = min(self.window, self.t)
    if k == 0:
        return 0.0
```

On the first day of every month the agent was told the market had no trend, no
volatility, and sat exactly on its own moving average, whatever had actually
happened the previous week. Momentum does not reset because January ended.

Measured across all 95 episodes with prior history:

| Quantity | Value |
|---|---|
| Days per episode receiving wrong values | 9.0 of ~21 |
| Mean fraction of each episode affected | **43.1%** |
| Worst case | 47.4% of the episode |
| Day-0 momentum was exactly zero | in **all 95 episodes** |
| Day-0 volatility was exactly zero | in **all 95 episodes** |
| True day-0 momentum, mean absolute size | 1.54% |
| Worst single error, momentum | 6.76 percentage points |
| Worst single error, volatility | 3.90 percentage points |
| Worst single error, trend gap | 3.49 percentage points |

The agent was blinded during the part of the episode where it chooses its opening
position, which is the most consequential decision it makes.

**Fix.** `features.compute_market_features` computes every market feature once on
the continuous series; `data.build_episodes` slices the resulting matrix into
months. A warm-up lead is fetched before the nominal start date and discarded, so
even the first retained episode carries a complete 20-day trend and 14-day
average true range. The environment now *receives* features and cannot compute a
window at all, which makes the defect structurally impossible rather than merely
fixed. Regression test: `test_rolling_features_survive_the_episode_boundary`.

---

## 2. The buy-and-hold baseline was a dollar-cost average

**Severity: high. Flattered every agent.**

`policy_bah` bought one slice whenever a Buy was legal, inside an environment
whose slice was a tenth of capital. That needs ten steps to reach full
investment, so across a twenty-day month it averaged 69% exposure. It was a
ten-day dollar-cost-average wearing the name of buy-and-hold, and in a market
that drifts upward it is materially weaker than the strategy it claimed to be.

Measured on the 24 test months, fee 5 bps:

| | Mean ΔW | Mean exposure | Trades | Intra-month MDD (mean) |
|---|---|---|---|---|
| Old "buy-and-hold" (slice schedule) | $141.31 | 0.694 | 11.0 | 2.35% |
| True buy-and-hold | **$180.25** | 0.913 | 2.0 | 3.10% |

The benchmark was understated by **$38.95 per month, or 21.6%**, and true
buy-and-hold beat the slice schedule in 17 of 24 months. Any claim of the form
"the agent matched buy-and-hold" was measured against a target a fifth too low.

**Fix.** Two references are now reported because they answer different questions.
`B1a_slice_bah` is the ceiling for a policy restricted to the agent's own action
granularity, which isolates skill from action-space advantage. `B1b_true_bah` is
fully invested at the first bar, implemented as the same policy in an environment
whose slice is all of capital, and is the benchmark an investor would use.
Beating B1a means timing better than a mechanical schedule with identical powers;
beating B1b means beating the market. Regression test:
`test_true_buy_and_hold_beats_the_slice_schedule_when_prices_rise`.

---

## 3. A drawdown-penalised reward was used with a state that omitted drawdown

**Severity: high. Invalidated the risk-aware experiment.**

`run_supplementary.py` shaped the reward with
`r' = ΔW − λ·max(0, D_t − D_{t−1})`, where `D_t` depends on running peak wealth.
Peak wealth was not among the nine features. The reward therefore depended on
information the agent could not observe, the Markov property failed, and the
derivations in Chapter 3 did not apply to the object being trained. This is a
plausible explanation for REINFORCE collapsing to zero trades under that reward.

**Fix.** The constraint is now enforced in code rather than in prose:

```python
if risk_lambda > 0.0 and "drawdown" not in self.feature_names:
    raise ValueError(...)
```

An inadmissible configuration cannot be constructed. Regression test:
`test_risk_penalty_requires_drawdown_in_the_state`.

---

## 4. Only closing prices were retained

**Severity: medium-high.**

`fetch_spy_daily.py` line 30 was `out = df[["Close"]]`. Open, high, low and
volume were downloaded and thrown away, which makes intraday range risk
unobservable *in principle* rather than merely absent: no function of the close
series can recover the day's high. A day that closed flat after a 1% range and a
day that closed flat after a 12% range were identical to the agent.

**Fix.** The pipeline retains adjusted OHLCV. `atr_norm` uses high and low;
`rel_vol` uses volume. Both are available to the ladder, with `rel_vol` held back
as an optional rung rather than a default.

---

## 5. The split was two-way, so no tuning claim was defensible

**Severity: medium-high.**

Training and test only. Learning rate, network width, step budget, epsilon
schedule and window lengths were all chosen somehow, and with no validation split
there is no way to demonstrate that the choice did not see the test months. An
examiner is entitled to treat every reported number as tuned.

**Fix.** Chronological three-way split with calendar boundaries, which are easier
to state and to defend than percentages:

| Split | Months | Span |
|---|---|---|
| train | 60 | 2018-01 to 2022-12 |
| validation | 12 | 2023-01 to 2023-12 |
| test | 24 | 2024-01 to 2025-12 |

All tuning happens on validation. The test years are evaluated once, at the end.

---

## 6. The data stopped at 2024

**Severity: low-medium.** 2025 is a complete calendar year and was omitted. The
series now runs to 2025-12, giving 96 months instead of 84.

---

## 7. The SB3 agent's seed did not drive month sampling

**Severity: medium. Understated one method's variance.**

`MonthSamplerFullEnv(..., sample_seed=0)` was constructed with a constant while
the scratch agents drew months from a generator seeded by the run seed. The SB3
agent therefore saw the same month order in every "seed", so its seed spread
measured network initialisation only and was not comparable to the spread
reported for REINFORCE and Deep Q.

**Fix.** `MonthSampler` takes the run seed and honours SB3's reseeding, so all
three methods vary over the same two sources.

---

## 8. The SB3 agent trained unmasked while the scratch agents trained masked

**Severity: medium. Different learning problems compared as if identical.**

REINFORCE and Deep Q set illegal logits and values to −∞ before choosing, so
probability mass is renormalised over legal actions. Plain PPO was trained
without masks and relied on the environment forcing a Hold, which is not the same
task: the policy can spend capacity on infeasible trades and receives no signal
distinguishing "I chose Hold" from "my Sell was refused".

**Fix.** `sb3-contrib` installed and `MaskablePPO` used, with `action_masks()`
exposed on the sampler. Verified: illegal-action rate is 0.00 for all three
methods in an end-to-end smoke run.

---

## 9. Drawdown was measured only across months, never within them

**Severity: medium. Understated risk.**

`summarize` built one equity curve by adding each month's profit to a constant
$10,000 stake and took the drawdown of that series. It says nothing about what
happened *inside* a month, which is where a trader's risk actually lives, and it
is not a compounded equity curve either.

On the test months, the old aggregate measure reports 4.67% for buy-and-hold
while the true mean intra-month drawdown is 3.10% and the worst single month
reaches **12.05%**. Across all 96 months the worst intra-month drawdown is
**22.11%**. The reported figure was not a conservative approximation of the real
one; it was a different quantity.

**Fix.** Both are reported and named for what they are: `mdd_intra_mean` and
`mdd_intra_max` for within-episode risk, `mdd_monthly_pnl` for month-to-month
consistency of profits on a constant stake.

---

## 10. `wealth_frac` carried no information

**Severity: medium, presentational.**

Feature 9 of the old state was `W_t/C_0 − 1`, which equals
`cash_frac + exposure_frac − 1` exactly. It is recoverable pointwise and is
therefore convenience, not observation.

**Fix.** Excluded from every ladder rung. It remains implemented so the
redundancy can be demonstrated rather than merely asserted. Regression test:
`test_wealth_frac_is_pointwise_recoverable`, paired with
`test_drawdown_is_not_pointwise_recoverable` which shows why drawdown is a
different case: two price paths ending at the same cash, exposure and wealth
produce different drawdowns, so no set of current-step features can recover it.

---

## 11. The accounting identity was checked against the shaped reward

**Severity: low, but it would have produced a false failure.**

Under a shaped reward the sum of rewards does not equal the wealth change, so
testing `sum(r) == W_T − W_0` fails for a *correct* environment whenever
`λ > 0`.

**Fix.** The environment returns `info["dw"]`, the pure wealth change, alongside
the shaped reward. The identity is checked on `dw` and holds to 0.0 across all 96
episodes.

---

## 12. Smaller items, fixed or documented

- **Feature scaling.** `rel_vol` and `vol_ratio` naturally centre on 1 while
  every other feature centres on 0. Both are now expressed as deviations from 1
  so no single input dominates the first layer.
- **Cost basis convention.** `entry_cost` records the pre-fee cost, so `pnl`
  measures the market move on the position rather than its net-of-cost profit.
  This is a deliberate convention and is now stated in the module docstring.
- **Dust.** A residual position worth less than the minimum trade value is
  flushed on a Sell, so "flat" means exactly flat and the terminal state is
  unambiguous.
- **`shares_raw` in rung S1.** Raw unit count is both badly scaled and
  uninformative about exposure, so S1's deficit relative to S2 is partly
  informational and partly one of scale. This is noted rather than engineered
  away, because the ladder's claim is precisely that a raw unit count is a poor
  state variable.
- **Hard-coded observation width.** `OBS_DIM = 9` was a module constant. Width is
  now derived from the resolved feature set, so every rung runs one code path and
  a difference between rungs cannot be an implementation difference.

---

## What the new package contains

| File | Purpose |
|---|---|
| `features.py` | feature definitions, canonical ordering, the ladder rungs S0–S5 |
| `data.py` | OHLCV fetch with warm-up lead, continuous features, three-way split |
| `env.py` | environment with selectable features, action model and reward |
| `baselines.py` | B0, B1a slice schedule, B1b true buy-and-hold, B2 random |
| `rollout.py` | episode rollout, corrected metrics |
| `agents.py` | REINFORCE and Deep Q from scratch, plus SB3 policy adapter |
| `month_sampler.py` | episode-sampling wrapper exposing `action_masks()` |
| `verify_fixes.py` | measures the magnitude of defects 1, 2, 9 and 11 |
| `test_v2.py` | 21 tests, each tied to a claim in Chapter 3 or 4 |

Status: 21 of 21 tests pass; data cached for 96 months; all three algorithms
train and evaluate end to end with a zero illegal-action rate.

---

## Still to do, once the state architecture is confirmed

1. `run_ladder.py` — rungs S0 to S5, one algorithm, three seeds, pre-declared
   saturation stopping rule.
2. `run_final.py` — three algorithms at the frozen state, three seeds, the
   three admissible state/reward cells, and the one action-model comparison.
3. Rewrite Chapter 3 §3.4 as the agreed design ladder; update Chapter 4's
   environment and verification sections; regenerate Chapter 5; update the
   Appendix A traceability rows against the new line numbers.
4. Verify and add the outstanding bibliography entries from
   `docs/state_design_ladder.md` §7.
