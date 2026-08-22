# State-space design: the ladder, the completeness argument, and the re-run plan

Working design note. Written 22 Aug 2026, ahead of the final Chapter 5 re-run.
Everything in Section 1 is taken from committed result files; Sections 3–6 are
design decisions for the re-run and are not yet implemented.

---

## 0. Why this note exists

Two separate things were being conflated:

1. **The ladder** — the historical progression of state representations actually
   built in this project, and what each one taught us.
2. **The completeness argument** — whether a state can "see the whole world" in
   the sense Nguyen asks for, and how to defend wherever we stop.

A Gemini-generated draft of a "complexity ladder" was considered and rejected.
It described an `R^2` Iteration 0 (`[ΔP, h]`) and an `R^7` Iteration 3 using
`RSI_14` and `ATR_14`. Neither exists in this repository: there is no 2-D
environment, RSI and ATR are never computed anywhere in the codebase, and the
final state is 9-D, not 7-D. Publishing that ladder would break the
equation → code → test traceability in Appendix A, which is the single thing
supervision has pushed hardest on. The *features* it suggests are reasonable
(ATR in particular is adopted below); the problem is claiming them without
implementing them.

---

## 1. The ladder as actually built

All figures below are from the committed JSON results, evaluated on the same
26 held-out test months (Nov 2022 – Dec 2024), `C_0 = $10,000`, fee = 5 bps.

### Iteration 1 — `experiments/iteration1/env.py` (`OneStockDiscreteEnv`)

State (3-D): `s = [ΔP_t, C_t, n_t]` — one-step return, raw cash, integer share count.
Actions: Buy one share / Sell one share / Hold, feasibility-masked.
Source: `experiments/iteration1/results/phase1_algo_ablation.json`

| Method | mean ΔW | std ΔW | trades/mo | months identical to B1b |
|---|---|---|---|---|
| B0 do-nothing | $0.00 | 0.00 | 0.0 | 0 |
| B1 buy-one-hold | $7.97 | 16.67 | 1.0 | 0 |
| B1b buy-max-hold | $76.91 | 178.68 | 19.1 | 26 |
| B2 random legal | $14.14 | 32.93 | 12.8 | 0 |
| REINFORCE | $76.91 | 178.68 | 19.1 | **26** |
| A2C | $76.91 | 178.68 | 19.1 | **26** |
| PPO | $76.91 | 178.68 | 19.1 | **26** |

**Finding.** Three structurally different optimisers produced *byte-identical*
behaviour to fully-invested buy-and-hold on every single test month. With no
clock, no entry reference and no market context beyond one lagged return, the
state offers nothing to condition on, so the only thing left to learn is the
unconditional corner solution: be maximally exposed to an asset that drifts up.

Secondary finding, worth keeping: against the weak baselines (B0, B1, B2) this
looks like a 5–10× outperformance. Against the honest baseline (B1b) it is
exactly zero. This is where the project's baseline discipline came from.

### Iteration 2 — `experiments/iteration1/env_v1.py` (`OneStockDiscreteEnvV1`)

State (5-D): `s = [ΔP_t, PnL_t, τ_t, C_t/C_0, n_t P_t/C_0]`.
Adds unrealized P&L versus average entry, an episode clock, and scaling by
initial capital. Actions unchanged (still whole-share).
Source: `experiments/iteration1/results/phase1_iteration2_results.json`

| Method | mean ΔW | std ΔW | trades/mo | months won vs B1b |
|---|---|---|---|---|
| PPO (5-D) | **$0.00** | 0.00 | **0.0** | 8 / 26 |
| B1b buy-max-hold | $76.91 | 178.68 | 19.1 | — |

**Finding.** The agent stopped trading entirely. It never opened a position in
any of the 26 test months. It "wins" 8 months only in the sense that holding
cash beats being long during a falling month.

This is the most instructive rung on the ladder. Adding *internal*
self-knowledge while leaving the agent blind to the market moved it from one
corner of the policy space (always fully invested) to the opposite corner
(always flat). Both corners are unconditional policies. The state changed; the
*conditionality* did not. That is what pointed to the diagnosis: the missing
ingredient was not more introspection but external market context, plus an
action space in which partial exposure is reachable.

### Iteration 3 — `experiments/final_model/env_full.py` (`FullStateTradingEnv`)

State (9-D): market `[ΔP, mom_5, vol_5, P/MA_10 − 1]`, clock `[τ]`,
balance sheet `[C/C_0, hP/C_0, PnL, W/C_0 − 1]`.
Actions: divisible asset, fixed monetary slice `Δ = 0.1·C_0`.
Source: `experiments/final_model/results/full_ablation_results.json`

| Method | mean ΔW | std ΔW | trades/mo | Sharpe (monthly) | MDD | win vs B1b | identical to B1b |
|---|---|---|---|---|---|---|---|
| B0 do-nothing | $0.00 | 0.00 | 0.0 | 0.000 | 0.0% | 30.8% | 0 |
| B1b buy-and-hold | $125.19 | 309.87 | 11.0 | 0.404 | 5.96% | — | 26 |
| B2 random legal | $37.97 | 80.43 | 14.2 | 0.472 | 0.93% | 30.8% | 0 |
| REINFORCE | $125.19 | 309.87 | 11.0 | 0.404 | 5.96% | 0.0% | **26** |
| DQN | $123.23 | 273.97 | 13.2 | 0.450 | **4.85%** | 34.6% | **0** |
| PPO | $55.67 | 110.45 | 4.0 | 0.504 | 1.85% | 30.8% | 0 |

**Finding.** DQN is the first agent on the ladder to produce genuinely
*interior* behaviour: it is not identical to buy-and-hold in any month, gives up
$1.96/month of mean return, and buys back 1.1 points of maximum drawdown. That
is a risk/return trade, discovered rather than imposed. REINFORCE still
reproduces buy-and-hold exactly, and PPO is bimodal across seeds.

**Conclusion of the ladder.** Once the state carries market context and the
action space permits partial exposure, the remaining corner-seeking is
attributable to the *reward*, not to the state. Under a risk-neutral
`r = ΔW` reward and an asset with positive drift, maximum exposure genuinely
*is* optimal, so an agent that finds buy-and-hold has not failed — it has
solved the problem as posed.

### Design flaw in the ladder as run

I3 changed **both** the state (5→9 features) **and** the action space
(whole-share → divisible slice) in one step. Its behaviour therefore cannot be
attributed to either change. The re-run must decouple them; see Section 5.

---

## 2. Can a trading state "see the whole world"?

The productive move is to stop treating the state as one object and split it by
who controls it.

### Block A — endogenous state (the account)

Cash, units held, cost basis, step index, and running peak wealth.

This block can be made **complete, and the claim is provable rather than
aspirational**. Conditional on the price path, the evolution of the account
under any sequence of actions is exactly determined by these quantities: there
is no hidden variable, no estimation, no noise. The transition is an accounting
identity that the environment computes in closed form. This is where Nguyen's
Flappy Bird standard is fully attainable, and the report should say so in those
terms.

### Block B — exogenous state (the market)

Here completeness is impossible **in principle**, not merely in practice. No
finite vector of price-derived observables makes the future conditionally
independent of the past: order-book depth and flow, news arrival, and the
positions and intentions of other participants are unobserved by construction.
This is the standard POMDP position (Kaelbling, Littman & Cassandra 1998) and,
read from the other side, the informational-efficiency argument (Fama 1970).

### The amendment the viva needs

The Flappy Bird analogy is pedagogically excellent and needs one honest caveat.
Flappy Bird's world is finite and deterministic, so a handful of variables is
literally sufficient. A market's is neither. The achievable target is therefore
not a sufficient state but an **approximate information state**: complete on
Block A, and rich enough on Block B that adding further features provably stops
changing behaviour. That second clause is what converts "we chose nine
features" into "we demonstrated saturation at N features", and there is formal
machinery for exactly this notion (Subramanian, Sinha, Seraj & Mahajan,
JMLR 2022).

---

## 3. Two defects in the current 9-D state

### (a) Feature 9 is a linear combination of features 6 and 7

```
W_t/C_0 − 1  =  C_t/C_0  +  h_t P_t/C_0  −  1
   s_9              s_6           s_7
```

It carries no information the network cannot already form with a single linear
layer. Keeping it is harmless and mildly helpful (it saves the network from
learning the identity), but it must be described in the report as a *derived
convenience feature*, not as added information. An examiner who notices this
before we do will enjoy it more than we will.

### (b) The risk-aware reward run in §5.6 was formally non-Markov

The shaped reward used in `run_supplementary.py` is

```
r'_t = ΔW_t − λ · max(0, D_t − D_{t−1}),   D_t = max_{u≤t} W_u − W_t
```

`D_t` depends on running peak wealth, which is **not in the 9-D state**. The
reward therefore depended on information the agent could not observe, and the
Markov property was violated. This is a plausible explanation for REINFORCE
degenerating to zero trades under that reward.

Rule to carry forward: **any quantity the reward depends on must be in the
state.** If the final report keeps a drawdown-penalised reward, drawdown (or
peak wealth) becomes a mandatory state feature, not an optional one. This is
worth reporting as a methodological finding rather than quietly fixing.

---

## 4. The agreed final architecture

Settled after review by two independent readers. My earlier proposal was 15
features; it was reduced on the argument that a state should be extended only to
resolve a stated limitation, and that folding the risk feature into the core
would make its effect unmeasurable. Implemented in
`experiments/final_v2/features.py`.

Core state, **S3 ∈ R⁹**:

| # | Feature | Definition | Block | Resolves |
|---|---|---|---|---|
| 1 | `ret_1` | `P_t/P_{t−1} − 1` | market | immediate price movement |
| 2 | `mom_k` | `P_t/P_{t−5} − 1` | market | trend, not just the last tick (Jegadeesh & Titman 1993) |
| 3 | `vol_k` | sd of last 5 one-step returns | market | current risk level; volatility clusters (Engle 1982) |
| 4 | `ma_gap` | `P_t/MA_20 − 1` | market | position relative to trend |
| 5 | `atr_norm` | `ATR_14/P_t` | market | intraday range risk, invisible in closes — **uses High/Low** |
| 6 | `clock` | `t/T` | time | time before forced liquidation |
| 7 | `cash_frac` | `C_t/C_0` | portfolio | funds available |
| 8 | `exposure_frac` | `h_t P_t/C_0` | portfolio | what the position is worth |
| 9 | `pnl` | return vs average entry | position | makes stop-loss / take-profit learnable |

Risk branch, **S4 ∈ R¹⁰**, adds `drawdown = (max_u W_u − W_t)/max_u W_u`.

Deliberately excluded, each for a stated reason:

- **`wealth_frac`** — pointwise recoverable, since `cash_frac + exposure_frac = W_t/C_0`.
- **RSI** — a function of the close series the agent already observes, so it
  re-represents rather than informs.
- **`rel_vol`, `vol_ratio`** — genuinely new information, but held back as
  optional rung S5 so their contribution is measured rather than assumed.
- **VIX and cross-asset context** — a second data dependency; future work.

### The test that replaced "is it a technical indicator?"

A feature is excluded only if it is **fully recoverable from data the agent
already observes**. That distinction is what separates RSI (excluded) from ATR
(included): no function of the close series can recover the day's high, so a day
that closed flat after a 1% range and one that closed flat after a 12% range are
otherwise identical to the agent.

### The binding constraint that must be stated

60 training months is roughly 1,250 daily transitions. At that sample size each
additional input buys variance rather than skill — the curse of dimensionality
(Bellman 1957) in its most practical form, and the backtest overfitting problem
López de Prado (2018) documents. Completeness is therefore established by
**demonstrated saturation**, never by adding everything available.

---

## 5. How to prove we stopped in the right place

### 5.1 The ladder as a designed experiment

Implemented as named rungs in `features.LADDER`, all sharing one environment
code path so a difference between rungs cannot be an implementation difference.

| Rung | Dim | Adds | Remaining limitation |
|---|---|---|---|
| S0 | 1 | `ret_1` | cannot tell whether it holds anything |
| S1 | 3 | cash, raw unit count | knows the count, not what it is worth |
| S2 | 4 | exposure, unrealised P&L | knows its position, not the time left |
| S3a | 5 | clock | market context is one lagged return |
| S3 | 9 | trend, risk, range | no memory of the path to current wealth |
| S4 | 10 | drawdown | no volume or regime information |
| S5 | 12 | relative volume, volatility ratio | order flow and news remain unobserved |

**Stopping rule, declared before running anything:** the ladder terminates at the
first rung where the next rung's change in mean ΔW, intra-month maximum drawdown
and trades per month all fall inside the seed-to-seed band over seeds
{42, 43, 44}. Pre-registering the rule makes the stopping point a result rather
than a time-budget excuse.

**Compute discipline.** The ladder is a claim about state, so the algorithm is
held fixed across rungs (Deep Q, three seeds). The full three-algorithm
comparison runs only at the frozen state. Six rungs × three algorithms × three
seeds would be 54 trainings and would not strengthen the argument.

### 5.2 The state × reward grid is not free

Applying the admissibility criterion rules out one cell rather than leaving it to
be explained away afterwards.

| | R1 = `ΔW` | R2 = `ΔW − λ·Δdrawdown` |
|---|---|---|
| S3 (R⁹) | valid — headline baseline | **inadmissible**: reward not measurable from state |
| S4 (R¹⁰) | valid — **control** | valid — treatment |

Three cells. `S4 × R1` is the control that answers the obvious examiner
question: under a pure wealth-change reward the agent does not need drawdown, so
comparing `S4 × R1` against `S3 × R1` measures whether the extra input costs
anything. The inadmissible cell is now impossible to construct, since
`TradingEnv` raises when `risk_lambda > 0` without `drawdown` in the state.

Baseline admissibility check for the headline cell: `cash_frac + exposure_frac =
W_t/C_0`, so `r_t = C_0[(c+v)_{t+1} − (c+v)_t]` is a function of
`(s_t, a_t, s_{t+1})` with no hidden variable. S3 is a valid MDP state under R1.

### 5.3 The action model is a second axis

State design and action design are separate representation choices, and the
earlier run changed both at once so neither could be attributed. One comparison
settles it: the frozen state under R1 with `action_model="share"` versus
`action_model="slice"`. Not crossed with every rung.

---

## 6. What stays unobservable, and how to report it

Even at 16 features, the agent cannot see order-book depth or order flow, news
text, other participants' positions and intentions, the latent market regime, or
scheduled macroeconomic releases. Two scope facts also need stating plainly,
because the project title says *portfolio*:

- **One asset.** A genuine portfolio state requires a weight vector across
  assets and their covariance structure (Markowitz 1952; Merton 1969).
- **Long-only.** No shorting, so the agent cannot express a negative view; it
  can only decline to participate.

The defensible closing claim is therefore: *complete on the account side by
construction, approximately sufficient and empirically saturated on the market
side, with residual partial observability acknowledged and its remedy —
recurrent policies or explicit belief states (Hausknecht & Stone 2015; Ni,
Eysenbach & Salakhutdinov 2022) — identified as future work.*

That is a stronger position than an unqualified completeness claim, and it is
the one the evidence actually supports.

---

## 7. Citation map

Each reference is tied to the specific claim it supports, so nothing sits in the
bibliography uncited.

| Claim | Reference | In bib? |
|---|---|---|
| Belief state / incomplete state information | Åström (1965) | add |
| Partial observability formalism | Kaelbling, Littman & Cassandra (1998) | have |
| Approximate information state; principled "approximately complete" state | Subramanian, Sinha, Seraj & Mahajan, JMLR (2022) | add |
| Informational efficiency of prices | Fama (1970) | have |
| Wealth as the endogenous state in portfolio choice | Merton (1969) | add |
| Portfolio state needs weights and covariance | Markowitz (1952) | have |
| Stylized facts justifying the market feature block | Cont (2001) | add |
| Volatility clustering ⇒ volatility features are predictive | Engle (1982); Bollerslev (1986) | add |
| Multi-horizon momentum | Jegadeesh & Titman (1993) | add |
| Volume carries information about price moves | Karpoff (1987) | add |
| Multi-horizon return features in deep RL trading | Zhang, Zohren & Roberts (2020) | add |
| Origin of ATR / RSI | Wilder (1978) | add |
| Position must be in the state for a trading agent | Moody & Saffell (2001) | have |
| Limited observation space causes overfitting | Théate & Ernst (2021) | have |
| Curse of dimensionality | Bellman (1957) | have |
| Backtest overfitting, feature discipline, sample size | López de Prado (2018) | have |
| Recurrent remedies for partial observability | Hausknecht & Stone (2015); Ni et al. (2022) | add |

Every "add" entry must be verified against its publisher record before it enters
`references.bib`, consistent with how the existing 34 entries were audited.

---

## 8. Execution checklist

Built on branch `final-v2-state-architecture`, in `experiments/final_v2/`.
The audit that accompanied this work is in `docs/audit_v2.md`.

- [x] Data extended through 2025 with OHLCV retained (96 months)
- [x] Market features computed on the continuous series, not per episode
- [x] Three-way chronological split: 60 train / 12 validation / 24 test
- [x] All ladder rungs S0–S5 defined and sharing one environment code path
- [x] Drawdown implemented; admissibility enforced at construction
- [x] Buy-and-hold baseline corrected, both variants reported
- [x] Intra-episode drawdown added to the metrics
- [x] `MaskablePPO` replacing unmasked PPO; sampler seeded by the run seed
- [x] 21 tests, each tied to a claim in Chapter 3 or 4
- [ ] `run_ladder.py` — saturation study with the pre-declared stopping rule
- [ ] `run_final.py` — three algorithms, three admissible cells, action-model comparison
- [ ] Verify and add the outstanding bibliography entries from Section 7
- [ ] Rewrite Chapter 3 §3.4 as the agreed design ladder
- [ ] Update Chapter 4 environment and verification sections
- [ ] Regenerate Chapter 5; update Appendix A traceability line numbers
