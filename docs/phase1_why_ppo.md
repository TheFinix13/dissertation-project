# Why PPO? — the evidence, not the habit
Last updated: 2026-08-06 · branch `simple-modelling-clean`

Nguyen will ask: *"Why PPO and not the others? Could the others trade?"*
This note is the complete answer, with numbers from our own runs.

---

## 1) Could each algorithm trade at all?

| Algorithm | Feasible on our MDP? | Efficient / suitable? |
|---|---|---|
| Tabular Q-learning | **No** | Cash/prices are continuous → the table needs infinite rows. We proved the point on CartPole, where it works *only after* hand-binning 4 numbers. |
| DQN | Yes | Works (Phase 0: CartPole 500, Lunar 178) but **value-based** — off the policy track Nguyen locked. Kept as Phase-0 comparator only. |
| REINFORCE | Yes | Policy-based; we coded it from scratch. High-variance updates — fine on short episodes, noisy in general. |
| A2C | Yes | Policy-based + critic, but **unclipped** — one bad update can wreck the policy (its LunarLander score −41 shows this). |
| **PPO** | **Yes** | Policy-based + critic + **clipped ratio** — one update cannot destroy the policy. |

## 2) Phase 0 evidence (games with known outcomes)

| Mean eval return | CartPole (≈475 = solved) | Flappy | LunarLander (≈200 = solved) |
|---|---:|---:|---:|
| Random | 28.8 | −7.4 | −183 |
| REINFORCE (scratch) | 292 | 7.1 | 15 |
| A2C | **500** | 4.6 | −41 |
| PPO | **500** | **12.6** | 176 |
| DQN | **500** | 7.0 | 178 |

PPO is the **only policy method that stays strong on all three games**.

## 3) Phase 1 evidence — the same-MDP ablation (the strong card)

Runner: `experiments/iteration1/run_phase1_algo_ablation.py`
Identical MDP, data split, fee (5 bps), cash ($10k), seed (42), budget (80k steps).
Scratch masked REINFORCE: `experiments/iteration1/reinforce_trading.py`.

| Method | Mean ΔW ($) | Trades/mo | Months identical to B1b |
|---|---:|---:|---:|
| B0 do-nothing | 0.00 | 0.0 | 0/26 |
| B2 random-masked | 14.14 | 12.8 | 0/26 |
| B1 buy-one-hold | 7.97 | 1.0 | 0/26 |
| B1b buy-max-hold | 76.91 | 19.1 | 26/26 (def.) |
| **REINFORCE (scratch)** | **76.91** | 19.1 | **26/26** |
| **A2C** | **76.91** | 19.1 | **26/26** |
| **PPO** | **76.91** | 19.1 | **26/26** |

**All three policy methods converge to the identical buy-max policy** —
the same executed action sequence on every one of the 26 held-out months.

Chart: `reports/generated/charts/phase1_algo_ablation_delta_w.png`
JSON: `experiments/iteration1/results/phase1_algo_ablation.json`

## 4) What the tie means (say this sentence)

> "The collapse to buy-and-hold is a property of the **objective**
> (pure ΔW in a mostly-rising window), not of PPO. Three different
> optimisers found the same optimum — so Iteration 3 changes the
> **reward**, not the algorithm."

And why still PPO as primary, given the tie on trading:

1. On **harder** problems (Flappy, Lunar) PPO was the only consistently strong
   policy method — trading with richer states/rewards will get harder, not easier.
2. Clipping gives stability that REINFORCE/A2C lack:

```text
ρ = π_new(a|s) / π_old(a|s)
score = min( ρ·A , clip(ρ, 1−ε, 1+ε)·A )   # don't trust big policy jumps
```

3. It is the standard in the finance-RL literature (FinRL et al.), so results
   are comparable to prior work.

## 5) The four beats, once more

Every method above is still: **predict → score → differentiate → update**.
Only the *score* differs:

```text
REINFORCE:  L = − Σ log π(a|s) · G
A2C:        L = − Σ log π(a|s) · A        # A = G − V(s)
PPO:        L = − Σ min(ρ·A, clip(ρ)·A)   # clipped
```
