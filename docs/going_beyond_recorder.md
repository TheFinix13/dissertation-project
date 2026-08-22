# Going beyond “recorder” — what we add that Nguyen did not spoon-feed
Last updated: 2026-07-30

Nguyen’s complaint: don’t just paste what he said and ask “did I record
correctly?” Show you **understood**, then **did something extra**.

## Already added (doer signals)

| Extra | Why it matters | Where |
|---|---|---|
| Explicit cash `C` + shares `n` (not vague Hold flag `H`) | He asked; we fixed and coded it | `env.py` |
| Accounting identity tests | Proves reward = ΔW always | `test_env.py` |
| Feasibility mask + illegal-action rate | Shows constraint thinking | Phase-1 results JSON |
| Chronological month split | No shuffle leakage | `fetch_spy_daily.py` |
| Baselines with named objectives | Exp outline he drew | results table |
| **Fair buy-max baseline (B1b)** | Caught PPO collapse to BAH | Phase-1 table |
| **Honest collapse writeup** | Doer: report “PPO = B1b” not fake win | memo §4 |
| Honest episode definition | Daily-month ≠ 390 minutes — we say so | meta JSON / memo |
| Phase 0 before Phase 1 | His homework, executed with numbers | CartPole curve |

## High-value next extras (pick 1–2 before the meeting if time)

1. **Tiny from-scratch REINFORCE** on Discrete(3) (no SB3) for one synthetic month  
   - Proves the update is not a black-box API (he attacked that).  
   - Keep SB3 for the main table; show REINFORCE as “I can derive it”.

2. **Reward variant row** (his MSE→CE analogy)

   ```text
   A1:   r  =  ΔW
   A1λ:  r  =  ΔW  −  λ · D     # drawdown penalty
   ```

   Same network, same data — only the score changes.

3. **Fee sensitivity** `c ∈ {0, 5bps, 20bps}`  
   - Table of mean ΔW — shows economic thinking, not only ML.

4. **Failure notebook**  
   - One test month where PPO loses to buy-and-hold; plot actions + wealth;  
     write 5 sentences on *why* (overtrading, missed drift, etc.).

5. **Minute-bar upgrade plan with one real day**  
   - Download one SPY day at 1-min; run same env with `T ≈ 390`;  
     prove the API is horizon-agnostic.

## What NOT to add yet (looks like scope creep / recording again)

- Uncertainty LSTM / dual-path / 70-ticker grid  
- Fancy architecture diagrams without new numbers  
- Claiming Phase-2 v0.17 results as Iteration-1 evidence  

## Viva one-liner

> “We didn’t only write down your MDP — we implemented it, tested the
> accounting identity, measured how often the policy proposes illegal
> trades, and compared against baselines with explicit objectives on
> held-out SPY months.”
