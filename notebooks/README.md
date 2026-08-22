# Colab notebooks (Nguyen demos)

Two notebooks, one per phase. Both default to `RUN_MODE = "cached"` so every
table and chart loads from committed repo results in seconds.

| Notebook | Covers |
|---|---|
| `phase0_games_colab.ipynb` | CartPole + Flappy, all algorithms, LaTeX objectives, charts |
| `phase1_trading_colab.ipynb` | Trading MDP, why-PPO ablation, Iterations 1–2, honest findings |

## How to open in Google Colab

1. Push / ensure the notebooks are on branch `simple-modelling-clean`.
2. Open on GitHub →
   `https://github.com/TheFinix13/dissertation-project/tree/simple-modelling-clean/notebooks`
3. Click “Open in Colab”, **or** upload the `.ipynb` to Colab.
4. Runtime → Run all. CPU is fine.

## Run modes

| Mode | Phase 0 | Phase 1 |
|---|---|---|
| `"cached"` (**meeting default**) | loads `suite_metrics.json` | loads `phase1_*.json` + charts |
| `"live_fast"` / `"live"` | retrains games with small budgets | retrains the full REINFORCE/A2C/PPO ablation (~1–5 min CPU) |
| `"full"` (Phase 0 only) | dissertation budgets | — |

## Phase 1 meeting flow (suggested)

1. §1 Why PPO — feasibility table + objectives in LaTeX
2. §2 MDP formal + `env.py` source shown in-notebook
3. §3 Protocol + baselines with named objectives (B1b is the star)
4. §4 Iteration 1 + **ablation table**: REINFORCE = A2C = PPO = buy-max on 26/26
5. §5 Iteration 2: 5-D state → mostly flat (0/26 identical, wins 8/26 falling months)
6. §6 The lesson: change the **reward** next (`ΔW − λD`), not the algorithm
7. §7 Viva checklist — rehearse the five answers

## Phase 0 meeting flow (suggested)

1. §1 Why games first
2. §2 LaTeX four beats
3. Part A CartPole — table + bars + curves; show REINFORCE source cell
4. Part B Flappy — PPO wins under budget
5. Bridge equations to the trading MDP
6. Viva checklist cell

## Rebuild after editing

```bash
python3 notebooks/_build_phase0_colab.py
python3 notebooks/_build_phase1_colab.py
```
