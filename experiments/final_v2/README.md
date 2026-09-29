# experiments/final_v2 — the two-study experiment package

This one package runs both studies in the dissertation. The studies
share the environment, the agents, the feature pipeline, and the test
suite, so the code is not split by study. The runners are.

**First study (real-data pilot, Chapter 5 Section 5.3).** Train on 60
real SPY months (2018–22) with a 128×128 network (~18,000 weights); test
on 2024–25. No learner beat buy-and-hold.

**Main study (simulated redo, Chapter 5 Sections 5.5–5.10).** Train on
3,000 balanced simulated episodes with a 64–32 pyramid network (~2,800
weights); test on 600 held-out simulated episodes per market condition,
then transfer unchanged to 180 real months.

## Shared layer (used by every runner)

| File | Role |
|---|---|
| `env.py` | Trading MDP: state, three actions, action mask, wealth reward |
| `agents.py` | Scratch REINFORCE and DQN |
| `harness.py` | Training/evaluation plumbing, seeds, checkpoints |
| `features.py` | The 9 state features (Table 3.1 of the dissertation) |
| `baselines.py` | Buy-and-hold, always-buy, never-trade |
| `rollout.py` | Episode rollout and metrics |
| `month_sampler.py` | Gymnasium wrapper for SB3 MaskablePPO |
| `test_v2.py` | 21 verification tests — run these first |

```bash
# from the repo root
source venv/bin/activate
cd experiments/final_v2
python -m pytest test_v2.py -q
```

## First study — real SPY data

```bash
python data.py                    # build data/spy_episodes.npz from SPY daily bars
python run_final.py               # 3 algorithms x cells -> results/final_results.json
python run_fees.py                # fee sensitivity      -> results/fee_results.json
python run_conditioning_v2.py     # state-dependence     -> results/conditioning_v2_results.json
python run_ladder.py              # state-ladder check   -> results/ladder_results.json
python calibrate_slice.py         # trade-size selection -> results/slice_calibration.json
```

Figures: `python scripts/make_ch5_figures.py` from the repo root writes
the pilot figures into `latex/dissertation/figs5/`.

## Main study — simulated data plus real transfer

```bash
python sim_data.py                # regime generator, calibrated on 2018-22 only
                                  #   -> data/sim_episodes.npz (3000/300/600 episodes)
python real_transfer.py           # 180 real months outside the calibration window
                                  #   -> data/spy_transfer.npz
python run_sim.py                 # main result -> results/sim_results.json
python run_sim_suite.py           # slice / lambda / cells / fees on sim data
                                  #   -> results/sim_{slice,lambda,cells,fees}.json
```

Figures: `python scripts/make_sim_figures.py` from the repo root writes
`fig5_{sim,transfer,lambda,fees}.pdf` into `latex/dissertation/figs5/`.

Headline numbers (6 seeds, 300k steps): DQN +$126.94/episode on the
balanced simulated test, state-dependent on all 6 seeds; transfer to 180
real months at +$60.28/month with Sharpe 0.224 against buy-and-hold's
0.198, holding zero exposure through October 2008.

## Post-viva extras — NOT in the submitted dissertation

These two runners were written after the viva (14 September 2026) and
their results are not in the submitted PDF.

```bash
python run_feature_ablation.py    # drop volatility features; DQN only
                                  #   -> results/feature_ablation.json
python run_other_assets.py        # AAPL/QQQ transfer for the demo page
                                  #   -> results/other_assets.json
python run_stats.py               # bootstrap CIs and paired tests, no training
                                  #   -> results/stats.json
python gate_audit.py              # Gate 2/4 measurements, no training
                                  #   -> results/gate_audit.json
python ppo_grad_diag.py           # why dollar-reward PPO does not learn (~1 min)
                                  #   -> results/ppo_grad_diag.json
python run_ppo_scale_check.py     # PPO dollars vs scaled, sim validation (~7 min)
                                  #   -> results/ppo_scale_check.json
python -m pytest -q test_v2.py test_learners.py   # 26 tests, ~35 s
```

The ablation answers a viva question: without the volatility features
the agent still sat out October 2008, but re-entered during the wider
September–December 2008 window (band loss −$180 → −$712) and its
down-condition loss grew by 64%.

## Housekeeping notes

- `run_conditioning.py` is the older generation of
  `run_conditioning_v2.py`; kept for the result files that cite it.
- `verify_fixes.py` measures the defects of the deleted `final_model`
  package that `final_v2` replaced.
- `make_tables.py` builds the per-month appendix tables.
- Large result JSONs (for example `results/sim_fees.json`) store
  per-step records; do not load them whole into an editor.
