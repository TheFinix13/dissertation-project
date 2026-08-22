# ai_context.md · simple-modelling-clean
Last updated: 2026-08-19 (post-Recording-52 pivot; full draft v1 compiled)

Compact state for the **Nguyen simple-modelling track**.
Branch `simple-modelling-clean` (orphan).

## 0) THE PIVOT (Recording 52, transcript in docs/)

Nguyen's directives, 19 Aug meeting (deadline **1 Sep 2026**):
- **No iteration-ladder experiments** ("don't do that") → one complete model.
- **Complete ~9-feature state**; divisible "Bitcoin-style" asset (fixed
  Discrete(3), fractional slice trades) fixes buy-feasibility.
- **Algorithms: scratch REINFORCE + scratch Deep Q** (PPO = stretch only).
- **Fix the Ch3 math**: θ parameterizes the POLICY not the reward; full
  likelihood-ratio derivation of why log π appears in the loss; empirical
  double-sum (1/N Σ_n Σ_t) as Monte-Carlo approximation.
- Episodes independent (game sessions); non-IID acknowledged as limitation.
- Write methodology/experiments first; intro/abstract last.
- Progression to 9-D state survives as *design rationale* + pilot study
  (the old 3-D collapse results), not as separate experiment tracks.

## 1) What is built and working (new final model)

- **Env** `experiments/final_model/env_full.py` — FullStateTradingEnv:
  9-D state [ΔP, mom5, vol5, magap10, τ, C/C0, hP/C0, PnL, W/C0−1],
  fractional units, $1000 trade slice, masking, forced end liquidation.
  8/8 unit tests pass (`test_env_full.py`).
- **Scratch DQN** `experiments/final_model/dqn_trading.py` — replay,
  target net, masked ε-greedy, masked TD targets.
- **Scratch REINFORCE** reused from `experiments/iteration1/reinforce_trading.py`
  (obs_dim=9).
- **Full ablation** (`experiments/final_model/run_full_ablation.py` →
  `results/full_ablation_results.json`, seeds 42/43/44, 80k steps,
  SPY 58/26 months, fee 5 bps; ~12 min total on CPU):
  - B1b buy&hold: mean ΔW **$125.19**, std 310, Sharpe 0.40, MDD 6.0%
  - REINFORCE: **identical to B1b to the cent, 26/26 months** (s42);
    s43/44 within a few $ of the corner — objective-driven collapse
  - DQN: ΔW **$123.23**, std 274, **Sharpe 0.45, MDD 4.8%**, wins 9/26
    vs B1b — "fully invested by default, trims around turbulence"
    (exposure chart shows 0.9→0.4 sell-down in falling month 2023-09)
  - **PPO (SB3, via MonthSamplerFullEnv): bimodal across seeds** — s42
    cautious low-exposure ($55.67, 4 tr/mo, MDD 1.9%); s43/44 = exact
    buy-and-hold 26/26. Clipping freezes policy near early-rollout corner.
  - **Fee sensitivity (s42, retrained per fee)**: B1b/REINFORCE $135→$125→$95
    at 0/5/20 bps; DQN adapts intensity 19.8→13.2→6.1 trades/mo
    (economically correct direction; still loses at 20 bps)
  - B2 random $37.97 · B0 $0 · accounting identity + 0 illegal everywhere
  - NB: B2 must use ONE rng across months (patch_b2_random.py fixed this)
  - Charts: `reports/generated/charts/ablation_{delta_w,wealth_paths,
    exposure_down_month,training_curves,fee_sensitivity}.png`
- **Pilot (kept for §5.3)**: 3-D state → REINFORCE/A2C/PPO all = buy-max
  26/26 ($76.91). From `results/phase1_algo_ablation.json`.

## 2) Dissertation draft v4 — `latex/final_dissertation/`

**54 pp, compiles clean**, no undefined refs/citations, no bad overfulls.
Full expansion pass done (all chapters + front matter + 3 appendices):
- Front matter: originality statement, acknowledgements, LoF/LoT.
- Ch1: lay motivation, scope/non-goals, reader's guide.
- Ch2: full lit review (~10pp) incl. quantitative critique of Théate &
  Ernst (their TDQN Sharpe on SPY = B&H's 0.834 exactly — corroborates
  our collapse finding; verified against published paper).
- Ch3: notation table, RL-loop figure, numeric fee example, γ note,
  Step-4 proof sketch, metric formulas, plain-language summaries.
- Ch4: pipeline + chrono-split TikZ figures (figs/ dir, fresh — old
  _docx_build figs are from dead LSTM design, don't use), DQN code
  listing, reproducibility section.
- Ch5: **PPO now 5 seeds** (3× exact B&H, 2× cautious ≈$56 — bimodal,
  both modes replicated) + **§5.6 risk-aware preliminary** (λ=0.25:
  DQN MDD 6.0→1.4%, Sharpe 0.40→0.68; REINFORCE → all-cash corner,
  0 trades). `run_supplementary.py` / `supplementary_results.json`.
- Appendices: A traceability, B per-month table (generated from JSON),
  C plain-language glossary.
- references.bib: 34 entries, all canonical, all cited; theate2021,
  fischer2018, towers2024gymnasium verified online this session.
- Examiner review + grades: `docs/external_review_v2.md` (overall 75).
Outline: `docs/dissertation_final_outline.md`.
- Ch3: full log-trick derivation (6 steps, exact vs approx labeled) +
  DQN formulation + **PPO clipped surrogate (sec:ppo)** + algorithm boxes.
- Ch4: PPO row protocol + masking-asymmetry disclosure (sec:impl_ppo).
- Ch5: Phase-0 figures wired; full ablation table (with Id.B1b column),
  wealth paths, exposure, training-curve noise discussion, seed-spread
  table, fee-sensitivity section, 4 key findings (incl. risk-aware-reward
  argument from DQN behaviour).
- Ch5 has real numbers only (old docx [SAMPLE] tables are dead — never
  submit those). Word drafts in Dissertation Drafts/ are superseded.

## 3) Next goals

1. Expand toward 60–80 pp: lengthen Ch2 lit review, add per-month result
   table appendix, code-listing appendix, project-plan Gantt/figure.
2. User to confirm registered title/objectives (starting afresh).
3. Viva pack: 12 slides, live regen demo (full ablation ~12 min CPU;
   single-seed subset < 2 min).
4. Broken old venv replaced: use `./venv` (python3 -m venv, requirements.txt).
