# ai_context.md · uncertainty-aware-portfolio-drl
Last updated: 2026-08-04 (simple-modelling track: Phase 0 multi-algo done, Iteration 2 run, Ch4/Ch5 drafted.)

Read this first in a fresh chat. Strictly technical state summary for the
EEEM004 dissertation implementation. Deeper history: `reports/PROJECT_STATUS.md`
and `docs/CHECKPOINT.md`.

## 1) What is built and working

- **Phase 2 complete and reconciled across the whole repo.** Full grid on
  `market_sample` (70 stocks × 10 seeds × 50,000 PPO timesteps per cell)
  plus four-fold walk-forward validation (320 cells). 2022–2025 hold-out:
  uncertainty-aware agents grow $1M → median **~$1.6M**, median Sharpe
  **+0.68 to +0.70**, win-rate **89–92%**. Baseline PPO is flat at
  **~$999,063** (Sharpe −0.04, win-rate 46.1%). Walk-forward 87.2% overall
  win-rate, probabilistic wins every fold.
- **Paired statistical inference (NEW v0.17, Section 5.6).** Wilcoxon
  signed-rank rejects B-vs-A and C-vs-A at **p < 10⁻¹²** on terminal
  value, Sharpe and MDD across the 70 matched tickers. 95% paired-bootstrap
  CI for the (B−A) terminal-value gap: **[+$559k, +$814k]**. Walk-forward
  (B vs A): terminal value p = 4.7×10⁻⁹, Sharpe p = 6.8×10⁻⁶. The original
  "aleatoric and epistemic are statistically indistinguishable" claim
  was wrong — B vs C is not significant on terminal value (p = 0.080) but
  IS detectable on Sharpe (**+0.014, p = 7×10⁻⁴**) and MDD (**−0.65 pp,
  p = 2×10⁻⁵**). Script: `experiments/stats_significance.py`. Outputs:
  `reports/generated/stats/phase2_statistical_tests.{json,md}`.
- **Two-piece ablation of Eq 3.9 (NEW v0.17, Section 5.7).** Three new
  arms on SPY at Phase-2 budget (10 seeds × 50k steps): state-only,
  guard-only, scaling-only. Honest finding: each single piece recovers
  ~95% of the terminal-value gain (all three pilot arms ≈ $1.63M vs full
  design's $1.65M; not separable, p = 1.0). BUT each single piece raises
  median MDD from **18.5% to 21–22%**, a 3 pp penalty that is
  significant at p = 2×10⁻³ on every contrast. The *drawdown-control*
  property is the specific contribution of the full Eq 3.9, not the
  return. This reframes the dissertation's contribution from "two-piece
  design that delivers return" to "two-piece design that delivers
  drawdown safety while preserving return". Runner:
  `experiments/runners/run_ablation.py`. EnvConfig now carries
  `enable_trade_scaling`, `enable_risk_on_guard`, `mask_uncertainty_in_state`
  flags. The 70-ticker grid extension is Colab-feasible (~2–3 h per arm).
- **Forecaster calibration figure (NEW v0.17, Section 5.2.5).** LSTM
  retrained on SPY 2009–2018, validated on 2019–2021. Empirical coverage
  at nominal 80% = **81.0%** (the threshold τ the policy actually uses).
  Coverage at 95% = 92.2% (mild under-coverage at the tail, consistent
  with Cont 2001 heavy-tail stylised facts). **ECE = 0.045**, below the
  0.10 typical of modern over-confident networks. Validation NLL =
  −3.63, RMSE = 0.0141. Reliability diagram + residual scatter at
  `reports/generated/charts/forecaster_calibration.{pdf,png}`. Script:
  `reports/builders/build_forecaster_calibration.py`.
- **Drawdown — honest framing (unchanged).** Proper benchmark is
  buy_and_hold (~25.9% median MDD), not the under-trading baseline PPO
  (~1.4%, an artefact of doing nothing). Uncertainty guard beats
  buy-and-hold on median (~22% / ~24%) **while growing capital to
  ~$1.6M**, but **loses on the worst-case tail** (guard worst ~57% vs
  buy-and-hold worst ~35%). The ablation now anchors the drawdown
  property to the *full* Eq 3.9 specifically.
- **Architecture (locked).** PPO (Stable-Baselines3) on Gymnasium
  `StockEnv`. LSTM forecaster, Gaussian NLL → μ, σ. Two uncertainty
  modes: aleatoric (Gaussian NLL) and epistemic (MC Dropout, T=20).
  Guard: if u > τ, scale or block new buys. Walk-forward: four rolling
  2-year folds 2018–2025.
- **Conceptual narrative (NEW v0.17.6).** Reader guide reframed as **"The core idea"**
  (two-module loop, soft dial vs hard guard, return-vs-drawdown ablation split,
  success/failure regimes, SPY/NVDA walkthrough pointers). Abstract, Ch1
  motivation/contributions, Ch3 overview + Eq 3.9 interpretation, Ch6 headline
  + Ch7 summary/conclusion now explain the *design idea* before the numbers.
  Terminology: **confidence signal** in prose, $u_t$ in equations.
- **Simple-modelling reset (NEW 2026-07-30).** After Dr Nguyen's July
  supervision, the methodology is being rebuilt progressively from a
  one-stock, three-action high-frequency MDP before any uncertainty or
  multi-asset layers are added. Iteration 1 uses state
  $s_t=[\Delta P_t,C_t,n_t]^\top$, actions {Hold, Buy one, Sell one},
  feasibility masking, wealth-change reward, and separate PPO policy/value
  updates. Word-ready Algorithm 3.1:
  `latex/tikz/algorithm3_1_training_pipeline_word.png`; canonical source:
  `latex/tikz/algorithm3_1_training_pipeline_standalone.tex`. The v0.17
  stack/results remain historical evidence, not automatically the final model.
- **Phase 0 CartPole PPO (NEW 2026-07-30).** Policy-based correctness
  sandbox per Nguyen Recording 47: 50k timesteps, random eval mean return
  **22.1** vs PPO **500.0** (solved). Curve:
  `reports/generated/charts/phase0_cartpole_learning_curve.png`. Scripts:
  `experiments/phase0_cartpole/{train_ppo,plot_learning}.py`. Env: `.venv311`.
- **Phase 1 SPY daily months (NEW 2026-07-30, branch `simple-modelling-iteration1`).**
  Chronological 58 train / 26 test monthly episodes (honest: daily bars,
  $T\approx 18$–$23$, not 390). Baselines B0 / B1(one) / **B1b(max)** /
  B2 random / A1 PPO (80k steps). Test mean ΔW: B0 **0**, B1 **\$8.0**,
  B1b **\$76.9**, B2 **\$14.1**, A1 **\$76.9**. **Honest finding:**
  deterministic PPO matched B1b on **26/26** months (collapsed to
  fully-invested buy-and-hold under pure $\Delta W$). Accounting tests
  pass; illegal-action rate **3.9%**. Charts:
  `reports/generated/charts/phase1_spy_daily_{delta_w,wealth_path}.png`.
  Runner: `experiments/iteration1/run_phase1_spy_daily.py`. Doer notes:
  `docs/going_beyond_recorder.md`.
- **Phase 0 multi-algorithm games (NEW 2026-08-04).** Random / tabular
  Q-learning / DQN / scratch-REINFORCE / A2C / PPO across three games,
  seed 42, 30-episode greedy eval. CartPole: random **28.8**, REINFORCE
  **292.2**, Q-table / DQN / A2C / PPO all **500** (Q-table needs
  6×6×12×12 binning — the anti-scaling argument). Flappy: random
  **−7.4**, PPO best **12.6**. LunarLander-v3 (150k steps): random
  **−183.4**, REINFORCE 15.3, A2C −41.2, PPO **176.4**, DQN 177.6
  (solved ≈ 200). Suites: `experiments/phase0_games/{suite,common,reinforce}.py`,
  runners `run_{cartpole,flappy,lunarlander}_suite.py`,
  `run_cartpole_qlearning.py`. Charts:
  `reports/generated/charts/phase0_{cartpole,flappy,lunarlander}_{comparison,curves}.png`.
  Viva doc: `docs/phase0_games_explained.md`.
- **Phase 1 Iteration 2 (NEW 2026-08-04).** `env_v1.py` 5-D scaled state
  $[\Delta P, \mathrm{PnL}, \tau, C/C_0, nP/C_0]$; same split/fee/budget/seed
  as Iter 1. **Honest finding:** breaks the buy-max collapse (0/26 months
  identical to B1b) but converges mostly-flat — deterministic ΔW **0.0**
  (0 trades; mean Hold prob 0.66), stochastic ΔW **+$1.87**, beats B1b in
  8/26 (falling) months. v1 accounting + feature unit tests pass.
  Results: `experiments/iteration1/results/phase1_iteration2_results.json`.
  Next lever per Ch5 analysis: reward $\Delta W - \lambda D_t$.
- **Simple-modelling Ch4+Ch5 drafted (NEW 2026-08-04).** Standalone LaTeX
  (does NOT touch the v0.17 build): `latex/simple_modelling/{main.tex,
  ch4_implementation_simple.tex,ch5_results_simple.tex}` → `main.pdf`
  (10 pp) + Word export `Chapter4-5_simple_modelling.docx`. All numbers
  real (no placeholder metrics).
- Branch `simple-modelling-iteration1` **pushed to origin** (visible on
  GitHub since 2026-08-04).
- Action plan + meeting memo: `docs/meeting_action_plan.md`,
  `docs/nguyen_meeting_memo_jul30.md` (Phase 0+1 numbers filled).
- **Canonical dissertation = LaTeX in `latex/`.** PDF: `latex/main.pdf`. Word: `dissertation.docx` at repo root (rebuilt via `latex/build_docx.sh`). Title: *AI-Driven Portfolio Protection: Balancing Growth and Limiting Large Losses Using Confidence Signals*. Author: Fiyinfoluwa Akano, URN 6962514. Ch 2 literature rebuild (71 bib entries, critical-comparison table Section 2.6.1, three-role
  positioning tags, safe-RL + distributional-RL + multiple-testing
  literatures explicit). Chapter 5 gained Sections 5.2.5
  (calibration), 5.6 (paired inference), 5.7 (ablation). Chapter 4
  gained Sections 4.10.1 (SPY 14-Jan-2022 blocked-trade walkthrough,
  verbatim from the Interim Review) and 4.10.2 (NVDA 25-May-2023
  capped-exposure losing-day walkthrough). Acknowledgements
  personalised; abstract rewritten in seven-rule voice; Group 2
  (Alpaca deployment) deleted from Ch 7 future-work and replaced with a
  one-paragraph out-of-scope statement. Eq 3.10 fixed to the
  Kendall–Gal-correct $\mathbb{E}_k[(\hat{\sigma}^{(k)})^2]$.
  Transaction-cost rate explicitly stated at $c = 0.0005$ (5 bps) with
  slippage explicitly out of scope.
- **Reproducible offline chart suite extended.** All ten Chapter 5
  figures now emit BOTH `.png` AND `.pdf` (vector) variants; LaTeX
  references switched to `.pdf`. Add the calibration figure and the
  ablation outputs to that set. `reports/builders/plot_phase2_charts.py`
  patched to dual-emit.
- **Two notebooks, both reconciled.** `notebooks/01_Project_Walkthrough.ipynb`
  (CPU smoke, ~5 min) and `notebooks/02_Full_Experiments.ipynb` (Colab
  T4/A100; SMOKE flag defaults True).
- **Supervisor checkpoints.** First Project Report ~69%; Interim Review
  "Satisfactory progress made but with some reservations" (Dr Cuong
  Nguyen). The Step-8 worked example is now re-inserted into Chapter 4
  per the eight Brain-Box action items.
- **Predicted grade (self-graded v0.17).** The v0.16 grade was 64–67/100
  (mid-Merit) bounded by the missing inference, ablation, lit-review,
  and worked-example items. With v0.17 closing all five Tier-A items
  (statistical-inference pass; two-piece ablation; lit review rebuild;
  Eq 3.10 fix; worked examples re-inserted) and four of seven Tier-B
  items (forecaster calibration, transaction-cost contradiction
  resolved, Sharpe-vs-terminal-value framing tightened, drawdown
  median footnote pending), the dissertation now sits at the
  Merit/Distinction boundary; the remaining gap is the 70-ticker grid
  extension of the ablation, which is Colab-feasible and is the next
  optional upgrade.

## 2) Key file paths

| Area | Files |
|---|---|
| Dissertation source (canonical) | `latex/main.tex`, `latex/chapters/abstract.tex`, `latex/chapters/reader_guide.tex` (The core idea), `ch1_introduction.tex` … `ch7_conclusion.tex`, `latex/chapters/acknowledgements.tex`, `latex/chapters/appendix_{a,b,c_glossary}.tex`, `latex/references.bib` (71 entries v0.17), `latex/tikz/{rl_loop,data_splits,training_pipeline,uncertainty_trade_scaling}.tex` |
| Deliverables | `latex/main.pdf` (96 pages v0.17, gitignored), `dissertation.docx` (gitignored) |
| Notebooks | `notebooks/01_Project_Walkthrough.ipynb`, `notebooks/02_Full_Experiments.ipynb` |
| Experiment runners | `experiments/runners/{run_baseline,run_probabilistic_agent,run_benchmarks,run_rule_baselines,run_walk_forward,run_extended_grid,run_ablation}.py` (last one NEW v0.17) |
| Statistical pass | `experiments/stats_significance.py`, `experiments/compute_spy_phase2_table.py`, `experiments/compute_spy_ablation_stats.py` (all NEW v0.17) |
| Shared experiment library | `experiments/common.py` (EnvConfig extended with ablation flags v0.17), `experiments/aggregate_results.py`, `experiments/configs/dissertation_protocol.json` |
| Phase-2 results (committed) | `experiments/results/per_cell/*.json` (2,749 cells incl. ablation v0.17), `experiments/results/wf_curves/*.csv` |
| Stats outputs (NEW v0.17) | `reports/generated/stats/phase2_statistical_tests.{json,md}`, `reports/generated/stats/forecaster_calibration_metrics.json`, `reports/generated/stats/spy_phase2_case_study.{json,md}`, `reports/generated/stats/spy_ablation_stats.json` |
| Chart suite | `reports/generated/charts/*.{png,pdf}` (10 charts dual-emit v0.17), `reports/builders/plot_phase2_charts.py`, `reports/builders/build_forecaster_calibration.py`, `reports/builders/gen_spy_repr_curve.py` |
| Simple-modelling track | branch `simple-modelling-iteration1` (on origin); `experiments/{phase0_cartpole,phase0_games,iteration1}/**`; `docs/{tutor_sl_to_rl_training_loop,iteration1_experiment_plan,nguyen_meeting_memo_jul30,going_beyond_recorder,phase0_games_explained}.md`; `latex/simple_modelling/**` (Ch4+Ch5 standalone); `latex/tikz/algorithm3_1_*`, `rl_training_loop_template_*`, `environment_cycle_word.png` |
| Cursor rules | `.cursor/rules/{use-brain-box,academic-writing,cross-pollination,ai-context-routine}.mdc` |
| Brain Box primary nodes | `~/Documents/GitHub/brain-box/school/surrey-msc-ai/dissertation-eeem004.md`, `~/Documents/GitHub/brain-box/life/finance-research/uncertainty-aware-portfolio-drl.md` |

## 3) Next immediate goal

**Rebuild and understand the simple model before restoring complexity.**

| Send this | Path | Why |
|---|---|---|
| **Primary (recommended)** | `dissertation.docx` (repo root) | Editable Word — Dr Nguyen can comment inline |
| **Secondary (print/formal)** | `latex/main.pdf` | Fixed 96-page PDF, same LaTeX source |

Immediate sequence:
- Bring to meeting: Ch4/Ch5 draft (`latex/simple_modelling/main.pdf` or
  `Chapter4-5_simple_modelling.docx`), Phase-0 game table, Iteration-1 vs
  Iteration-2 story (buy-max collapse → mostly-flat collapse).
- Ask Nguyen: Iteration 3 = drawdown-penalised reward $\Delta W-\lambda D_t$
  (Ch5 analysis argues state alone only moves exposure, not timing).
- Then: multi-seed CIs, fee grid, one SPY minute-day ($T=390$) episode.

Parked (do not start without discussion):
- Live broker integration of any kind (forbidden in this repo per `.cursor/rules/use-brain-box.mdc`).
- Restoring the full v0.17 uncertainty dual-path as the *meeting story*
  (keep as historical evidence only until Iteration-1 is understood).
- Deflated Sharpe ratio — only if Dr Nguyen requests it.
