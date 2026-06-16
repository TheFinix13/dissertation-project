# ai_context.md · uncertainty-aware-portfolio-drl
Last updated: 2026-06-16 (v0.17.2 — repo pruned: Phase-1 DOCX pipeline removed; LaTeX canonical only.)

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
- **Canonical dissertation = LaTeX in `latex/`.** PDF: `latex/main.pdf` (96 pp, 0 undefined refs). Word: `dissertation.docx` at repo root (rebuilt via `latex/build_docx.sh`). Title: Fiyinfoluwa Akano, URN 6962514. Local folder: `uncertainty-aware-portfolio-drl` (GitHub: `TheFinix13/dissertation-project`).
  entries, critical-comparison table Section 2.6.1, three-role
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
| Dissertation source (canonical) | `latex/main.tex`, `latex/chapters/abstract.tex` … `ch7_conclusion.tex`, `latex/chapters/acknowledgements.tex`, `latex/chapters/appendix_{a,b,c_glossary}.tex`, `latex/references.bib` (71 entries v0.17), `latex/tikz/{rl_loop,data_splits,training_pipeline,uncertainty_trade_scaling}.tex` |
| Deliverables | `latex/main.pdf` (96 pages v0.17, gitignored), `dissertation.docx` (gitignored) |
| Notebooks | `notebooks/01_Project_Walkthrough.ipynb`, `notebooks/02_Full_Experiments.ipynb` |
| Experiment runners | `experiments/runners/{run_baseline,run_probabilistic_agent,run_benchmarks,run_rule_baselines,run_walk_forward,run_extended_grid,run_ablation}.py` (last one NEW v0.17) |
| Statistical pass | `experiments/stats_significance.py`, `experiments/compute_spy_phase2_table.py`, `experiments/compute_spy_ablation_stats.py` (all NEW v0.17) |
| Shared experiment library | `experiments/common.py` (EnvConfig extended with ablation flags v0.17), `experiments/aggregate_results.py`, `experiments/configs/dissertation_protocol.json` |
| Phase-2 results (committed) | `experiments/results/per_cell/*.json` (2,749 cells incl. ablation v0.17), `experiments/results/wf_curves/*.csv` |
| Stats outputs (NEW v0.17) | `reports/generated/stats/phase2_statistical_tests.{json,md}`, `reports/generated/stats/forecaster_calibration_metrics.json`, `reports/generated/stats/spy_phase2_case_study.{json,md}`, `reports/generated/stats/spy_ablation_stats.json` |
| Chart suite | `reports/generated/charts/*.{png,pdf}` (10 charts dual-emit v0.17), `reports/builders/plot_phase2_charts.py`, `reports/builders/build_forecaster_calibration.py`, `reports/builders/gen_spy_repr_curve.py` |
| Cursor rules | `.cursor/rules/{use-brain-box,academic-writing,cross-pollination,ai-context-routine}.mdc` |
| Brain Box primary nodes | `~/Documents/GitHub/brain-box/school/surrey-msc-ai/dissertation-eeem004.md`, `~/Documents/GitHub/brain-box/life/finance-research/uncertainty-aware-portfolio-drl.md` |

## 3) Next immediate goal

**Your read-through, then ship to Dr Cuong Nguyen.**

| Send this | Path | Why |
|---|---|---|
| **Primary (recommended)** | `dissertation.docx` (repo root) | Editable Word — Dr Nguyen can comment inline |
| **Secondary (print/formal)** | `latex/main.pdf` | Fixed 96-page PDF, same LaTeX source |

Before sending: skim acknowledgements, abstract, Table 5.3 (stats), Section 5.7 (ablation), Ch4 worked examples.

After Dr Nguyen's feedback:
- Optional: extend ablation from SPY pilot to full 70-ticker grid on Colab (~2–3 h per arm).
- Re-open Cursor workspace from **`~/Documents/GitHub/uncertainty-aware-portfolio-drl`** (folder renamed).

Parked (do not start without discussion):
- Live broker integration of any kind (forbidden in this repo per `.cursor/rules/use-brain-box.mdc`).
- Deflated Sharpe ratio (Bailey & López de Prado 2014) — listed as named limitation in Section 5.8; only revisit if Dr Nguyen requests it.
- Rewriting any chapter that v0.17 has already touched, unless Dr Nguyen's feedback forces it.
