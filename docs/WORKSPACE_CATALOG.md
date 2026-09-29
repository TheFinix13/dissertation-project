# Workspace catalog

This file maps the whole project: every study the dissertation went
through, and every file that lives on `main` today. Part 1 covers the
historical studies, which survive only as git tags. Part 2 labels the
current working tree by study and by type of work.

Last updated: 16 September 2026, after the viva (14 September 2026).

---

## Part 1 — Historical studies (git tags, not on `main`)

The project changed direction three times before the submitted
dissertation. Each earlier study is preserved as an annotated git tag.
The branches that once held them were deleted on 7 September 2026, so
the tags are the only way in. To inspect one:

```bash
git checkout era1-uncertainty-ppo    # detached HEAD; look around, then
git checkout main                    # come back
```

| # | Study | Period | Was branch | Tag | Useful now? |
|---|---|---|---|---|---|
| 1 | Uncertainty-aware PPO ("AI-Driven Portfolio Protection") | Feb–Jun 2026 | `epistemic-uncertainty` / old `main` | `era1-uncertainty-ppo` | Archive only |
| 1b | Live-trading scaffold | May 2026 | `live-trading` (dup: `fiyins-portfolio`) | `era1-live-scaffold` | No |
| 1c | Pre-orphan mix | Aug 2026 | old `main` before the orphan restart | `pre-orphan-simple-modelling` | File recovery only |
| 2 | Simple-modelling pivot (Nguyen) | Jul–Aug 2026 | `simple-modelling-clean` | `era2-simple-modelling` | History; produced the MDP |
| 3 | `final_model` + kitchen-sink tree | Aug–Sep 2026 | `final-v2-state-architecture` | `era3-pre-restructure-snapshot` | File recovery only |

### 1. Uncertainty-aware PPO — the original idea

The README at the tag is titled "AI-Driven Portfolio Protection". The
study trained an uncertainty-aware PPO agent for capital preservation.
A DeepAR-style probabilistic LSTM produced aleatoric and epistemic
uncertainty estimates that fed the policy. The grid covered 70 tickers,
10 seeds, and 50,000 PPO steps per cell, with four-fold walk-forward
validation. The tag README claims the uncertainty agents grew $1M to a
median of about $1.6M on the held-out 2022–25 window, against a flat
baseline PPO.

Dr Nguyen ended this study in August 2026 (Recording 52): build one
complete simple model instead of the uncertainty stack. The tag holds
4,210 files, most of them per-cell result JSONs. Keep it as the record
of the original idea; do not merge any of it back.

### 1b. Live-trading scaffold — a side project

Alpaca paper trading, a nightly decision-support advisor, scheduled
execution, and a plain-English companion document. This was never part
of the assessed dissertation, and later project notes bar live brokers
from this repository. Keep the tag; nothing to reuse.

### 1c. Pre-orphan mix — a transitional tree

Phase-0 games and early SPY iteration work bolted onto the still-present
uncertainty tree. Its README still carries the era-1 title. It is not a
study in its own right. Use it only if a file exists on neither clean
tag.

### 2. Simple-modelling pivot — the Nguyen track

An orphan branch with no shared history with the old `main`. Three
stages: Phase-0 games (CartPole, Flappy Bird, LunarLander with random,
tabular Q-learning, DQN, scratch REINFORCE, A2C, and PPO); Iteration 1,
a 3-D SPY monthly MDP where PPO collapsed to buy-max on 26 of 26 test
months; and Iteration 2, a 5-D state that broke the collapse but
converged mostly flat. Draft Chapters 4–5 lived in
`latex/simple_modelling/`. This track produced the trading MDP that the
final study rebuilt. The Phase-0 code was removed from `main` in the
7 September cleanup; recover it from this tag if ever needed.

### 3. `final_model` and the pre-cleanup snapshot

`experiments/final_model/` (`env_full.py`, `run_final_experiments.py`)
was the first full SPY package. The audit in [audit_v2.md](audit_v2.md)
found defects, and `final_v2` replaced it. The tag
`era3-pre-restructure-snapshot` holds the entire 313-file tree from just
before the 7 September cleanup, including `final_model`, Phase-0 code,
and old LaTeX builds. Recovery only.

---

## Part 2 — The current tree on `main`

The submitted dissertation contains two studies. Both run from one
shared package, `experiments/final_v2/`. Do not split that package; the
studies share the environment, the agents, and the test suite.

**First study (real-data pilot).** 60 real SPY training months
(2018–22), a 128×128 network of about 18,000 weights, tested on
2024–25. No learner beat buy-and-hold ($180.25/month). The
policy-gradient seeds were state-independent.

**Main study (simulated redo).** 3,000 balanced simulated episodes, a
64–32 pyramid network of about 2,800 weights. DQN earned
+$126.94/episode with a state-dependent policy on all 6 seeds, and
transferred to 180 real months with a higher Sharpe ratio than
buy-and-hold (0.224 vs 0.198).

### Shared modelling (both studies)

| Path | What it is |
|---|---|
| `experiments/final_v2/env.py` | Trading MDP: state, actions, action mask, reward |
| `experiments/final_v2/agents.py` | Scratch REINFORCE and DQN (pyramid 64–32 net) |
| `experiments/final_v2/harness.py` | Shared training and evaluation plumbing |
| `experiments/final_v2/features.py` | The 9 state features |
| `experiments/final_v2/baselines.py` | Buy-and-hold, always-buy, never-trade |
| `experiments/final_v2/rollout.py` | Episode rollout and metrics |
| `experiments/final_v2/month_sampler.py` | SB3 episode wrapper for MaskablePPO |
| `experiments/final_v2/test_v2.py` | 21 verification tests |
| `experiments/final_v2/conftest.py` | pytest configuration |
| `experiments/final_v2/verify_fixes.py` | Quantifies the defects that motivated final_v2 |
| `experiments/final_v2/make_tables.py` | Per-month appendix tables |

### First study — data, training, results, figures

| Type | Path |
|---|---|
| Data pipeline | `experiments/final_v2/data.py` |
| Data | `experiments/final_v2/data/spy_episodes.npz`, `spy_meta.json` |
| Training | `experiments/final_v2/run_final.py` (3 algorithms, cells, risk axis) |
| Training | `experiments/final_v2/run_fees.py` (fee sensitivity) |
| Training | `experiments/final_v2/run_conditioning.py`, `run_conditioning_v2.py` (state-dependence diagnostics) |
| Training | `experiments/final_v2/run_ladder.py` (state-ladder saturation) |
| Training | `experiments/final_v2/calibrate_slice.py` (trade-size calibration) |
| Results | `results/final_results.json`, `final_results_risk.json`, `fee_results.json`, `fee_results_pre_ppo.json`, `conditioning_results.json`, `conditioning_v2_results.json`, `ladder_results.json`, `slice_calibration.json`, `verify_fixes.json` + logs |
| Figures | `scripts/make_ch5_figures.py` → `latex/dissertation/figs5/fig5_{cumwealth,months,mscatter,seeds,slice,state,effic}.pdf` |
| Archived chapter | `notes/archive/ch5_pilot_only_2026-09-07.tex` (Chapter 5 before the sim rewrite) |

### Main study — data, training, results, figures

| Type | Path |
|---|---|
| Data generator | `experiments/final_v2/sim_data.py` (regime-switching, calibrated on 2018–22 only) |
| Transfer set | `experiments/final_v2/real_transfer.py` (180 real months, 2006–17 + 2023–25) |
| Data | `experiments/final_v2/data/sim_episodes.npz`, `sim_meta.json`, `spy_transfer.npz`, `spy_transfer_meta.json` |
| Training | `experiments/final_v2/run_sim.py` (main train + per-regime test + real transfer) |
| Training | `experiments/final_v2/run_sim_suite.py` (slice, lambda, cells, fees rerun on sim data) |
| Results | `results/sim_results.json`, `sim_slice.json`, `sim_lambda.json`, `sim_cells.json`, `sim_fees.json` + logs |
| Figures | `scripts/make_sim_figures.py` → `latex/dissertation/figs5/fig5_{sim,transfer,lambda,fees}.pdf` |

### Dissertation (both studies)

| Path | What it is |
|---|---|
| `latex/dissertation/main_full.tex` + `ch1.tex`–`ch6.tex` | Submitted final draft, 80-page body |
| `latex/dissertation/appendix_*.tex` | Appendices A–E |
| `latex/dissertation/main_full.pdf` | Compiled PDF (~101 pages) |
| `latex/dissertation/references.bib` | Bibliography |
| `latex/dissertation/build.py` | FROZEN — old Word-to-LaTeX regenerator; do not run |
| `latex/dissertation/figs5/`, `figs/`, `media*/` | Figures |
| `scripts/make_ch4_timeline.py` | Chapter 4 timeline figure |
| `scripts/make_appendix_transfer.py` | Transfer-year appendix table |
| `latex/dissertation/make_appendix_permonth.py` | Per-month appendix helper |

### Viva pack (14 September 2026)

| Path | What it is |
|---|---|
| `latex/viva/viva_deck.pptx` | Frozen 11-slide main deck (delivered) |
| `latex/viva/viva_deck_extended.pptx` | Extended deck with backup slides |
| `latex/viva/figs/`, `figs_extra/` | Slide figures (PNG) |
| `scripts/make_viva_figures.py`, `make_viva_extra_figures.py` | Figure generators |
| `scripts/make_viva_pptx.py`, `make_viva_pptx_extended.py` | Deck builders |
| `scripts/make_transcript_html.py` → `notes/viva/transcript.html` | Read-along teleprompter |
| `scripts/make_qa_html.py` → `notes/viva/qa.html` | Question-card prompter |
| `scripts/make_demo_page.py` → `demo/index.html` | QR demo page (SPY transfer animation) |
| `latex/viva/_archive_beamer/` | The earlier Beamer version of the deck |
| `notes/viva/transcript.md` | The spoken script, built from rehearsal memos |
| `notes/viva/viva_qa_prep.md`, `nguyen_questions_compiled.md` | Q&A preparation |

### Post-viva engineering (not in the submitted PDF)

| Path | What it is |
|---|---|
| `experiments/final_v2/run_feature_ablation.py` | Volatility-feature ablation (answers Nguyen's "how do you know?") |
| `results/feature_ablation.json` + log | Ablation results |
| `experiments/final_v2/run_other_assets.py` | AAPL/QQQ transfer for the demo page |
| `results/other_assets.json` | Other-asset results |
| `experiments/final_v2/run_stop_loss.py` | Tuned stop-loss baseline (answers Nikitopoulos's "why not a simple rule?") |
| `results/stop_loss.json` | Stop-loss results |
| `experiments/final_v2/run_stats.py` | Bootstrap intervals and paired tests on saved results (SPY, AAPL, QQQ); no training |
| `results/stats.json` | Statistics results |
| `notes/post-viva/2026-09-28-improvement-plan.md` | Plan after the final feedback: tracks, owners, order |
| `notes/post-viva/2026-09-28-statistics.md` | What the statistical tests show |
| `experiments/final_v2/gate_audit.py` | Gate 2 and 4 measurements (capacity, balance, signal content, realism, coverage, seeds against months); no training |
| `results/gate_audit.json` | Gate audit results |
| `experiments/final_v2/test_learners.py` | Known-answer tests every learner must pass before tuning |
| `experiments/final_v2/ppo_grad_diag.py` | Measures why PPO on dollar rewards does not learn (gradient norms, clip factor, Adam epsilon) |
| `results/ppo_grad_diag.json` | PPO gradient diagnostic results |
| `experiments/final_v2/run_ppo_scale_check.py` | PPO dollar vs scaled rewards on simulated validation months; `--algo dqn/reinforce` for references |
| `results/ppo_scale_check.json`, `results/scale_check_dqn.json`, `results/scale_check_reinforce.json` | Scale check results |
| `notes/post-viva/2026-09-29-gates-2-4-audit.md` | Gates 2 to 4 for Track B steps 3 to 5, including the PPO reward-scale finding |
| `notes/post-viva/2026-09-29-experiment-designs.md` | Pre-registered designs: fair tuning with 20 seeds, and a hidden-state simulator |
| `experiments/final_v2/freeze_post_viva_data.py` | Froze the fresh simulated tuning/test sets and the SPY 1993–2005 hold-out (run once) |
| `experiments/final_v2/data/sim_post_viva.npz`, `spy_holdout_1993_2005.npz`, `spy_holdout_1993_2005_raw.csv`, `post_viva_data_meta.json` | Frozen post-viva evaluation data; the hold-out is sealed until Experiment 1's final policies are fixed |
| `notebooks/reproduce_main_study.ipynb` | Reproduction notebook for outside researchers (verify → recompute → retrain); Colab-ready |
| `notes/post-viva/improvement-backlog.md` | Ranked post-viva improvement backlog |
| `notes/post-viva/lit-review-right-data.md` | Gate-1 literature review: what makes the right data |

### Notes and supervision

| Path | What it is |
|---|---|
| `notes/supervision/2026-09-07-nguyen-meeting-*.txt` | The meeting that triggered the main study |
| `notes/supervision/2026-09-08-nguyen-briefing.md` | Briefing note |
| `notes/lit-methodology-review.md` | Literature note |
| `notes/viva/2026-09-14-*.md`, examiner emails from 10 Sep on | Local only; not for the public repository |

### Archived planning (docs/archive/)

Superseded planning documents from earlier eras. Kept for the record;
their file paths no longer exist on `main`.

| Path | Era |
|---|---|
| `docs/archive/EXPLAIN_THE_EXPERIMENTS.md` | Era 2 Phase-0 briefing |
| `docs/archive/nguyen_meeting_memo_jul30.md` | Era 2 CartPole memo |
| `docs/archive/meeting_action_plan.md` | Era 2 Phase-0 plan |
| `docs/archive/dissertation_final_outline.md` | Pre-sim outline (cites deleted `final_model`) |
| `docs/archive/OUTSTANDING_SECTIONS.md` | Word-to-LaTeX sync tracker (obsolete) |

### Still-current docs

| Path | What it is |
|---|---|
| `docs/audit_v2.md` | The audit that led to final_v2 |
| `docs/external_review_v2.md` | External-style review of the draft |
| `docs/nguyen_recording52_transcript.txt` | The recording that locked the outline |
