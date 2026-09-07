# ai_context.md · final submission state
Last updated: 2026-09-07 evening (post Nguyen meeting; sim-data redo underway)

Compact state of the project for future AI sessions.

## 1) What this project is

MSc dissertation (EEEM004, Surrey; supervisor Dr Cuong Nguyen).
Question: can REINFORCE / DQN / PPO learn profitable, **state-dependent**
trading policies for SPY versus buy-and-hold on held-out 2024–2025 data?
Headline answer: no learned agent beat buy-and-hold (+$180.25/mo);
best was REINFORCE (+$171.51) but behaviourally state-independent
(0/12 policy-gradient seeds state-dependent vs 6/6 for DQN).

## 2) Live artifacts (everything else was deleted in the restructure)

- **Dissertation**: `latex/dissertation/` — hand-edited LaTeX, compiles
  with `xelatex main_full.tex` (110 pp total, 87 pp body, no errors).
  `build.py` there is **frozen** (guarded with --force); it regenerated
  chapters from old Word sources and must not overwrite the hand edits.
  `OUTSTANDING_SECTIONS.md` in that folder tracks remaining items.
- **Experiments**: `experiments/final_v2/` — env, scratch REINFORCE/DQN,
  SB3 PPO, 21 verification tests (`test_v2.py`), runners
  (`run_final.py`, `run_fees.py`, `run_conditioning_v2.py`), results JSONs.
- **Figures**: `scripts/make_ch5_figures.py` regenerates Chapter 5 figures
  into `latex/dissertation/figs5/`.
- **Venv**: `./venv` (Python 3.14, torch 2.13). `.venv311` was deleted.

## 3) Writing voice

Write prose in Fiyin's voice per the always-on workspace rule
(`.cursor/rules/brain-box-writing.mdc`): plain verbs, no metaphors,
short paragraphs, chapter-level cross-refs, modest claims.

## 4) Git state

Single branch: `main` (the final study and cleaned layout). The three
project eras are preserved as annotated tags, documented in README.md:
`era1-uncertainty-ppo`, `era1-live-scaffold`, `era2-simple-modelling`,
`era3-pre-restructure-snapshot` (full pre-cleanup snapshot; every file
deleted in the restructure lives there).

## 5) Nguyen meeting 2026-09-07 and the simulated-data redo

Transcripts: `notes/supervision/2026-09-07-nguyen-meeting-{main,end}.txt`.
Diagnosis: 60 monthly train episodes vs ~16k-parameter MLP = memorisation;
always-buy comes from rising-market class imbalance; DQN worst because value
estimation needs a stable state distribution. The "DQN uses state" Chapter 5
explanation was rejected as surface-level. Fix: simulated data (control),
smaller net, per-regime controlled testing; real data only as transfer test.

Redo (in `experiments/final_v2/`, primary study going forward):
- `sim_data.py`: regime-switching generator (up/down/flat), calibrated on
  real SPY train months only; 3000/300/600 balanced episodes; same episode
  format as `data.py`.
- `agents.py`/`harness.py`: pyramid MLP (64, 32) ≈ 2.8k params (was 128x128).
- `run_sim.py`: trains on sim, reports per-regime sim-test results + real
  2024-25 transfer test -> `results/sim_results.json` (+ `sim_run.log`).
- On balanced sim test, buy-and-hold LOSES (-$60/mo): always-buy no longer
  wins by construction. Report asterisk fix done (ch4 3×10^-4, recompiled).
- RESULTS (300k steps, 6 seeds): DQN +$126.94/ep (std 7.3), 6/6 state-dep,
  up +473.72 / down -103.35 / flat +10.45, real transfer +$130.61/mo (vs
  BAH +180.25). REINFORCE -5.71 (1/6 state-dep; 3 abstain, 2 always-buy,
  1 conditioned at +83). PPO -11.16 (0/6). Ranking inverts vs real study.
- Chapter ripple DONE (all six chapters, 2026-09-07 evening): ch1 RQ5+O8+
  contributions 4&5; ch2 sec 2.3.1 "Data requirements and simulated
  markets" + gap; ch3 "The Simulated Market Model" (3.2.x) + net sizes;
  ch4 "Simulated Market Data" section (Table 4.1 calibration) + config;
  ch5 "The Data Bottleneck" + "Simulated-Data Study" (Table 5.8, Figure
  5.8 via scripts/make_sim_figures.py) + rewritten Discussion/Summary;
  ch6 fifth conclusions subsection, appraisal, limitations, future work,
  closing. PDF 120pp, 0 errors. yoon2019timegan added to references.bib.
- EC decision pending. Viva prep: key numbers above; DQN-worst-on-real
  explained by value estimation needing stable/large data, not "uses state".

## 6) Known open items

- External-review self-grade ≈72 (report). Highest-value improvements:
  bootstrap CIs over existing per-month/per-seed results; one simple
  active baseline (e.g. moving-average rule) on the same test months.
- Viva pack (slides + demo) not started.
- Source Word drafts live outside the repo in
  `~/Documents/University of Surrey/... /Dissertation Drafts/`; the
  LaTeX in `latex/dissertation/` is canonical and ahead of them.
