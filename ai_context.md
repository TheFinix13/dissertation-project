# ai_context.md · final submission state
Last updated: 2026-09-07 (post workspace restructure)

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

## 5) Known open items

- External-review self-grade ≈72 (report). Highest-value improvements:
  bootstrap CIs over existing per-month/per-seed results; one simple
  active baseline (e.g. moving-average rule) on the same test months.
- Viva pack (slides + demo) not started.
- Source Word drafts live outside the repo in
  `~/Documents/University of Surrey/... /Dissertation Drafts/`; the
  LaTeX in `latex/dissertation/` is canonical and ahead of them.
