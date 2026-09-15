# ai_context.md · final submission state
Last updated: 2026-09-08 (sim-data redo COMPLETE + examiner-readability pass:
abstract rewritten, "pilot" renamed "the first study" everywhere, two-study
design introduced in ch1 before first use, ch3 formulas moved to display
equations. PDF 114pp, 0 errors.)

Compact state of the project for future AI sessions.
NOTE: the post-viva state (viva 14 Sep 2026 and after) is maintained
locally in `notes/viva/ai_context_post_viva.md`, which is not tracked.
Read that file first if it exists. The workspace map is in
`docs/WORKSPACE_CATALOG.md`.

## 1) What this project is

MSc dissertation (EEEM004, Surrey; supervisor Dr Cuong Nguyen).
Question: can REINFORCE / DQN / PPO learn profitable, **state-dependent**
trading policies for SPY versus buy-and-hold?
Headline answer (two studies): the real-data PILOT (60 train months,
test 2024-25) failed — no agent beat buy-and-hold, policy-gradient seeds
state-independent. The MAIN sim-data study (3000 balanced episodes,
2.8k-param pyramid net) succeeded: DQN +$126.94/ep on all 6 seeds where
no fixed pattern profits, and transferred to 180 real months (2006-17 +
2023-25) profitably with a higher Sharpe than buy-and-hold (0.224 vs
0.198), holding zero exposure through Oct 2008. Data was the bottleneck.

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
  up +473.72 / down -103.35 / flat +10.45. REINFORCE -5.71 (1/6 state-dep;
  3 abstain, 2 always-buy, 1 conditioned at +83). PPO -11.16 (0/6).
  Ranking inverts vs the real-data pilot.
- FULL SUITE RERUN on sim data (`run_sim_suite.py`, results in
  `results/sim_{slice,lambda,cells,fees}.json`): trade-size re-selects
  0.25 C0; drawdown feature again no effect under wealth reward; lambda
  sweep selects 0.25 — risk reward now WORKS for DQN (down-loss −73%,
  return −1.5%, Sharpe up), overturning the pilot conclusion; fee sweep:
  DQN stays +$95.27/ep at 50 bps (10× fee) while BAH goes negative.
- EXTENDED REAL TRANSFER SET (`real_transfer.py` -> `data/spy_transfer.npz`):
  180 SPY months outside the 2018-22 calibration window (2006-2017 +
  2023-2025), incl. the 2008 crisis. DQN transfer: +$60.28/mo (BAH +80.24)
  but Sharpe 0.224 vs 0.198, profitable on all 6 seeds, ZERO exposure
  through Oct 2008 (crisis Sep08–Mar09: agents −$180 vs BAH −$3,046).
- Chapter restructure DONE 2026-09-08 (per user: eliminate old content,
  don't append). ch5 REWRITTEN from scratch (999 lines vs 1579): pilot
  compressed to one ~5pp section (5.3), then bottleneck (5.4), sim
  trade-size (5.5), main regime comparison (5.6), state (5.7), risk
  reward (5.8), fees (5.9), real transfer + 2008 worked example (5.10),
  discussion (5.11), summary (5.12). Old ch5 archived at
  `notes/archive/ch5_pilot_only_2026-09-07.tex`. ch1/2/3/4/6 rewritten
  for consistency (ch4 has "The Extended Transfer Set" + pyramid-net
  config; ch6 conclusions/limitations/future-work now transfer-aware).
  Figures: `scripts/make_sim_figures.py` -> figs5/fig5_{sim,transfer,
  lambda,fees}.pdf. PDF 113pp, 0 errors, no undefined refs.
- EC decision pending. Viva prep: key numbers above; DQN-worst-on-real
  explained by value estimation needing stable/large data, not "uses state".
- PAGE CUT DONE 2026-09-08: main body now exactly 80 printed pages
  (ch1 p1 -> ch6 ends p80; refs p81; appendices A-E after). Was 92. How:
  ch3 algorithm sections 3.6-3.9 compressed (derivations cited to
  Williams/Sutton-Barto/Mnih instead of reproduced; all project-specific
  settings kept: gamma=0.99, 50k buffer/64 batch, 500-step target sync,
  eps 1.0->0.05, Adam 1e-3, eps-clip 0.2, GAE, both network sizes);
  staged-state summary table and Flappy-Bird table folded into prose;
  notation table moved to NEW Appendix E (ch3 state table renumbered
  3.2->3.1); ch4 verification/corrections deduplicated, config bullet
  list inlined, fig 4.2 at 0.72\textwidth; ch5 sec 5.2 state-dependence
  told once, discussion "Overall" passage deduplicated vs summary
  ("kept its edge"/"learned edge" slang also fixed); tocdepth=1 (ToC
  2pp). appendix_algorithms ref updated 3.8.5->3.8. 0 errors, bibtex OK.

## 5b) FINAL DRAFT STATUS (9 Sep 2026)

- The LaTeX build (`latex/dissertation/main_full.pdf`) is the FINAL DRAFT
  for submission: main body exactly 80 printed pages (handbook guideline),
  101 physical pages total incl. refs + appendices A-D, 0 LaTeX errors,
  bibtex clean, no undefined refs.
- Voice: three full style audits against
  `brain-box/school/methodology/fiyin-writing-style.md` plus ~15
  user-flagged passages fixed on 8 Sep (bimodal seeds, single principle,
  earned its place, isolates/mean per month, informative, carries,
  price of a controlled experiment). Remaining agent-register markers
  sweep clean except the deliberate glossed "regime-switching models"
  citation in ch2.
- Any further edit before submission should be compiled twice + bibtex
  and re-checked to stay at 80 body pages.

## 6) Known open items

- External-review self-grade ≈72 (report). Highest-value improvements:
  bootstrap CIs over existing per-month/per-seed results; one simple
  active baseline (e.g. moving-average rule) on the same test months.
- Viva pack (slides + demo) not started.
- Source Word drafts live outside the repo in
  `~/Documents/University of Surrey/... /Dissertation Drafts/`; the
  LaTeX in `latex/dissertation/` is canonical and ahead of them.
