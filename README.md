# Evaluating Reinforcement Learning Algorithms for Portfolio Trading

MSc dissertation project (EEEM004, University of Surrey; supervisor
Dr Cuong Nguyen). The dissertation asks whether REINFORCE, DQN, and PPO
can learn profitable, state-dependent trading policies for the SPY
exchange-traded fund against a buy-and-hold benchmark. It answers with
two studies.

**First study — real data.** Train on 60 real SPY months (2018–22),
test on 2024–25. No learner beat buy-and-hold ($180.25/month), and the
policy-gradient methods learned state-independent policies. The
diagnosis: 60 episodes cannot train an 18,000-weight network, and the
rising market makes always-buy unbeatable by construction.

**Main study — controlled data.** Train on 3,000 balanced simulated
episodes with a right-sized 2,800-weight network, where no fixed action
pattern is profitable. DQN earned +$126.94/episode with a
state-dependent policy on all 6 seeds. Transferred unchanged to 180
real months (2006–17 + 2023–25), it stayed profitable on every seed,
achieved a higher Sharpe ratio than buy-and-hold (0.224 vs 0.198), and
held zero exposure through October 2008.

The viva took place on 14 September 2026.

## Repository layout

| Path | Contents |
|---|---|
| `latex/dissertation/` | Submitted dissertation LaTeX source and compiled `main_full.pdf` (80-page body) |
| `experiments/final_v2/` | The two-study experiment package: environment, agents, tests, runners, results. See its own README |
| `scripts/` | Figure generators (dissertation and viva), deck builders, prompter/demo page builders |
| `latex/viva/` | Viva slide decks (PPTX) and slide figures |
| `demo/` | QR demo page shown at the viva (SPY transfer animation) |
| `notebooks/` | Reproduction notebook (verify results, recompute baselines, retrain guide) |
| `notes/` | Supervision notes and viva preparation |
| `docs/` | Project documentation, including the full workspace catalog |

The complete file-by-file map, including which files belong to which
study, is in [docs/WORKSPACE_CATALOG.md](docs/WORKSPACE_CATALOG.md).

## Compile the dissertation

```bash
cd latex/dissertation
xelatex -interaction=nonstopmode main_full.tex
bibtex main_full
xelatex -interaction=nonstopmode main_full.tex
xelatex -interaction=nonstopmode main_full.tex
```

Note: `build.py` in that folder is frozen (it regenerated chapters from
old Word sources and would overwrite the hand-edited files). Compile
with `xelatex` directly as above.

## Run the experiments

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cd experiments/final_v2
python -m pytest test_v2.py -q        # 21 verification tests

# First study (real data)
python run_final.py

# Main study (simulated data + real transfer)
python run_sim.py
```

The full runner list for both studies is in
[experiments/final_v2/README.md](experiments/final_v2/README.md).

## Reproduce it yourself

The guided path for anyone who wants to verify this work is
[notebooks/reproduce_main_study.ipynb](notebooks/reproduce_main_study.ipynb)
— it runs on
[Google Colab](https://colab.research.google.com/github/TheFinix13/dissertation-project/blob/main/notebooks/reproduce_main_study.ipynb)
with no local setup. It works in three levels: verify the committed
result files against the dissertation's headline numbers (seconds),
recompute the deterministic baselines from the committed episode data
and run the 21 verification tests (about two minutes), and a documented
full retrain (under an hour of training for the main study). Every
result file embeds its provenance: git commit, library versions,
seeds, and wall-clock time.

## Project history

The project changed direction three times before the two-study
dissertation. Each earlier era is preserved as an annotated git tag
rather than a branch, so `main` stays one line of development. To read
an old era:

```bash
git checkout era1-uncertainty-ppo     # look around (detached HEAD)
git checkout main                     # come back
```

| Tag | Period | What the project was at that time |
|---|---|---|
| `era1-uncertainty-ppo` | Feb–Jun 2026 | "AI-Driven Portfolio Protection": an uncertainty-aware PPO agent fed by a DeepAR-style probabilistic forecaster, evaluated on a 70-ticker grid with walk-forward validation. |
| `era1-live-scaffold` | May 2026 | Side artifact of era 1: an Alpaca paper-trading scaffold and personal-portfolio companion. Never part of the assessed work. |
| `pre-orphan-simple-modelling` | Aug 2026 | Transitional tree: Phase-0 games added on top of the era-1 code, before the history restart. |
| `era2-simple-modelling` | Jul–Aug 2026 | The pivot directed by Dr Nguyen: one complete simple model instead of the uncertainty stack. Phase-0 games (CartPole, Flappy Bird, LunarLander), then a single-stock monthly trading MDP with scratch REINFORCE and DQN. |
| `era3-pre-restructure-snapshot` | Sep 2026 | The full tree immediately before the 7 September cleanup; includes the superseded era-2 code, the `final_model` package, and old LaTeX builds removed from `main`. |

The final study on `main` grew out of era 2: the state representation
was rebuilt (`final_v2`), PPO was added back as a third algorithm, and
the evaluation-first protocol, behavioural analysis, and verification
tests were added on top. After the first study's negative result, the
main study replaced the training data with a calibrated simulator and
kept the real data as a transfer test.

## Student

Fiyinfoluwa Akano · URN 6962514 · EEEM004 · Supervisor Dr Cuong Nguyen
