# Evaluating Reinforcement Learning Algorithms for Portfolio Trading

MSc dissertation project (EEEM004, University of Surrey). The study tests
whether REINFORCE, DQN, and PPO can learn profitable, state-dependent
trading policies for the SPY exchange-traded fund, evaluated against a
buy-and-hold benchmark on a held-out 2024–2025 test period.

## Repository layout

| Path | Contents |
|---|---|
| `latex/dissertation/` | Final dissertation LaTeX source and compiled `main_full.pdf` |
| `experiments/final_v2/` | Environment, agents, feature pipeline, verification tests (`test_v2.py`), experiment runners, and results JSONs |
| `scripts/make_ch5_figures.py` | Regenerates the Chapter 5 figures into `latex/dissertation/figs5/` |
| `docs/` | Supervisor meeting notes and project planning documents |

## Compile the dissertation

```bash
cd latex/dissertation
xelatex -interaction=nonstopmode main_full.tex
bibtex main_full
xelatex -interaction=nonstopmode main_full.tex
xelatex -interaction=nonstopmode main_full.tex
```

Note: `build.py` in that folder is frozen (it regenerated chapters from the
old Word sources and would overwrite hand-edited files). Compile with
`xelatex` directly as above.

## Run the experiments

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cd experiments/final_v2
python -m pytest test_v2.py -q        # 21 verification tests
python run_final.py                   # main training and evaluation
python run_fees.py                    # fee sensitivity
python run_conditioning_v2.py         # behavioural (state-dependence) analysis
```

## Project history

The project went through three distinct ideas before the final study.
Each era is preserved under a git tag rather than a branch, so `main`
stays the single line of development:

| Tag | Period | What the project was at that time |
|---|---|---|
| `era1-uncertainty-ppo` | Feb–Jun 2026 | "AI-Driven Portfolio Protection": an uncertainty-aware PPO agent fed by a DeepAR-style probabilistic forecaster, evaluated on a 70-ticker grid. |
| `era1-live-scaffold` | May 2026 | Side artifact of era 1: a live-trading scaffold and personal-portfolio companion. |
| `era2-simple-modelling` | Jul–Aug 2026 | The pivot directed by Dr Nguyen: one complete simple model instead of the uncertainty stack. Phase-0 games (CartPole, Flappy Bird, LunarLander), then a single-stock monthly trading MDP with scratch REINFORCE and DQN. |
| `era3-pre-restructure-snapshot` | Sep 2026 | The final study (this codebase) immediately before the workspace cleanup; includes the superseded era-2 code and old LaTeX builds that were removed from `main`. |

The final study on `main` grew out of era 2: the state representation was
rebuilt (`final_v2`), PPO was added back as a third algorithm, and the
evaluation-first protocol, behavioural analysis, and verification tests
were added on top.
