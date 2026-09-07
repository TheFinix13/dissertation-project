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

## Branches

- `workspace-restructure` — cleaned final layout (this branch).
- `final-v2-state-architecture` — full pre-restructure snapshot, including
  earlier phase-0 games, iteration-1 experiments, and old LaTeX/Word builds.
- Older branches (`simple-modelling-clean`, `epistemic-uncertainty`,
  `fiyins-portfolio`, `live-trading`, `main`) predate the final study.
