# Simple-modelling track (Nguyen, July 2026)

**Branch:** `simple-modelling-clean`  
**GitHub:** https://github.com/TheFinix13/dissertation-project/tree/simple-modelling-clean

This branch is an **orphan** — it has **no shared history with `main`**.
It contains only the progressive simple-modelling dissertation work directed
by Dr Cuong Nguyen after July 2026 feedback. It does **not** include the
older uncertainty / dual-path / Phase-2 v0.17 stack.

| What lives here | What does **not** |
|---|---|
| Phase 0 games (CartPole, Flappy, LunarLander) | Live brokers / API keys |
| Iteration-1 / Iteration-2 one-stock trading MDP | FinRL / DeepAR / GluonTS stack |
| Scratch REINFORCE + PPO / A2C / DQN suites | 70-ticker Phase-2 grid |
| Standalone Chapters 4–5 under `latex/simple_modelling/` | Old `latex/chapters/` v0.17 writeup |

## Quick start

```bash
git clone -b simple-modelling-clean https://github.com/TheFinix13/dissertation-project.git
cd dissertation-project
python3.11 -m venv .venv311 && source .venv311/bin/activate
pip install -r requirements.txt
pip install flappy-bird-gymnasium "gymnasium[box2d]"   # Phase 0 games
```

## Run experiments

```bash
# Phase 0 — games
python experiments/phase0_games/run_cartpole_suite.py
python experiments/phase0_games/run_cartpole_qlearning.py
python experiments/phase0_games/run_flappy_suite.py
python experiments/phase0_games/run_lunarlander_suite.py
python reports/builders/build_phase0_games_charts.py

# Phase 1 — SPY monthly episodes
python experiments/iteration1/fetch_spy_daily.py
python experiments/iteration1/test_env.py
python experiments/iteration1/run_phase1_spy_daily.py      # Iteration 1 (3-D)
python experiments/iteration1/run_phase1_iteration2.py    # Iteration 2 (5-D)
```

## Chapters (NEW drafts — do not overwrite supervisor skeletons)

| Deliverable | Path |
|---|---|
| Combined PDF | `latex/simple_modelling/main.pdf` |
| Combined Word | `latex/simple_modelling/Chapter4-5_simple_modelling.docx` |
| Chapter 4 Word (new) | `latex/simple_modelling/Chapter_4_Implementation.docx` |
| Chapter 5 Word (new) | `latex/simple_modelling/Chapter_5_Empirical_Results.docx` |
| LaTeX sources | `latex/simple_modelling/ch{4,5}_*.tex` |

Rebuild:

```bash
cd latex/simple_modelling && pdflatex main.tex && pdflatex main.tex
# Word export helpers: see docs/build_chapter_word.md
```

## Student

Fiyinfoluwa Akano · URN 6962514 · EEEM004 · Supervisor Dr Cuong Nguyen
