#!/usr/bin/env python3
"""Build notebooks/phase1_trading_colab.ipynb (run once after editing)."""
from __future__ import annotations

import json
import uuid
from pathlib import Path

OUT = Path(__file__).resolve().parent / "phase1_trading_colab.ipynb"


def md(text: str) -> dict:
    src = [ln + "\n" for ln in text.strip("\n").split("\n")]
    if src:
        src[-1] = src[-1].rstrip("\n") + "\n"
    return {
        "cell_type": "markdown",
        "metadata": {},
        "id": uuid.uuid4().hex[:8],
        "source": src,
    }


def code(text: str) -> dict:
    src = [ln + "\n" for ln in text.strip("\n").split("\n")]
    if src:
        src[-1] = src[-1].rstrip("\n") + "\n"
    return {
        "cell_type": "code",
        "metadata": {},
        "id": uuid.uuid4().hex[:8],
        "execution_count": None,
        "outputs": [],
        "source": src,
    }


cells: list[dict] = []

# ---------------------------------------------------------------- intro
cells.append(md(r"""
# Phase 1 — A simple trading MDP, modelled honestly (Nguyen demo)

**Student:** Fiyinfoluwa Akano · URN 6962514 · EEEM004
**Branch:** `simple-modelling-clean`
**Prerequisite:** Phase 0 (`notebooks/phase0_games_colab.ipynb`) proved the policy-RL training loop on games with known outcomes.

**Modelling goal (locked with Dr Nguyen):** one stock (SPY), discrete actions
**Hold / Buy one / Sell one** with a feasibility mask, reward = **change in wealth**,
learning = **policy-based RL**. No indicators, no uncertainty layers — the point is to
understand what a *simple* model does before adding anything.

**The two findings so far (both honest, both useful):**

1. **Iteration 1 (3-D state):** every policy-gradient method — REINFORCE, A2C **and** PPO —
   converged to *fully-invested buy-and-hold* on all 26 held-out months.
   The naive "maximise ΔW" objective makes buy-max the optimum; the algorithms found it.
2. **Iteration 2 (5-D state):** a richer state broke the buy-max collapse but produced a
   *mostly-flat* policy. State alone does not create timing skill — the **reward** must change next.

> **Meeting tip:** keep `RUN_MODE = "cached"` — every table and chart loads from the
> repo's committed results in seconds. Switch to `"live"` to retrain the whole
> policy-family ablation in front of him (~1–5 min on Colab CPU).
"""))

# ---------------------------------------------------------------- setup
cells.append(md(r"""
## 0 — Setup
"""))

cells.append(code(r"""
# Installs (Colab). Safe to re-run.
import sys, subprocess

def pip_install(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])

IN_COLAB = "google.colab" in sys.modules
print("IN_COLAB =", IN_COLAB)

pip_install(
    "gymnasium==0.29.1",
    "stable-baselines3==2.3.2",
    "torch",
    "matplotlib",
    "pandas",
    "numpy",
)
print("Installs OK")
"""))

cells.append(code(r"""
# Config
from pathlib import Path
import json, sys, subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# "cached" → load committed dissertation results (seconds) — MEETING DEFAULT
# "live"   → retrain the REINFORCE/A2C/PPO ablation now (~1–5 min CPU)
RUN_MODE = "cached"

REPO_URL = "https://github.com/TheFinix13/dissertation-project.git"
BRANCH = "simple-modelling-clean"
CLONE_DIR = Path("/content/dissertation-project") if IN_COLAB else Path(".").resolve()

def ensure_repo() -> Path:
    here = Path(".").resolve()
    for p in [here, *here.parents]:
        if (p / "experiments" / "iteration1").exists():
            print("Using local repo:", p)
            return p
    if IN_COLAB:
        if CLONE_DIR.exists():
            subprocess.check_call(["rm", "-rf", str(CLONE_DIR)])
        subprocess.check_call([
            "git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, str(CLONE_DIR)
        ])
        print("Using cloned repo:", CLONE_DIR)
        return CLONE_DIR
    raise FileNotFoundError("Run from the dissertation repo, or open in Colab.")

ROOT = ensure_repo()
ITER1 = ROOT / "experiments" / "iteration1"
RES = ITER1 / "results"
CHARTS = ROOT / "reports" / "generated" / "charts"
sys.path.insert(0, str(ITER1))
print("RUN_MODE =", RUN_MODE)
"""))

# ---------------------------------------------------------------- why ppo
cells.append(md(r"""
## 1 — Why PPO? (and could the others trade?)

We do **not** use PPO "because FinRL does". The choice is justified twice:

**(a) Phase 0 evidence — same algorithms, games with known outcomes**

| Mean eval return | CartPole (solved ≈ 475) | Flappy | LunarLander (solved ≈ 200) |
|---|---:|---:|---:|
| Random | 28.8 | −7.4 | −183 |
| REINFORCE (scratch) | 292 | 7.1 | 15 |
| A2C | **500** | 4.6 | −41 |
| PPO | **500** | **12.6** | 176 |
| DQN | **500** | 7.0 | 178 |

PPO was the **most consistent across all three games** — A2C fell apart on the harder
ones, REINFORCE learns but is high-variance.

**(b) Phase 1 evidence — same algorithms, same trading MDP (Section 4 below)**

We re-ran **REINFORCE and A2C on the identical trading MDP** (same state, reward,
fees, data split, seed, 80k-step budget) so the comparison is an ablation, not an opinion.

**Could the other algorithms trade at all?**

| Algorithm | Feasible on our MDP? | Verdict |
|---|---|---|
| Tabular Q-learning | **No** | cash/prices are continuous — the table needs infinitely many rows (we showed this on CartPole by having to hand-bin the state) |
| DQN | Yes | works, but **value-based** — off the policy-based track Nguyen locked; kept as a Phase-0 comparator only |
| REINFORCE | Yes | policy-based, we coded it from scratch — high-variance updates |
| A2C | Yes | policy-based with a critic — but **unclipped** updates |
| **PPO** | **Yes** | policy-based, critic, **clipped** updates → stable; Phase-0 winner |

**Objectives (the "score" beat of the four-beat loop):**

REINFORCE:

$$
L(\theta) = -\sum_t \log\pi_\theta(a_t\mid s_t)\, G_t
$$

A2C (advantage replaces the raw return):

$$
L(\theta) = -\sum_t \log\pi_\theta(a_t\mid s_t)\, A_t,
\qquad A_t = G_t - V_\phi(s_t)
$$

PPO (clip the policy ratio so one update cannot destroy the policy):

$$
L^{\mathrm{CLIP}}(\theta)=\mathbb{E}_t\!\left[\min\bigl(\rho_t A_t,\;
\mathrm{clip}(\rho_t,\,1-\varepsilon,\,1+\varepsilon)\,A_t\bigr)\right],
\qquad
\rho_t=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}
$$
"""))

# ---------------------------------------------------------------- MDP
cells.append(md(r"""
## 2 — The trading MDP, formally

$$
s_t = \begin{bmatrix}\Delta P_t \\ C_t \\ n_t\end{bmatrix}
\qquad
\Delta P_t = \frac{P_t - P_{t-1}}{P_{t-1}}
$$

- $\Delta P_t$ — one-step price return, $C_t$ — cash, $n_t$ — shares held
  (explicit cash + share **count**, not a vague hold flag $H\in\{0,1\}$ — Nguyen's correction).

**Actions** $\;a_t \in \{0:\text{Hold},\; 1:\text{Buy one},\; 2:\text{Sell one}\}$

**Feasibility mask** — Buy needs $C_t \ge P_t(1+c)$; Sell needs $n_t \ge 1$; illegal → forced Hold.

**Wealth and reward:**

$$
W_t = C_t + n_t P_t
\qquad\qquad
r_t = W_{t+1} - W_t
$$

**Episode:** one calendar month of SPY daily closes ($T \approx 18$–$23$ steps).
Honest disclaimer: the target design is one trading *day* at minute bars ($T=390$);
daily months are the interim protocol and we say so.

The whole return of an episode **telescopes** to the wealth change — this gives a
built-in accounting test:

$$
\sum_{t=0}^{T-1} r_t = W_T - W_0 = \Delta W
$$

If the sum of rewards ever differs from $\Delta W$, the environment has a bug.
Every run below checks this identity (`accounting_ok`).
"""))

cells.append(code(r"""
# The environment source — the exact class used for every result below
src = (ITER1 / "env.py").read_text()
print(src[: src.find("def make_synthetic_day")])
"""))

# ---------------------------------------------------------------- protocol
cells.append(md(r"""
## 3 — Experiment protocol (identical for every method)

| Item | Value |
|---|---|
| Data | SPY adjusted daily closes, 2018–2024 |
| Split | **chronological**: 58 train months, 26 test months (2022-11 → 2024-12); never shuffled |
| Fee | 5 bps per trade ($c = 0.0005$) |
| Initial cash | \$10{,}000 |
| Training budget | 80{,}000 env steps, seed 42 — same for every learner |
| Evaluation | deterministic (greedy) policy + feasibility mask on all 26 test months |
| Honesty checks | accounting identity per episode; illegal-action-request rate; fair buy-max baseline |

### Baselines — each with a *named objective* (Nguyen's requirement)

| ID | Rule | Objective |
|---|---|---|
| **B0** do-nothing | always Hold | no exposure — the zero line |
| **B1** buy-one-hold | buy 1 share, hold | capture the month's move with one share |
| **B1b** buy-max-hold | invest all cash, hold | the **fair** buy-and-hold — full capital at work |
| **B2** random-masked | uniform over legal actions | chance floor |

B1b is the baseline that matters: with \$10k cash and SPY ≈ \$400–\$500, "one share"
buy-and-hold uses <5% of capital. Beating B1 while losing to B1b would be a
false win — adding B1b is what caught the collapse below.
"""))

# ---------------------------------------------------------------- iter1
cells.append(md(r"""
## 4 — Iteration 1 results (3-D state) + the policy-family ablation

**Finding 1:** under the naive objective "maximise $\Delta W$", the learned policy
**is** fully-invested buy-and-hold. Not approximately — the executed action
sequence is *identical* to B1b on **26 / 26** test months.

**Finding 1b (the ablation):** this is **not a PPO quirk**. REINFORCE (our scratch
implementation) and A2C converge to the *same* buy-max policy on the same MDP.
The collapse is a property of the **objective**, not the optimiser — which is
exactly why Iteration 3 changes the reward, not the algorithm.
"""))

cells.append(code(r"""
if RUN_MODE == "live":
    # Retrain REINFORCE + A2C (80k steps each) and re-evaluate everything.
    # PPO is loaded from the committed Iteration-1 model for protocol identity.
    print("Running the full ablation live — expect ~1–5 min on CPU ...")
    subprocess.check_call([sys.executable, str(ITER1 / "run_phase1_algo_ablation.py")])

abl = json.loads((RES / "phase1_algo_ablation.json").read_text())
iter1 = json.loads((RES / "phase1_spy_daily_results.json").read_text())

order = ["B0_do_nothing", "B2_random", "B1_buy_one_hold", "B1b_buy_max_hold",
         "A1_reinforce", "A1_a2c", "A1_ppo"]
rows = []
for k in order:
    s = abl["summary"][k]
    rows.append({
        "method": k,
        "mean ΔW ($)": round(s["mean_delta_w"], 2),
        "std": round(s["std_delta_w"], 1),
        "trades/mo": round(s["mean_trades"], 1),
        "win vs B1b": f"{s['win_rate_vs_B1b_max']:.0%}",
        "months identical to B1b": f"{s['months_identical_to_B1b']}/26",
        "accounting ok": s["accounting_ok"],
    })
display(pd.DataFrame(rows).set_index("method"))

ill = iter1["doer_extras"]["ppo_illegal_action_request_rate"]
print(f"\nPPO illegal-action-request rate on test months: {ill:.1%} (env forces Hold)")
print("Budget:", abl["meta"]["timesteps_budget"], "steps · seed", abl["meta"]["seed"],
      "· fee", abl["meta"]["fee"], "· cash $", abl["meta"]["initial_cash"])
"""))

cells.append(code(r"""
# Charts: ablation bar chart + a representative wealth path
fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
for ax, png, title in [
    (axes[0], CHARTS / "phase1_algo_ablation_delta_w.png", "Policy-family ablation"),
    (axes[1], RES / "wealth_path.png", "Wealth path — PPO vs B1b (mid test month)"),
]:
    if png.exists():
        ax.imshow(mpimg.imread(png))
        ax.set_title(title, fontsize=10)
    ax.axis("off")
plt.tight_layout()
plt.show()
"""))

cells.append(md(r"""
### How to read this

- All three learners sit at **exactly** B1b's mean ΔW (\$76.9), std, and trade count —
  because they execute the same actions. Column "months identical to B1b" is the proof.
- Random (B2) trades a lot and earns little; do-nothing (B0) earns nothing.
- `accounting_ok = True` everywhere: the telescoping identity $\sum r_t = \Delta W$ held
  on every one of the $26 \times 7$ episodes.

**Why this is a result, not a failure:** 2022-11 → 2024-12 was mostly a rising market.
With reward $= \Delta W$ and no risk term, *being fully invested is genuinely optimal in
expectation* — the agents solved the MDP we posed. The MDP was the wrong question,
and only the fair B1b baseline exposed that. This is the modelling lesson Nguyen
wanted the project to surface.
"""))

# ---------------------------------------------------------------- iter2
cells.append(md(r"""
## 5 — Iteration 2 (5-D state): does a richer state fix it?

Everything held fixed (reward, data, fees, budget, seed) **except the state**:

$$
s_t = \begin{bmatrix}
\Delta P_t \\[2pt]
\mathrm{PnL}_t \\[2pt]
\tau_t \\[2pt]
C_t / C_0 \\[2pt]
n_t P_t / C_0
\end{bmatrix}
\qquad
\begin{aligned}
&\mathrm{PnL}_t &&\text{— unrealised return vs average entry price} \\
&\tau_t = t/T &&\text{— progress through the month} \\
&C_t/C_0 &&\text{— cash as fraction of starting capital} \\
&n_t P_t/C_0 &&\text{— position value as fraction of starting capital}
\end{aligned}
$$

Any change in behaviour is attributable to the state alone.
"""))

cells.append(code(r"""
it2 = json.loads((RES / "phase1_iteration2_results.json").read_text())

s2 = it2["summary"]["A2_ppo_iter2_5d"]
b = it2["behaviour"]
df2 = pd.DataFrame([
    {"method": "B1b buy-max hold", "mean ΔW ($)": 76.91, "trades/mo": 19.1,
     "months identical to B1b": "26/26 (by definition)"},
    {"method": "PPO Iter-2 (5-D state)", "mean ΔW ($)": round(s2["mean_delta_w"], 2),
     "trades/mo": round(s2["mean_trades"], 1),
     "months identical to B1b": f"{b['months_identical_to_buy_max']}/{b['n_test_months']}"},
]).set_index("method")
display(df2)
print(f"Iter-2 PPO beats B1b on {b['win_months_vs_buy_max']}/{b['n_test_months']} months "
      "(the falling months, where staying in cash wins).")

png = CHARTS / "phase1_iteration2_delta_w.png"
if png.exists():
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.imshow(mpimg.imread(png))
    ax.axis("off")
    plt.tight_layout()
    plt.show()
"""))

cells.append(md(r"""
### Finding 2 — from one extreme to the other

The 5-D state **did** break the collapse: 0/26 months identical to buy-max.
But the greedy policy became **mostly flat** (deterministic ΔW = 0.0; it wins against
B1b only on the 8 falling months, by not being invested).

So under pure $\Delta W$:

- 3-D state → **always invested** (copy buy-max)
- 5-D state → **almost never invested** (sit in cash)

Both are degenerate corner solutions. Richer *perception* did not create timing skill,
because the *objective* still says nothing about risk or opportunity cost.
"""))

# ---------------------------------------------------------------- next
cells.append(md(r"""
## 6 — What we learned, and the next lever

**The staged-modelling story (say it in this order):**

1. Phase 0: the training loop is correct — verified on games with known outcomes.
2. Algorithm choice: policy family, PPO primary — justified by Phase 0 consistency **and**
   a same-MDP ablation where all three policy methods found the same optimum.
3. Iteration 1: naive reward → buy-max collapse. Caught only by the fair B1b baseline.
4. Iteration 2: richer state → flat policy. State was not the binding constraint.
5. Therefore **Iteration 3 changes the reward**, e.g.

$$
r_t = \Delta W_t - \lambda\, D_t
$$

where $D_t$ is drawdown from the in-month high-water mark. That gives the agent a
*reason* to be invested sometimes and flat sometimes — the behaviour neither state
could produce. (Iteration 3 is planned; not part of this notebook's results.)

**One-line summary for the meeting:**
> "Every algorithm we tried solved the MDP we posed; the MDP was the wrong question.
> Iterations 1–2 isolated *why* — so Iteration 3 changes the objective, not the model."
"""))

# ---------------------------------------------------------------- viva
cells.append(md(r"""
## 7 — Viva checklist (answer out loud, no notes)

1. **Why PPO and not DQN or tabular Q?** Policy-based track (Nguyen), Phase-0
   consistency, clipped updates; tabular Q cannot index continuous cash/prices;
   DQN is value-based and kept as a Phase-0 comparator.
2. **If all three policy methods gave the same trading answer, why prefer PPO?**
   Because on *harder* problems (Flappy, LunarLander) PPO was the only consistently
   strong policy method — and its clipping gives stability guarantees the others lack.
   The trading tie itself is informative: the objective, not the optimiser, is binding.
3. **Why is "PPO = buy-and-hold" a result and not a bug?** Accounting identity passed,
   masks enforced, action sequences audited — the agent genuinely maximised the
   reward we specified. The specification was the problem.
4. **Why did the 5-D state go flat?** With no risk/opportunity term, being in cash is
   as defensible as being invested; richer features let the policy find the *other*
   corner solution.
5. **What exactly changes in Iteration 3?** Only the reward ($\Delta W - \lambda D$);
   state, data, protocol stay fixed — same one-change-at-a-time discipline.
"""))

cells.append(md(r"""
---
## Repo map (if he asks "where is the code?")

| What | Path |
|---|---|
| This notebook | `notebooks/phase1_trading_colab.ipynb` |
| Iteration-1 env (3-D) | `experiments/iteration1/env.py` |
| Iteration-2 env (5-D) | `experiments/iteration1/env_v1.py` |
| Scratch masked REINFORCE | `experiments/iteration1/reinforce_trading.py` |
| Ablation runner | `experiments/iteration1/run_phase1_algo_ablation.py` |
| Iteration-1 runner | `experiments/iteration1/run_phase1_spy_daily.py` |
| Iteration-2 runner | `experiments/iteration1/run_phase1_iteration2.py` |
| Env unit tests | `experiments/iteration1/test_env.py` |
| Results JSON | `experiments/iteration1/results/phase1_*.json` |
| Why-PPO note | `docs/phase1_why_ppo.md` |
| Phase 0 notebook | `notebooks/phase0_games_colab.ipynb` |
"""))

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        "colab": {"provenance": [], "toc_visible": True},
    },
    "cells": cells,
}

OUT.write_text(json.dumps(nb, indent=1))
print(f"Wrote {OUT} ({len(cells)} cells)")
