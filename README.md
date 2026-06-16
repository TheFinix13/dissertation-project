# Probabilistic DRL for Portfolio Risk Analysis

EEEM004 research project: an uncertainty-aware PPO policy for
capital preservation under regime stress.

**Project status — Phase-2 complete.** The full experimental grid has been run
across the `market_sample` universe (70 stocks, 10 seeds, 50,000 PPO time-steps
per cell) plus a four-fold walk-forward validation. On the held-out 2022–2025
window the uncertainty-aware agents grow a $1M portfolio to a **median of
~$1.6M** (median Sharpe **+0.68 to +0.70**, win-rate **89–92%**) versus the
baseline PPO's **~$999,063** (Sharpe **−0.04**, win-rate **46.1%**). The
**canonical dissertation is the LaTeX source in [`latex/`](latex/)**, reconciled
to these results; run [`latex/build_docx.sh`](latex/build_docx.sh) for an
editable Word copy. (On drawdown the honest benchmark is buy-and-hold, ~25.9%
median: the guard beats it on the typical stock and the index basket, ~22–24%,
but not on the worst-case tail.)

## For supervisors — two notebooks

Both notebooks live in `notebooks/` and are designed to run top to bottom with one click.

* **`01_Project_Walkthrough.ipynb`** — Phase-1 demonstration, runs locally in ~5 minutes. Loads the SPY test-window data, trains the probabilistic LSTM forecaster in both *aleatoric* and *epistemic* modes, trains three PPO variants (baseline, aleatoric, epistemic), renders the headline comparison table and equity curves, and finishes with a day-by-day replay of the trained agent's decisions. Designed to be readable end-to-end without prior knowledge of the project. Ships with executed outputs.

  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TheFinix13/dissertation-project/blob/main/notebooks/01_Project_Walkthrough.ipynb)

* **`02_Full_Experiments.ipynb`** — Phase-2 production grid, designed for Google Colab with GPU. *Runtime → T4 GPU → Run all*. Clones the repo, smoke-tests the GPU on SPY, then runs the full 70-stock × 10-seed × 50,000-step grid for all three agent variants, plus walk-forward folds and bootstrap augmentation. Aggregates everything into a single results CSV and zips it for download. ~3–6 hours on T4, ~1–2 hours on A100. *Do not run on a CPU laptop.*

  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TheFinix13/dissertation-project/blob/main/notebooks/02_Full_Experiments.ipynb)

For a local run instead of Colab:

```bash
git clone https://github.com/TheFinix13/dissertation-project.git
cd dissertation-project
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install jupyter
jupyter notebook notebooks/01_Project_Walkthrough.ipynb
```

## The dissertation (canonical)

The **canonical Master's dissertation is the LaTeX source in [`latex/`](latex/)**,
reconciled to the completed Phase-2 results.

```bash
cd latex && ./build.sh          # PDF → latex/main.pdf
latex/build_docx.sh             # Word → dissertation.docx at repo root
```

Heaviest experiments (market-sample × 10-seed × 50k-step extended grid, walk-forward
across all four folds, bootstrap-augmented training) live in
`notebooks/02_Full_Experiments.ipynb` and run on a Colab T4/A100 GPU runtime — see
"Phase-2 (Colab GPU) pipeline" below.

## Project Structure

```
dissertation-project/
├── latex/                   # Canonical dissertation source (PDF + DOCX export)
├── experiments/
│   ├── runners/             # CLI entry points (run_baseline.py, run_ablation.py, ...)
│   ├── common.py            # Shared library: env, metrics, data, training helpers
│   ├── aggregate_results.py # Pools per-cell JSON results into median + IQR summaries
│   ├── configs/             # dissertation_protocol.json
│   └── results/             # Per-cell JSON + canonical Phase-2 CSVs
├── reports/
│   ├── builders/            # plot_phase2_charts.py, build_forecaster_calibration.py
│   ├── generated/           # charts/, stats/ (Phase-2 figures and inference outputs)
│   └── templates/           # viva Q&A notes
├── notebooks/               # 01_Project_Walkthrough (local) + 02_Full_Experiments (Colab)
├── scripts/                 # Lab-machine helpers (run_phase2.py, sync_results.py)
├── docs/CHECKPOINT.md       # Deep-state snapshot (updated at major divergences)
├── ai_context.md            # Compact state summary for fresh chat sessions
└── requirements.txt
```

## Experiment Pipeline

### Phase-1 (CPU) — runs on a laptop in 25–35 minutes

```bash
source venv/bin/activate

python experiments/runners/run_baseline.py
python experiments/runners/run_probabilistic_agent.py
python experiments/runners/run_benchmarks.py
python experiments/runners/run_rule_baselines.py
python experiments/aggregate_results.py
MPLBACKEND=Agg MPLCONFIGDIR=.mplconfig venv/bin/python reports/builders/plot_phase2_charts.py
```

- Protocol config: `experiments/configs/dissertation_protocol.json`
- Artifacts: `experiments/results/per_cell/`

### Phase-2 (Colab GPU) — heavy lifting only

Anything that takes more than ~1 hour on CPU lives in
`notebooks/02_Full_Experiments.ipynb`. Runtime preset: *T4 GPU* for the headline
market-sample grid (~5–7 h), *A100* if you also want the full market-sample walk-forward
(~12–14 h on A100).

Or from the command line on a leased GPU node:

```bash
python experiments/runners/run_extended_grid.py \
    --tickers market_sample --seeds extended --folds all \
    --timesteps 50000 --bootstrap-paths 16 --tag colab_70_extended
```

### Statistical inference and ablation (CPU, seconds to minutes)

```bash
venv/bin/python experiments/stats_significance.py
venv/bin/python experiments/compute_spy_ablation_stats.py
venv/bin/python reports/builders/build_forecaster_calibration.py
MPLBACKEND=Agg MPLCONFIGDIR=.mplconfig venv/bin/python reports/builders/plot_phase2_charts.py
```

## CLI flag reference (every runner)

| Flag | Default | Notes |
|---|---|---|
| `--tickers` | legacy single ticker | Comma-separated, a CLI alias such as `basket` (8-ticker sub-universe) or a named group from `data.named_groups` in the protocol — `market_sample` (the market sample (70 stocks)), `market_sample_stocks` (41 single names) or `market_sample_etfs` (29 ETFs). |
| `--seeds` | `[7, 19, 42]` | Comma-separated, or `default` / `extended` (10 seeds). |
| `--folds` | legacy single test window | Comma-separated fold ids from `walk_forward_folds`, or `all`. (Walk-forward runner only honours this.) |
| `--timesteps` | from protocol | PPO training budget per cell. |
| `--initial-balance` | $1,000,000 | Starting capital in USD; metric ratios (Sharpe, MDD, preservation) are unit-free. |
| `--bootstrap-paths` | 0 | Politis & Romano (1994) stationary block-bootstrap synthetic training paths. |
| `--tag` | none | Optional suffix appended to output filenames. |
| `--agents` | `baseline,probabilistic` | Walk-forward only; subset to run. |
