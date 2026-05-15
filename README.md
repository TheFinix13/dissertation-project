# Probabilistic DRL for Portfolio Risk Analysis

EEEM004 research project: an uncertainty-aware PPO policy for
capital preservation under regime stress.

## For supervisors — two notebooks

* **Single-ticker walkthrough (CPU/Colab, narrative):** [`notebooks/Dissertation_Walkthrough.ipynb`](notebooks/Dissertation_Walkthrough.ipynb) — open in Colab and *Run all*. Loads the SPY test-window dataset, builds the DeepAR-style probabilistic forecaster, prints the uncertainty values, states the mathematics that differentiates the probabilistic agent from the baseline PPO, and renders the comparison tables and equity-curve plots. Ships with executed outputs so the notebook is readable without running anything.

  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TheFinix13/dissertation-project/blob/main/Dissertation_Walkthrough.ipynb)

* **Heavy experiments runner (Colab GPU only):** [`notebooks/Run_Full_Experiments.ipynb`](notebooks/Run_Full_Experiments.ipynb) — *Runtime → T4 GPU → Run all*. Clones the repo, smoke-tests the GPU, runs the full broad-universe × 10-seed × 50 000-step extended grid + walk-forward folds + bootstrap, rebuilds the Word document with the new numbers, then offers a one-click zip download. ~5–7 hours on T4, ~2 hours on A100. *Do not run on a CPU laptop.*

For a local run instead of Colab:

```bash
git clone https://github.com/TheFinix13/dissertation-project.git
cd dissertation-project
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install jupyter
jupyter notebook notebooks/Dissertation_Walkthrough.ipynb
```

The interim review draft (Surrey form) lives at
`reports/generated/interim_review_draft.md`.

## Generated artifacts (under `reports/generated/exports/`)

- `Main_Dissertation_Draft.docx` — the **academic** Master's dissertation. Headline robustness evidence is the four-agent comparison on a broad universe of 70 diversified stocks (Section 5.5) with the full per-ticker table in Appendix B; supplementary studies are an extended seed-stability check on a representative eight-ticker sub-universe (Section 5.5.1) and a four-fold walk-forward grid on a four-ticker subset (Section 6.4). Title page, abstract, 7 chapters, references, two appendices.
- `InterimReview.docx` — the formal Surrey Interim Review form.
- `equations/` — individual PNGs for every equation in the docx.

To regenerate every document:

```bash
venv/bin/python reports/builders/build_main_dissertation_docx.py       # academic dissertation
venv/bin/python reports/builders/build_interim_review_docx.py          # interim review form
```

Heaviest experiments (broad-universe × 10-seed × 50k-step extended grid, walk-forward
across all four folds, bootstrap-augmented training) live in
`notebooks/Run_Full_Experiments.ipynb` and run on a Colab T4/A100 GPU runtime — see
"Phase-2 (Colab GPU) pipeline" below.

## Project Structure

```
dissertation-project/
├── experiments/
│   ├── runners/             # CLI entry points (run_baseline.py, run_probabilistic_agent.py, ...)
│   ├── common.py            # Shared library: env, metrics, data, training helpers
│   ├── aggregate_results.py # Pools per-cell JSON results into median + IQR summaries
│   ├── configs/             # dissertation_protocol.json
│   └── results/             # Per-cell JSON + CSV outputs from runners
├── reports/
│   ├── builders/            # All build_*.py / generate_*.py / plot_*.py scripts
│   ├── generated/           # Outputs (markdown, charts/, exports/)
│   └── templates/           # Markdown templates and viva notes
├── notebooks/               # Dissertation_Walkthrough + Run_Full_Experiments
├── requirements.txt
└── README.md
```

## Experiment Pipeline

### Phase-1 (CPU) — runs on a laptop in 25–35 minutes

```bash
source venv/bin/activate

# 1) Baseline PPO with deterministic seeds
python experiments/runners/run_baseline.py

# 2) Probabilistic DeepAR-style uncertainty + PPO
python experiments/runners/run_probabilistic_agent.py

# 3) Buy-and-hold and all-cash benchmarks
python experiments/runners/run_benchmarks.py

# 4) Rule-based trailing stop-loss comparator (5 % and 10 % variants)
python experiments/runners/run_rule_baselines.py

# 5) Markdown summary, supervisor pack, plots
python reports/builders/generate_dissertation_report.py
python reports/builders/build_supervisor_pack.py
python reports/builders/plot_dissertation_visuals.py

# 6) Word documents (dissertation + interim review)
python reports/builders/build_main_dissertation_docx.py
python reports/builders/build_interim_review_docx.py
```

- Protocol config: `experiments/configs/dissertation_protocol.json`
- Artifacts: `experiments/results/`
- Report output: `reports/generated/dissertation_results.md`

### Phase-1 — broad-universe run (CPU, ~25–35 min)

```bash
python experiments/runners/run_benchmarks.py        --tickers broad_universe --tag broad
python experiments/runners/run_rule_baselines.py    --tickers broad_universe --tag broad
python experiments/runners/run_baseline.py          --tickers broad_universe --tag broad
python experiments/runners/run_probabilistic_agent.py --tickers broad_universe --tag broad

# Walk-forward subset (96 trainings, ~6–8 hours CPU)
python experiments/runners/run_walk_forward.py --tickers SPY,QQQ,XLK,XLF

# Extended seed-stability check on representative sub-universe (80 trainings, ~4–5 hours CPU)
python experiments/runners/run_probabilistic_agent.py --tickers basket --seeds extended --timesteps 50000 --tag extbasket

# Build the dissertation
python reports/builders/build_main_dissertation_docx.py
python reports/builders/build_interim_review_docx.py
```

### Phase-2 (Colab GPU) — heavy lifting only

Anything that takes more than ~1 hour on CPU lives in
`notebooks/Run_Full_Experiments.ipynb`. Runtime preset: *T4 GPU* for the headline
broad-universe grid (~5–7 h), *A100* if you also want the full broad-universe walk-forward
(~12–14 h on A100).

Or to drive the same heavy run from the command line (e.g. on a leased GPU node):

```bash
python experiments/runners/run_extended_grid.py \
    --tickers broad_universe --seeds extended --folds all \
    --timesteps 50000 --bootstrap-paths 16 --tag colab_70_extended
```

## CLI flag reference (every runner)

| Flag | Default | Notes |
|---|---|---|
| `--tickers` | legacy single ticker | Comma-separated, a CLI alias such as `basket` (8-ticker sub-universe) or a named group from `data.named_groups` in the protocol — `broad_universe` (the broad test universe (70 stocks)), `broad_stocks` (41 single names) or `broad_etfs` (29 ETFs). |
| `--seeds` | `[7, 19, 42]` | Comma-separated, or `default` / `extended` (10 seeds). |
| `--folds` | legacy single test window | Comma-separated fold ids from `walk_forward_folds`, or `all`. (Walk-forward runner only honours this.) |
| `--timesteps` | from protocol | PPO training budget per cell. |
| `--initial-balance` | $1,000,000 | Starting capital in USD; metric ratios (Sharpe, MDD, preservation) are unit-free. |
| `--bootstrap-paths` | 0 | Politis & Romano (1994) stationary block-bootstrap synthetic training paths. |
| `--tag` | none | Optional suffix appended to output filenames. |
| `--agents` | `baseline,probabilistic` | Walk-forward only; subset to run. |
