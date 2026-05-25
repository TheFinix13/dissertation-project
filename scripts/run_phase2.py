"""Run all Phase-2 experiments with full logging to file.

Usage on a lab machine:

    python scripts/run_phase2.py

What it does
- Runs all four phase-2 jobs in sequence: baseline, aleatoric,
  epistemic, walk-forward.
- Streams output to the terminal AND mirrors it to a timestamped log
  file under experiments/results/logs/.
- If one job crashes, it logs the failure and moves on to the next so
  a single bad arm does not block the others.
- Honours the per-cell checkpointing: re-running this script is safe
  and will resume from where it left off.

Why a Python wrapper instead of `tee`
- Works the same on Linux, macOS and Windows.
- No shell-portability issues if the lab machine is Windows / PyCharm.
- The log files travel with the result files when you push to the
  branch, so the dissertation analysis has a full record of every run.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = REPO_ROOT / "experiments" / "results" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Tweak any of these to change the grid. The defaults match the
# Configuration cell of notebooks/02_Full_Experiments.ipynb.
TICKERS = "market_sample"
SEEDS = "extended"
TIMESTEPS = "50000"
BOOTSTRAP = "32"
DEVICE = "cpu"

WF_TICKERS = "basket"
WF_FOLDS = "all"
WF_BOOTSTRAP = "16"

JOBS = [
    (
        "phase2_arm_a_baseline",
        [
            sys.executable, "experiments/runners/run_baseline.py",
            "--tickers", TICKERS, "--seeds", SEEDS,
            "--timesteps", TIMESTEPS,
            "--tag", "phase2", "--device", DEVICE,
        ],
    ),
    (
        "phase2_arm_b_aleatoric",
        [
            sys.executable, "experiments/runners/run_probabilistic_agent.py",
            "--tickers", TICKERS, "--seeds", SEEDS,
            "--timesteps", TIMESTEPS, "--bootstrap-paths", BOOTSTRAP,
            "--uncertainty-mode", "aleatoric",
            "--tag", "phase2_aleatoric", "--device", DEVICE,
        ],
    ),
    (
        "phase2_arm_c_epistemic",
        [
            sys.executable, "experiments/runners/run_probabilistic_agent.py",
            "--tickers", TICKERS, "--seeds", SEEDS,
            "--timesteps", TIMESTEPS, "--bootstrap-paths", BOOTSTRAP,
            "--uncertainty-mode", "epistemic",
            "--tag", "phase2_epistemic", "--device", DEVICE,
        ],
    ),
    (
        "phase2_walk_forward",
        [
            sys.executable, "experiments/runners/run_walk_forward.py",
            "--tickers", WF_TICKERS, "--folds", WF_FOLDS, "--seeds", SEEDS,
            "--timesteps", TIMESTEPS, "--bootstrap-paths", WF_BOOTSTRAP,
            "--agents", "baseline,probabilistic",
            "--tag", "phase2_wf", "--device", DEVICE,
        ],
    ),
]


def run_job(name: str, cmd: list[str]) -> int:
    log_path = LOG_DIR / f"{name}_{STAMP}.log"
    header = (
        f"\n=== {name} ===\n"
        f"Started: {datetime.now().isoformat(timespec='seconds')}\n"
        f"Command: {' '.join(cmd)}\n"
        f"Log:     {log_path}\n"
        f"{'=' * 60}\n"
    )
    print(header)
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(header)
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        proc.wait()
        footer = (
            f"\n{'=' * 60}\n"
            f"Finished: {datetime.now().isoformat(timespec='seconds')}\n"
            f"Exit code: {proc.returncode}\n"
        )
        log.write(footer)
        print(footer)
    return proc.returncode


def main() -> None:
    print(f"Phase-2 wrapper starting at {STAMP}")
    print(f"Repository: {REPO_ROOT}")
    print(f"Log directory: {LOG_DIR}")
    print(f"Total jobs: {len(JOBS)}\n")

    summary = []
    for name, cmd in JOBS:
        rc = run_job(name, cmd)
        summary.append((name, rc))

    print("\n" + "=" * 60)
    print("Phase-2 wrapper finished. Summary:")
    for name, rc in summary:
        status = "OK" if rc == 0 else f"EXIT {rc}"
        print(f"  {name:<28} {status}")
    print("=" * 60)
    print(f"\nLogs:    {LOG_DIR}")
    print(f"Results: {REPO_ROOT}/experiments/results/")
    print("\nTo upload to GitHub:")
    print("  export GH_TOKEN=ghp_yourPersonalAccessToken")
    print("  python scripts/sync_results.py")


if __name__ == "__main__":
    main()
