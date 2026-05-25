"""Push Phase-2 results, per-cell checkpoints and logs to GitHub.

Usage:

    export GH_TOKEN=ghp_yourPersonalAccessToken
    python scripts/sync_results.py

What it does
- Stages experiments/results/ (consolidated files, per_cell/, logs/)
- Commits with a timestamped message
- Pushes to the epistemic-uncertainty branch using the token

Why this exists
- Lab machines without an interactive editor (no Cursor, no SSH key
  setup) still need a one-command path to get results into the repo.
- Logs travel with the results, so any failure is reproducible from
  the branch alone without the user having to copy-paste error text.

Get a Personal Access Token at:
    https://github.com/settings/tokens (use 'classic', scope: repo)
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "experiments" / "results"
BRANCH = "epistemic-uncertainty"
GH_REPO = "TheFinix13/dissertation-project"
COMMITTER_NAME = "Fiyin Akano"
COMMITTER_EMAIL = "43079183+TheFinix13@users.noreply.github.com"


def run(cmd: list[str], **kwargs) -> str:
    return subprocess.check_output(cmd, cwd=REPO_ROOT, text=True, **kwargs).strip()


def main() -> int:
    token = os.environ.get("GH_TOKEN")
    if not token:
        print("ERROR: GH_TOKEN environment variable is not set.")
        print()
        print("Get a token at https://github.com/settings/tokens")
        print("  Token type:  classic")
        print("  Scope:       repo (full)")
        print()
        print("Then run:")
        print("  export GH_TOKEN=ghp_yourTokenHere   # macOS/Linux")
        print("  $env:GH_TOKEN = 'ghp_yourTokenHere' # Windows PowerShell")
        print("  python scripts/sync_results.py")
        return 1

    if not RESULTS_DIR.exists():
        print(f"ERROR: {RESULTS_DIR} does not exist. Did you run anything yet?")
        return 1

    # Make sure git is sane in this checkout
    try:
        run(["git", "rev-parse", "--git-dir"])
    except subprocess.CalledProcessError:
        print(f"ERROR: {REPO_ROOT} is not a git checkout.")
        print("If you downloaded the source as a zip from GitHub, you")
        print("need to clone it instead so it has git history attached:")
        print(f"  git clone https://github.com/{GH_REPO}.git")
        print(f"  cd {GH_REPO.split('/')[1]}")
        print(f"  git checkout {BRANCH}")
        return 1

    subprocess.check_call(["git", "config", "user.email", COMMITTER_EMAIL], cwd=REPO_ROOT)
    subprocess.check_call(["git", "config", "user.name", COMMITTER_NAME], cwd=REPO_ROOT)
    subprocess.check_call(["git", "add", str(RESULTS_DIR)], cwd=REPO_ROOT)

    status = run(["git", "status", "--porcelain"])
    if not status:
        print("Nothing new to push. Branch already up to date.")
        return 0

    # Quick summary of what's about to be committed
    n_per_cell = len(list(RESULTS_DIR.glob("per_cell/*.json")))
    n_failed = len(list(RESULTS_DIR.glob("per_cell/*.failed.json")))
    n_logs = len(list(RESULTS_DIR.glob("logs/*.log")))
    n_consol = len(list(RESULTS_DIR.glob("*phase2*.json"))) + len(
        list(RESULTS_DIR.glob("*phase2*.csv"))
    )

    stamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    msg = (
        f"Add Phase-2 lab-machine results {stamp}\n\n"
        f"- per-cell checkpoints: {n_per_cell - n_failed} ok, {n_failed} failed\n"
        f"- consolidated files:   {n_consol}\n"
        f"- run logs:             {n_logs}"
    )
    subprocess.check_call(["git", "commit", "-m", msg], cwd=REPO_ROOT)

    push_url = f"https://x-access-token:{token}@github.com/{GH_REPO}.git"
    subprocess.check_call(
        ["git", "push", push_url, f"HEAD:{BRANCH}"],
        cwd=REPO_ROOT,
    )

    print()
    print("Pushed:")
    print(f"  per-cell checkpoints: {n_per_cell - n_failed} ok, {n_failed} failed")
    print(f"  consolidated files:   {n_consol}")
    print(f"  run logs:             {n_logs}")
    print()
    print(f"Visible at https://github.com/{GH_REPO}/tree/{BRANCH}/experiments/results")
    return 0


if __name__ == "__main__":
    sys.exit(main())
