# Lab-machine workflow

You are running on a machine that does not have Cursor (school computer,
shared workstation, etc). This directory has two scripts that let you run
the full Phase-2 grid and ship the results back to the repo without ever
having to copy and paste error text by hand.

## One-time setup

1. Clone the repo (do not download the zip — you need the `.git/`
   directory so you can push results back):

   ```bash
   git clone https://github.com/TheFinix13/dissertation-project.git
   cd dissertation-project
   git checkout epistemic-uncertainty
   pip install -r requirements.txt
   ```

2. Get a GitHub Personal Access Token so the lab machine can push
   results back. https://github.com/settings/tokens, type **classic**,
   scope **repo**, and copy the token (looks like `ghp_xxx...`).

   Save it as an environment variable so the sync script can use it:

   ```bash
   export GH_TOKEN=ghp_yourTokenHere          # Linux / macOS
   $env:GH_TOKEN = "ghp_yourTokenHere"        # Windows PowerShell
   ```

## Running the experiments

```bash
python scripts/run_phase2.py
```

This runs the four Phase-2 jobs in sequence:

1. Arm A — baseline PPO
2. Arm B — probabilistic PPO with aleatoric uncertainty
3. Arm C — probabilistic PPO with epistemic uncertainty (MC Dropout)
4. Walk-forward evaluation

Two things make this safe to leave running for hours or days:

- Each (ticker, seed) cell is checkpointed to disk the moment it
  finishes. If the machine reboots or you kill the script, the next
  run picks up exactly where it left off.
- Output is mirrored to `experiments/results/logs/<job>_<stamp>.log`.
  No more copy-pasting screenshots into email when something breaks —
  the log file has the full output and travels with the results when
  you push.

If you want to interrupt and resume later, just `Ctrl+C`. Re-run the
script and it'll skip everything already done.

## Pushing results back

When the run finishes (or any time you want to checkpoint progress):

```bash
python scripts/sync_results.py
```

This commits and pushes:

- the per-cell JSON checkpoints (so any future session can resume)
- the consolidated headline files (the deliverable)
- the run logs (for diagnosing failures)

Files appear at
[`epistemic-uncertainty/experiments/results/`](https://github.com/TheFinix13/dissertation-project/tree/epistemic-uncertainty/experiments/results)
on GitHub, and Cursor on your Mac can read them after a `git pull`.

## Diagnosing a failed cell

If a single (ticker, seed) cell crashes during training, the runner
writes a `.failed.json` marker next to where the result would have
been, prints a one-line summary, and moves on. The full traceback
sits in the run log under `experiments/results/logs/`.

To inspect:

```bash
ls experiments/results/per_cell/*.failed.json
cat experiments/results/per_cell/probabilistic__NVDA__seed7__phase2_aleatoric.failed.json
```

To force a retry of a failed cell after a code fix, delete the marker
and re-run. To force-retry every cell (cached or failed), pass
`--no-skip` to the underlying runner.
