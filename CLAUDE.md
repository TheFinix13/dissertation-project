# CLAUDE.md · shared entry point for Claude Code

Two agents work in this repository: Cursor's agent and Claude Code.
Both read the same state files and follow the same rules. Neither agent
keeps private memory of the project; the files below are the memory.

## Read first, in this order

1. @notes/viva/ai_context_post_viva.md — current project state. Start at
   the last numbered section ("CURRENT STATE" and anything after it).
   This file is gitignored and must never be committed.
2. @ai_context.md — the submitted-dissertation state (frozen 9 Sep 2026).
3. @docs/WORKSPACE_CATALOG.md — map of every live file by study.
4. @notes/post-viva/2026-09-28-improvement-plan.md — the current plan and
   which agent owns which track. **Claude Code owns Track A (report
   revision)** and edits LaTeX only in the worktree at
   `/Users/the1finix/Documents/GitHub/dissertation-revision` (branch
   `revision/post-viva`). The Cursor agent owns Track B on `main`.
5. @notes/post-viva/improvement-backlog.md — older ranked work items.

## Rules that apply to every session

- @.cursor/rules/brain-box-writing.mdc — report-writing standards.
- @~/.cursor/rules/fiyin-writing-style.mdc — Fiyin's voice for any prose.
- @~/.cursor/rules/literature-first-research.mdc — the five-gate research
  protocol. No experiment runs before its gates are documented.

## Working agreement between the two agents

- One agent owns a file at a time. Before editing a file the other agent
  touched this session, re-read it from disk.
- Record every finished piece of work as a dated bullet in the current
  section of `notes/viva/ai_context_post_viva.md`, including which agent
  did it. That file is the handoff.
- The submitted dissertation in `latex/dissertation/` stays frozen.
  Revisions happen on a separate branch opened on Fiyin's instruction.
- Never commit `notes/viva/ai_context_post_viva.md`, the viva transcript,
  or supervision emails. Check `.gitignore` before any `git add`.
- Python work uses `./venv` (Python 3.14).
