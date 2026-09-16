# Post-viva improvement backlog

The viva took place on 14 September 2026. This file ranks the technical
work that applies the examiners' feedback. It contains work items only.

Standing rule: the submitted dissertation under `latex/dissertation/` is
frozen. No LaTeX edits until marks are released or a revision branch is
explicitly opened. New work lands in `experiments/final_v2/` and in this
folder.

Standing rule (added 16 Sep 2026): all further work follows the
literature-first protocol. Literature first, then verify the data, then
verify the implementation, then verify that data and implementation are
compatible, and only then experiment fully. No stage proceeds on
assumption; each stage leaves a dated document. The protocol lives in
the brain-box methodology folder
(`school/methodology/literature-first-research-protocol.md`).

## The primary track: literature-first redo (16 Sep 2026 onward)

The improvement study now runs in gate order:

1. **Literature review — "what makes the right data"** for a
   reinforcement-learning trading agent. Working document:
   `notes/post-viva/lit-review-right-data.md`. This answers
   Nikitopoulos's question with named characteristics and citations.
2. **Data and state design from the review.** What enters the state and
   the dataset (fundamentals, rates, VIX, regime markers); volume
   checked against model capacity; balance and signal content measured
   before any training.
3. **Implementation and compatibility checks.** Entropy in PPO handled
   explicitly (the main study ran SB3's default `ent_coef=0`, so
   exploration collapsed); every other library default audited; network
   sized to the measured data volume.
4. **Full experiment.** Only after 1–3 are documented.

Items 2–4 of the finishing backlog below stay open and can run in
parallel; item 5 is absorbed into this track.

## Where the feedback came from

Two lines of critique, one from each examiner.

Prof Nikitopoulos argued from value: prices move on actual and potential
revenue, the state features are all derived from price and volume, so
the inputs omit the drivers. He also asked what a learned agent adds
over a simple stop-loss rule, and noted the closing slide never stated
the contribution in one sentence.

Dr Nguyen argued from mechanism: the work tried existing models rather
than building one, the conclusions leaned harder than six seeds and a
three-condition simulator can support, and the viva answer to "why did
Deep Q beat PPO" was a guess. The feature ablation run on the evening of
the viva (`experiments/final_v2/results/feature_ablation.json`) settled
that question with numbers.

## The backlog, in order

### 1. Tuned stop-loss baseline — DONE 16 Sep 2026

A trailing/fixed stop-loss rule evaluated on the same episode format the
agents trade, tuned on the 300 simulated validation episodes only, then
reported once on the held-out sets.

- Code: `experiments/final_v2/run_stop_loss.py`
- Output: `experiments/final_v2/results/stop_loss.json`
- Selected on validation: trailing 5% stop (best of 10 configurations).

**Finding: the simple rule does not do what the agent does.** Headline
numbers, held out:

| Strategy | Sim test $/ep | Transfer $/mo | Transfer Sharpe | Crisis band Sep 08–Mar 09 | Oct 2008 |
|---|---:|---:|---:|---:|---:|
| Trailing 5% stop | +59.46 | +45.26 | 0.118 | −$3,105 | −$986 |
| Buy-and-hold | −60.29 | +80.24 | 0.198 | −$3,046 | −$1,665 |
| DQN (6-seed mean) | +126.94 | +60.28 | ~0.224 | −$180 | ≈ $0 |

The stop rule cuts the single worst month (Oct 2008: −$986 against
buy-and-hold's −$1,665) but it re-enters fully at the start of every
month, so across the seven crisis months it loses $3,105 — slightly more
than buy-and-hold itself. The learned agent avoided the band almost
entirely (−$180) because it read the volatility features and declined to
enter. A stop-loss can only exit after losses begin; it cannot decline
to enter. That is the answer to the viva question, with a number.

Honest caveats to carry into any write-up: the stop rule beats
buy-and-hold on the balanced simulated test (+$59.46 against −$60.29),
so the rule is not a strawman; and on raw transfer return buy-and-hold
still beats both the rule and the agent. The agent's edge is
risk-adjusted (Sharpe 0.224 against 0.198) and concentrated in the
crisis.

### 2. Bootstrap and significance tests on existing results

Per-seed and per-month numbers already exist in
`results/sim_results.json`. Bootstrap confidence intervals on the
transfer means and a paired test on DQN versus buy-and-hold months cost
minutes of compute. This addresses "conclusions stronger than the
evidence."

### 3. Feature-ablation revision note

The ablation results exist but are not in the submitted PDF. Write the
Chapter-5-style section (protocol, table, interpretation) as a note in
this folder, ready to drop into a revision branch. Key numbers: without
volatility features the agent still avoided October 2008 itself, but
re-entered across September–December 2008 (band loss −$180 → −$712),
down-condition loss grew 64%, and seed spread nearly doubled.

### 4. One-sentence contribution statement

Draft: "returns alone cannot tell you whether a trading agent has
learned anything; this work builds the test that can." Refine it and
keep it ready for any revision closing section, talk, or application.

### 5. Fundamentals and causal drivers — absorbed into the primary track

Superseded on 16 Sep 2026: this item is now the literature-first track
at the top of this file. The earlier design sketch (index earnings and
revisions, short and long rates, VIX, an equity-versus-bond alternative,
slower decision frequency, a linear-model check before any RL) is input
to the data-design gate, but the literature review runs first and
decides it.

## Done so far

- 14 Sep 2026 (evening): feature ablation ran and confirmed the
  volatility-feature claim directionally
  (`results/feature_ablation.json`).
- 16 Sep 2026: workspace catalogued and published
  (`docs/WORKSPACE_CATALOG.md`); this backlog opened.
