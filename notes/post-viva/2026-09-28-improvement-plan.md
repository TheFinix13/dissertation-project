# Improvement plan after the viva and final feedback

Written 28 September 2026. The dissertation has been examined and marked,
so this plan covers Fiyin's own continuation of the study. It answers
three sources of feedback: the viva on 14 September, the examiners'
emails afterwards, and the final report and interview comments on the
project portal. The submitted PDF stays as the record of what was
submitted; all changes land on branches or in new files.

## What the feedback asks for

The report feedback rated the methodology as outstanding and the
literature review as fully comprehensive. It asked for three writing
fixes: one spelling standard, no one-sentence or over-split paragraphs,
and less repetition. It also named four ways to strengthen the evidence.
These are a fair tuning of all three algorithms, a more realistic market
simulation, statistical tests over the seed variation, and tests on more
than one asset.

The viva and the emails asked a deeper question. Both examiners argued
that the inputs describe the surface of the price rather than its cause.
Dr Nguyen framed this as a hidden Markov model, where an unobserved state
drives the price and the observed data is noise around it. Prof
Nikitopoulos named the cause directly as actual and expected revenue. The
interview comment asks for domain knowledge before choosing a tool.

## How the work is split

Fiyin chose two tracks in parallel, one per agent, with the literature
review running alongside. Each agent owns its track's files, and every
finished item is logged in the private state file
(`notes/viva/ai_context_post_viva.md`, Section 8) with the agent's name.

| Track | Owner | Where the work lives |
|---|---|---|
| A. Report revision | Claude Code | git worktree `../dissertation-revision`, branch `revision/post-viva` |
| B. Strengthen the evidence | Cursor agent | `experiments/final_v2/` and `notes/post-viva/` on `main` |
| C. Literature for the redo | Cursor agent, then shared | `notes/post-viva/lit-review-right-data.md` |

The worktree keeps the two agents out of each other's files. Track A
edits LaTeX only in the worktree, and Track B never touches
`latex/dissertation/`.

## Track A: report revision

This track fixes what the examiners flagged in the writing. It changes
no numbers, so it needs no new experiments. The steps run in this order.

1. Standardise on British spelling. Seven American forms were found on
   28 September (utilize, utilization, penalizing, organized,
   optimization, minimize, finalized). Words inside citation titles and
   package names such as `\normalsize` stay as they are.
2. Merge one-sentence paragraphs under the updated voice rule. The rough
   count was 27 of 47 prose paragraphs in Chapter 1 and 66 of 177 in
   Chapter 3. Related sentences are joined, and no sentence is made
   longer to do it.
3. Remove repeated content. The first candidates are results told in
   both Chapter 5 and Chapter 6, and the two-study design explained in
   several chapters.
4. Regroup the methods as Dr Nguyen asked. Deep Q-learning sits on one
   side as the value-based method, and REINFORCE and PPO sit on the
   other as policy-gradient methods that share one mechanism. Each gets
   a short reason for being chosen.
5. Compile twice with bibtex after every chapter and check for errors.

## Track B: strengthen the evidence

This track answers the four points in the report feedback. The first
two steps only re-analyse saved results, so they need no gate documents.
The later steps train new models and follow the five-gate protocol.

1. **Statistics on the saved results.** Bootstrap confidence intervals
   for every mean in the main results table, a paired test of the agent
   against buy-and-hold on the 180 real months, and a Sharpe-ratio
   comparison with a proper test. The per-month and per-seed data is
   already in `results/sim_results.json`.
2. **Risk-adjusted results on other assets.** AAPL and QQQ were run
   after the viva (`results/other_assets.json`) but only raw means were
   read. Sharpe ratios, drawdowns and the 2008 crisis band are computed
   the same way as for SPY.
3. **Fair tuning of all three algorithms.** Gate documents first. Each
   algorithm gets its own validation search, and PPO gets the entropy
   sweep already specified in the literature review (entropy coefficient
   0, 1e-4, 1e-3 and 1e-2, with entropy and KL logged).
4. **More seeds.** Twenty seeds per algorithm instead of six, so the
   intervals in step 1 become meaningful.
5. **A more realistic simulator.** Gate documents first. Regime changes
   inside a month, volatility that clusters over time, and continuous
   episodes where each month starts from the last one's holdings.

## Track C: literature for the domain-first redo

This is the long-term part of the work and follows the September
semester. It starts with reading, not code. The anchor papers in the
literature review get full-text reads, and the review gains a section on
regime-switching and hidden-state models. The first experiment after the
review is a plain linear regression of forward returns on the drivers,
as Fiyin proposed to Prof Nikitopoulos. Reinforcement learning is only
used if that check finds signal and the decision becomes complex enough
to need it.

Data access was checked on 28 September. Surrey provides WRDS through
free online registration with a Surrey email, and Datastream through the
institutional login. Datastream normally carries index-level I/B/E/S
forward earnings estimates. Bloomberg terminals are in room 12MS01 and
need a tutor's approval. The free sources are FRED for rates and credit
spreads, CBOE or FRED for VIX, and Robert Shiller's dataset for monthly
reported S&P 500 earnings.

## Order of work

Track B steps 1 and 2 start first because they cost minutes and change
how every later result is read. Track A starts once Claude Code is
signed in. Track B steps 3 to 5 start only after their gate documents
are written.

Steps 1 and 2 were finished on 28 September
(`notes/post-viva/2026-09-28-statistics.md`). The simulated-test result
is significant, but the real-market advantage over buy-and-hold is not.
That makes the gate documents for steps 3 to 5 more important, because
more seeds alone cannot fix a gap that comes from having only 180 real
months.

The gate documents for steps 3 to 5 were written on 29 September
(`2026-09-29-gates-2-4-audit.md` and `2026-09-29-experiment-designs.md`)
and reviewed by a GPT model. The audit found that the dissertation's PPO
did not learn because its dollar rewards, combined with SB3's joint
gradient clipping, pushed its policy gradient below Adam's epsilon. With
rewards scaled to a fraction of capital, PPO learned a state-dependent
policy on all three validation seeds. Steps 3 and 4 are merged into one
pre-registered experiment, fair tuning followed by 20 seeds. Step 5
follows it. Both are ready to run.
