# Critique of `Main_Dissertation_Draft.docx`

This is an honest read of the current dissertation draft against the six writing rules distilled from the EEEM072 research-proposal feedback (Group 30 = 69/100, Group 33 = 80/100). It is meant to be read alongside `02_outline.md`, which proposes the clean rewrite.

## The six rules I'm grading against

1. **Define-before-use.** Every technical term is defined in the same sentence it's introduced.
2. **One idea per sentence.** If a sentence has more than two commas it gets split.
3. **Work-package structure.** Plans use WP1/WP2/WP3/WP4 with tickable tasks.
4. **Conservative claims.** Claims are bounded; every number has a citation or a code pointer.
5. **Bounded scope.** Every chapter states what it does *not* claim.
6. **No bracketed labels.** Headings are plain.

## TL;DR — what's good, what's broken

**What's good (keep these — don't rewrite them):**
- The reframing from "capital preservation as objective" to "drawdown as constraint, risk-adjusted return as objective" (Section 1.2 "A note on framing"). This is the dissertation's intellectual move. It works.
- Section 3.1.6 (Soft enforcement of the preservation constraint) — explains *why* the Lagrangian approach was rejected. This is exactly how technical decisions should be written up.
- Objectives O1 through O5 (Section 1.3) — well-bounded, modular, in answering order.
- Future work Group 1 / Group 2 / Group 3 (Chapter 7.2) — explicit dates, tickable boxes, and go/no-go criteria. This is the closest thing to Talha's WP structure already in the document.
- Equation 3.9 is named as *the* mathematical contribution. Conservative, defensible.

**What's broken (priority fixes):**
- The abstract is a single 460-word block. It mentions Sharpe ratio = 0.85 before defining Sharpe ratio. It uses "DeepAR-style", "PPO", "Gaussian over the next-step log return", "predictive standard deviation normalised to a unit-interval uncertainty score" without any in-line definitions.
- Chapter 1.1 (Background) drops "CTA hedge fund", "MDP", "DeepAR-style probabilistic LSTM", "PPO" without defining them on first appearance.
- Chapter 1.1 contains six factual claims with no citation: CalPERS drawdown limits, Yale/Harvard endowment drawdown reporting, Bridgewater's $150B AUM at peak, prospect-theory's loss-aversion ratio, "billions of pounds under management", and "the literature on how a sequential decision policy might fill that gap is thin".
- Chapter 3 opens with notation immediately ("MDP with state space S, action space A, transition kernel P, reward function R and discount factor gamma in (0,1]"). A non-technical reader has no on-ramp.
- "Note on framing" reads as meta-commentary about the previous draft. An examiner doesn't need to see the author's revision history — the current framing should stand on its own.

## Issue-by-issue against the six rules

### Rule 1 — Define-before-use

Specific failures, all from the first 50 paragraphs:

| First appearance | What's missing |
|---|---|
| "CTA hedge fund" (Abstract, line 1) | No definition. The parenthetical "(CTA hedge fund)" later in Chapter 1.1 still doesn't say what a CTA does. |
| "PPO" (Abstract) | Acronym only. Never expanded inline. |
| "DeepAR-style probabilistic LSTM" (Abstract) | No definition of DeepAR. No definition of what "style" means here. |
| "Gaussian over the next-step log return" (Abstract) | Assumes reader knows Gaussian, log return. |
| "Markov decision process (MDP)" (Section 3.1) | Acronym expanded but not defined in plain English. |
| "Lagrangian penalty" (Section 3.1.6) | Undefined technical jargon. |
| "credit assignment" (Section 3.1.6) | Undefined RL term. |
| "transition kernel P" (Section 3.1) | Math notation, no plain-English wrapper. |
| "policy-gradient methods" (Section 2.3 heading) | Not on-ramped from the previous paragraph. |

**Fix:** every first appearance gets *either* a one-clause definition in the same sentence ("a Markov decision process, the standard mathematical frame for sequential decision problems under uncertainty"), *or* a footnote-style aside, *or* a forward-reference to a chapter section ("defined in Section 2.2 below"). The walkthrough notebook already does this well — the dissertation should match its voice.

### Rule 2 — One idea per sentence

The worst offenders:

> "A risk-constrained investor — a pension fund, a CTA hedge fund, a family office, or a behaviourally loss-averse retail account — has to satisfy two requirements at once: not lose more than a stated percentage from peak, and earn a return above cash." *(Abstract, sentence 1)*

This is **three ideas**: who has the constraint, the constraint shape, the return requirement. Each deserves its own sentence.

> "The combined system is evaluated against three named comparators on a single fully reproducible protocol: passive buy-and-hold, a rule-based trailing stop-loss policy of the kind a discretionary investor would actually use, and a baseline PPO that sees no uncertainty signal." *(Abstract)*

This is fine as one sentence because it's a list — but lists should be set as bullet lists, not jammed into prose. Talha's proposal did exactly this on his evaluation criteria.

**Fix:** abstract becomes 6–8 short paragraphs, not one block. Bullet lists wherever a sentence enumerates more than two items.

### Rule 3 — Work-package structure

The dissertation already does this in the *future work* (Group 1, Group 2, Group 3 with checkboxes) but does not do it in the *methodology* or *implementation* chapters. Chapter 3 is organised by mathematical object (decision variables → state → reward → objective), not by work package. That works for the methodology but not for someone trying to assess what the project actually did, in what order, with what deliverables.

**Fix:** add a single section at the top of Chapter 3 or 4 that lays out the project as four work packages:

- **WP1 — Data and forecaster.** What we fetched, what we trained, what came out.
- **WP2 — Environment and policy.** What the environment looks like, how the agent interacts with it.
- **WP3 — Experimental grid.** Which agents we trained, on what data, with what seeds.
- **WP4 — Evaluation and reporting.** Which metrics, which baselines, which artefacts.

Each WP gets 2–4 tasks. Each task is something the supervisor can tick off.

### Rule 4 — Conservative claims (and citations)

Unsourced claims in Chapter 1.1 alone:

1. *"CalPERS … documents an explicit drawdown limit in its governance papers"* — needs a citation to the governance document.
2. *"University endowments such as Yale's and Harvard's report drawdown alongside return as their headline performance measure"* — citation needed.
3. *"Bridgewater Associates' All Weather fund, which managed over 150 billion US dollars at peak"* — citation needed.
4. *"the mandates that create it are billions of pounds under management"* — vague; either be specific with a citation or remove.
5. *"the literature on how a sequential decision policy might fill that gap is thin"* — this is the kind of claim that demands a literature-survey footnote.
6. *"Kahneman and Tversky's (1979) prospect-theory result establishes that retail investors feel losses about twice as painfully as equivalent gains"* — Kahneman & Tversky 1979 is cited but the "twice as painful" number is more typically attributed to Tversky & Kahneman 1992 (cumulative prospect theory). Pin the right citation to the right number.

**Fix:** any factual claim in Chapter 1 that doesn't come from this dissertation's own experiments must have a citation. If the claim can't be sourced cleanly, soften the language or remove it.

### Rule 5 — Bounded scope

The dissertation does this *moderately* well. Examples that work:

- *"The contribution claimed is deliberately modest — a careful empirical study of a specific combination on a reproducible protocol with the rule-based comparator that the DRL-finance literature mostly leaves out, not a new algorithm."* (Abstract)
- *"the dissertation's strongest piece of evidence"* (Contributions, regarding the 70-stock generalisation study)

Examples that don't:

- The abstract claims *"the highest Sharpe (0.85), the highest terminal value, and the smallest path drawdown of any policy that participates in market upside enough to beat cash"* on the test window. This is a *Phase-1, 3-seed, 10,000-step* number. It should be flagged as such *in the abstract*, not just buried in Chapter 5. The reader who scans the abstract will read it as the headline result of the whole dissertation, not as a preliminary finding.
- Chapter 1 makes the leap from a single-asset case study (SPY) to a 70-stock universe without a sentence saying "the SPY result motivates the broader study; the 70-stock result is the headline".

**Fix:** every chapter introduction has one sentence saying what it doesn't claim. The abstract has *one* sentence saying "all numbers in this abstract are at Phase-1 budget; Phase-2 at extended budget is scheduled and described in Chapter 7".

### Rule 6 — No bracketed labels

The dissertation **passes** this rule. There are no `[The Goal]` or `[The Innovation]` labels in section headings — the writing has matured past the proposal style. Good.

The one remaining offender is the *"A note on framing"* paragraph in Section 1.2. It's not a bracketed heading, but it functions like one. The examiner doesn't need to see the revision archaeology.

**Fix:** delete *"A note on framing"* and the explanation of the previous wrong framing. Start Section 1.2 directly with the current, correct problem statement.

## Top-priority rewrites — ranked by impact-per-effort

The list below is roughly in order of impact for the time invested. The first three would move the dissertation from "competent" to "polished" in maybe 4 hours of focused work.

1. **Rewrite the abstract.** One 460-word block becomes seven short paragraphs. Every technical term defined on first use. One sentence flagging that the headline numbers are Phase-1. Aim: a competent third-year undergraduate in finance or computer science should understand what the dissertation does after reading the abstract.

2. **Rewrite Section 1.1 (Background).** Remove the "Note on framing". Source or remove the six unsourced claims. Define CTA, MDP, PPO, DeepAR on first appearance. Add a one-paragraph "Glossary of terms used in this dissertation" right after the contents page (or as a fold-out appendix), with PPO, LSTM, MDP, Sharpe, drawdown, preservation, aleatoric, epistemic — all defined in one sentence each. The walkthrough notebook already has this glossary; lift it.

3. **Add a work-package summary table** to Chapter 4 (Implementation). Four rows — WP1 data + forecaster, WP2 environment + policy, WP3 experimental grid, WP4 evaluation. Each row names the deliverable artefact (a file, a notebook, a results CSV).

4. **Rewrite Chapter 3.1 opening paragraph** to give a non-technical reader an on-ramp. Three sentences in plain English before the formal MDP notation: *"The trading task is set out as a sequential decision problem. Every day the agent sees a snapshot of the market and its own portfolio, chooses how much to buy or sell, and is rewarded for growing the portfolio. Mathematically this is a Markov decision process, the standard frame for problems where today's action affects tomorrow's state."*

5. **Audit every figure caption.** A figure caption should be readable on its own — the reader who flips to a chart should know what they're looking at without going back to the prose. Several captions in the current draft are placeholder-y (`Figure 3.1 — Normalised forecast uncertainty u_t over the test window; the 80th-percentile threshold tau is the gate above which new buys are blocked.`) — that one's actually OK; check the others.

6. **Bound the scope of the headline claim in the abstract.** One sentence: *"The headline numbers in this abstract come from a Phase-1 budget (3 seeds, 10,000 PPO training steps). The Phase-2 budget (10 seeds, 50,000 steps, 70 stocks, walk-forward) is scheduled for the Colab GPU runtime and is described in Chapter 7."*

7. **Replace every Markdown-y emphasis** like *italics for emphasis* in the body text with proper italics applied only to maths and titles. The Word doc has some loose styling that the LaTeX rewrite is a chance to fix.

8. **Delete or soften the "deliberately modest" framing.** The dissertation says this twice — once in the abstract, once in the contributions list. Once is good academic humility; twice reads as anxious.

## What the abstract should look like (illustrative)

The example below is *not* the final wording — it's an illustration of the target voice. Compare it to the current abstract (one block, 460 words) and Talha's proposal opening (short paragraphs, definitions inline).

```
A pension fund manager has two obligations at once. The first is to not
lose more than a stated percentage of the fund's value from its recent
peak — a drawdown limit. The second is to earn a return above cash.
Standard tools force a trade-off. Buy-and-hold violates the drawdown
limit routinely. Trailing stop-losses fire after the drawdown has
already happened. Mean-variance optimisation, the textbook approach,
assumes a single period and is blind to the running peak entirely.

This dissertation studies whether a small machine-learning agent can do
better. The agent in question is a Proximal Policy Optimization (PPO)
policy — PPO is the standard learning algorithm for sequential decision
problems where the action affects the next state. The agent is paired
with a separate forecaster, a probabilistic Long Short-Term Memory
network (LSTM, a neural network that learns from sequences). The
forecaster emits not just a prediction of tomorrow's return, but a
confidence interval around that prediction. The width of that interval
is fed to the PPO agent as a separate signal called the uncertainty
score.

The uncertainty score is used in two places. As a coordinate of the
agent's state, so the agent knows when its forecaster is uncertain. And
as a hard guard, so that when the forecaster is uncertain above a
threshold, new buys are blocked.

The agent is measured against three named comparators on the same data
and the same time window: passive buy-and-hold, a trailing-stop-loss
rule of the kind a discretionary investor would actually use, and a
baseline PPO that sees no uncertainty signal. The protocol fixes the
train/validation/test splits, the random seeds, and the metric set, so
that all four agents are compared like-for-like.

The headline numbers in this abstract are from a Phase-1 budget (three
random starting points, ten thousand training rounds per agent). On the
held-out 2022–2025 window the uncertainty-aware agent earns the highest
risk-adjusted return (Sharpe ratio 0.85) and the smallest drawdown of
the four agents that participate in the market enough to beat cash.
The trailing-stop rule's drawdown comes out worse than buy-and-hold's
on this window — the stop fires late and the re-entry rule sits in
cash through the recovery.

The contribution is a careful empirical study, not a new algorithm.
What's new is the controlled four-agent comparison on a single
reproducible protocol, with the trailing-stop comparator that most of
the deep-RL-for-finance literature leaves out. The Phase-2 budget (ten
seeds, fifty thousand training rounds, seventy stocks, four-fold
walk-forward) is scheduled for the Colab GPU runtime and is described
in Chapter 7.
```

Six paragraphs. Every technical term defined on first use. The Phase-1 disclaimer is in the headline-numbers paragraph itself, not buried in Chapter 7. The contribution claim is bounded. No "deliberately modest" — the modesty is shown, not stated.

## Section-by-section verdict

| Section | Status | What to do |
|---|---|---|
| Abstract | Rewrite | One block → seven paragraphs; define-before-use; flag Phase-1 |
| 1.1 Background | Rewrite | Remove "Note on framing"; source the six unsourced claims; define CTA/PPO/MDP/DeepAR |
| 1.2 Problem statement | Trim | Delete the meta-paragraph; start directly with the current problem statement |
| 1.3 Aims and objectives | Keep | O1–O5 are clean; minor wording polish only |
| 1.4 Contributions | Trim | Delete one of the two "deliberately modest" framings |
| 1.5 Structure | Keep | Already terse and useful |
| Chapter 2 (Background) | Light edit | Add a plain-English sentence on the front of 2.2 (RL fundamentals) and 2.5 (probabilistic forecasting) |
| 3.1 Problem formulation | Add on-ramp | Three plain-English sentences before the MDP notation |
| 3.1.6 Soft enforcement | Keep verbatim | Best-written section in the draft |
| 3.2–3.4 | Light edit | First-use definitions for "yfinance", "DeepAR-style", "min-max normalised" |
| 3.5 Trading environment | Light edit | Keep the equation; add a one-sentence plain-English version above it |
| 3.7 Evaluation protocol | Keep | Already clean |
| Chapter 4 (Implementation) | Add WP table | New work-package summary table at the top |
| 4.4 Day-by-day walkthrough | Keep | Already strong — lifted from the walkthrough notebook |
| Chapter 5 (Results) | Light edit | Each table caption needs a one-sentence "what to look for" line |
| Chapter 6 (Discussion) | Keep | Already does the "where it wins, where it loses" job well |
| Chapter 7 (Conclusion) | Keep | Group 1/2/3 structure is already the WP voice |

The dissertation is in much better shape than the EEEM072 proposal was. The proposal's 69/100 came from depth-and-clarity failings on 4–5 axes; the dissertation has fixed most of them already. The remaining failings are concentrated in the abstract and Chapter 1, which is the highest-impact place to focus.
