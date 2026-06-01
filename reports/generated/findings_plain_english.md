# What this dissertation actually does, in plain English

> **For anyone who is not a reinforcement-learning person and not a finance person.**
> If you only have 5 minutes, read this. The dissertation itself is the academic version of the same content.

---

## The problem in one paragraph

If you put $1,000,000 into the US stock market in January 2022 and held on, by January 2025 you would have had ~$1.52M. Good. But on the way there, your account would have shown $750,000 at one point — a 25% drop from peak. For a normal person that 25% drop is scary. For a pension fund, an endowment, or an institutional asset manager, that 25% drop **breaks contractual constraints** and forces them to sell at the worst possible moment. The financial industry calls this a "drawdown". Most pensions and endowments operate under explicit drawdown rules: don't lose more than X% from the peak, ever. Buy-and-hold violates those rules constantly. Manually setting a "stop loss" (sell if it drops 5%) violates them too — it sells late and buys back even later. **The question this dissertation answers is: can a small AI model do better than either of those two options?**

---

## The solution in one paragraph

We built a two-piece AI agent. **Piece 1** is a small neural network that watches recent price moves and outputs not just a prediction, but a confidence number. When it's not confident, it says so. **Piece 2** is a reinforcement-learning policy that reads that confidence number and decides how much money to put into the market. When confidence is low, the policy automatically reduces position size; when confidence is high, it goes back in. The whole thing is trained to maximise return *subject to* a 95%-of-peak floor — i.e., never let the account drop more than 5% below its all-time high without taking action. This is the exact constraint a real pension fund operates under. **The novelty is not the neural network or the reinforcement learning — those exist in the literature — it is the explicit, formal coupling of the two for drawdown control with confidence-aware position sizing.**

---

## Does it work? Three results, in dollars and percentages.

### Result 1 — An early single-ticker case study on the broad US market (SPY ETF, 2022–2025)

This first result is an **illustrative single-ticker case study** — a useful
warm-up before the real headline (Result 2). It is reported on one ETF (SPY) at
a small training budget, so treat it as an example, not the main evidence.
Starting from $1,000,000:

| Strategy | Final value | Worst drawdown |
|---|---:|---:|
| Buy-and-hold (just hold the ETF) | $1.52M | -25% |
| Manual 5% stop-loss rule | $1.15M | -20% |
| **Probabilistic AI agent (this dissertation)** | **$1.62M** | **-18%** |

On this single ticker the AI made about $100K more than buy-and-hold *and* took
roughly 6 percentage points less drawdown. The manual stop-loss rule trimmed
drawdown a little but gave up about $370K versus buy-and-hold — it sells late
and buys back even later. This is only one stock, though; the dissertation's
actual headline is the full grid below.

### Result 2 — On the full market sample of 70 stocks (the dissertation's headline)

This is the **completed Phase-2 result** and the headline of the dissertation.
Every agent is trained across the `market_sample` universe (70 diversified
stocks — 41 single-name US large-cap stocks across technology, payments and
financial services, healthcare, consumer and industrials, plus 29 ETFs
spanning broad-market, sector, dividend, thematic and commodity exposures),
with **ten random starting points (seeds)** and **50,000 training rounds** per
combination. Each run starts from $1M and is tested on the 2022–2025 window.
"Median" means the typical (middle) stock; "win-rate" means the share of runs
that finish with more money than they started.

| Strategy | Median final value | Median Sharpe | Win-rate (finish above $1M) |
|---|---:|---:|---:|
| Baseline PPO (no confidence signal) | $999,063 | −0.04 | 46% |
| **Probabilistic AI agent — aleatoric** | **$1.61M** | **+0.70** | **89%** |
| **Probabilistic AI agent — epistemic** | **$1.62M** | **+0.68** | **92%** |

The big finding: **the uncertainty-aware AI grows a typical $1M portfolio to
about $1.6M — a 60% median gain — and finishes ahead of where it started about
nine times out of ten (89–92%).** The baseline PPO, which sees no confidence
signal, finishes essentially flat at about $999,000 and only wins 46% of the
time: without a confidence signal it learns to barely trade at all. The two
ways of measuring uncertainty (aleatoric = "how noisy is the market?" and
epistemic = "how unfamiliar is today?") give almost identical results.

**What about drawdown — does the AI lose less?** On the *typical* stock and on
the index basket, yes: the AI's median worst-drop is about **22–24%**, a few
points *better* than just holding the market (**~26%**), and far better than
the trailing stop-loss rules (**26%** and **33%**). But this is an honest, not
a perfect, story:

- The right thing to compare against is **buy-and-hold** (which is fully
  invested and so actually carries market risk), **not** the baseline PPO.
- The baseline PPO's tiny ~1.4% drawdown looks great but is a **side-effect of
  barely investing** — there is almost nothing to lose. It is not risk skill.
- The AI does **not** win on the worst case. On the most volatile individual
  stocks its deepest drop (about **57%**) is worse than buy-and-hold's worst
  (about **35%**). So the claim is "the AI takes less drawdown on the typical
  stock and on the index basket" — **never** "it reduces drawdown on every
  single stock".

---

## Why drawdown control is the right thing to optimise

Three reasons, each one taken from a body of finance literature that is older than RL.

1. **Real money is run under drawdown rules, not Sharpe ratios.** Endowments operate under spending policies that are tied to how far the portfolio is below its high-water mark. Hedge funds charge their performance fee only above the high-water mark. Pension funds have funded-ratio targets that effectively cap drawdown. If the academic objective is "useful in finance", the academic objective should be drawdown control. (Chekhlov, Uryasev & Zabarankin 2005 formalised this with CDaR; Markowitz 1952 wrote the original variance-based version; both are in the dissertation Section 2.1.)

2. **Buy-and-hold's 25% drawdown actually happens, and it actually breaks people.** Behavioural finance has measured this for 40 years: real investors sell at the bottom, not because they are stupid, but because their constraints (margin calls, redemption requests, pension funding ratios) force them to. A 25% drawdown is the *minimum* risk on the broad US market in 2022. A drawdown-aware overlay is not a fancy optimisation — it is the difference between staying invested and being forced out.

3. **Manual stop-loss rules don't work, but people use them anyway.** The reason people use them is that there's no good alternative. The dissertation's contribution is the alternative: a confidence-aware overlay that does what the stop-loss tries to do, but smarter — it reduces position size *before* the drawdown rather than after, and it scales the reduction to confidence rather than firing all-or-nothing.

---

## What is "AI" in this project? Specifically.

There are two AI components, both small.

- **The forecaster** is an LSTM neural network with about 9,500 parameters. It reads the last 20 daily returns of a stock and outputs a mean prediction *and* a variance. The variance is the model's own admission of uncertainty. This is "DeepAR-style probabilistic forecasting" (Salinas et al. 2020).
- **The trader** is a Proximal Policy Optimisation (PPO) reinforcement-learning agent. It reads (today's price, recent price history, the forecaster's confidence number) and outputs an action: buy, hold, or sell, scaled by a fraction. PPO is a 2017 algorithm from OpenAI; it is the standard choice for continuous-action control problems.

Total parameter count is well under 50,000. A real production trading system would have 100x more. The model is small *on purpose* — the dissertation is about the methodology, not the size of the network.

---

## What is *not* claimed (honesty)

- **This is not a complete trading system.** It is the risk-control layer of one. The stock-picking is assumed; the layer being studied is what to do with positions you've already chosen.
- **The novelty is modest.** Probabilistic forecasting exists. Drawdown-constrained optimisation exists. RL-based trading exists. The contribution is putting the three together into one explicit, formal, reproducible pipeline and measuring the result. It is one paper's worth of contribution, not a thesis chapter's worth of breakthrough.
- **2022–2025 is a 4-year test window, but it is not the only one tested.** A four-fold walk-forward check (training on an earlier period and testing on a strictly later one) has now been run across the grid — 320 runs spanning 2018–2025, including the COVID crash and the 2022 inflation shock. The AI beats the baseline on **all four** time windows, with an overall **87% win-rate** and a median final value of about **$1.17M** out-of-sample. So the advantage is not an accident of one test period.
- **The full grid is complete.** The headline numbers above come from the finished Phase-2 grid (10 seeds × 50,000 training rounds × 70 stocks). This is no longer "scheduled" or "planned" — it has been run, and the dissertation reports the results.
- **Single-asset environment.** The agent runs on one ticker at a time. A true multi-asset version that watches the running peak of the *whole portfolio* is the natural next step and is in the future-work plan.

---

## What you can show Dr Nguyen, today

- **The LaTeX dissertation in `latex/`** — this is now the canonical, full
  dissertation, reconciled to the completed Phase-2 results. An editable Word
  copy can be produced at any time with `latex/build_docx.sh`. The chapters
  that directly answer the supervisor's previous-meeting feedback are Chapter 1
  (formal problem statement and objectives), Chapter 2 (finance background with
  notation), Chapter 3 (explicit objective function), Chapter 5 (the full
  Phase-2 grid across 70 stocks, the honest drawdown comparison, and the
  walk-forward validation), and Chapter 6 (honest discussion of where the agent
  wins and where it fails).
- **`reports/generated/charts/`** — the reproducible Phase-2 chart suite,
  including the equity curves, the outcome and Sharpe distributions, the
  drawdown comparison, and the walk-forward-by-fold breakdown.
- **`reports/generated/dissertation_results.md`** — a one-page summary of the
  Phase-2 headline numbers.
- **`reports/generated/findings_plain_english.md`** — this document.
