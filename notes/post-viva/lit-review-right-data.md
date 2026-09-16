# Literature review: what makes the right data for an RL trading agent

Started 16 September 2026. Gate 1 of the literature-first protocol.
Status: **search phase done, extraction phase open.** The claims below
come from abstracts and key passages, not yet from full-text reads. Each
anchor paper needs a full read before its claim is used in a design
decision.

## 1. The question

Prof Nikitopoulos asked in the viva: "what is the right data?" and
rejected "more volume" as an answer. This review answers with named
characteristics. Framed as a review question:

**For a reinforcement-learning agent trading a broad equity index, which
input characteristics does the literature establish as (a) predictive of
returns, (b) predictive of risk, and (c) sufficient in volume for the
model class — and what does it establish about training such an agent
without exploration collapse?**

Five sub-questions, one per section below: what predicts returns; what
predicts risk; how much data a given model size needs; what simulated
data buys; and how PPO entropy must be handled.

## 2. Search record (reproducible, per the brain-box protocol)

Web searches run 16 Sep 2026. Strings, verbatim:

1. "how much data does deep reinforcement learning need sample
   efficiency overfitting small datasets finance trading"
2. "Gu Kelly Xiu empirical asset pricing machine learning which
   predictors matter fundamentals macroeconomic variables stock return
   predictability"
3. "PPO entropy coefficient exploration collapse premature convergence
   deterministic policy reinforcement learning"
4. "synthetic market data simulator training reinforcement learning
   trading agents generalization sim-to-real transfer finance"
5. "Welch Goyal comprehensive look empirical performance equity premium
   prediction out-of-sample Campbell Thompson"
6. "deep reinforcement learning trading portfolio management survey
   2024 2025 state representation feature engineering challenges
   non-stationarity"
7. "neural network sample size requirements events per parameter
   overfitting small datasets tabular data rule of thumb"
8. "volatility managed portfolios Moreira Muir VIX regime timing equity
   returns risk management improves Sharpe"

Next search round (snowballing from the anchors, not yet done): forward
citations of Gu, Kelly and Xiu (2020); Goyal, Welch and Zafirov (2021)
"A Comprehensive Look ... II"; offline RL for finance; distributional
shift in offline RL (Levine et al. 2020).

## 3. Theme A — what predicts returns (the core of "the right data")

**Welch and Goyal (2008, Review of Financial Studies 21(4), 1455–1508).**
The reference test of every popular aggregate-market predictor:
dividend-price ratio, earnings-price ratio, book-to-market, interest
rates, and more. Out of sample, none beat the historical mean return.
Most hurt. This is the strongest version of Nikitopoulos's own point,
and it cuts against naive fundamentals too, not just price statistics.
Adding a valuation ratio to the state is not automatically "the right
data."

**Campbell and Thompson (2008, Review of Financial Studies 21(4),
1509–1531).** The rebuttal. With economically motivated restrictions
(coefficient signs fixed by theory, forecasts floored at zero), many
predictors do beat the historical mean out of sample. The monthly
out-of-sample R-squared is under one percent, but that is still
economically meaningful for a mean-variance investor. Two lessons for
the redo: theory-restricted inputs beat unrestricted ones, and the
honest expectation for return prediction is an R-squared near zero.
Any design that needs strong return forecasts to work is wrong.

**Gu, Kelly and Xiu (2020, Review of Financial Studies 33(5),
2223–2273, DOI 10.1093/rfs/hhaa009).** Machine learning on 94 firm
characteristics plus eight macro predictors, whole US cross-section.
Every model class agrees on the dominant predictors, in order: price
trends (momentum, industry momentum, short-term reversal), liquidity
(market value, dollar volume, bid-ask spread), and volatility (total
and idiosyncratic volatility, beta). Fundamentals are present but not
dominant. This complicates the viva critique in our favour: the
best-attested predictable signal in modern ML asset pricing IS built
from past prices and volume. What the paper adds is that the gains come
from nonlinear interactions and heavy regularization, and that the
predictable component is small.

The Theme A synthesis: the literature does not say "swap price data for
fundamentals." It says return predictability is tiny from ANY input
set, price-trend and volatility variables are the strongest of a weak
field, and restrictions from theory are what make weak signals usable.

## 4. Theme B — what predicts risk (where the real signal is)

**Moreira and Muir (2017, Journal of Finance 72(4), 1611–1644, DOI
10.1111/jofi.12513).** Scaling equity exposure inversely to lagged
realized variance raises Sharpe ratios and produces large positive
alphas, for the market and for most major factors. The mechanism:
volatility is strongly forecastable at monthly horizons, expected
returns are not, and the two do not move proportionally. Taking less
risk when volatility is high is therefore close to a free improvement.

This is the single most important paper for the defence. The DQN
agent's learned behaviour — read volatility features, refuse exposure
in the 2008 band — is the Moreira-Muir strategy discovered from data.
The right data for risk timing is exactly what the state already
carries: realized volatility. A follow-up (Wang, DeMiguel et al. line;
"VIX-managed portfolios", International Review of Financial Analysis
2024) finds implied volatility (VIX) versions give steadier weights and
better after-cost alphas, which supports adding VIX to the state.

The Theme B synthesis: return prediction and risk prediction are
different problems with different data requirements. Volatility is
predictable; returns barely are. An agent scored on risk-adjusted
outcomes can earn its keep from the predictable half. This reframes the
contribution one level up from "beating buy-and-hold."

## 5. Theme C — how much data the model class needs

**Van der Ploeg, Austin and Steyerberg (2014, BMC Medical Research
Methodology 14:137).** Simulation study across model classes. Logistic
regression reaches a stable out-of-sample AUC at roughly 20–50 events
per variable. Neural networks and random forests remain unstable and
optimistic even past 200 events per variable — more than ten times the
classical requirement.

**Riley et al. (2020, BMJ 368:m441).** Sample-size rules like "10
events per parameter" are context-dependent, but the direction is
uncontested: when candidate parameters are large relative to
observations, overfitting is guaranteed, and the apparent (training)
performance will overstate the true performance.

**Silvey et al. (2024, JMIR-published empirical study on tabular
clinical data).** Across 16 real tabular datasets, a one-hidden-layer
network needed a median of roughly 13,000 observations for stable
discrimination, versus roughly 700 for logistic regression.

**Zhang et al. (2023, IJCAI, "Towards Generalizable Reinforcement
Learning for Trade Execution", DOI 10.24963/ijcai.2023/553).** The
finance-specific version. They model trading as offline RL with dynamic
context: the market variables in the state evolve independently of the
agent, and the dataset contains a finite number of context sequences
(historical price paths). Their generalization bound shows overfitting
is driven by large context space against few context sequences, and
they show deep RL policies memorizing the single best price in the
training data. Their fixes: compact context representations and more
independent sequences — not more replays of the same ones.

The Theme C synthesis, applied to us: the first study trained roughly
18,000 weights on 60 real episodes. The clinical literature says a
network of that size wants observations in the tens of thousands; the
RL-specific paper says the binding count is INDEPENDENT price paths,
not steps. Both were violated. The main study's fix (10,000 simulated
episodes) matches the Zhang et al. prescription almost exactly:
independent sequences at the scale the model class needs. That defence
now has citations. For the redo: count independent episodes first, then
size the network so the ratio is defensible, and prefer a linear or
tiny-network policy check before any deep model (the Campbell-Thompson
and van-der-Ploeg logic both point the same way).

## 6. Theme D — what simulated data buys

FinRL-Meta (AI4Finance Foundation) builds market environments and a
train-test-trade pipeline explicitly to close the gap between
backtesting and deployment. ABIDES-Gym (Amrouni et al. 2021, arXiv
2110.14771, J.P. Morgan) wraps a multi-agent limit-order-book
simulator into Gym for training investor agents. MarS (Microsoft,
arXiv 2409.07486) generates controllable order-level market
trajectories with a generative model and trains RL agents in them. An
AAAI 2026 paper trains an ensemble of imitation-learned regime
specialists (bull, bear) and shows RL policies trained in the
responsive simulator get superior downside protection out of sample.

The Theme D synthesis: simulator-first training is an established,
current line of work, and its stated purpose matches the main study's
design — many independent, regime-diverse paths where history offers
one. The literature also flags the cost: a simulator only teaches what
it contains, so the transfer test on real data stays mandatory. The
main study already does both. The redo can strengthen the simulator
side by making regime diversity (not just path count) the design
target.

## 7. Theme E — entropy in PPO

**Stable-Baselines3 documentation:** PPO's default entropy coefficient
is `ent_coef=0` — no entropy bonus at all — and there is no KL limit by
default (`target_kl=None`). Both defaults are decisions the main study
inherited without checking.

**OpenAI Spinning Up (PPO page):** PPO explores only through the
randomness of its own stochastic policy; over training the policy
becomes progressively less random, and it can get trapped in local
optima. This is the generic mechanism of the collapse.

**Briola et al. (2021, arXiv 2101.07107, "Deep Reinforcement Learning
for Active High Frequency Trading").** Reports the finance-specific
failure verbatim: their PPO agent "sporadically ... converges to
sub-optimal policy such as not trading at all," which they attribute to
PPO's exploration policy meeting the negative expected return of early
exploration. That is the main study's always-hold PPO, observed
independently. Not trading is a locally safe attractor, and a policy
with no entropy bonus cannot climb out of it.

Practice from tuning guides: sweep `ent_coef` over {0, 1e-4, 1e-3,
1e-2}, optionally decay it over training, set `target_kl` around
0.01–0.05 as a safety stop, and always log policy entropy, approximate
KL, and clip fraction so collapse is visible while it happens.

The Theme E synthesis: the PPO redo experiment is well-defined — same
environment, entropy coefficient swept instead of defaulted, entropy
logged. The literature predicts the collapse and names the cure; the
experiment tests whether the cure closes the DQN-PPO gap.

## 8. What this means for Gate 2 (data and state design) — provisional

Not decisions yet. These become decisions only after the full-text
reads confirm the claims.

1. **Keep volatility features; add implied volatility (VIX) if the
   data source allows.** Risk is the predictable quantity
   (Moreira-Muir); the state must carry it.
2. **Keep momentum and short-term-reversal features.** They are the
   top-ranked ML predictors (Gu-Kelly-Xiu), which defends the existing
   price-derived state better than the viva answer did.
3. **Add at most a small set of theory-restricted slow variables** —
   a valuation ratio, term spread, default spread — with
   Campbell-Thompson-style sign restrictions, and expect near-zero
   return R-squared from them. They address the "drivers of value"
   critique, but Welch-Goyal says unrestricted versions will hurt.
4. **Count independent episodes before sizing anything.** Parameters
   scale to episode count, not the reverse. A linear policy baseline
   runs first as the capacity floor.
5. **Evaluate on risk-adjusted outcomes.** The predictable edge is in
   risk timing, so raw return comparisons understate a correct agent.
6. **PPO runs with an entropy sweep, KL guard, and entropy logging.**
   Never again on unexamined defaults.

## 9. Open work in this gate

- Full-text reads of the five anchors: Welch-Goyal 2008,
  Campbell-Thompson 2008, Gu-Kelly-Xiu 2020, Moreira-Muir 2017,
  Zhang et al. 2023. Extraction table (problem, method, dataset,
  metric, result, limitation) per the brain-box protocol.
- Snowball round: Goyal-Welch-Zafirov 2021 (the update), forward
  citations of Gu-Kelly-Xiu, offline-RL data-coverage literature.
- A short written verdict per sub-question, each one phrased as the
  answer Fiyin would give if an examiner asked it.
