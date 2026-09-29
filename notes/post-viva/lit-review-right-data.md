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

Second round, run 29 Sep 2026 for Themes F and G:

9. "Rydén Teräsvirta Åsbrink 1998 stylized facts of daily return series
   and the hidden Markov model Journal of Applied Econometrics"
10. "Nystrup Madsen Lindström dynamic portfolio optimization across
    hidden market regimes Quantitative Finance 2018"
11. "Andrychowicz 2021 What matters in on-policy reinforcement learning
    large-scale empirical study ICLR observation reward normalization
    value loss"
12. "Agarwal 2021 deep reinforcement learning at the edge of the
    statistical precipice few runs stratified bootstrap interquartile
    mean NeurIPS"

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

## 8. Theme F — regimes as a hidden state

This theme answers Dr Nguyen's framing in the viva and his email. He
described the market as a hidden Markov model, where an unobserved state
drives the price and the observed data is noise around it.

**Hamilton (1989, Econometrica 57(2), 357–384).** The founding
regime-switching model. The economy moves between a small number of
states under a Markov chain, and each state has its own mean growth.
The state is never observed. It is inferred from data as a probability,
and that probability is the useful output.

**Rydén, Teräsvirta and Åsbrink (1998, Journal of Applied Econometrics
13(3), 217–244).** They fitted hidden Markov models to about 17,000
daily S&P 500 returns. A mixture of normal distributions switching
under a hidden chain reproduced most of the known properties of daily
returns. It failed on one: the slow decay in the autocorrelation of
squared returns, which is volatility clustering over long horizons.
The parameters also changed a lot between subperiods.

**Nystrup, Madsen and Lindström (2015, Quantitative Finance; 2018,
Quantitative Finance 18(1), 83–95).** The 2015 paper extends the
Rydén result. Three states and heavier-tailed state distributions fit
the stylised facts better than two normal states. The 2018 paper uses
a hidden Markov model with time-varying parameters to forecast the mean
and variance of returns, then re-optimises the allocation each day.
After costs and a one-day delay, it earned a higher return and lower
risk than buy-and-hold on several major indices. This is the closest
published analogue to what the main study's agent attempts, and it
works through the variance, not the mean.

**Ang and Timmermann (2012, Annual Review of Financial Economics 4,
313–337).** A survey of regime changes in financial markets. Regimes
are persistent, volatility differs more between regimes than the mean
does, and regime probabilities are a natural input to allocation.

The Theme F synthesis: the literature treats the regime as a hidden
state that must be inferred, and it finds the regime shows up mainly
in volatility. The main study's simulator does the opposite. Each
episode is one regime, drawn independently, and its warm-up history
shares the regime, so the state is close to observed from the first
bar. The measured consequence is in the Gate 2 audit
(`2026-09-29-gates-2-4-audit.md`). A more realistic simulator should
draw regimes from a persistent Markov chain that runs across months,
with regime-dependent volatility, and should let the agent infer the
state rather than read it.

## 9. Theme G — comparing reinforcement-learning methods fairly

This theme answers the report feedback on fair tuning and seed
statistics, and Dr Nguyen's point that PPO suits larger problems.

**Henderson et al. (2018, AAAI, "Deep Reinforcement Learning that
Matters").** The same algorithm gave very different results under
different codebases, hyperparameters and random seeds. Comparisons
that tune one method and not the other are not evidence about the
methods. Reward scale was one of the settings that changed rankings.

**Engstrom et al. (2020, ICLR, "Implementation Matters in Deep Policy
Gradients") and Huang et al. (2022, ICLR Blog Track, "The 37
Implementation Details of Proximal Policy Optimization").** Much of
PPO's reported advantage comes from code-level choices outside the core
algorithm. Reward scaling, value-target normalisation and gradient
clipping are among them. Huang et al. list reward scaling as a standard
part of a correct PPO implementation.

**Andrychowicz et al. (2021, ICLR, "What Matters in On-Policy
Reinforcement Learning?").** They trained over 250,000 agents across 50
implementation choices. Observation normalisation was crucial almost
everywhere. Normalising the value targets had a strong effect, helping
on some tasks and hurting on others, so they recommend checking it per
task. A learning rate near 3e-4 and a GAE lambda near 0.9 were good
starting points.

**Agarwal et al. (2021, NeurIPS, "Deep Reinforcement Learning at the
Edge of the Statistical Precipice").** Point estimates from a handful
of runs gave conclusions that reversed under proper interval estimates.
They recommend bootstrap confidence intervals, the interquartile mean
and performance profiles. Their percentile intervals had good coverage
from about 10 runs.

The Theme G synthesis: a fair comparison gives each method its own
tuning budget of equal size, fixes reward scaling and normalisation as
part of the implementation, and reports interval estimates over at
least 10 seeds. The dissertation gave PPO none of these. The Gate 3
probe in the audit document shows the reward scale alone decided
whether PPO could learn a trivial problem.

## 10. What this means for Gate 2 (data and state design) — provisional

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

## 11. Open work in this gate

- Full-text reads of the five anchors: Welch-Goyal 2008,
  Campbell-Thompson 2008, Gu-Kelly-Xiu 2020, Moreira-Muir 2017,
  Zhang et al. 2023. Extraction table (problem, method, dataset,
  metric, result, limitation) per the brain-box protocol.
- Snowball round: Goyal-Welch-Zafirov 2021 (the update), forward
  citations of Gu-Kelly-Xiu, offline-RL data-coverage literature.
- A short written verdict per sub-question, each one phrased as the
  answer Fiyin would give if an examiner asked it.
- Themes F and G (added 29 Sep 2026) rest on abstracts and the search
  pages, plus the normalisation section of Andrychowicz et al. Full
  reads are still owed for Nystrup et al. 2018, Ang-Timmermann 2012 and
  Huang et al. 2022 before the simulator and tuning designs are final.
