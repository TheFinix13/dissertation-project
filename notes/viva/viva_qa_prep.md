# Viva preparation — EEEM004, September 2026

## Format and marking (from the Project Handbook)

- Presentation: no more than 20 minutes, no more than 12 slides.
  The deck (`latex/viva/viva_deck.tex`) has exactly 12.
- Examiners: two, one of whom is Dr Nguyen.
- Viva mark: technical achievement 33%, methodology 34%,
  interview performance (reasoned argument, logical thought) 33%.
- Possible outcomes: pass; pass with minor corrections (40 days);
  fail with resubmission within 6 months (capped at 50 unless the
  viva was passed).

## The ten numbers to know cold

| Number | What it is |
|---|---|
| +$126.94/ep (std $7.30) | DQN on the balanced sim test, 6/6 seeds |
| +$473.72 / −$103.35 / +$10.45 | DQN per condition: up / down / flat |
| −$60.29 | Buy-and-hold per episode on the balanced test |
| +$180.25 | Buy-and-hold per month, first study 2024–25 |
| 60 vs ~18,000 | First study: episodes vs parameters |
| 3,000 vs ~2,800 | Main study: episodes vs parameters (~20:1 transitions) |
| 73% for 1.5% | Risk reward: down-loss cut for mean-return cost (λ=0.25) |
| +$95.27/ep at 50 bps | DQN profit at ten times the assumed fee |
| 0.224 vs 0.198 | Transfer Sharpe: DQN vs buy-and-hold, 180 months |
| $1,665 vs $0 | October 2008: buy-and-hold loss vs DQN (zero exposure) |

Also: transfer raw return +$60.28 vs +$80.24 (buy-and-hold wins raw);
crisis Sep 08–Mar 09: −$180.34 vs −$3,045.66; benchmark bug caught by
tests: $38.95/month (21.6%).

## Anticipated questions and answers

### Ownership and derivation (examiners confirm authorship here)

**Derive the REINFORCE update.**
The objective is expected return J(θ) = E[G₀]. Only the action
probabilities depend on θ, so the likelihood-ratio identity gives
∇J = E[Σₜ ∇log π(aₜ|sₜ) · Gₜ]. Each action is weighted by the
reward-to-go, not the whole episode return, because an action cannot
influence rewards already received. Implemented as a negative-log-
likelihood loss weighted by standardised returns, Adam at 1e-3.

**Why is the DQN target y = r + γ(1−d)·max over legal actions?**
The max is taken over legal actions only, because the agent should not
assign value to actions it cannot take. The (1−d) term matters in this
environment because every episode force-closes on the last day: with
d=1 the future term vanishes, otherwise the target would pretend
further decisions were possible.

**Why 0.25·C₀ as the trade size?**
Selected on validation by an exposure rule, not by test performance:
the smallest size whose attainable exposure reaches 90% of
buy-and-hold's. 0.10 caps exposure at 0.69 and cannot compete even in
principle; 0.50 adds little ceiling but halves position control. The
rule re-selected 0.25 on the simulated validation episodes.

**Why γ = 0.99 for 21-day episodes?**
0.99²⁰ ≈ 0.82, so end-of-month rewards keep ~82% of their value.
Mild discounting keeps the standard formulation while the objective
stays close to end-of-month wealth.

### Methodology attacks

**Only six seeds — why should we believe small differences?**
We don't, and the dissertation says so. The +$4.37 drawdown-feature
change sits inside a $7 seed std and is reported as no effect. The
headline result does not rest on small differences: DQN's +$126.94 has
a $7.30 std and every other policy loses money or breaks even. The
limitation section names bootstrap CIs and more seeds as future work.

**Your balanced test only proves you beat *constant* policies. Where
is the active baseline (e.g. a moving-average rule)?**
Correct, and it is the fairest criticism of the study. The benchmarks
were chosen to test the behavioural question — does the agent read its
state — which fixed patterns answer exactly. A momentum rule on the
same episodes is the right next comparison and is future work. Two
mitigations: the transfer test pits the agent against real markets,
and the fee sweep shows the profit is not an artefact of free trading.

**Isn't the agent just classifying your generator's three regimes
rather than learning to trade?**
Partly, and that is by design: the controlled experiment makes market
condition the variable being learned. The check on the claim is the
transfer test: the regularities the agent learned (volatility and drift
features signalling decline) carried to 180 real months it never saw,
including October 2008. The imperfect timing around the Dec 08/Mar 09
rebounds shows exactly where the generator's simplicity costs accuracy
— the dissertation reports this as the domain gap, measured.

**The policy-gradient collapse — did you tune them before concluding?**
No, and the dissertation states the finding as behaviour under shared
settings, not a bound on tuned methods. PPO ran with entropy
coefficient 0; entropy regularisation might keep exploration alive.
One REINFORCE seed found the conditioned policy (+$83.13), so the
optimum is reachable by policy gradients — the claim is only that they
do not find it reliably here.

**Why monthly independent episodes instead of a continuous portfolio?**
Control. Same length, same starting capital, no cross-month coupling,
so results are comparable across months and seeds. The cost — no
compounding, no carried positions — is in the limitations, and a
continuous environment is future work.

### Results attacks

**Buy-and-hold still beats you on raw return. So did this fail?**
On raw return over two mostly-rising decades, yes — consistent with
market efficiency, and the dissertation never claims otherwise. The
claim is risk-adjusted: Sharpe 0.224 vs 0.198 at three-quarters of the
exposure and two-thirds of the drawdown, with five of six seeds at or
above the benchmark. The agent earns less but takes far less risk to
earn it.

**Why did DQN fail on real data but win on simulated data?**
Value estimation needs many visits to similar states under a stable
distribution. 60 shifting months provide neither, so DQN conditioned
on noisy estimates and earned the least. 3,000 balanced episodes make
the estimates reliable, and epsilon-greedy exploration means DQN never
stops visiting both actions — unlike a near-deterministic policy
gradient, whose gradient signal dies at the corners.

**Your risk-aware reward result contradicts your first study. Which
is right?**
Both, on their own data — that is the point. The penalty can only
produce selective behaviour if the value estimates can tell a falling
market from a rising one. Estimates trained on 60 months could not
(participation tax); estimates trained on 3,000 balanced episodes
could (73% down-loss cut for 1.5% return). The disagreement is itself
evidence for the data conclusion.

**October 2008: luck or learning?**
Neither seed nor month specific: all six seeds held zero exposure the
whole month, and the same de-risking appears across Sep 08–Mar 09 and
in the 2011 and 2015–16 declines. The agents respond to volatility and
drift features matching the simulated down condition. The honest
counterpart is reported: they also sat out the two rebound months.

### Structure and process

**Why are the algorithm formulations in Chapter 3 and not the
background?** (Nguyen raised this on 7 Sep.)
Chapter 2 covers the ideas and lineage — what the families are and why
they differ. Chapter 3 contains only what was implemented and trained:
every equation there corresponds to code that ran, listed in
Appendix B. The split is background-by-idea versus
methodology-by-implementation.

**What would you do differently?**
Budget the data before the model. The project spent most of its effort
on state, reward, and verification; all necessary, none sufficient. The
binding constraint was 60 episodes feeding 18,000 parameters. Counting
episodes against parameters on day one would have moved the simulator
to the start of the project. Second: bootstrap CIs and one active
baseline were cheap and should have been in the submitted document.

**What is the original contribution?**
The controlled combination, not an invention: a mask-conditioned
state-dependence test reported per seed next to returns; a two-study
design that diagnoses a negative result and then removes its cause; and
transfer evidence that simulator-trained policies de-risked through a
real crisis outside their calibration window.

## Presentation delivery notes

- 12 slides / ~18 minutes: ~90 s per slide; slides 8 and 10 (main
  result, transfer) can take 2–3 min each; slides 2–3 under a minute.
- Say the negative result plainly and early (slide 5); examiners reward
  the diagnosis, not spin.
- On any "did it make money" question, give both halves in one breath:
  lower raw return, higher Sharpe, zero exposure October 2008.
- Bring per-month Appendix C numbers printed; if pressed on any month,
  the table answers it.
