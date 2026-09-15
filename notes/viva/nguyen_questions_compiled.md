# Nguyen's questions — every one, with a prepared answer

Source: nine supervisor recordings from 3 July through 8 September 2026,
transcribed with whisper small.en. The raw question list is in the
subagent output; this file is the ANSWER SIDE — what you say when he
asks each thing in the viva.

The document is organised by the ten patterns Nguyen returned to
across every meeting. Under each pattern: (a) the questions he
actually asked, distilled, and (b) the fluent answer to give.
Read this out loud once before the viva. If you cannot say the
answer without checking, it is not ready.

---

## Pattern 1 · "Write it in mathematics, not English"

The most repeated demand across ALL meetings. Nguyen refuses
plain-English hand-waves for the RL objective, state, reward,
uncertainty. You must have equations ready.

### Q: "What is the objective function of REINFORCE?"

> REINFORCE maximises the expected discounted return of a trajectory
> sampled from the current policy:
>
> J(θ) = 𝔼_{τ ~ π_θ} [ Σ_{t=0}^{T-1} γ^t r_t ]
>
> The gradient estimator, from the policy-gradient theorem, is
>
> ∇_θ J(θ) = 𝔼_{τ ~ π_θ} [ Σ_t ∇_θ log π_θ(a_t | s_t) · G_t ]
>
> where G_t is the return from step t onward. In practice we
> subtract a baseline b(s_t) — the mean return — to reduce variance.

### Q: "What is the objective function of reinforcement learning — in general?"

> All of reinforcement learning optimises one quantity: the expected
> discounted return.
>
> J(θ) = 𝔼_{τ ~ π_θ} [ Σ_{t=0}^{T-1} γ^t r_t ]
>
> In this study r_t is the day's change in portfolio wealth, so the
> objective reads: prefer the policy whose average month ends with
> the most money.
>
> The three methods share this objective and differ only in how they
> climb it. REINFORCE and PPO move the policy parameters directly up
> an estimate of the gradient of J. Deep Q-learning never touches J
> itself — it learns the action-value function Q(s,a) through the
> Bellman equation and reads a policy off it by picking the
> best-valued legal action.

### Q: "What is the loss function of REINFORCE — and how is it used here?"

> The loss my code minimises is the negative log-likelihood of the
> actions taken, weighted by the return that followed each action:
>
> L_{RF}(θ) = − Σ_t log π_θ(a_t|s_t) · Ĝ_t
>
> where Ĝ_t is the discounted return from step t, standardised
> within the episode — subtract the mean, divide by the standard
> deviation. The standardisation IS the baseline. Minimising this
> loss is the same as ascending the policy-gradient estimator:
> actions followed by above-average returns gain probability,
> below-average lose it.
>
> How it is used here, step by step. One episode is one simulated
> month, about 21 trading days. The agent plays the month; each day
> pays a dollar reward. At month end the code discounts the rewards
> into returns G_t, standardises them, multiplies each by its
> action's log-probability, sums, and takes ONE Adam step. Then the
> trajectory is thrown away — REINFORCE is on-policy, so each
> episode buys exactly one update.

### Q: "What is the objective function of DQN?"

> DQN minimises the temporal-difference error between the current
> Q-network Q_θ and a bootstrapped target computed from a delayed
> copy Q_{θ⁻}:
>
> L(θ) = 𝔼_{(s,a,r,s') ~ 𝒟} [ (r + γ · max_{a'} Q_{θ⁻}(s', a') − Q_θ(s, a))² ]
>
> The replay buffer 𝒟 gives the (s,a,r,s') samples and the
> target network is copied from Q_θ every N steps.

### Q: "What is the objective function of PPO?"

> PPO maximises a clipped surrogate objective. Define the
> importance ratio r_t(θ) = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t).
> Then
>
> L^{CLIP}(θ) = 𝔼_t [ min( r_t(θ) · Â_t, clip(r_t(θ), 1−ε, 1+ε) · Â_t ) ]
>
> where Â_t is an advantage estimate (I use GAE-λ) and ε = 0.2.
> The clip prevents the policy moving too far from the previous
> iterate — that is the entire point of PPO. It is a first-order
> approximation to a KL-constrained optimisation via a Lagrangian
> reformulation.

### Q: "How is generalisation defined mathematically?"

> Generalisation error is the gap between empirical risk on the
> training set and the true risk under the data distribution:
>
> 𝓡(f) = 𝔼_{(x,y) ~ 𝒟} [ ℓ(f(x), y) ]
> 𝓡̂_n(f) = (1/n) Σ_i ℓ(f(x_i), y_i)
>
> Generalisation = 𝓡(f) − 𝓡̂_n(f). You minimise 𝓡̂_n at
> train time but ultimately want 𝓡 to be small. Nguyen pressed
> me on this because if I can't write it, I can't defend why
> 60 real months over-fit an 18,000-parameter network.

---

## Pattern 2 · "Define state, action, reward precisely (with dimensions)"

Nguyen went round in circles on this in Recording 47 and made
Fiyin re-derive H (holdings) as two variables (cash + shares).

### Q: "What is your state? What are its components and what dimension?"

> Nine real numbers in the main studies — s_t ∈ ℝ⁹ — and a tenth,
> drawdown, added only in the risk experiments (ℝ¹⁰):
>
> - five MARKET features: daily return, 5-day momentum, 5-day
>   volatility, gap to the 20-day average, and true range (ATR₁₄/Pₜ);
> - three PORTFOLIO features: cash fraction Cₜ/C₀, position value
>   hₜPₜ/C₀, and open profit/loss against the entry price;
> - one TIME feature: t/T — how far through the month we are;
> - risk experiments only: drawdown ddₜ = (W_peak − Wₜ)/W_peak.
>
> The action MASK is NOT part of the state. It is derived from the
> portfolio (cash sets the buy switch, holdings set the sell switch)
> and applied as a constraint on action selection — it adds no
> information the state does not already contain.

### Q: "What is your action space?"

> Discrete, three actions: Buy, Hold, Sell — cardinality fixed at
> exactly 3, as Nguyen said to do in Recording 49. Buy commits one
> fixed slice — a quarter of starting capital, 0.25·C₀ — at the
> current price; Sell closes one slice symmetrically. Any open
> position is closed at month end, so no exposure carries between
> episodes. The quarter slice was chosen on validation: the smallest
> slice that reaches at least 90% of buy-and-hold's exposure within
> a month.

### Q: "What is your reward?"

> Change in portfolio wealth after the transaction fee is paid:
>
> r_t = W_{t+1} − W_t
>
> where W_t = cash_t + shares_t · price_t. The transaction fee is
> 0.05 % of trade value and is deducted from cash on Buy/Sell.
> The episode-terminal reward is W_T − W_0.

### Q: "Why not use log-return as the reward?"

> Tried it in an ablation. Log-return smooths out the reward scale
> across up-markets and flat markets, which speeds convergence,
> but it flattens the penalty for large drawdowns — a $1000 loss
> and a $10 loss look proportionally similar. Since the whole
> point of the study is caution in falling markets, I use raw
> wealth change so the penalty for a big drop is genuinely big.

### Q: "H must be a function of s, but H is part of s — that's circular. Explain."

> Nguyen caught this in Recording 47. The fix in the final
> submission: I split "holdings" into two separate state components
> — cash and share-count — and never write H = f(s). H doesn't
> exist as a single scalar; the two components live independently
> in the state vector, and the reward is a function of both plus
> the current price.

---

## Pattern 3 · "60 episodes cannot train 18,000 parameters"

The single largest technical criticism. Recording 55 and 57 both.

### Q: "How many training episodes do you have and how many parameters in your network?"

> First study: 60 real monthly episodes, first network was 128×128
> MLP with ≈18,000 parameters. That is the core cause of the
> first-study failure — the ratio was about 3 examples per
> parameter, so the network was free to memorise.
>
> Main study: 3,000 balanced simulated episodes, network is a
> 64 → 32 pyramid with ≈2,800 parameters. Ratio is now above one
> to one in favour of data.

### Q: "How did you get from 60 to 3,000?"

> A Gym-style simulator. I fit three regime-conditional geometric
> Brownian motions on 2018-2022 real returns — up, down, flat —
> and generate 1,000 episodes per regime for a balanced 3,000.
> No real month outside 2018-2022 was used to calibrate the
> simulator, so all pre-2018 and post-2022 real months are
> available as unseen transfer test data.

### Q: "AlphaGo had millions of games, not thousands. Is 3,000 enough?"

> The comparison is not one-to-one — AlphaGo has a branching
> factor of ~250 per move and games of length ~200, so its
> effective state space is astronomically larger than my 21-step
> episode with a nine-number state. For a 2,800-parameter network
> on this state space, 3,000 balanced episodes is sufficient to
> reach state-dependent behaviour on all six seeds. That is the
> empirical finding.

### Q: "Why do you keep sticking with the data that you have?"

> I don't any more — that was the change I made between the
> first study and the main study. The first study defended real
> data because the problem statement was about real markets;
> the second acknowledges the data was the bottleneck and
> substitutes simulation.

---

## Pattern 4 · "Your train/test split is unfair"

Recording 55. Nguyen argued training 2018-2022 and testing
2024-2025 samples from different regimes.

### Q: "How is training 2018-2022 and testing 2024-2025 a fair split?"

> For the first study, honestly, it is not fair by machine-learning
> standards — the training window skewed toward rising and flat
> markets and the test window was almost entirely rising. That is
> one of the reasons the state-independent "always buy" policies
> looked competitive. I documented this in Chapter 5 as the primary
> lesson of the first study.
>
> For the main study, the fix is: I calibrate a simulator on
> 2018-2022 real data, then generate 1,000 balanced episodes in
> each of three regimes. Every simulator episode is independent
> and identically distributed within its regime; the training and
> test sets are DRAWN from the same distribution, and I stratify
> by regime.

### Q: "What about the transfer test — isn't that still an unfair split?"

> The transfer test is deliberately unfair, and that is the point.
> The agent is trained on the balanced simulator and then run on
> 180 real months (2006-2017 + 2023-2025). Those months were never
> seen by the calibrator OR the agent. It is the strongest
> possible test — can what the agent learned on the simulator
> generalise to genuinely out-of-distribution real markets? The
> answer is: Sharpe 0.224 vs 0.198 for buy-and-hold, and zero
> shares held through October 2008.

---

## Pattern 5 · "Understand PPO, DQN, REINFORCE at the equation level"

Nguyen made Fiyin read the PPO paper on the spot in Recording 44.

### Q: "What is PPO actually doing?"

> PPO is a first-order approximation to TRPO's constrained
> optimisation. TRPO solves
>
> max_θ 𝔼[ π_θ(a|s) / π_{θ_old}(a|s) · Â ] subject to KL(π_θ || π_{θ_old}) ≤ δ
>
> That's a constrained problem. TRPO uses a natural gradient with
> a line search. PPO drops the KL constraint and replaces it with
> a clipped surrogate objective — L^{CLIP} above. The clip acts
> as a soft trust region: if the new policy tries to increase r_t
> beyond 1+ε when Â > 0, the objective saturates. Empirically the
> clip works about as well as the natural gradient at much lower
> compute.

### Q: "Is PPO a model?"

> No. PPO is an optimisation algorithm for a policy. The MODEL is
> the neural network π_θ that maps state to action distribution.
> PPO is the recipe by which you update θ. Adam is the optimiser
> below PPO. Nguyen made this distinction very carefully in
> Recording 49: model vs optimiser vs training-algorithm.

### Q: Why is DQN the best? Why does Deep Q win here when REINFORCE and PPO don't?

> SAY THIS FIRST, about twenty seconds:
>
> All three methods see the same state, the same reward, and the
> same 3,000 episodes. The difference is what they have to
> estimate, and how they reuse the data. Deep Q-learning is
> off-policy: every transition goes into a replay buffer and can
> be learned from many times, while epsilon-greedy keeps buying
> and selling even after a habit forms. REINFORCE and PPO are
> on-policy: each trajectory gives one gradient update and is
> thrown away, and once the policy is near-certain the gradient
> dies. On balanced data no fixed habit makes money, so they
> collapse to never-trade or always-buy. Deep Q cannot take that
> shortcut — it has to keep estimating the value of Buy, Hold and
> Sell in each state. That is why it outperforms the other
> algorithms here.

> The numbers. Balanced test: Deep Q +$126.94 an episode, six
> seeds out of six state-dependent, seed spread $7.30. REINFORCE
> −$5.71, spread $52.17 because its seeds split between corners —
> five collapsed, one escaped and earned $83.13. PPO −$11.16: five
> seeds at $0.00, one always-buy at −$66.99. Transfer test, 180
> unseen months: Deep Q $60.28 a month, REINFORCE $31.66, PPO
> $8.32. And October 2008: only Deep Q went to zero shares.

> The ranking flipped once. In the first study Deep Q was the
> WORST, not the best. Sixty real months cannot support
> bootstrapped value estimates, so it conditioned on noise. The
> policy-gradient methods had an easy answer — always buy, because
> the market mostly rose — and that scored well without reading
> the state. Give them 3,000 balanced episodes and that easy
> answer disappears; the value estimates become reliable; Deep Q
> wins every seed. Together the two rankings say the deciding
> variable was data volume and balance, not a magic algorithm.

> Honest caveat if they press "you just tuned Deep Q": Chapter 6
> says the policy-gradient methods were not separately tuned, and
> PPO's entropy coefficient was left at the library default of
> zero. An entropy bonus might have kept PPO exploring. That is
> future work. What I can claim is that, at the same network, the
> same data, and the same budget, the method that reuses
> experience and never stops exploring was the only one that
> found the state-dependent policy.

### Q: "How can you use PPO if you don't understand it?"

> I used the sb3-contrib implementation of MaskablePPO for the
> comparison, exactly because I wanted a reference implementation
> of a modern method against my from-scratch DQN and REINFORCE.
> Chapter 3 of the report presents the mathematical formulation
> alongside my code — the clipped objective, the advantage
> estimator, the actor-critic architecture I used.

### Q: "What is a policy?"

> A policy is a mapping from state to a distribution over actions:
> π_θ : S → Δ(A). "Deterministic policy" means the distribution
> is a Dirac delta. Both REINFORCE and PPO learn stochastic
> policies. DQN doesn't learn a policy directly — it learns Q(s,a)
> and derives a policy by taking argmax over actions.

---

## Pattern 6 · "Model uncertainty as a distribution, not a number"

Recording 44, the very first meeting.

### Q: "How did you model uncertainty?"

> Honest answer: in the FINAL submission there is no explicit
> uncertainty model in the policy — that was the direction of the
> original proposal but I pivoted after Recording 44 to focus on
> state-dependence and controlled experimentation.
>
> The closest thing that remains in the work is:
>
> 1. The seed spread — I report mean and range across six seeds,
>    which is empirical uncertainty over the training procedure.
> 2. The volatility feature in the state — the agent effectively
>    conditions on realised uncertainty in the market.
>
> If I had time, I would have added an ensemble of DQNs
> (Osband-style bootstrapped DQN) or a distributional Q-network
> (C51/QR-DQN) so the agent could learn σ² alongside μ. That is
> in the future work section of Chapter 6.

---

## Pattern 7 · "Domain-knowledge honesty — this is ML, not finance"

Nguyen's steady line since Recording 44: don't oversell the
finance side.

### Q: "This is a finance problem you don't have background for. Why?"

> Right. I framed it as a machine-learning study that uses
> stock trading as its testbed. The contribution is the
> methodological finding that small-data RL researchers should
> check whether the DATA can support their model before blaming
> the algorithm. The finance-specific claims are deliberately
> narrow — I do not claim to have beaten the market on raw
> return over 20 years, and Chapter 6 lists what a genuine
> finance contribution would require (slippage modelling,
> order-book depth, T+2 settlement, higher-frequency data).

### Q: "What is buy-and-hold, and why do you use it?"

> Buy-and-hold means investing all the starting capital on day one,
> holding through the entire episode, and marking to market at
> the end. It is the simplest possible passive policy — it makes
> no decisions and pays exactly one transaction fee. In an
> up-trending market it is famously hard to beat; that is why
> I use it as the reference. Also, it is a state-independent
> policy by construction, which makes the state-dependence test
> meaningful — any policy that beats buy-and-hold BY reading
> its state has learned something buy-and-hold cannot.

---

## Pattern 8 · "The state-dependence test — explain this properly"

Recording 55 and 57 both. Also the centre of the main study.

### Q: "What is the state-dependence test?"

> A per-seed statistical test that asks: given the AVAILABLE
> actions (the mask), does the agent's chosen action depend on
> the state?
>
> Procedure. For each seed, I collect (mask, state, action)
> triples from evaluation. For every pair of triples where the
> mask is identical, I count whether the action was identical.
> Under the null hypothesis that the agent ignores state, action
> should be identical (up to ε for stochastic policies) whenever
> mask is identical. A Fisher exact test rejects the null when
> action varies significantly across identical-mask states.
>
> The score "6 / 6" means all six seeds' tests rejected the null,
> so the agent is genuinely state-dependent. "0 / 6" means no
> seed rejected — the agent is a fixed-mask-conditioned rule.

### Q: "Why is this better than just looking at returns?"

> Returns are confounded by market direction. In an up-trending
> market, an "always buy" policy earns money without being an
> agent in any meaningful sense — it is a fixed rule wearing
> agent's clothing. The state-dependence test separates
> genuinely learned behaviour from lucky-with-market-trend.

---

## Pattern 9 · "Insight over recital — don't sound like ChatGPT"

Recording 47 and 55. This is a MANNER instruction.

### How to talk in the viva

> - If you know the answer, give it plainly with equations where
>   asked and a number where possible.
> - If you don't know, say "I don't have that in my head — my
>   best guess would be X, but I'd need to check". Never invent
>   a number.
> - If you're asked a definitional question and you have the
>   definition wrong, correct yourself out loud rather than
>   double-down. Nguyen respects that.
> - Never begin an answer with "Great question". Never say
>   "as I mentioned earlier". Just answer.
> - Every methodological choice needs a WHY, not a definition.

---

## Pattern 10 · "Controlled experiment, not random experiment"

Recording 55.

### Q: "What did your agent do on the drawdown period vs the rally?"

> Broken out by simulator regime on the balanced test:
>
> - UP condition (1,000 episodes): DQN agent mostly Buys early
>   and Holds through the run — it keeps $473.72 of the $477.22 the
>   always-buy rule earns, 99% of the gains (the slide's numbers).
> - DOWN condition (1,000 episodes): DQN agent enters Sell early
>   and holds cash. Always-buy loses $663.91 per episode; DQN loses
>   $103.35 — an 84% cut. Against TRUE buy-and-hold (one lump on
>   day 1, −$710.06) the cut is 85% — same story, different
>   baseline.
> - FLAT condition (1,000 episodes): DQN agent trades more
>   actively, exploiting short reversals: +$10.45 against
>   buy-and-hold's +$19.00.
>
> Aggregated: +$126.94/episode for DQN, −$60.29/episode for
> buy-and-hold — on balanced data no fixed pattern survives.
> On the transfer test the same behavioural pattern appears —
> the "hold-cash-through-drawdown" instinct is what earns the
> higher Sharpe.

---

## The one honest limitation Nguyen wants you to state

Nguyen said in Recording 55: "if you put a lot of limitations,
which means you almost do nothing." Pick ONE and defend it.

### The honest limitation to say

> The single biggest limitation is that the simulator is a
> three-regime geometric Brownian motion — it does not produce
> regime SHIFTS. A real 2008-style event is a sharp change in
> regime, not a slower down-regime. The agent's transfer
> performance on October 2008 is real but partly coincidental —
> the volatility feature spiked into DOWN-condition territory
> and the agent stepped aside for the right reason but by an
> imperfect mechanism. A more realistic generator (regime-switching
> GBM with transition probabilities, or a Hidden Markov process
> over states) is the top future-work item.

---

## The one directive Nguyen gave that DIDN'T get done

Nguyen asked in Recording 57 to add the PPO objective explicitly to
the report. It IS in Chapter 3 in the final PDF. If asked "did you
add the PPO math?" the answer is YES, Section 3.5.

He also asked to drop validation or move it to training. In the
final submission I still use a small validation set (12 months,
2023) for hyperparameter selection only — the test set is used
ONCE. If asked, be honest: the validation set is small but
necessary for the transfer test to have any meaning at all.

---

## Pattern 11 · "Tuning and exploration" (from the 13 Sep practice run)

### Q: What ways would you tune REINFORCE and PPO?

> Both failed here for sample-efficiency reasons, so tuning starts
> with variance, not with the learning rate.

> For REINFORCE, three concrete changes. First, a learned baseline —
> a small value network — which turns it into an actor-critic; that
> cuts gradient variance without biasing the estimate. Second, batch
> several episodes per update: my implementation updates on a single
> month, so every gradient is one noisy sample. Third, an entropy
> bonus so the policy cannot collapse to a fixed action early.

> For PPO, the knobs are the entropy coefficient, the clip range ε,
> GAE-λ, and the rollout length. I would raise the entropy
> coefficient first, because the observed failure mode was collapse
> to "hold" or "always buy".

> In the study all three methods ran on matched budgets — 300,000
> steps, the same 64→32 network, the same six seeds — precisely so
> the comparison measures the algorithm and not the tuning effort.
> A per-algorithm hyperparameter search is future work, and I say
> that plainly.

### Q: How would you make REINFORCE explore, like epsilon-greedy?

> REINFORCE already explores: it samples from its softmax policy, so
> every action with non-zero probability gets tried. The problem is
> that this exploration decays with the policy itself — as the
> probabilities sharpen, exploration vanishes.

> You cannot simply bolt ε-greedy on. REINFORCE's gradient assumes
> the actions were drawn from the policy being updated; acting
> ε-greedily makes the data off-policy and biases the gradient
> unless you add importance weights.

> The clean equivalents are: an entropy bonus added to the
> objective, which keeps probabilities away from zero and one; or a
> temperature on the softmax, annealed over training. That is
> exactly why ε-greedy lives naturally in value-based, off-policy
> methods like my DQN — the replay buffer never assumes the data
> came from the current policy.

> In my DQN, ε starts at 1.0 and decays linearly to 0.05 across the
> first half of training, and it explores only over LEGAL actions
> under the mask.

---

## Pattern 12 · "Why these numbers — episodes and parameters"

### Q: Why 3,000 episodes? Why not more from the balanced data?

> 3,000 episodes is roughly 63,000 daily decisions against 2,800
> network parameters — about twenty examples per parameter. The
> first study had it the other way round: three examples per
> parameter.

> More draws add little. Each regime is a geometric Brownian motion
> with two fitted parameters — a drift and a volatility. A thousand
> draws per regime already samples that two-parameter distribution
> densely; the 3,001st up-month is statistically a near-copy of one
> I already have.

> The binding constraint is the simulator's realism — three fixed
> regimes, no regime shifts — and no episode count fixes that.

> Honest caveat: I did not run a formal ablation over episode count.
> The evidence of sufficiency is the outcome: six seeds out of six
> state-dependent, with a seed spread of about $7 per episode.

### Q: The network shrank to 2,800 parameters — why not double the episodes too?

> Because the extra episodes carry almost no new information — the
> generator has only six fitted numbers in total, two per regime.

> There is also a budget interaction. Training is fixed at 300,000
> environment steps, which is about 14,000 episode plays; over 3,000
> episodes that is roughly five visits each. Doubling the pool
> halves the visits without adding variety.

> What I would actually buy with more compute is a richer generator
> — regime switching within an episode — not more draws from the
> same three distributions.

### Q: Why a pyramid network — 64 then 32? Why not 64×64 or 32×32?

> The parameter budget came first, the shape second. The design rule
> after the first study's failure was: keep trainable weights well
> below the number of training decisions. With a 9-feature input and
> 3 outputs, the three candidates cost:
>
> - 64×32 — about 2,800 parameters (the choice)
> - 64×64 — about 5,000, nearly double, so the examples-per-weight
>   ratio drops from ~22 to ~13
> - 32×32 — about 1,500, which fits the budget too, but narrows the
>   FIRST layer, and the first layer is the one reading all nine
>   features at once
>
> The funnel shape is the standard encoder argument: a wide first
> layer forms combinations of the raw features — momentum with
> volatility, exposure with trend; a narrower second layer compresses
> those combinations toward what the three action values need. Going
> 64 then 32 spends the budget where the feature mixing happens.
>
> The honest caveat: I did not run an architecture sweep. 64×32 is a
> heuristic that respects the data budget, and I would expect 32×32
> to behave similarly — the first study showed capacity RELATIVE to
> data is what matters, at 18,000 weights against 60 months. What I
> can say is that the choice was fixed once, before the test data
> was touched, and every method used the same layout — so the
> comparison between algorithms never depends on it.

---

## Pattern 13 · "Action masking — define it, implement it"

### Q: What is action masking? How do you implement it in the agents?

> Masking is a hard rule that removes impossible actions before the
> agent chooses: you cannot buy with no cash, you cannot sell shares
> you do not hold.

> In the environment it is a three-element boolean vector. Hold is
> always legal. Buy is legal while at least one dollar of cash
> remains. Sell is legal while the position is worth at least one
> dollar.

> Implementation, per method:

> - REINFORCE: illegal actions' logits are set to −10⁹ before the
>   softmax, so their probability is exactly zero and no gradient
>   ever flows through them.
> - DQN: illegal Q-values are set to −10⁹ in TWO places — the greedy
>   argmax, and inside the bootstrapped target's max, so the target
>   never credits an action the next state cannot take. The ε-greedy
>   step also samples only legal actions.
> - PPO: sb3-contrib's MaskablePPO applies the same logit masking
>   natively.

> The environment double-checks: an illegal action that somehow
> arrives executes as Hold and is logged. In the reported runs this
> never fires.

> Why mask rather than penalise: a penalty makes the agent LEARN a
> rule I already know, spending scarce samples on it. And the mask
> is not extra state information — it is derived from cash and
> holdings, which are already in the state.

### Q: Write action masking mathematically — the ML form.

> Define the legal-action set from the portfolio part of the state:
>
> A(s_t) = {hold} ∪ {buy if C_t ≥ slice cost} ∪ {sell if h_t > 0}
>
> Buy is legal when cash covers one quarter-of-capital slice plus
> its fee; Sell is legal when shares are held; Hold is always legal.
>
> For the policy methods, masking is an additive term on the logits
> before the softmax:
>
> π_θ^{mask}(a|s) = softmax(z_a + m_a), m_a = 0 if a ∈ A(s), −10⁹ otherwise
>
> An illegal action's probability is then exactly zero — and because
> it is never sampled, no gradient ever flows through it. The
> network never has to learn the rule; the rule is imposed.
>
> For Deep Q-learning the mask enters twice — action selection AND
> the bootstrapped target:
>
> y = r + γ max_{a'∈A(s')} Q_{θ⁻}(s',a'), a_t = argmax_{a∈A(s_t)} Q_θ(s_t,a)
>
> Masking the target matters: without it, the target can credit a
> next-state action the agent would not be allowed to take, and the
> value estimates inherit that phantom value. In code both are the
> same one-line operation — set the illegal entries to −10⁹ before
> the max.

---

## Pattern 14 · "Signals" — Nikitopoulos's home ground

### Q: How did you make the signals?

> All five market signals come from one daily price-volume series —
> SPY open, high, low, close, volume — computed once on the
> continuous series and only then sliced into monthly episodes.

> ret_1 = P_t/P_{t-1} − 1, mom_5 = P_t/P_{t-5} − 1, ma_gap = P_t/MA_20 − 1

> vol_5 = std(r_{t-4..t}), atr_norm = TR14/P_t

> Two engineering rules. First, windows never reset at a month
> boundary — computed inside the episode, a 5-day momentum has no
> history on day 1, so the agent would be told the market has no
> trend exactly when it opens its position. Second, every signal is
> strictly causal: the value at time t uses data up to t only, and
> an automated test truncates the series and checks nothing changes.

> Window choices: 5 days — a trading week — for momentum and
> volatility; 20 days — a trading month — for the trend reference;
> 14 days for the true-range average, which is Wilder's convention.
> All are centred near zero so no input dominates the first layer.

### Q: Define ATR — the average true range.

> ATR measures how far the price actually travels in a day —
> including any overnight gap — averaged over fourteen days. Closes
> alone can look calm on a day that swung four percent intraday;
> this signal is the one that sees it.

> One day's true range is the largest of three spans:

> TR_t = max(H_t − L_t, |H_t − C_{t−1}|, |L_t − C_{t−1}|)

> Worked example: the high is $102, the low is $99, yesterday's
> close was $100. The three spans are $3, $2 and $1 — the true
> range is $3.

> The feature averages fourteen of these and divides by price, so
> the number is comparable across price levels:

> ATR_14 = (1/14)·ΣTR, atr_norm = ATR_14/P_t

> A $3 range on a $100 asset reads 0.03 — the same as a $30 range
> on a $1,000 asset. One honesty point if pressed: this is a simple
> 14-day rolling mean of the true range, not Wilder's recursive
> smoother. The window length, 14 days, is Wilder's convention.

### Q: Define the moving average — and the ma_gap signal.

> The 20-day moving average is the mean closing price over the
> last trading month. It is the study's trend reference.

> MA_20 = (1/20)·ΣP, ma_gap = P_t/MA_20 − 1

> The state never feeds the average itself — it feeds the gap:
> where today's price sits relative to its own trend. If SPY closes
> at $105 and its 20-day mean is $100, ma_gap is +5% — stretched
> above trend. Negative means trading below trend.

> Why 20 days: one trading month, which also matches the episode
> length. And why a ratio rather than the raw average: a raw price
> level means nothing across episodes, but "5% above trend" means
> the same thing in 2006 and in 2020.

### Q: Define volatility — what exactly is vol_5?

> Volatility here is realised volatility: the standard deviation of
> the last five daily returns. It answers one question — how large
> has a typical daily move been this week?

> σ_5 = sqrt((1/5)·Σ(r_t−i − r̄)²), r_t = P_t/P_{t−1} − 1

> Worked example: five daily returns of +1%, −2%, +1.5%, −1% and
> +0.5% give σ₅ of roughly 1.3% per day — a hectic week. A calm
> market sits nearer 0.3%.

> Two details if pressed. It is the population standard deviation —
> divide by five, not four — because this is a state feature, not a
> statistical estimator; consistency matters more than
> unbiasedness. And it is close-to-close, which is why atr_norm
> exists alongside it: a day can close where it opened and still
> swing wide intraday. vol_5 misses that day; the range signal
> catches it.

### Q: List all nine state features — with their formulas.

> Nine numbers: five describe the market, four describe the agent's
> own situation. Five market signals first —

> ret_1 = P_t/P_{t−1} − 1, mom_5 = P_t/P_{t−5} − 1, ma_gap = P_t/MA_20 − 1

> vol_5 = std(r_{t−4..t}), atr_norm = ATR_14/P_t

> In words: yesterday's return; the one-week trend; price against
> its one-month trend; how large a typical daily move has been this
> week; and how wide the daily trading range is, gaps included.

> Then the four account features —

> clock = t/T, cash_frac = C_t/C_0, exposure_frac = h_t·P_t/C_0, pnl = (P_t − P_entry)/P_entry

> In words: how much of the month is gone; cash left as a fraction
> of the starting $1,000; the market value of the open position on
> the same scale; and the unrealised return since entry.

> The optional tenth, used only in the risk-aware experiment:

> drawdown = (W_peak − W_t)/W_peak

> — the fall from the episode's peak wealth. It is path-dependent,
> which is exactly why it stays out of the base state: the other
> nine can be read off today's prices and account, this one needs
> the episode's history. Every feature is a ratio or a return, so
> everything sits near zero and no single input dominates the
> network's first layer.

### Q: What other signals are in the study, or would you add?

> Two more are implemented but kept out of the headline state:
> relative volume — today's volume against its 20-day mean — and the
> volatility ratio σ₅/σ₂₀ − 1, which flags a regime shift. They form
> the top rung of the state ladder; the headline state stops one
> rung below, at five market signals.

> What I would add next, and did not: frequency-domain features — an
> autocorrelation or spectral estimate of the return series would
> expose cyclic structure that fixed time-domain windows cannot;
> cross-asset context such as VIX or rates; and order-flow data.

> On filtering: there is no smoothing beyond the rolling means. A
> low-pass or Kalman filter on the trend is a sensible extension,
> but each filter adds a lag parameter that must be validated, and
> with my data budget I kept the signal set small and interpretable.

### Q: Why daily bars? Why that sampling rate?

> Daily close data is the coarsest rate that still gives the agent
> about 21 decisions per episode, and the finest that is free,
> clean, and adjusted for splits and dividends.

> The rate also matches the question: a retail investor deciding
> once a day. Intraday data would turn this into a market
> microstructure problem — order books, slippage, latency — which is
> a different dissertation.

### Q: (Niko) "Your signals are all technical. Why no fundamentals, no VIX, no rates?"

> A practical constraint, not a preference. The main study trains on
> simulated months, so every signal in the state must be computable
> from what the simulator generates — a single daily price-volume
> path. A VIX level or an earnings figure cannot be produced by a
> simulator calibrated on one asset's prices, so a policy trained
> with those inputs could never run on the simulated episodes.
>
> The scope matches the research question too: can the agent read
> the market's CONDITION from price behaviour alone? Cross-asset
> context — VIX, rates, breadth — is named in Chapter 6 as future
> work, and it would need either real-data-only training or a much
> richer simulator.

### Q: (Niko) "vol_5 and atr_norm both measure volatility. Isn't one redundant?"

> They measure different volatility. vol_5 is the standard deviation
> of five close-to-close returns — it only sees where each day ends.
> atr_norm is a 14-day average of the true range over price — it
> sees the intraday span and any overnight gap. A day can swing 3%
> intraday and close flat: vol_5 records calm, atr_norm records the
> swing. They are correlated, but not substitutes — and Wilder's
> true range exists precisely because close-to-close measures miss
> that intraday risk.

### Q: (Niko) "How do you know the signals carry information at all?"

> Three pieces of evidence, in increasing strength.
>
> 1. The state-dependence test: all six Deep Q seeds change their
>    decisions when the market features change, against matched
>    portfolio contexts. If the signals carried nothing, acting on
>    them could not have cut down-market losses by 84%.
> 2. The drawdown-feature experiment: adding a tenth feature moved
>    the mean by $4.37 against a seed spread of about $7 — so a
>    signal only shows value when the OBJECTIVE gives the agent a
>    reason to use it. Information content and information use are
>    different things, and the study separates them.
> 3. The transfer test: the agent held zero shares through October
>    2008 — months it had never seen, in a year it had never seen.
>    The volatility features of that period matched the falling
>    condition it had learned on the simulator. That is a signal
>    carrying information out of sample.

---

## Pattern 15 · "The design on the page"

### Q: What is this 9-state and 10-state design? Why isn't it on the slide?

> Two experiments, two states. The main result uses nine features.
> The risk experiment adds a tenth — drawdown, the fall from the
> episode's peak wealth:

> dd_t = (W_peak − W_t)/W_peak

> The tenth is not decoration. A drawdown-PENALISED reward that the
> agent cannot see makes the process non-Markov — the reward would
> depend on a running peak that is absent from the state. My
> environment enforces this: setting the risk penalty without the
> drawdown feature raises an error at construction.

> Why it is not on the state slide: each slide shows the state of
> ONE experiment. The nine-feature table belongs to the main result;
> the tenth feature is introduced on the risk slide. Putting both in
> one table invited exactly the confusion this question is about.

### Q: What are these different rewards?

> Three rewards, one per purpose.

> The main reward is the day's wealth change in dollars, net of
> fees:

> r_t = W_{t+1} − W_t

> It telescopes: summed over the month it is exactly the episode
> profit, W_T − W_0.

> The risk-aware reward subtracts a penalty on any INCREASE in
> drawdown:

> r_t^{risk} = ΔW_t − λ·C_0·max(0, dd_{t+1} − dd_t)

> The increase in fractional drawdown is multiplied by the starting
> capital so both terms are in dollars and λ stays dimensionless.
> λ = 0.25 was selected on the validation split.

> Third, a log-return reward was tried in an ablation and rejected:
> it flattens the penalty for large losses, and caution in falling
> markets is the point of the study.

> One thing that is NOT a different reward: the learner-side reward
> scaling in DQN. It divides rewards before they enter the replay
> buffer to keep targets near the range the learning rate expects —
> preprocessing, not a change to the MDP. Every reported number is
> in dollars.

### Q: What are the contributions of your work?

> Four, in order of importance.

> 1. An empirical finding: on small balanced data, the value-based
>    off-policy method learns a state-dependent policy where the
>    policy-gradient methods collapse to fixed rules — six seeds out
>    of six against zero.
> 2. A diagnostic method: the mask-conditioned state-dependence
>    test, which separates genuinely learned behaviour from a fixed
>    rule that happens to profit. Returns alone cannot make that
>    distinction.
> 3. A controlled protocol: a regime-balanced simulator for
>    training, then a transfer test on 180 genuinely unseen real
>    months — Sharpe 0.224 against buy-and-hold's 0.198, holding
>    zero shares through October 2008.
> 4. A documented failure analysis: the first study's collapse
>    diagnosed as a data-model mismatch — three examples per
>    parameter — rather than an algorithm failure. That diagnosis is
>    reusable by anyone doing small-data RL.

### Q: What makes the value of this work, considering it didn't beat buy-and-hold? What's the point if it lost to buy-and-hold in dollars?

> SAY THIS FIRST, about twenty seconds:
>
> It did not beat buy-and-hold in dollars. I say that on the
> slide. The value was never beating the market. Buy-and-hold
> wins a twenty-year uptrend by construction — one fee, maximum
> exposure. The question I actually asked is: can an agent learn
> a policy that changes with market state, on realistic data, and
> how would you even know. Deep Q did, six seeds out of six. The
> other two collapsed to fixed habits that dollar returns would
> have hidden. That test is the contribution.

> Then, if they press "but it still lost":
>
> On the 180 unseen months the agent earned $60.28 a month
> against buy-and-hold's $80.24. On the risk-adjusted measure it
> is ahead — Sharpe 0.224 against 0.198 — because it held zero
> shares through October 2008 and buy-and-hold rode the whole
> fall. Giving up some dollars to avoid the crash is the point,
> not a failure.

> Three numbers show where that policy is worth something.
> Balanced data, equal up, down and flat months: buy-and-hold
> loses $60.29 an episode, the agent makes $126.94. Down months:
> the agent cuts the passive loss by about 85%, −$103.35 against
> −$710.06. Fees: buy-and-hold is already negative by 10 basis
> points on balanced data; the agent is still profitable at 50,
> ten times the study's cost.

> If beating buy-and-hold in dollars were the bar, the honest
> move would be to pick a different test window. That would be
> curve-fitting, not science. A passive rule wins the specific
> twenty years it was lucky to sit in. The work's value is that
> the decisions change with conditions, and I built the test that
> proves those decisions are real. That test, the protocol, and
> the first study's failure diagnosis still hold whichever way
> the market went.

### Q: How do you build the trading environment?

> A Gymnasium environment, about three hundred lines, and the same
> code path for every experiment.

> One episode is one calendar month of daily bars — 18 to 23 steps.
> Reset gives $10,000 cash and no shares. Each step the agent sees
> its state, the mask removes impossible actions, and Buy commits a
> quarter of starting capital at the current price with a 0.05% fee
> paid out of the amount; Sell is the mirror image. At month end any
> open position is liquidated, so nothing carries overnight.

> Two design details worth stating. First, market features are
> precomputed on the continuous series and PASSED IN — the
> environment cannot truncate a rolling window at an episode
> boundary because it never computes one. Second, the environment
> checks reward admissibility at construction: a drawdown penalty
> without the drawdown feature raises an error, so the MDP argument
> in Chapter 3 is enforced by code, not asserted in prose.

> The quarter slice is calibrated, not assumed: a tenth of capital
> needs ten steps to reach full exposure and caps mean exposure near
> 0.69 against buy-and-hold's 0.91 — under that setting no agent can
> win however well it trades, and reporting that failure would
> report arithmetic, not learning.

---

## Pattern 16 · "Straight off the slides"

### Q: Slide 6 — all three algorithms failed the first study. Why show a failure?

> Because the failure is the finding that motivates everything
> after. All three methods scored zero out of six on the
> state-dependence test — profitable sometimes, but by fixed rules,
> not by reading the market.

> The honest diagnosis — 60 episodes cannot constrain 18,000
> parameters — dictated both fixes in the main study: more data
> through simulation, fewer parameters through the smaller network.
> Hiding the failure would hide the method.

### Q: Slide 9 — how exactly is the simulator calibrated?

> From the 60 real months of 2018-2022. Each month is labelled up,
> down or flat by its return; a geometric Brownian motion is fitted
> per regime — one drift, one volatility each. Then 1,000 synthetic
> months per regime for training, with separate seed blocks for the
> validation and test splits, so train and test are independent
> draws from the same distribution.

> No real month outside 2018-2022 touches the calibration — that is
> what makes the 180-month transfer test genuinely unseen.

### Q: Slide 10 — the Sharpe gap looks small. Is 0.224 vs 0.198 significant?

> On 180 months, a Sharpe gap of that size is suggestive, not
> decisive, and I say so. The stronger evidence is behavioural: on
> the balanced test, where market direction is controlled, the agent
> cuts down-market losses by 85% while keeping 93% of up-market
> gains — and on the real transfer it holds zero shares through
> October 2008. The Sharpe number is the headline; the behaviour is
> the result.

### Q: Slide 11 — how did you choose λ = 0.25?

> On the validation split, never the test set. I swept λ, and 0.25
> gave the best trade-off: down-market losses cut from −$87.28 to
> −$23.26 — a 73% reduction — for a 1.5% cost to the mean (+$129.38
> against +$131.31). Validation Sharpe rose from 0.285 to 0.332.
> Larger λ kept cutting losses but started eating the up-market
> gains.

---

## Pattern 17 · "Joel's catches" — from the 13 Sep practice run

### Q: What is the Sharpe ratio? How do you calculate it?

> The Sharpe ratio is return per unit of risk: the average excess
> return over the risk-free rate, divided by the standard deviation
> of those returns.

> Sharpe = (R̄_p − R_f) / σ_p

> In this study: each month's return is the wealth change divided by
> the $10,000 starting stake; the Sharpe is the mean of those
> monthly returns over their standard deviation, with the risk-free
> rate set to zero, reported monthly and not annualised.

> Why it matters here: buy-and-hold earns MORE raw dollars on the
> transfer test, but it rides every crash, so its volatility is
> higher. The agent's 0.224 against buy-and-hold's 0.198 says the
> agent earned its returns with less risk per dollar — that is the
> risk-adjusted story, and it is the honest framing of the result.

### Q: What do you mean by "memorised from insufficient data"?

> Formally: over-fitting. The first study gave an 18,000-parameter
> network only 60 real training months — about three examples per
> parameter — so minimising training error does not constrain the
> network's behaviour on unseen months. The generalisation gap can
> be arbitrarily large.

> And the data was imbalanced as well as small: the 2018-2022
> window skews toward rising and flat markets. The cheapest fit to
> an imbalanced sample is a fixed rule — "always buy" — which looks
> like learning but is a lookup.

> The evidence it happened: zero of six seeds passed the
> state-dependence test in the first study. The policies were not
> reading the market; they had settled into fixed rules that the
> training window happened to reward.

### Q: The timeline slide shows 144 months — you keep saying 180. Which is it?

> Both, and the slide now says so: the transfer test is 180 unseen
> months in TWO grey bands — 144 months from 2006 to 2017, which
> includes the 2008 crash, plus 36 months from 2023 to 2025. The
> calibration window 2018-2022 sits between them and is excluded.
> 144 plus 36 is 180.

---

## Pattern 18 · "How does this sit in the literature?"

### Q: What did the literature use for methodology, data and networks — and why didn't you follow it?

> The reviewed studies vary widely, and that variety is itself a
> finding of Chapter 2.

> Methodology: Moody and Saffell trade single assets with direct
> reinforcement — gradients straight through the trading rule.
> Deng et al. combine that with deep recurrent networks. Jiang et
> al. learn continuous portfolio WEIGHTS over many assets with a
> CNN. Théate and Ernst — the closest study to mine — train a DQN
> per stock, allow short positions, and test on one continuous
> period against active strategies.

> Data: mostly long continuous price histories with a single
> chronological train/test split — and, in Jiang's case, intraday
> crypto data. Networks: substantially larger than mine —
> recurrent and convolutional architectures with tens of
> thousands of parameters and more.

> Why I did not follow them, three reasons.

> 1. The research question is different. They ask "how much can
>    an agent earn?" — I ask "does the learned policy actually
>    read its state?" That needs controlled market conditions and
>    a behavioural test, which none of the reviewed studies
>    perform: Théate and Ernst inspect trading trajectories, Guan
>    and Liu use feature attribution, and neither tells a learned
>    policy apart from a fixed rule that happens to profit.
> 2. The data budget is different. One asset's monthly history is
>    small; a recurrent network of their size on my 60 real
>    months is exactly the 18,000-parameter memorisation failure
>    my first study documented. The network must be sized to the
>    data, so mine is a 64→32 MLP with about 2,800 parameters.
> 3. Evaluation discipline. None of the reviewed studies describe
>    selecting every design decision on a validation period
>    before touching the test set, and results are often single
>    runs. I fixed every knob on validation, froze it, and report
>    six seeds with their spread.

> One honest consequence: my numbers cannot be compared directly
> with theirs — different benchmarks, different constraints. I
> state that in Chapter 2 rather than pretend a comparison.

### Q: Which methodology DID you emulate from the literature?

> Four things, deliberately.

> 1. From Théate and Ernst: the overall evaluation shape — a DQN
>    trading agent against a buy-and-hold benchmark with explicit
>    transaction costs on real equity data, including SPY. Their
>    observation that agents drift toward passive behaviour, and
>    their honesty about variance across identical training runs,
>    directly motivated my six-seed reporting and the
>    state-dependence test.
> 2. From Mnih et al.: the DQN mechanics exactly as published —
>    experience replay and a target network — implemented from
>    scratch so Chapter 3 can derive what my code runs.
> 3. From Williams and from Schulman et al.: textbook REINFORCE
>    and reference PPO (sb3-contrib MaskablePPO), so the
>    policy-gradient side is standard, not my invention. The
>    masking approach follows Huang and Ontañón.
> 4. From Moody and Saffell: the idea that the objective should
>    carry risk, not just return — their differential Sharpe is
>    the ancestor of my drawdown-penalised reward in the risk
>    experiments.

> So the components are all standard and traceable; the
> contribution is the controlled data and the behavioural
> evaluation wrapped around them.

---

## Pattern 19 · "Artefacts — what and where"

### Q: "What are artefacts in the first place — and where are yours?"

> An artefact is the tangible product of the project beyond the
> report — the software and data outputs an examiner can run or
> inspect. The dissertation is the argument; the artefacts are the
> evidence it points at.
>
> Mine live in one repository, described in Chapter 4:
>
> 1. The trading environment — a Gymnasium-style environment where
>    the reward, the accounting and the action mask live.
> 2. From-scratch REINFORCE and Deep Q-learning implementations,
>    plus a harness that trains all three methods — including the
>    library PPO — through one code path.
> 3. The calibrated market simulator and the feature pipeline.
> 4. The automated test suite — the accounting, causality and
>    masking guards listed in the appendix.
> 5. The results files. Every table and figure in Chapter 5 is
>    generated by script from those JSON files with fixed seeds, so
>    every number in the report can be regenerated end to end.
>
> If they want to SEE one: the demo page at the end of the deck runs
> the transfer-test result live, and I can walk through any module
> on screen.

---

## Pattern 20 · "Where is the reward function? Show me."

### Q: "Where is the reward function in the respective algorithms — and how is it used?"

> The reward is defined in exactly ONE place — the environment's
> step function — and in none of the algorithms. That is deliberate:
> all three methods are paid by the same rule, so they differ only
> in how they learn, never in what they are rewarded for.
>
> r_t = W_{t+1} − W_t ,   W_t = C_t + h_t P_t
>
> the day's change in portfolio wealth, net of the transaction fee.
> The risk-aware variant adds a drawdown penalty:
>
> r_t^{risk} = ΔW_t − λ C_0 max(0, dd_{t+1} − dd_t) ,   λ = 0.25
>
> Each algorithm then consumes the same number differently:
>
> 1. REINFORCE discounts the month's rewards into returns G_t,
>    standardises them, and weights the action log-probabilities.
> 2. Deep Q-learning stores (s, a, r, s') in the replay buffer; the
>    reward enters the TD target y = r + γ max Q over legal actions.
> 3. PPO turns the rewards into GAE advantages Â_t inside the
>    clipped objective.
>
> One detail worth offering: the code REFUSES a risk-penalised
> reward unless drawdown is in the state — otherwise the reward is
> no longer a function of the state and the process stops being an
> MDP. The validity argument in Chapter 3 is enforced by the code,
> not just asserted in prose.

---

## Pattern 21 · "PPO's entropy bonus, and the dying gradient"

### Q: "Didn't you use an entropy bonus for PPO? Isn't that the standard implementation?"

> No — and I can say precisely why. I used the sb3-contrib
> MaskablePPO implementation with the library defaults, changing
> only the network size, and Stable-Baselines3's default entropy
> coefficient is zero. "Standard" depends on the domain: the
> original PPO paper sets it to 0.01 on Atari but zero on the
> continuous-control benchmarks. So no entropy bonus was active.
>
> The bonus adds c · H(π_θ) to the objective, where
>
> H(π_θ) = − Σ_a π_θ(a|s) log π_θ(a|s)
>
> Its gradient pushes the policy toward uniform and never vanishes,
> so it is exactly the kind of change that could have kept PPO
> exploring here. This connects to a limitation Chapter 6 states
> plainly: the policy methods were not separately tuned — validation
> choices were made with Deep Q-learning. Whether an entropy bonus
> rescues PPO on this environment is a testable question, and it is
> future work, not a claim.

### Q: "If PPO is more stable than REINFORCE, why was it the WORST result in the main study? Shouldn't restricting the update help here?"

> Because stability and exploration are different axes, and this
> environment punishes lost exploration, not instability.
>
> The clip limits how far one update can move the policy. That
> protects against a catastrophic jump — but it does nothing to
> restore exploration once the policy is already near-deterministic.
> On balanced data, never trading is the locally safe habit: it
> earns nothing but cannot lose. Once PPO drifts near that habit,
> the trust region holds it there. The same mechanism that makes PPO
> safe makes it commit to its first bad habit.
>
> The seeds say exactly this. Five of six PPO seeds ended at $0.00 —
> never trading — and the sixth at minus $66.99, a fixed always-buy
> habit. REINFORCE's noisier updates occasionally kick a seed out of
> the trap: one of its six escaped, read its state, and earned
> $83.13 per episode. No PPO seed escaped. The averages — minus
> $5.71 for REINFORCE, minus $11.16 for PPO — follow from that.

### Q: "How do you preserve the gradient signal? Is there a way to stop it decaying?"

> First, why it decays. The policy-gradient estimator weights
> ∇ log π by the return:
>
> ∇_θ J(θ) = 𝔼_{τ ~ π_θ} [ Σ_t ∇_θ log π_θ(a_t | s_t) · G_t ]
>
> As the policy becomes near-certain, two things die at once: the
> log-probability gradient of the chosen action goes to zero, and
> the alternatives stop being sampled, so nothing generates contrast
> in the returns. The estimator still runs — it just carries almost
> no signal.
>
> Four known remedies, none of which I used:
>
> 1. Entropy regularisation — a bonus term whose gradient pushes
>    toward uniform and never vanishes.
> 2. A probability floor — an epsilon-soft policy that must give
>    every action some minimum probability. That is the
>    policy-gradient analogue of epsilon-greedy.
> 3. Off-policy exploration data with importance weighting — let a
>    more exploratory behaviour policy collect the trajectories.
> 4. Reshape the reward so the extremes stop being locally optimal —
>    a small penalty for never trading, the same way balancing the
>    data removed the always-buy shortcut. Chapter 6 lists this
>    under better reward functions.
>
> I kept the algorithms exactly as Chapter 3 derives them, so the
> comparison measures the core mechanisms rather than my tuning.
> Deep Q-learning's epsilon-greedy rule is the built-in version of
> remedy two — which is why it never collapsed.

---

## Pattern 22 · "Not on this board" — the unseen-question protocol

### Q: The question is NOT on this board — what do I do?

> Four moves, in order. This is the whole protocol.
>
> 1. REPEAT the question back in your own words — "So you are asking
>    whether…". It buys ten seconds, confirms you understood, and
>    feeds the board's matcher a second chance to find a neighbour.
>
> 2. ANCHOR it to the nearest thing you know cold. Every question in
>    this viva reduces to one of six anchors: the state (9 features),
>    the reward (ΔW in dollars), the data (60 real vs 3,000 balanced),
>    the three algorithms' mechanisms, the seeds (6), or the two
>    studies' results. Name the anchor out loud — "that touches the
>    reward design" — and walk from there.
>
> 3. GIVE one concrete instance: October 2008 and the flat Deep Q
>    line; seed 45, the one REINFORCE escape; λ = 0.25 cutting the
>    down-market loss 73%. A specific number beats a general claim.
>
> 4. CLOSE honestly if it is beyond the study: "The study did not
>    test that. To answer it properly I would…" — and name the
>    experiment. That is a strong answer, not a weak one; it is the
>    answer the examiners themselves would give.
>
> NEVER invent a number. If it is not in the dissertation's tables
> or on this board, say you would check. One made-up figure costs
> more than ten honest "I'd need to verify" answers.

---

## Final rehearsal checklist (do these once)

1. Read every equation above out loud. If any of them makes you
   pause, memorise it separately.
2. Say the elevator pitch (in `prompter.md`) from memory. Under
   30 seconds.
3. Say the answers to Patterns 3, 5, 8 without looking. These are
   the three Nguyen returns to most.
4. Rehearse ONE full pass through slides 1-12 with a timer, aiming
   for 13-15 minutes. Do NOT rush.
5. Rehearse the panic phrases from `prompter.md` — say them out
   loud so they feel natural when you need them.
