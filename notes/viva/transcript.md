# Viva read-along transcript

Your own spoken words, from your rehearsal recordings, cleaned up so you can
read them out exactly as written. Anything I had to correct from the tape is
flagged in a CAREFUL block under that slide — read those once now, so the fix
doesn't surprise you mid-viva.

Status: complete — all 11 slides are from your recordings.

Timings assume a calm reading pace (~130 words per minute).
The full read is roughly 17–18 minutes against the 20-minute limit. If a
rehearsal runs long, trim from the contributions detail on slide 11 first.

---

## Slide 1 · Title (~15 sec)

Good morning, professors. Today I'll be presenting my dissertation, which is
an evaluation of reinforcement learning algorithms for portfolio management —
the applicable field here being finance: trading and stock portfolios.

---

## Slide 2 · The problem (~55 sec)

So what exactly is the problem, or the research question?

A regular person who invests in stocks can decide on a day-to-day basis
whether to buy more shares of that stock, leave it and do nothing, or sell —
depending on a number of reasons. Let's say the market is declining and you
want to protect your money, or you just need cash at hand for an emergency.

This leads to the main research question: can a reinforcement learning
algorithm learn a strategy that is overall better than simply buying and
holding an asset for the long term?

Performance here can be measured by how much more profitable the algorithm is
compared to just buying and holding — or by whether the learned strategy
reduces its exposure to the asset during a market decline, limiting the losses
in that period.

> **CAREFUL — changed from your recording.** You said reducing exposure in a
> decline means "maximising the most profits at that period". In a decline the
> honest claim is *limiting losses*, not maximising profit. Reworded.

---

## Slide 3 · The trading environment (~70 sec)

To understand the trading environment, we first need to understand
reinforcement learning. Unlike supervised learning, where a model is given
true labels to predict from a dataset, reinforcement learning learns by trial
and error — from the rewards it gets for taking a specific action in its
current state.

Now, I've said a lot of things there, so let me break it down into simpler
terms. We have an agent and an environment. The agent is a small neural
network — in our case a policy network, or a Q-network — which outputs an
action — buy, hold, or sell — for its current state.

The environment is the market and the portfolio together, and the agent
interacts with it in episodes. An episode is a sequence of trading days —
the trajectory, in more technical terms. In this case an episode represents
one month of trading, which is typically 18 to 23 trading days depending on
the month. So over an episode, the environment feeds the agent the daily
price of the asset as it moves, and at the end of the month it reports the
total change in portfolio wealth.

> **CAREFUL — changed from your recording.** Three things. You said the
> environment watches the daily price "increase" — prices also fall, so it
> now says "move". You said "the environment represents the episode" — the
> environment is the market plus the portfolio; the episode is one month of
> interaction with it. Saying they're the same invites a definition question
> from Nguyen. And the dissertation says a month is **18 to 23** trading
> days, not 16 to 23.

---

## Slide 4 · The three algorithms (~75 sec)

Moving on to the reinforcement learning algorithms. There are two families
here: the value-based methods, like Deep Q-learning, and the policy-based
methods, where REINFORCE and PPO fall.

Deep Q-learning learns the value of an action. In essence, it estimates how
good each action is, and selects the highest estimate. So the agent is
asking: how good is buying in this scenario? How good is selling? And it
picks whichever value is highest.

REINFORCE, on the other hand, learns the policy directly. The policy is a
probability distribution over the actions — say, a probability of 0.6 for
buying, 0.3 for selling, and 0.1 for holding. After each episode it adjusts
those probabilities based on the outcome that followed: actions that led to
good outcomes become more likely, and actions that led to poor outcomes
become less likely. The outcome here is the reward — in our case, the change
in the portfolio's wealth. So the more profitable an action turns out to be,
the higher its probability becomes, and the less profitable, the lower.

PPO is like REINFORCE, but it learns the policy in a safer manner. It has a
clipping parameter that limits how large any single update to the policy can
be, so the new policy never strays too far from the old one.

> **CAREFUL — changed from your recording.** You said REINFORCE "selects which
> probability gives the highest reward". REINFORCE does not select a maximum —
> it *shifts probabilities* toward actions that led to better outcomes. The
> wording above is safe if Nguyen pushes on the mechanics.

---

## Slide 5 · Dataset splits (~80 sec)

Now for the dataset splits. The first study used real data from the SPY index,
to be as realistic as possible. This didn't turn out great — which led to the
main study we'll come back to.

We used 2018 to 2022 as the training period: 60 months. This is what the
agents were trained on, and later what the simulator was calibrated on. For
validation we used the whole of 2023. This is where hyperparameters were
fine-tuned into one fixed setting used for all tests, so the comparisons stay
equal along the way. And the test period was 2024 to 2025 — which also had
issues we'll come to.

Because of these dataset constraints, we then moved to a simulator and a
transfer test. From the same 2018-to-2022 training months we generated 3,000
episodes across different market conditions: 1,000 for rising markets, 1,000
for falling markets, and 1,000 for consolidation, where prices move within
roughly the same range.

Transfer testing was then done on 180 real months the agents had never seen:
2006 to 2017 — which covers the 2008 market crash — plus 2023 to 2025. Those
are the two grey bands on the timeline: 144 months before 2018, and 36 after
2022 — 180 in total. All of this is to prevent look-ahead leakage, and to
stop the agent memorising patterns.

> **CAREFUL — changed from your recording.** Two slips on tape: you said the
> transfer months run "from 2008" (they run from **2006**; 2006–2017 plus
> 2023–2025 is what makes 180 months), and you once said the training period
> was "2020 to 2022" (it is **2018 to 2022**).

---

## Slide 6 · First study result (~90 sec)

So for the first study, trained on the 60 real months: no agent actually beat
the simple buy-and-hold benchmark. If you had bought at the beginning of
January 2024 and held to the end of 2025, you would have earned an average of
$180.25 per month. The closest learned agent was REINFORCE at $171.51, then
PPO at $127.35, and Deep Q-learning at the bottom with $74.68 per month.

At first glance you could say REINFORCE was nearly efficient. But looking
closer at the seed distribution, we noticed that REINFORCE and PPO — the
policy methods — weren't state-dependent. So what do we mean by
state-dependent? It's a behavioural check on whether the agent actually uses
the information from its current state to change its decisions. In
reinforcement learning, every agent is given the state information — the
question is how much that information actually drives its decision-making.

A quick word on seeds: a seed is the random starting point of a training
run. We trained each algorithm six separate times, with six different
seeds — six independent repeats of the same experiment. None of those six
runs showed REINFORCE or PPO acting on the information provided by their
states: they mostly moved to always buying, or never trading at all — the
two extremes.

Part of the blame is the testing period itself: 2024 to 2025 was a strong
rising market, so always-buy was close to the best possible fixed pattern.
But there was also a deeper underlying cause — overfitting — which I'll show
on the next slide.

> **CAREFUL — changed from your recording.** Your latest take said "the
> closest was PPO at 171.51". **$171.51 is REINFORCE**, and REINFORCE was the
> closest; PPO earned $127.35. Your earlier take had the order right. Also
> kept Nguyen's grammar rule: "acting on the information provided by their
> states", never "ignored its state".

---

## Slide 7 · The diagnosis (~90 sec)

So why did this happen? In the first study we used a multi-layer perceptron
with two hidden layers of 128 units each — around 18,000 trainable
parameters — trained on only 60 data points. With far more trainable weights
than training months, the network can memorise all 60 months perfectly.
Nothing forces it to learn a rule that works in a different market condition.
It's like a student preparing for an exam using the demo questions, then
getting to the exam and seeing completely different questions.

What added to this is that the training months were imbalanced: 29 of the 60
were rising markets, and only 16 were falling markets — like the COVID crash
in 2020. That imbalance constantly teaches the agent to always buy, since
that returns the highest reward for the policy methods. Coupled with
memorisation from insufficient data, the policies never needed to rely on the
information provided by the state — even though the state contains multiple
features that describe the market and the portfolio. And it explains why the
policy methods appeared to do well in the test period, which was also a
strongly rising market.

Deep Q-learning, on the other hand, came lowest because estimating the worth
of an action needs many similar situations to average over. Sixty months that
never repeat, without balanced market conditions, make its value estimates
very noisy.

> **CAREFUL — changed from your recording.** Three slips on tape: the hidden
> layers are **128** units (you said 120), the training set is **60** data
> points (transcription heard "16"), and **16** of the 60 months were falling
> (you said 19).

---

## Slide 8 · The main study (~75 sec)

After understanding the limitations of the first study — and accepting the
corrections from my supervisor, thank you, Doctor — I built a small market
simulator that generates monthly prices for the main study. It is still
calibrated on the same training period, 2018 to 2022 — deliberately, because
the simulator should only ever see the data the agents were already allowed
to see. That keeps the validation and test years untouched, and it keeps the
comparison with the first study fair: same information, more of it.

The good thing is that we can now recreate the different market conditions
ourselves. And the training window gives the simulator real examples of each
condition to calibrate from: the steady rises of 2019 and 2021, the COVID
crash and its sharp recovery in 2020, and the 2022 decline.

So we created 3,000 balanced training episodes: 1,000 per condition, for the
rising market, the falling market, and the consolidation market where prices
move side to side.

Now that we had substantial volume in the data, the other thing to fix was
the network. We reduced the MLP's hidden layers to 64 and 32 units — about
2,800 parameters in total — which sits comfortably with the 3,000 training
episodes: roughly 20 training examples per weight.

And from the simulator we don't just get volume — we get labelled market
conditions, so we can distinguish rising from falling markets and evaluate
how the agent acts in each. With these changes came very different results,
and different behaviours — which is the next slide.

> **CAREFUL — changed from your recording.** You said "the same training
> period from 2018 to 2012" — it is 2018 to **2022**. Transcription also heard
> "labour market conditions"; it now reads "labelled market conditions".
> On the noteworthy periods you suggested: there was no "2022 recovery" —
> **2022 was a decline** (SPY fell roughly 19%), and 2018 ended slightly down
> after a late-year drop. The safe examples are the ones now in the text:
> rises in 2019 and 2021, the 2020 COVID crash and recovery, the 2022 decline.

---

## Slide 9 · The result flips (~150 sec)

The main result shows that once we balance the data, the always-buy strategy
can't work the way it did before. The market is no longer constantly rising —
there is a mix of different market conditions. The agent now has to learn a
policy that works in each of these conditions, and that forces it to rely on
the information provided by the state.

The chart on the left shows the average earned in each market condition —
rising, falling and consolidating — for every method. In this simulated
held-out test set, Deep Q-learning performed well in all market conditions.
In rising markets it kept $473.72 of the $477.22 that the
benchmark always-buy strategy earns — that is 99% of the gains. And in
falling markets it lost $103.35 where always-buy lost $663.91 — an 84%
reduction in losses.

The chart on the right shows all 600 test episodes overall. Deep Q-learning
was the only method that landed a stable policy: it earns $126.94 per
episode, across all six seeds. Because this is simulated, balanced data —
with no constantly rising market — the fixed patterns the policy methods
developed would lose to falling markets, to consolidations, or to
transaction fees: the cost of making a trade.

As for REINFORCE and PPO ending in negative values — minus $5.71 and minus
$11.16 per episode — they could not develop a fixed strategy suitable for
all markets, which caused a collapse. On balanced data, even the benchmark —
always-buy — loses money. So never trading becomes the locally safe answer:
it earns nothing, but it cannot lose. And once a policy-gradient method
drifts close to either extreme — always buying or never trading — its
gradient carries no signal. It stops exploring, and it gets stuck there.
Deep Q-learning doesn't have this failure mode, because its epsilon-greedy
exploration keeps trying every action a fraction of the time, no matter what
the current policy prefers.

It is noteworthy that one of the six REINFORCE seeds actually did rely on
information from its state, gaining $83.13 per episode for that seed alone.
But with five of the six collapsing, we can't consider that a reliable
result.

> **CAREFUL — changed from your recording.** Transcription garble cleaned:
> "epsilon gradient" is **epsilon-greedy**, "always mine" is always-buy, and
> the conditioned REINFORCE seed earned **$83.13**. All the numbers you spoke
> were verified against Table 5.3 and are correct: $473.72 of $477.22 (99%),
> $103.35 vs $663.91 (84% less), $126.94 overall.

---

## Slide 10 · The transfer test (~140 sec)

For the final transfer test, I took the policies trained on the simulator
and ran them, unchanged, on 180 real months that the training never saw.
That is 2006 to 2017, plus 2023 to 2025 — everything outside the
2018-to-2022 window that was used to calibrate the simulator.

The left chart is the actual real SPY data, showing all the algorithms. Note
that buy-and-hold still ends the highest by 2025: we couldn't beat the
market, as it rose almost constantly over the span of two decades. On the
right side, Deep Q-learning had the highest average of the learned methods
at $60.28 per month, followed by REINFORCE at $31.66 and PPO at $8.32.
Buy-and-hold still gave the highest return, at $80.24 a month. The
policy-gradient methods are weaker here because most of their seeds
collapsed to fixed patterns.

Coming back to the main graph, we notice an unexpected behaviour in Deep
Q-learning. During the 2008 financial crisis, buy-and-hold lost $1,665 in
October alone — you can see the line plunge downwards. But all six of our
Deep Q seeds held zero shares through that period, so their line barely
dips — through the shaded band it actually sits highest. The agent reduced
its losses by itself. It was never given this data at all, not even in the
first study. So a safe assumption is that the volatility features it
observed matched the falling condition it had learned, and it stepped aside
for that period. In turn it gave up some market momentum and didn't gain
profit — but it avoided most of the crash, unlike REINFORCE and PPO.

> **CAREFUL — changed from your recording.** Three fixes that matter in front
> of examiners. You said "the Great Depression of 2008" — the Great
> Depression was the 1930s; it is now "the **2008 financial crisis**".
> Buy-and-hold's October 2008 loss is **$1,665**, not 1,655. And the
> calibration window is **2018 to 2022** — on tape you said "2008 to 2022".

---

## Slide 11 · Conclusions (~220 sec)

To conclude — we found some interesting points over the course of this
dissertation.

First, the findings. The algorithms were never the problem; the data was.
As with any deep learning method, data is essential to a decent result:
60 real months could never teach an 18,000-weight network, but 3,000
balanced simulated episodes could, to a significant degree.

Next, the algorithms behaved very differently, and that traces back to
their core mechanisms. REINFORCE and PPO, the policy methods, drifted to
fixed habits — always buying, or never trading — because once a policy
becomes near-certain, its gradient carries almost no signal, and it stops
exploring. Deep Q-learning, on the other hand, kept exploring through its
epsilon-greedy rule, and it was the only method that learned a policy that
changes with the state — on all six seeds.

And that learned behaviour carried into real markets: run unchanged on 180
unseen months, it stayed profitable, and it stepped aside through the
October 2008 crash.

What does this work contribute to the body of literature? First, a careful
way to evaluate. Every decision was calibrated on validation data alone —
the trade size, the cost of trading, the risk coefficient. We used six
seeds rather than relying on one result, and automated tests guarded the
accounting, combined with action masking that prevents unavailable actions
from being executed.

Second, a behavioural test, which tells whether a policy genuinely reads
the market information provided to it, apart from executing a fixed habit
the way REINFORCE and PPO did. This changed how every result was read:
without it, we would have read REINFORCE as very profitable, not knowing it
was overfitting to rising markets. The test doesn't tell us which piece of
information was used — volatility, moving average, or true range — and that
goes to future work.

Lastly, a calibrated simulator, which gives us volume, control and balance
in our data. We can replicate multiple market conditions, and test exactly
where needed.

For our limitations: we studied one asset only — SPY, which tracks the
S&P 500 — so the findings do not automatically carry to other markets. We
used a deliberately simple simulator: months are categorised as rising,
falling or consolidating, with no market change inside a month. The
transfer test shows the cost of that simplicity — the gap between the
simulated results and the real-month results is the price we paid for it.
The policy methods were also not separately tuned: validation choices were
made with Deep Q-learning, the method most responsive to its state, so we
cannot tell whether the policy methods would have done better with their
own tuning. And the evidence is modest — six seeds, with no formal
significance tests.

For future work: more seeds, with confidence intervals, and other assets,
for stronger evidence. Better simulators, that represent realistic market
change rather than fixed categories — and continuous months: real trading
is continuous, months carry over into each other, and we treated each month
independently to keep the modelling simple. And better reward functions, so
that an agent cannot benefit from never trading — the same way balancing
the data stopped the always-buy habit.

So, coming back to the research question: can reinforcement learning learn
a strategy that beats buy-and-hold? On raw returns, on one asset, it did
not. But given the right data, it learned something buy-and-hold can never
do: to read the market's condition, and step aside when it turns.

What these methods learn is decided by the data they are given.

Thank you. I'm happy to take your questions.

> **CAREFUL — gaps filled and fixes from your recording.** The "dash dash
> dash" gaps are filled with the mechanism: policy methods stall because a
> near-certain policy has no gradient signal; Deep Q keeps exploring via
> **epsilon-greedy**. The outro you asked for is the last four paragraphs —
> it answers the research question directly before the one-line takeaway.
> Transcription fixes: "60 Romans" = 60 real months, "rainforest" =
> REINFORCE, "work forward through calibration" = walk-forward
> recalibration, "gained by refusing to trade" = **gamed**. The asset: say
> "SPY, which tracks the S&P 500" (on tape you said "SBI"). I dropped
> "trading grows exponentially" — hard to defend. And "the cost of
> simplicity" you queried now has its plain meaning in the text: the gap
> between simulated and real results is the price of the simple simulator.
