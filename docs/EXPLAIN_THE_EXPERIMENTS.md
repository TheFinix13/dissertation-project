# How to explain the whole experiment to Nguyen (plain English)
Last updated: 2026-08-05 · branch `simple-modelling-clean`

Read this out loud once. If you can tell this story without notes, you are ready.

---

## The one-sentence story

> We first proved our reinforcement-learning code works on simple games with
> known answers (Phase 0). Then we used the same learning method on a tiny
> one-stock trading model (Phase 1). The honest finding so far: with a plain
> “make more money” reward, the agent either copies buy-and-hold or sits in
> cash — so the next step is to change the reward, not just add more features.

That is the whole dissertation track in three sentences.

---

## Why games before stocks? (Nguyen’s homework)

Trading is hard to debug: if the agent loses money, you cannot tell whether
**the code is wrong** or **the market is hard**.

A game has a known correct outcome (“keep the pole up”, “don’t hit the pipe”).
If the agent fails the game, the *implementation* is broken. If it passes,
we trust the training loop enough to move to stocks.

That is Phase 0.

---

## What is CartPole? (say this in 30 seconds)

Imagine a cart on a track with a pole balanced on top, like balancing a broom
on your hand.

- **What the agent sees (state):** 4 numbers — where the cart is, how fast it
  moves, how tilted the pole is, how fast the tilt is changing.
- **What it can do (actions):** push the cart left, or push right.
- **Reward:** +1 every moment the pole stays up. Episode ends when it falls
  (or hits a 500-step cap).
- **“Solved”:** average return about **475–500**. Random play gets ~**29**.

We used CartPole because:
1. episodes are short (train in minutes, not days);
2. everyone in RL uses it as a sanity check;
3. continuous numbers in / discrete choices out — same *shape* as our trading
   actions (Hold / Buy / Sell), without market noise.

**Mapping to trading:** pole angle ≈ “how the market just moved”; push left/right
≈ Hold/Buy/Sell; +1 per step ≈ wealth change; one game ≈ one trading period.

---

## What is Flappy Bird? (and LunarLander)

Same idea, harder games — to show the algorithms still work when the task
gets messier.

| Game | In one line | Why we added it |
|---|---|---|
| **Flappy Bird** | Bird must flap or fall; dodge pipes | Sparse “pipe passed” rewards — harder exploration than CartPole |
| **LunarLander** | Land a craft with 4 thruster actions | Closest game to “managing a position” with several discrete choices |

If someone asks “why not only CartPole?”: Nguyen asked for a game first;
Flappy is the side-scroller he named; LunarLander is the third check that
multi-action control still learns.

---

## The algorithms — families, not jargon soup

There are two families. Nguyen locked us onto the **policy** family for trading.

### Family A — Value-based (“how good is each action?”)
1. **Tabular Q-learning** — a spreadsheet of scores for every (state, action).
   Update rule: move the score toward “reward + best future score”.
   Problem: CartPole’s state is continuous, so we had to **bin** it into boxes
   first. Trading cash/prices would need infinite boxes → tables don’t scale.
2. **DQN** — same idea, but a neural net replaces the spreadsheet.

### Family B — Policy-based (“what should I do?”) ← our main path
3. **REINFORCE** — directly learns probabilities of actions. After an episode,
   increase probability of actions that led to high total reward.
   We coded this **from scratch** (not a black-box library) so you can say
   you understand the update.
4. **A2C** — same idea + a second network (critic) that estimates “how good is
   this situation?”, which reduces noise in the updates.
5. **PPO** — A2C’s big brother: it **clips** how much the policy can change in
   one update so training doesn’t suddenly collapse. This is what we take into
   trading.

**Random** is not an algorithm — it is the floor. Everything must beat random.

### Phase 0 results (what to say)

| | CartPole | Flappy | LunarLander |
|---|---:|---:|---:|
| Random | 28.8 | −7.4 | −183 |
| REINFORCE | 292 | 7.1 | 15 |
| A2C | **500** | 4.6 | −41 |
| PPO | **500** | **12.6** | 176 |
| DQN | **500** | 7.0 | 178 |

Viva line: “All learners beat random on CartPole; PPO was the most consistent
across harder games, so we used PPO for trading.”

---

## Phase 1 — the trading experiment (Iteration 1 → 2)

### The toy market model (Iteration 1)

One stock (SPY). Each step the agent may:

- **Hold**, **Buy one share**, or **Sell one share** (illegal moves blocked).

State (3 numbers): recent price change, cash, number of shares.  
Reward: change in wealth \(W = \text{cash} + \text{shares}×\text{price}\).  
Episode: one calendar month of daily prices (honest: not yet 390 minutes).

Baselines (each has a clear objective):
- **B0** do nothing  
- **B1** buy one share and hold  
- **B1b** invest as much cash as possible and hold ← the *fair* buy-and-hold  
- **B2** random legal actions  
- **A1** PPO

### Honest Iteration-1 finding
PPO’s mean wealth change matched **B1b exactly** and produced the **same
action sequence on all 26 test months**. Under “just maximise ΔW”, it
rediscovered fully invested buy-and-hold. That is not a failure of the
experiment — it is the scientific result. Without B1b we would have falsely
claimed a win against the weak one-share baseline.

### Iteration 2 — richer state
We added: unrealized profit/loss, time through the month, and scaled cash /
position value (5 numbers). Same reward, same data split, same PPO budget.

Result: it **stopped** copying buy-and-hold, but became **mostly flat**
(almost never trades under greedy evaluation). So richer state only moved us
from “always invested” to “almost never invested” — both extremes. It did
**not** create smart timing.

### What that means for Iteration 3 (next)
Change the **reward**, e.g. wealth change minus a drawdown penalty. That gives
a reason to be invested sometimes and flat sometimes. State alone was not enough.

---

## How you would walk Nguyen through it (5 minutes)

1. **Board story:** ML = predict → score → update. RL is the same without labels;
   reward replaces the label.
2. **Phase 0:** “We verified the loop on CartPole (pole balance), then Flappy and
   LunarLander. PPO solved / nearly solved them; scratch REINFORCE also learns.”
3. **Show one chart:** CartPole comparison bar — random low, PPO at 500.
4. **Phase 1 model:** state cash+shares (not a vague Hold bit), discrete actions,
   ΔW reward, mask for illegal trades.
5. **Honest table:** Iteration 1 PPO = buy-max; Iteration 2 mostly flat.
6. **Ask him:** “Next lever — drawdown-penalised reward, or minute bars first?”

---

## Questions he may ask — short answers

**Q: Why not Q-learning for stocks?**  
A: Tables need discrete states; prices/cash are continuous. Nguyen also asked
for a *policy*. We showed Q-learning only on binned CartPole as a contrast.

**Q: Why is PPO better than REINFORCE?**  
A: REINFORCE is noisy; PPO clips updates and uses a critic, so it is more stable.
Our Flappy/Lunar numbers show PPO more consistent.

**Q: Did the trading agent beat the market?**  
A: Not yet in a meaningful way. Iteration 1 copied buy-and-hold; Iteration 2
avoided the market. That is the finding we are analysing, not hiding.

**Q: Why monthly daily bars not 390 minutes?**  
A: Honest interim. Same code works for minute bars; we state the gap and will
upgrade after the reward iteration.

---

## Where to point if he asks for files

| Thing | Path |
|---|---|
| This explanation | `docs/EXPLAIN_THE_EXPERIMENTS.md` |
| CartPole one-pager | `docs/phase0_cartpole_explained.md` |
| Games + algorithms | `docs/phase0_games_explained.md` |
| Meeting memo | `docs/nguyen_meeting_memo_jul30.md` |
| Ch4 / Ch5 drafts | `latex/simple_modelling/Chapter_4_Implementation.docx`, `Chapter_5_Empirical_Results.docx` |
| CartPole chart | `reports/generated/charts/phase0_cartpole_comparison.png` |
| Trading chart | `reports/generated/charts/phase1_spy_daily_delta_w.png` |
