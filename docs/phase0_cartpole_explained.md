# Phase 0 explained — CartPole PPO (for viva / Nguyen)

CartPole is the “robot in a room” stand-in: known rules, known success.
If this learns, our **policy** training loop is trustworthy before trading.

## MDP pieces (say these out loud)

| Piece | CartPole | Maps to trading Iteration 1 |
|---|---|---|
| State | cart pos, cart vel, pole angle, pole ang. vel | `s = [ΔP, C, n]`  (price change, cash, shares) |
| Action | push left / push right | Hold / Buy one / Sell one |
| Reward | +1 each step pole stays up | `r = W_next − W`  (wealth change) |
| Episode | until pole falls (or 500 cap) | one trading day (T bars) |
| Policy | `π_θ(a \| s)` via PPO | same algorithm family |

Wealth reminder:

```text
W  =  C  +  n · P
# wealth = cash + (number of shares × price)
```

## Four beats (Nguyen’s board → this script)

1. **Predict** — PPO policy network outputs action probabilities from state.
2. **Score** — clipped PPO objective + value loss (not MSE to a label `y`).
3. **Differentiate** — autograd / `loss.backward()`.
4. **Update** — Adam step on θ (and critic φ).

## Why not Q-learning here?

Nguyen: learn a **policy**. PPO directly parameterises `π_θ`.
Q-learning learns action-values first; different story for the viva.
(We still *ran* Q-learning on binned CartPole as a comparator — see the boards doc.)

## Success criterion

PPO mean return ≫ random mean return on the same eval episodes.
Learning curve should trend upward (see `reports/generated/charts/phase0_cartpole_learning_curve.png`).
