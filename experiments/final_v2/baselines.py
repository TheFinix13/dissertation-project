"""Reference strategies, including a corrected buy-and-hold.

STUDY: shared — the benchmarks both studies are measured against.

The defect being fixed
----------------------
The earlier "buy-and-hold" baseline was a policy that bought one slice whenever
a Buy was legal, inside an environment whose slice was a tenth of capital. That
policy needs ten steps to become fully invested, so across a twenty-day month it
holds an average exposure near 0.75 rather than 1.0. It is a ten-day
dollar-cost-average, not buy-and-hold, and in a market that drifts upward it is
systematically weaker than the strategy it was named after. Every agent measured
against it was therefore flattered.

Two references are now reported and they answer different questions.

``B1a_slice_bah``
    Buy a slice whenever legal, at the agent's own granularity. This is the
    ceiling for a policy restricted to the same action set, so it isolates skill
    from action-space advantage.

``B1b_true_bah``
    Fully invested at the first bar and held to forced liquidation, implemented
    as the same policy in an environment whose slice is all of capital. This is
    the benchmark an investor would actually compare against.

Reporting both is the honest choice: beating B1a means the agent times better
than a mechanical schedule with identical powers, while beating B1b means it
beats the market.
"""
from __future__ import annotations

import numpy as np

# ------------------------------------------------------------------ policies


def policy_hold(obs, mask) -> int:
    """B0: never trade. Preserves cash exactly; anything below this destroys value."""
    return 0


def policy_buy_when_legal(obs, mask) -> int:
    """Buy whenever a Buy is feasible, otherwise hold.

    In a slice environment this dollar-cost-averages in; in an environment whose
    slice is all of capital it invests fully at the first bar and then holds,
    because no cash remains for a second Buy.
    """
    return 1 if mask[1] else 0


class StopLossPolicy:
    """Post-viva baseline: buy-and-hold protected by a stop-loss rule.

    Enter fully at the first legal Buy (run it in an environment whose slice
    is all of capital, like ``B1b_true_bah``). Exit the whole position when
    the open position falls `threshold` below its entry price (fixed stop)
    or below its running peak since entry (trailing stop), then stay in cash
    until the episode's forced liquidation.

    The rule reads only two observation features: ``pnl`` for the position
    and ``clock`` to detect a fresh episode (``clock`` is zero only at
    reset, so the policy re-arms itself without needing a reset call). Both
    are in the S3 state, so the rule sees strictly less than the agents do.

    This is the "a simple rule can cut crash losses without AI" benchmark
    Prof Nikitopoulos asked for in the viva (14 Sep 2026). It is not in the
    BASELINES registry so the dissertation's committed baseline set stays
    exactly as reported; `run_stop_loss.py` evaluates it separately.
    """

    def __init__(self, feature_names, *, threshold: float, trailing: bool):
        names = list(feature_names)
        for needed in ("clock", "pnl", "exposure_frac"):
            if needed not in names:
                raise ValueError(f"StopLossPolicy needs {needed!r} in the state")
        self._i_clock = names.index("clock")
        self._i_pnl = names.index("pnl")
        self._i_expo = names.index("exposure_frac")
        self.threshold = float(threshold)
        self.trailing = bool(trailing)
        self._stopped = False
        self._peak_pnl = 0.0

    def __call__(self, obs, mask) -> int:
        if float(obs[self._i_clock]) == 0.0:  # new episode: re-arm
            self._stopped = False
            self._peak_pnl = 0.0

        if self._stopped:
            return 0

        exposure = float(obs[self._i_expo])
        if exposure <= 0.0:
            return 1 if mask[1] else 0  # enter at the first legal Buy

        pnl = float(obs[self._i_pnl])
        if self.trailing:
            self._peak_pnl = max(self._peak_pnl, pnl)
            drop = (1.0 + pnl) / (1.0 + self._peak_pnl) - 1.0
        else:
            drop = pnl
        if drop <= -self.threshold and mask[2]:
            self._stopped = True
            return 2
        return 0


def make_random_policy(seed: int = 0):
    """B2: uniform over legal actions.

    One generator is threaded through every episode of an evaluation, so the
    baseline is reproducible and is not re-seeded per month.
    """
    rng = np.random.default_rng(seed)

    def _policy(obs, mask) -> int:
        return int(rng.choice(np.flatnonzero(mask)))

    return _policy


# ---------------------------------------------------------------- registry

#: name -> (policy factory, environment overrides, description)
#: The overrides are applied on top of the run's environment configuration, so a
#: baseline can differ from the agent only in the way its description states.
BASELINES: dict[str, dict] = {
    "B0_do_nothing": {
        "policy": lambda: policy_hold,
        "env_overrides": {},
        "description": "always Hold; preserves cash",
    },
    "B1a_slice_bah": {
        "policy": lambda: policy_buy_when_legal,
        "env_overrides": {},
        "description": "buy one slice whenever legal, at the agent's granularity",
    },
    "B1b_true_bah": {
        "policy": lambda: policy_buy_when_legal,
        # `action_model` is pinned as well as the slice. Buy-and-hold is a
        # market benchmark, not a policy drawn from the agent's action set, so it
        # must mean the same thing in every configuration. Left unpinned, a
        # whole-share configuration would silently reduce it to one share per
        # step, which reaches only about half exposure across a month and is
        # indistinguishable from B1a.
        "env_overrides": {"slice_frac": 1.0, "action_model": "slice"},
        "description": "fully invested at the first bar, held to liquidation",
    },
    "B2_random": {
        "policy": lambda: make_random_policy(0),
        "env_overrides": {},
        "description": "uniform over legal actions, single generator across months",
    },
}
