"""Reference strategies, including a corrected buy-and-hold.

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
        "env_overrides": {"slice_frac": 1.0},
        "description": "fully invested at the first bar, held to liquidation",
    },
    "B2_random": {
        "policy": lambda: make_random_policy(0),
        "env_overrides": {},
        "description": "uniform over legal actions, single generator across months",
    },
}
