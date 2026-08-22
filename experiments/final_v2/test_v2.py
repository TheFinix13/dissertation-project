"""Tests for the final environment, features and baselines.

Each test corresponds to a claim made in Chapter 3 or Chapter 4, so a failing
test points at a specific sentence in the dissertation. The synthetic series
below is used only to make the tests independent of network access; no result
reported in the dissertation comes from it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import data as datamod
from baselines import policy_buy_when_legal, policy_hold
from env import MIN_TRADE_VALUE, TradingEnv
from features import (
    LADDER,
    MARKET_FEATURES,
    MOM_WINDOW,
    compute_market_features,
    order_features,
    resolve_rung,
)
from rollout import run_episode

CASH0 = 10_000.0


# --------------------------------------------------------------- fixtures

def synthetic_ohlcv(n_days: int = 300, seed: int = 0, start: str = "2017-09-01") -> pd.DataFrame:
    """Deterministic synthetic OHLCV on business days. Not market data."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n_days)
    rets = rng.normal(0.0004, 0.009, size=n_days)
    close = 250.0 * np.cumprod(1.0 + rets)
    spread = np.abs(rng.normal(0.006, 0.003, size=n_days)) * close
    high = close + spread
    low = close - spread
    open_ = close - rng.normal(0.0, 0.004, size=n_days) * close
    volume = rng.integers(4e7, 9e7, size=n_days).astype(float)
    return pd.DataFrame(
        {"Open": open_, "High": np.maximum(high, np.maximum(open_, close)),
         "Low": np.minimum(low, np.minimum(open_, close)),
         "Close": close, "Volume": volume},
        index=idx,
    )


@pytest.fixture(scope="module")
def episodes() -> list[dict]:
    return datamod.build_episodes(synthetic_ohlcv())


@pytest.fixture
def episode(episodes) -> dict:
    return episodes[1]  # not the first, so history definitely precedes it


def make_env(episode, rung="S3", **kw) -> TradingEnv:
    kw.setdefault("initial_cash", CASH0)
    return TradingEnv(episode["prices"], episode["market"], resolve_rung(rung), **kw)


# ------------------------------------------------------- feature contracts

def test_features_use_no_future_data():
    """Truncating the series must not change any earlier feature value."""
    df = synthetic_ohlcv()
    full = compute_market_features(df)
    cut = len(df) - 30
    truncated = compute_market_features(df.iloc[:cut])
    pd.testing.assert_frame_equal(full.iloc[:cut], truncated, check_exact=False,
                                  rtol=1e-12, atol=1e-12)


def test_rolling_features_survive_the_episode_boundary(episodes):
    """The defect this pipeline exists to fix.

    Momentum on the first day of an episode must be the true k-day return, which
    reaches back into the previous month, and must not be zeroed because a new
    month started.
    """
    df = synthetic_ohlcv()
    feats = compute_market_features(df)
    mom_col = list(MARKET_FEATURES).index("mom_k")

    ep = episodes[2]
    first_day_mom = ep["market"][0, mom_col]

    # Locate that day in the continuous series and recompute independently.
    close = df["Close"].astype(float)
    pos = int(np.flatnonzero(np.isclose(close.to_numpy(), ep["prices"][0]))[0])
    expected = close.iloc[pos] / close.iloc[pos - MOM_WINDOW] - 1.0

    assert first_day_mom == pytest.approx(expected, rel=1e-9)
    assert first_day_mom != 0.0, "momentum was reset at the episode boundary"
    assert feats.iloc[pos]["mom_k"] == pytest.approx(first_day_mom, rel=1e-9)


def test_no_nans_or_infs_in_any_episode(episodes):
    for ep in episodes:
        assert np.all(np.isfinite(ep["market"])), f"non-finite features in {ep['id']}"
        assert np.all(ep["prices"] > 0)


def test_canonical_order_is_independent_of_input_order():
    a = order_features(["pnl", "ret_1", "clock"])
    b = order_features(["clock", "pnl", "ret_1"])
    assert a == b == ("ret_1", "clock", "pnl")


def test_ladder_dimensions_are_as_documented():
    expected = {"S0": 1, "S1": 3, "S2": 4, "S3a": 5, "S3": 9, "S4": 10, "S5": 12}
    for rung, dim in expected.items():
        assert len(resolve_rung(rung)) == dim, rung


def test_every_rung_is_a_subset_of_the_next():
    chain = ["S0", "S1", "S2", "S3a", "S3", "S4", "S5"]
    # S1 -> S2 deliberately replaces shares_raw with exposure_frac + pnl, so
    # only monotone growth from S2 onward is asserted.
    for lo, hi in zip(chain[2:], chain[3:]):
        assert set(LADDER[lo]) <= set(LADDER[hi]), f"{lo} not contained in {hi}"


# --------------------------------------------------------- MDP contracts

def test_accounting_identity_holds_for_every_policy(episodes):
    for rung in ("S0", "S3", "S4"):
        for policy in (policy_hold, policy_buy_when_legal):
            for ep in episodes[:4]:
                row = run_episode(make_env(ep, rung), policy)
                assert row["accounting_gap"] < 1e-9, (rung, ep["id"])


def test_reward_equals_wealth_change_when_risk_lambda_is_zero(episode):
    env = make_env(episode, "S4", risk_lambda=0.0)
    obs, _ = env.reset()
    done = False
    while not done:
        obs, reward, term, trunc, info = env.step(1 if env.action_masks()[1] else 0)
        assert reward == pytest.approx(info["dw"], abs=1e-12)
        done = term or trunc


def test_risk_penalty_requires_drawdown_in_the_state(episode):
    """Admissibility enforced in code: the reward must be a function of the state."""
    with pytest.raises(ValueError, match="drawdown"):
        make_env(episode, "S3", risk_lambda=1.0)
    make_env(episode, "S4", risk_lambda=1.0)  # must not raise


def test_risk_penalty_never_increases_the_reward(episode):
    plain = make_env(episode, "S4", risk_lambda=0.0)
    shaped = make_env(episode, "S4", risk_lambda=2.0)
    for env in (plain, shaped):
        env.reset()
    done = False
    while not done:
        a = 1 if plain.action_masks()[1] else 0
        _, r_plain, t1, _, _ = plain.step(a)
        _, r_shaped, t2, _, _ = shaped.step(a)
        assert r_shaped <= r_plain + 1e-12
        done = t1 or t2


def test_episode_always_ends_flat(episodes):
    for ep in episodes[:5]:
        env = make_env(ep, "S3")
        env.reset()
        done = False
        while not done:
            _, _, term, trunc, info = env.step(1 if env.action_masks()[1] else 0)
            done = term or trunc
        assert info["units"] == 0.0
        assert info["exposure"] == 0.0


# ---------------------------------------------------- feature relationships

def test_wealth_frac_is_pointwise_recoverable(episode):
    """Why wealth_frac is excluded from every rung: it is c_t + v_t - 1."""
    names = order_features(["cash_frac", "exposure_frac", "wealth_frac"])
    env = TradingEnv(episode["prices"], episode["market"], names, initial_cash=CASH0)
    obs, _ = env.reset()
    i = {n: k for k, n in enumerate(names)}
    done = False
    while not done:
        assert obs[i["cash_frac"]] + obs[i["exposure_frac"]] - 1.0 == pytest.approx(
            obs[i["wealth_frac"]], abs=1e-5)
        obs, _, term, trunc, _ = env.step(1 if env.action_masks()[1] else 0)
        done = term or trunc


def test_drawdown_is_not_pointwise_recoverable(episodes):
    """Two paths can share cash, exposure and wealth yet differ in drawdown."""
    rising = np.array([100.0, 110.0, 120.0, 110.0, 110.0])
    flat = np.array([100.0, 100.0, 100.0, 110.0, 110.0])
    market = np.zeros((5, len(MARKET_FEATURES)))
    names = order_features(["cash_frac", "exposure_frac", "drawdown"])

    seen = []
    for prices in (rising, flat):
        env = TradingEnv(prices, market, names, initial_cash=CASH0, fee=0.0)
        env.reset()
        env.step(1)  # take a position, then hold to the end
        done = False
        while not done:
            obs, _, term, trunc, info = env.step(0)
            done = term or trunc
        seen.append((obs, info))

    (obs_a, info_a), (obs_b, info_b) = seen
    i = {n: k for k, n in enumerate(names)}
    assert obs_a[i["cash_frac"]] == pytest.approx(obs_b[i["cash_frac"]], abs=1e-6)
    assert info_a["drawdown"] != pytest.approx(info_b["drawdown"], abs=1e-6), \
        "drawdown failed to distinguish two different paths"


# --------------------------------------------------------------- masking

def test_mask_blocks_infeasible_trades(episode):
    env = make_env(episode, "S3")
    env.reset()
    assert env.action_masks().tolist() == [True, True, False], "cannot sell from flat"
    while env.action_masks()[1]:  # spend all cash
        env.step(1)
    assert not env.action_masks()[1], "Buy still legal with no cash"
    assert env.action_masks()[2], "Sell should be legal while holding"


def test_illegal_action_is_forced_to_hold_and_flagged(episode):
    env = make_env(episode, "S3")
    env.reset()
    _, _, _, _, info = env.step(2)  # Sell from flat
    assert info["illegal"] is True
    assert info["action_executed"] == 0
    assert info["n_trades"] == 0


def test_mask_is_a_deterministic_function_of_observed_features(episode):
    """Chapter 3 claims masking constrains without informing. This checks it."""
    names = order_features(["cash_frac", "exposure_frac"])
    env = TradingEnv(episode["prices"], episode["market"], names, initial_cash=CASH0)
    obs, info = env.reset()
    threshold = MIN_TRADE_VALUE / CASH0
    done = False
    while not done:
        mask = env.action_masks()
        predicted = np.array([True, obs[0] >= threshold, obs[1] >= threshold])
        assert mask.tolist() == predicted.tolist()
        obs, _, term, trunc, info = env.step(1 if mask[1] else (2 if mask[2] else 0))
        done = term or trunc


# ---------------------------------------------------------- action models

def test_share_action_model_buys_one_unit(episode):
    env = make_env(episode, "S3", action_model="share", fee=0.0)
    env.reset()
    env.step(1)
    assert env.units == pytest.approx(1.0, rel=1e-9)


def test_slice_action_model_spends_a_fixed_sum(episode):
    env = make_env(episode, "S3", action_model="slice", slice_frac=0.10, fee=0.0)
    env.reset()
    cash_before = env.cash
    env.step(1)
    assert cash_before - env.cash == pytest.approx(0.10 * CASH0, rel=1e-9)


# --------------------------------------------------------------- baselines

def test_true_buy_and_hold_beats_the_slice_schedule_when_prices_rise(episodes):
    """The baseline defect: a slice schedule under-invests early and so understates
    buy-and-hold in a rising market."""
    prices = np.linspace(100.0, 130.0, 21)
    market = np.zeros((len(prices), len(MARKET_FEATURES)))
    names = resolve_rung("S3")

    slice_env = TradingEnv(prices, market, names, initial_cash=CASH0, slice_frac=0.10)
    true_env = TradingEnv(prices, market, names, initial_cash=CASH0, slice_frac=1.0)
    slice_row = run_episode(slice_env, policy_buy_when_legal)
    true_row = run_episode(true_env, policy_buy_when_legal)

    assert true_row["delta_w"] > slice_row["delta_w"]
    assert true_row["n_trades"] == 2, "true buy-and-hold should buy once and liquidate once"
    assert true_row["mean_exposure"] > slice_row["mean_exposure"]


def test_do_nothing_preserves_cash_exactly(episodes):
    row = run_episode(make_env(episodes[0], "S3"), policy_hold)
    assert row["delta_w"] == pytest.approx(0.0, abs=1e-12)
    assert row["n_trades"] == 0
    assert row["fees_paid"] == 0.0


# ----------------------------------------------------------- data pipeline

def test_split_is_chronological_and_disjoint():
    # Long enough to span train, validation and test under the calendar rule.
    parts = datamod.split(datamod.build_episodes(synthetic_ohlcv(n_days=2200)))
    train_ids, val_ids, test_ids = (
        [e["id"] for e in parts[k]] for k in ("train", "val", "test"))
    assert max(train_ids) <= datamod.TRAIN_END < min(val_ids)
    assert max(val_ids) <= datamod.VAL_END < min(test_ids)
    assert not set(train_ids) & set(val_ids) & set(test_ids)
