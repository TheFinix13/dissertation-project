"""Feature definitions and the state-design ladder, in one place.

Two rules are enforced here and they are the reason this module exists.

1. Market features are computed ONCE on the continuous price series and only
   then sliced into monthly episodes. Computing a rolling window inside an
   episode makes the first days of every month degenerate: a 5-day momentum
   has no 5 days of history on day 1, so the agent is told the market has no
   trend and no volatility exactly when it chooses its opening position.
   Momentum does not reset because a new month started.

2. Every feature uses data up to and including time t only. `test_features.py`
   verifies this by truncating the series and checking the values are
   unchanged.

Canonical ordering is fixed by CANONICAL_ORDER. A ladder rung is a subset of
feature names; the observation vector is always emitted in canonical order
filtered by that subset, so two runs with the same rung always agree on which
column means what.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- windows

MOM_WINDOW = 5       # short-horizon momentum
VOL_WINDOW = 5       # short-horizon realised volatility
VOL_LONG = 20        # long-horizon volatility, for the regime ratio
MA_WINDOW = 20       # trend reference
ATR_WINDOW = 14      # rolling arithmetic mean of true range (see true_range)
VOLUME_WINDOW = 20   # relative-volume reference

#: Rows of continuous history needed before any feature is fully defined.
WARMUP_BARS = max(MOM_WINDOW, VOL_LONG, MA_WINDOW, ATR_WINDOW, VOLUME_WINDOW) + 1

# ---------------------------------------------------------------- registry

#: Features computed from market data alone (precomputed on the full series).
MARKET_FEATURES = (
    "ret_1",      # P_t / P_{t-1} - 1
    "mom_k",      # P_t / P_{t-k} - 1
    "vol_k",      # std of last k one-step returns
    "ma_gap",     # P_t / MA_m(P) - 1
    "atr_norm",   # ATR_14 / P_t          (requires High/Low)
    "rel_vol",    # V_t / mean(V, 20) - 1 (requires Volume)
    "vol_ratio",  # sigma_5 / sigma_20 - 1
)

#: Features computed from the agent's own account at run time.
ACCOUNT_FEATURES = (
    "clock",          # t / T
    "cash_frac",      # C_t / C_0
    "exposure_frac",  # h_t P_t / C_0
    "shares_raw",     # h_t, unnormalised (ladder rung S1 only)
    "pnl",            # (P_t - avg_entry) / avg_entry
    "drawdown",       # (peak W - W_t) / peak W   <- path-dependent
    "wealth_frac",    # W_t / C_0 - 1             <- pointwise redundant
)

CANONICAL_ORDER: tuple[str, ...] = (
    "ret_1", "mom_k", "vol_k", "ma_gap", "atr_norm", "rel_vol", "vol_ratio",
    "clock", "cash_frac", "exposure_frac", "shares_raw", "pnl", "drawdown",
    "wealth_frac",
)

FEATURE_BLOCK = {
    "ret_1": "market", "mom_k": "market", "vol_k": "market", "ma_gap": "market",
    "atr_norm": "market", "rel_vol": "market", "vol_ratio": "market",
    "clock": "time",
    "cash_frac": "portfolio", "exposure_frac": "portfolio", "shares_raw": "portfolio",
    "pnl": "position",
    "drawdown": "risk",
    "wealth_frac": "portfolio",
}

#: Human-readable purpose, used to generate the Chapter 3 / Chapter 4 tables.
FEATURE_PURPOSE = {
    "ret_1": "immediate price movement",
    "mom_k": "short-horizon trend",
    "vol_k": "recent market variability",
    "ma_gap": "price relative to its recent trend",
    "atr_norm": "intraday range risk invisible in closes",
    "rel_vol": "conviction and liquidity behind the move",
    "vol_ratio": "shift between short- and long-horizon volatility",
    "clock": "time remaining before forced liquidation",
    "cash_frac": "liquid funds available",
    "exposure_frac": "market value of the open position",
    "shares_raw": "units held, without price context",
    "pnl": "unrealised return on the open position",
    "drawdown": "fall from the episode's peak wealth",
    "wealth_frac": "episode performance so far (derived)",
}

# ------------------------------------------------------------ ladder rungs

#: The state-design ladder. Each rung is a subset of CANONICAL_ORDER.
#: S3 is the core dissertation state; S4 adds the path-dependent risk feature
#: required to make a drawdown-penalised reward admissible.
LADDER: dict[str, tuple[str, ...]] = {
    "S0": ("ret_1",),
    "S1": ("ret_1", "cash_frac", "shares_raw"),
    "S2": ("ret_1", "cash_frac", "exposure_frac", "pnl"),
    "S3a": ("ret_1", "clock", "cash_frac", "exposure_frac", "pnl"),
    "S3": ("ret_1", "mom_k", "vol_k", "ma_gap", "atr_norm",
           "clock", "cash_frac", "exposure_frac", "pnl"),
    "S4": ("ret_1", "mom_k", "vol_k", "ma_gap", "atr_norm",
           "clock", "cash_frac", "exposure_frac", "pnl", "drawdown"),
    "S5": ("ret_1", "mom_k", "vol_k", "ma_gap", "atr_norm", "rel_vol", "vol_ratio",
           "clock", "cash_frac", "exposure_frac", "pnl", "drawdown"),
}

LADDER_LIMITATION = {
    "S0": "cannot tell whether it holds anything, or whether it can afford to buy",
    "S1": "knows the unit count but not what that position is worth",
    "S2": "knows its financial position but not where it is in the episode",
    "S3a": "market context is still a single lagged return",
    "S3": "no memory of the path taken to the current wealth",
    "S4": "no volume or regime information; no cross-asset context",
    "S5": "order flow, news and other participants remain unobserved",
}


def order_features(names) -> tuple[str, ...]:
    """Return `names` in canonical order, rejecting anything unknown."""
    unknown = set(names) - set(CANONICAL_ORDER)
    if unknown:
        raise ValueError(f"unknown feature(s): {sorted(unknown)}")
    return tuple(n for n in CANONICAL_ORDER if n in set(names))


def resolve_rung(rung: str) -> tuple[str, ...]:
    if rung not in LADDER:
        raise ValueError(f"unknown rung {rung!r}; expected one of {sorted(LADDER)}")
    return order_features(LADDER[rung])


# ------------------------------------------------- market feature matrix

def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Wilder's true range: the day's span, extended to include an overnight gap."""
    prev_close = close.shift(1)
    a = high - low
    b = (high - prev_close).abs()
    c = (low - prev_close).abs()
    return pd.concat([a, b, c], axis=1).max(axis=1)


def compute_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute every market feature on the CONTINUOUS series.

    `df` must be indexed by date and contain Open/High/Low/Close/Volume.
    The result is aligned to `df.index`; the first WARMUP_BARS rows contain
    partially-defined values and are dropped by the data pipeline, not here.

    Every column at row t is a function of df rows <= t only.
    """
    required = {"High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns for market features: {sorted(missing)}")

    close = df["Close"].astype(float)
    ret = close.pct_change()

    out = pd.DataFrame(index=df.index)
    out["ret_1"] = ret
    out["mom_k"] = close / close.shift(MOM_WINDOW) - 1.0
    out["vol_k"] = ret.rolling(VOL_WINDOW).std(ddof=0)
    out["ma_gap"] = close / close.rolling(MA_WINDOW).mean() - 1.0

    # Simple rolling mean of TR over n=14 days (not Wilder's recursive smoother).
    atr = true_range(df["High"].astype(float), df["Low"].astype(float), close) \
        .rolling(ATR_WINDOW).mean()
    out["atr_norm"] = atr / close

    volume = df["Volume"].astype(float)
    out["rel_vol"] = volume / volume.rolling(VOLUME_WINDOW).mean() - 1.0

    vol_long = ret.rolling(VOL_LONG).std(ddof=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = out["vol_k"] / vol_long
    out["vol_ratio"] = ratio.replace([np.inf, -np.inf], np.nan) - 1.0

    # Centre everything near zero so no single input dominates the first layer.
    # ret_1, mom_k, ma_gap, rel_vol and vol_ratio are already centred by
    # construction; vol_k and atr_norm are non-negative scales left as-is.
    return out[list(MARKET_FEATURES)]
