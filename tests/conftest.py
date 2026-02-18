import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    """Generate 400 bars of synthetic minute-level OHLCV data.

    Simulates a random walk with realistic OHLCV relationships.
    """
    np.random.seed(42)
    n = 400
    base = 100.0
    returns = np.random.normal(0, 0.001, n)
    close = base * np.exp(np.cumsum(returns))

    high = close * (1 + np.abs(np.random.normal(0, 0.002, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.002, n)))
    open_ = low + (high - low) * np.random.uniform(0.3, 0.7, n)
    volume = np.random.randint(1000, 50000, n).astype(float)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )


@pytest.fixture
def synthetic_daily_window() -> pd.DataFrame:
    """Generate 90 days of synthetic daily OHLCV data for daily context."""
    np.random.seed(123)
    n = 90
    base = 95.0
    returns = np.random.normal(0.001, 0.015, n)
    close = base * np.exp(np.cumsum(returns))

    high = close * (1 + np.abs(np.random.normal(0, 0.01, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.01, n)))
    open_ = low + (high - low) * np.random.uniform(0.3, 0.7, n)
    volume = np.random.randint(100_000, 5_000_000, n).astype(float)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )
