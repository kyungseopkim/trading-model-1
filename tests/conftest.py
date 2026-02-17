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
