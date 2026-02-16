# Gymnasium Trading Environment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a modular single-stock intraday Gymnasium trading environment with Sharpe-hybrid reward, discrete action space with position sizing, and full technical indicator observation space.

**Architecture:** Composition-based — `TradingEnv` composes four independent components (`FrictionModel`, `ActionMapper`, `FeatureEngine`, `RewardCalculator`). Each component is built test-first, then wired together in the env. A thin `DataLoader` handles CSV-to-DataFrame conversion.

**Tech Stack:** Python 3.13, gymnasium, numpy, pandas, ta (technical analysis), pytest, uv

---

### Task 1: Project Scaffolding

**Files:**
- Modify: `pyproject.toml`
- Create: `src/trading_model/__init__.py`
- Create: `src/trading_model/env/__init__.py`
- Create: `src/trading_model/data/__init__.py`
- Create: `tests/__init__.py`

**Step 1: Update pyproject.toml**

```toml
[project]
name = "trading-model"
version = "0.1.0"
description = "RL intraday trading environment"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "gymnasium>=1.0.0",
    "numpy>=2.0.0",
    "pandas>=2.2.0",
    "ta>=0.11.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
]
db = [
    "pymysql>=1.1.2",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/trading_model"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

**Step 2: Create package directories and __init__.py files**

```bash
mkdir -p src/trading_model/env src/trading_model/data tests
touch src/trading_model/__init__.py
touch src/trading_model/env/__init__.py
touch src/trading_model/data/__init__.py
touch tests/__init__.py
```

**Step 3: Install dependencies**

```bash
uv sync --all-extras
```

Expected: resolves and installs gymnasium, numpy, pandas, ta, pytest

**Step 4: Verify pytest runs**

```bash
uv run pytest --co -q
```

Expected: `no tests ran` (no test files yet, but pytest itself works)

**Step 5: Commit**

```bash
git add pyproject.toml uv.lock src/ tests/
git commit -m "feat: scaffold trading-model package structure"
```

---

### Task 2: FrictionModel

**Files:**
- Create: `src/trading_model/env/frictions.py`
- Create: `tests/test_frictions.py`

**Step 1: Write failing tests**

```python
# tests/test_frictions.py
from trading_model.env.frictions import FrictionModel


class TestFrictionModel:
    def test_zero_trade_returns_zero_cost(self):
        fm = FrictionModel(fee_rate=0.001)
        assert fm.calculate_cost(0.0) == 0.0

    def test_positive_trade_value(self):
        fm = FrictionModel(fee_rate=0.001)
        assert fm.calculate_cost(10_000.0) == 10.0

    def test_negative_trade_value_uses_abs(self):
        fm = FrictionModel(fee_rate=0.001)
        assert fm.calculate_cost(-5_000.0) == 5.0

    def test_custom_fee_rate(self):
        fm = FrictionModel(fee_rate=0.01)
        assert fm.calculate_cost(1_000.0) == 10.0

    def test_default_fee_rate_is_0001(self):
        fm = FrictionModel()
        assert fm.fee_rate == 0.001
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_frictions.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'trading_model.env.frictions'`

**Step 3: Write minimal implementation**

```python
# src/trading_model/env/frictions.py


class FrictionModel:
    def __init__(self, fee_rate: float = 0.001):
        self.fee_rate = fee_rate

    def calculate_cost(self, trade_value: float) -> float:
        return abs(trade_value) * self.fee_rate
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_frictions.py -v
```

Expected: 5 passed

**Step 5: Commit**

```bash
git add src/trading_model/env/frictions.py tests/test_frictions.py
git commit -m "feat: add FrictionModel with flat-fee cost calculation"
```

---

### Task 3: ActionMapper

**Files:**
- Create: `src/trading_model/env/actions.py`
- Create: `tests/test_actions.py`

**Step 1: Write failing tests**

```python
# tests/test_actions.py
import pytest
from trading_model.env.actions import ActionMapper, Action


class TestAction:
    def test_action_values(self):
        assert Action.HOLD == 0
        assert Action.BUY_25 == 1
        assert Action.BUY_50 == 2
        assert Action.BUY_100 == 3
        assert Action.SELL_25 == 4
        assert Action.SELL_50 == 5
        assert Action.SELL_100 == 6


class TestActionMapper:
    def setup_method(self):
        self.mapper = ActionMapper()

    def test_hold_returns_zero_deltas(self):
        shares_d, cash_d = self.mapper.map_action(0, cash=50_000, shares=10, price=100)
        assert shares_d == 0.0
        assert cash_d == 0.0

    def test_buy_25_spends_quarter_of_cash(self):
        shares_d, cash_d = self.mapper.map_action(1, cash=40_000, shares=0, price=100)
        assert shares_d == 100.0  # 10_000 / 100
        assert cash_d == -10_000.0

    def test_buy_50_spends_half_of_cash(self):
        shares_d, cash_d = self.mapper.map_action(2, cash=40_000, shares=0, price=200)
        assert shares_d == 100.0  # 20_000 / 200
        assert cash_d == -20_000.0

    def test_buy_100_spends_all_cash(self):
        shares_d, cash_d = self.mapper.map_action(3, cash=10_000, shares=0, price=50)
        assert shares_d == 200.0  # 10_000 / 50
        assert cash_d == -10_000.0

    def test_sell_25_sells_quarter_of_shares(self):
        shares_d, cash_d = self.mapper.map_action(4, cash=0, shares=100, price=50)
        assert shares_d == -25.0
        assert cash_d == 1_250.0  # 25 * 50

    def test_sell_50_sells_half_of_shares(self):
        shares_d, cash_d = self.mapper.map_action(5, cash=0, shares=100, price=50)
        assert shares_d == -50.0
        assert cash_d == 2_500.0

    def test_sell_100_sells_all_shares(self):
        shares_d, cash_d = self.mapper.map_action(6, cash=0, shares=100, price=50)
        assert shares_d == -100.0
        assert cash_d == 5_000.0

    def test_buy_with_no_cash_is_hold(self):
        shares_d, cash_d = self.mapper.map_action(1, cash=0, shares=0, price=100)
        assert shares_d == 0.0
        assert cash_d == 0.0

    def test_sell_with_no_shares_is_hold(self):
        shares_d, cash_d = self.mapper.map_action(4, cash=1000, shares=0, price=100)
        assert shares_d == 0.0
        assert cash_d == 0.0
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_actions.py -v
```

Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/trading_model/env/actions.py
from enum import IntEnum


class Action(IntEnum):
    HOLD = 0
    BUY_25 = 1
    BUY_50 = 2
    BUY_100 = 3
    SELL_25 = 4
    SELL_50 = 5
    SELL_100 = 6


_BUY_FRACTIONS = {Action.BUY_25: 0.25, Action.BUY_50: 0.50, Action.BUY_100: 1.0}
_SELL_FRACTIONS = {Action.SELL_25: 0.25, Action.SELL_50: 0.50, Action.SELL_100: 1.0}


class ActionMapper:
    def map_action(
        self, action: int, cash: float, shares: float, price: float
    ) -> tuple[float, float]:
        """Map a discrete action to (shares_delta, cash_delta).

        Positive shares_delta = buying, negative = selling.
        Cash delta is the inverse (spend cash to buy, receive cash on sell).
        Invalid actions (buy with no cash, sell with no shares) return (0, 0).
        """
        act = Action(action)

        if act == Action.HOLD:
            return 0.0, 0.0

        if act in _BUY_FRACTIONS:
            if cash <= 0 or price <= 0:
                return 0.0, 0.0
            cash_to_spend = cash * _BUY_FRACTIONS[act]
            shares_to_buy = cash_to_spend / price
            return shares_to_buy, -cash_to_spend

        if act in _SELL_FRACTIONS:
            if shares <= 0:
                return 0.0, 0.0
            shares_to_sell = shares * _SELL_FRACTIONS[act]
            cash_received = shares_to_sell * price
            return -shares_to_sell, cash_received

        return 0.0, 0.0
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_actions.py -v
```

Expected: 10 passed

**Step 5: Commit**

```bash
git add src/trading_model/env/actions.py tests/test_actions.py
git commit -m "feat: add ActionMapper with discrete buy/hold/sell sizing"
```

---

### Task 4: FeatureEngine

**Files:**
- Create: `src/trading_model/env/features.py`
- Create: `tests/test_features.py`
- Create: `tests/conftest.py` (shared fixture for synthetic OHLCV)

**Step 1: Write shared fixture for synthetic data**

```python
# tests/conftest.py
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
```

**Step 2: Write failing tests**

```python
# tests/test_features.py
import numpy as np
import pytest

from trading_model.env.features import FeatureEngine


class TestFeatureEngine:
    def test_precompute_sets_num_steps(self, synthetic_ohlcv):
        fe = FeatureEngine()
        fe.precompute(synthetic_ohlcv)
        assert fe.num_steps == 400

    def test_warmup_period_is_26(self):
        assert FeatureEngine.WARMUP_PERIOD == 26

    def test_obs_dim_is_18(self):
        assert FeatureEngine.OBS_DIM == 18

    def test_get_market_features_shape(self, synthetic_ohlcv):
        fe = FeatureEngine()
        fe.precompute(synthetic_ohlcv)
        features = fe.get_market_features(30)
        assert features.shape == (14,)
        assert features.dtype == np.float32

    def test_get_market_features_no_nans_after_warmup(self, synthetic_ohlcv):
        fe = FeatureEngine()
        fe.precompute(synthetic_ohlcv)
        for step in range(FeatureEngine.WARMUP_PERIOD, 400):
            features = fe.get_market_features(step)
            assert not np.any(np.isnan(features)), f"NaN at step {step}"

    def test_build_observation_shape(self, synthetic_ohlcv):
        fe = FeatureEngine()
        fe.precompute(synthetic_ohlcv)
        obs = fe.build_observation(
            step=30, position=1.0, unrealized_pnl_pct=0.05,
            cash_ratio=0.5, trade_duration=0.1,
        )
        assert obs.shape == (18,)
        assert obs.dtype == np.float32

    def test_build_observation_includes_agent_state(self, synthetic_ohlcv):
        fe = FeatureEngine()
        fe.precompute(synthetic_ohlcv)
        obs = fe.build_observation(
            step=30, position=1.0, unrealized_pnl_pct=0.05,
            cash_ratio=0.5, trade_duration=0.1,
        )
        # Last 4 elements are agent state
        assert obs[14] == pytest.approx(1.0)   # position
        assert obs[15] == pytest.approx(0.05)  # unrealized_pnl_pct
        assert obs[16] == pytest.approx(0.5)   # cash_ratio
        assert obs[17] == pytest.approx(0.1)   # trade_duration

    def test_rsi_in_zero_one_range(self, synthetic_ohlcv):
        fe = FeatureEngine()
        fe.precompute(synthetic_ohlcv)
        for step in range(FeatureEngine.WARMUP_PERIOD, 400):
            features = fe.get_market_features(step)
            rsi = features[8]  # index 8 is RSI
            assert 0.0 <= rsi <= 1.0, f"RSI out of range at step {step}: {rsi}"

    def test_get_close_price(self, synthetic_ohlcv):
        fe = FeatureEngine()
        fe.precompute(synthetic_ohlcv)
        price = fe.get_close_price(0)
        assert price == pytest.approx(synthetic_ohlcv.iloc[0]["close"])
```

**Step 3: Run tests to verify they fail**

```bash
uv run pytest tests/test_features.py -v
```

Expected: FAIL — `ModuleNotFoundError`

**Step 4: Write implementation**

```python
# src/trading_model/env/features.py
import numpy as np
import pandas as pd
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands


class FeatureEngine:
    WARMUP_PERIOD = 26
    OBS_DIM = 18

    def __init__(self):
        self._features: pd.DataFrame | None = None

    def precompute(self, ohlcv: pd.DataFrame) -> None:
        """Precompute all technical indicators for a day's OHLCV data.

        Args:
            ohlcv: DataFrame with columns: open, high, low, close, volume.
                   Rows are minute bars in chronological order.
        """
        df = ohlcv[["open", "high", "low", "close", "volume"]].copy()
        n = len(df)

        # Price features: normalize by previous close
        prev_close = df["close"].shift(1)
        df["norm_open"] = df["open"] / prev_close - 1
        df["norm_high"] = df["high"] / prev_close - 1
        df["norm_low"] = df["low"] / prev_close - 1
        df["norm_close"] = df["close"] / prev_close - 1
        df["log_volume"] = np.log1p(df["volume"])

        # MACD (12, 26, 9)
        macd_ind = MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
        df["macd"] = macd_ind.macd()
        df["macd_signal"] = macd_ind.macd_signal()
        df["macd_hist"] = macd_ind.macd_diff()

        # RSI (14), scaled to [0, 1]
        rsi_ind = RSIIndicator(df["close"], window=14)
        df["rsi"] = rsi_ind.rsi() / 100.0

        # Bollinger Bands (20)
        bb_ind = BollingerBands(df["close"], window=20)
        df["bb_upper_pct"] = (bb_ind.bollinger_hband() - df["close"]) / df["close"]
        df["bb_lower_pct"] = (bb_ind.bollinger_lband() - df["close"]) / df["close"]
        df["bb_width"] = bb_ind.bollinger_wband()

        # Time embeddings
        t = np.arange(n, dtype=np.float64) / n
        df["time_sin"] = np.sin(2 * np.pi * t)
        df["time_cos"] = np.cos(2 * np.pi * t)

        # Fill NaN from warmup with 0
        df = df.fillna(0.0)

        self._features = df

    @property
    def num_steps(self) -> int:
        if self._features is None:
            raise RuntimeError("Call precompute() first")
        return len(self._features)

    def get_close_price(self, step: int) -> float:
        return float(self._features.iloc[step]["close"])

    def get_market_features(self, step: int) -> np.ndarray:
        """Return the 14 market features at the given step."""
        row = self._features.iloc[step]
        return np.array(
            [
                row["norm_open"],
                row["norm_high"],
                row["norm_low"],
                row["norm_close"],
                row["log_volume"],
                row["macd"],
                row["macd_signal"],
                row["macd_hist"],
                row["rsi"],
                row["bb_upper_pct"],
                row["bb_lower_pct"],
                row["bb_width"],
                row["time_sin"],
                row["time_cos"],
            ],
            dtype=np.float32,
        )

    def build_observation(
        self,
        step: int,
        position: float,
        unrealized_pnl_pct: float,
        cash_ratio: float,
        trade_duration: float,
    ) -> np.ndarray:
        """Build the full 18-dim observation vector."""
        market = self.get_market_features(step)
        agent_state = np.array(
            [position, unrealized_pnl_pct, cash_ratio, trade_duration],
            dtype=np.float32,
        )
        return np.concatenate([market, agent_state])
```

**Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/test_features.py -v
```

Expected: 9 passed

**Step 6: Commit**

```bash
git add src/trading_model/env/features.py tests/test_features.py tests/conftest.py
git commit -m "feat: add FeatureEngine with MACD, RSI, Bollinger, time embeddings"
```

---

### Task 5: RewardCalculator

**Files:**
- Create: `src/trading_model/env/rewards.py`
- Create: `tests/test_rewards.py`

**Step 1: Write failing tests**

```python
# tests/test_rewards.py
import pytest
from trading_model.env.rewards import RewardCalculator


class TestRewardCalculator:
    def test_first_step_returns_weighted_return(self):
        rc = RewardCalculator(alpha=0.6, beta=0.5, window=60)
        rc.reset(100_000.0)
        # First step: risk_adjusted = 0 (not enough data), no drawdown
        reward = rc.calculate(100_100.0, step_return=0.001)
        assert reward == pytest.approx(0.6 * 0.001, abs=1e-6)

    def test_drawdown_penalty_applied(self):
        rc = RewardCalculator(alpha=0.6, beta=0.5, window=60)
        rc.reset(100_000.0)
        # First step: go up
        rc.calculate(101_000.0, step_return=0.01)
        # Second step: go down — triggers drawdown increase
        reward = rc.calculate(100_000.0, step_return=-0.0099)
        # dd = (101_000 - 100_000) / 101_000 ≈ 0.0099
        # dd_increase = 0.0099 - 0 = 0.0099
        assert reward < 0.6 * (-0.0099)  # penalty makes it more negative

    def test_no_drawdown_penalty_when_value_increases(self):
        rc = RewardCalculator(alpha=0.6, beta=0.5, window=60)
        rc.reset(100_000.0)
        rc.calculate(101_000.0, step_return=0.01)
        reward = rc.calculate(102_000.0, step_return=0.0099)
        # No drawdown increase — penalty term is 0
        # reward = alpha * return + (1-alpha) * risk_adj - 0
        assert reward > 0

    def test_risk_adjusted_component_after_enough_steps(self):
        rc = RewardCalculator(alpha=0.6, beta=0.5, window=5)
        rc.reset(100_000.0)
        value = 100_000.0
        # Feed 5 positive returns to build rolling stats
        for _ in range(5):
            value *= 1.001
            rc.calculate(value, step_return=0.001)
        # 6th step: risk-adjusted component should be nonzero
        value *= 1.001
        reward = rc.calculate(value, step_return=0.001)
        # With constant positive returns, mu/sigma > 0
        assert reward > 0.6 * 0.001  # risk component adds to reward

    def test_reset_clears_state(self):
        rc = RewardCalculator(alpha=0.6, beta=0.5, window=60)
        rc.reset(100_000.0)
        rc.calculate(101_000.0, step_return=0.01)
        rc.reset(50_000.0)
        # After reset, peak should be 50_000 not 101_000
        reward = rc.calculate(50_100.0, step_return=0.002)
        assert reward == pytest.approx(0.6 * 0.002, abs=1e-6)
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_rewards.py -v
```

Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# src/trading_model/env/rewards.py
from collections import deque

import numpy as np


class RewardCalculator:
    def __init__(self, alpha: float = 0.6, beta: float = 0.5, window: int = 60):
        self.alpha = alpha
        self.beta = beta
        self.window = window
        self._returns: deque[float] = deque(maxlen=window)
        self._peak_value = 0.0
        self._prev_drawdown = 0.0

    def reset(self, initial_value: float) -> None:
        self._returns.clear()
        self._peak_value = initial_value
        self._prev_drawdown = 0.0

    def calculate(self, portfolio_value: float, step_return: float) -> float:
        """Compute Sharpe-hybrid reward with drawdown penalty.

        R_t = alpha * r_t + (1 - alpha) * (mu / sigma) - beta * max(0, DD_t - DD_{t-1})
        """
        self._returns.append(step_return)

        # Drawdown tracking
        self._peak_value = max(self._peak_value, portfolio_value)
        current_dd = (self._peak_value - portfolio_value) / self._peak_value
        dd_increase = max(0.0, current_dd - self._prev_drawdown)
        self._prev_drawdown = current_dd

        # Risk-adjusted component (rolling Sharpe)
        if len(self._returns) < 2:
            risk_adjusted = 0.0
        else:
            arr = np.array(self._returns)
            mu = arr.mean()
            sigma = arr.std()
            risk_adjusted = mu / sigma if sigma > 1e-8 else 0.0

        return self.alpha * step_return + (1 - self.alpha) * risk_adjusted - self.beta * dd_increase
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_rewards.py -v
```

Expected: 5 passed

**Step 5: Commit**

```bash
git add src/trading_model/env/rewards.py tests/test_rewards.py
git commit -m "feat: add RewardCalculator with Sharpe-hybrid and drawdown penalty"
```

---

### Task 6: DataLoader

**Files:**
- Create: `src/trading_model/data/loader.py`
- Create: `tests/test_loader.py`

**Step 1: Write failing tests**

```python
# tests/test_loader.py
import pandas as pd
import pytest

from trading_model.data.loader import load_days_from_dataframe


class TestLoadDaysFromDataFrame:
    def test_splits_by_date(self):
        df = pd.DataFrame({
            "datetime": pd.date_range("2025-01-02 09:30", periods=800, freq="min"),
            "open": range(800),
            "high": range(800),
            "low": range(800),
            "close": range(800),
            "volume": range(800),
        })
        days = load_days_from_dataframe(df)
        assert len(days) >= 1
        for day in days:
            assert list(day.columns) == ["open", "high", "low", "close", "volume"]

    def test_skips_short_days(self):
        # Day 1: 400 bars, Day 2: 10 bars (should be skipped)
        dt1 = pd.date_range("2025-01-02 09:30", periods=400, freq="min")
        dt2 = pd.date_range("2025-01-03 09:30", periods=10, freq="min")
        df = pd.DataFrame({
            "datetime": dt1.append(dt2),
            "open": range(410),
            "high": range(410),
            "low": range(410),
            "close": range(410),
            "volume": range(410),
        })
        days = load_days_from_dataframe(df, min_bars=30)
        assert len(days) == 1

    def test_returns_sorted_within_day(self):
        # Create data out of order
        df = pd.DataFrame({
            "datetime": pd.to_datetime(["2025-01-02 10:00", "2025-01-02 09:30",
                                         "2025-01-02 09:31"] + [f"2025-01-02 09:{32+i}" for i in range(47)]),
            "open": range(50),
            "high": range(50),
            "low": range(50),
            "close": range(50),
            "volume": range(50),
        })
        days = load_days_from_dataframe(df, min_bars=1)
        assert len(days) == 1
        # First row should be 09:30 (was index 1 in original)
        assert days[0].iloc[0]["open"] == 1
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_loader.py -v
```

Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# src/trading_model/data/loader.py
import pandas as pd


def load_days_from_dataframe(
    df: pd.DataFrame, min_bars: int = 30
) -> list[pd.DataFrame]:
    """Split a multi-day OHLCV DataFrame into a list of per-day DataFrames.

    Args:
        df: Must contain columns: datetime, open, high, low, close, volume.
        min_bars: Days with fewer bars than this are skipped.

    Returns:
        List of DataFrames, each with columns [open, high, low, close, volume],
        sorted chronologically within each day, index reset.
    """
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["_date"] = df["datetime"].dt.date

    days = []
    for _, group in df.groupby("_date"):
        if len(group) < min_bars:
            continue
        day = (
            group.sort_values("datetime")
            .reset_index(drop=True)[["open", "high", "low", "close", "volume"]]
        )
        days.append(day)

    return days
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_loader.py -v
```

Expected: 3 passed

**Step 5: Commit**

```bash
git add src/trading_model/data/loader.py tests/test_loader.py
git commit -m "feat: add DataLoader to split multi-day OHLCV into per-day frames"
```

---

### Task 7: TradingEnv

**Files:**
- Create: `src/trading_model/env/trading_env.py`
- Create: `tests/test_env.py`

This is the largest task. We compose all four components into a Gymnasium environment.

**Step 1: Write failing tests**

```python
# tests/test_env.py
import gymnasium as gym
import numpy as np
import pytest

from trading_model.env.trading_env import TradingEnv


@pytest.fixture
def env(synthetic_ohlcv):
    """Create an env with 1 day of synthetic data."""
    return TradingEnv(data_days=[synthetic_ohlcv], initial_cash=100_000.0, shuffle=False)


class TestTradingEnvSpaces:
    def test_observation_space_shape(self, env):
        assert env.observation_space.shape == (18,)

    def test_action_space_size(self, env):
        assert env.action_space.n == 7


class TestTradingEnvReset:
    def test_reset_returns_observation_and_info(self, env):
        obs, info = env.reset(seed=42)
        assert obs.shape == (18,)
        assert isinstance(info, dict)

    def test_reset_observation_no_nans(self, env):
        obs, _ = env.reset(seed=42)
        assert not np.any(np.isnan(obs))

    def test_reset_agent_state_initial(self, env):
        obs, _ = env.reset(seed=42)
        # Agent state is last 4: position=0, pnl=0, cash_ratio=1, duration=0
        assert obs[14] == pytest.approx(0.0)   # no position
        assert obs[15] == pytest.approx(0.0)   # no unrealized pnl
        assert obs[16] == pytest.approx(1.0)   # all cash
        assert obs[17] == pytest.approx(0.0)   # no trade duration


class TestTradingEnvStep:
    def test_hold_does_not_change_portfolio(self, env):
        env.reset(seed=42)
        obs, reward, terminated, truncated, info = env.step(0)  # HOLD
        assert obs.shape == (18,)
        assert not terminated
        assert not truncated

    def test_buy_then_sell_full_episode(self, env):
        env.reset(seed=42)
        # Buy 100%
        obs, reward, terminated, truncated, info = env.step(3)
        assert obs[14] == pytest.approx(1.0)  # has position
        assert obs[16] < 0.01  # almost no cash left (just rounding)

        # Sell 100%
        obs, reward, terminated, truncated, info = env.step(6)
        assert obs[14] == pytest.approx(0.0)  # no position
        assert obs[16] == pytest.approx(1.0)  # all cash

    def test_episode_terminates_at_end_of_day(self, env):
        env.reset(seed=42)
        terminated = False
        steps = 0
        while not terminated:
            _, _, terminated, _, _ = env.step(0)
            steps += 1
        # Should run for num_bars - warmup_period steps
        assert steps == 400 - 26

    def test_force_liquidation_at_episode_end(self, env):
        env.reset(seed=42)
        # Buy on first step
        env.step(3)
        # Hold until end
        terminated = False
        while not terminated:
            _, _, terminated, _, info = env.step(0)
        # Portfolio value should reflect liquidation
        assert "portfolio_value" in info

    def test_portfolio_value_in_info(self, env):
        env.reset(seed=42)
        _, _, _, _, info = env.step(0)
        assert "portfolio_value" in info
        assert info["portfolio_value"] == pytest.approx(100_000.0, rel=0.01)


class TestTradingEnvGymCompliance:
    def test_check_env(self, synthetic_ohlcv):
        """Verify the env passes Gymnasium's built-in validation."""
        env = TradingEnv(data_days=[synthetic_ohlcv], shuffle=False)
        # Just check reset/step cycle works without error
        obs, info = env.reset(seed=0)
        assert env.observation_space.contains(obs)
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_env.py -v
```

Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# src/trading_model/env/trading_env.py
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from trading_model.env.actions import ActionMapper
from trading_model.env.features import FeatureEngine
from trading_model.env.frictions import FrictionModel
from trading_model.env.rewards import RewardCalculator


class TradingEnv(gym.Env):
    """Single-stock intraday trading environment.

    One episode = one trading day of minute-level OHLCV data.
    The agent trades with discrete buy/hold/sell actions with position sizing.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        data_days: list[pd.DataFrame],
        initial_cash: float = 100_000.0,
        fee_rate: float = 0.001,
        shuffle: bool = True,
        reward_alpha: float = 0.6,
        reward_beta: float = 0.5,
        reward_window: int = 60,
    ):
        super().__init__()

        self.data_days = data_days
        self.initial_cash = initial_cash
        self.shuffle = shuffle

        self.feature_engine = FeatureEngine()
        self.action_mapper = ActionMapper()
        self.reward_calc = RewardCalculator(
            alpha=reward_alpha, beta=reward_beta, window=reward_window
        )
        self.friction = FrictionModel(fee_rate=fee_rate)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(FeatureEngine.OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(7)

        # Episode state
        self._day_index = 0
        self._day_order: list[int] = list(range(len(data_days)))
        self._step = 0
        self._cash = initial_cash
        self._shares = 0.0
        self._entry_price = 0.0
        self._trade_start_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self._day_index >= len(self.data_days):
            self._day_index = 0
            if self.shuffle:
                self.np_random.shuffle(self._day_order)

        day_idx = self._day_order[self._day_index]
        self._day_index += 1

        self.feature_engine.precompute(self.data_days[day_idx])
        self._step = FeatureEngine.WARMUP_PERIOD
        self._cash = self.initial_cash
        self._shares = 0.0
        self._entry_price = 0.0
        self._trade_start_step = self._step

        self.reward_calc.reset(self._cash)

        return self._get_obs(), {}

    def step(self, action: int):
        price = self.feature_engine.get_close_price(self._step)

        # Map action to trade
        shares_delta, cash_delta = self.action_mapper.map_action(
            action, self._cash, self._shares, price
        )

        # Compute portfolio value before trade (for return calculation)
        prev_value = self._cash + self._shares * price

        # Apply friction on trade
        if shares_delta != 0:
            trade_value = abs(shares_delta * price)
            cost = self.friction.calculate_cost(trade_value)
            self._cash -= cost

        # Update position
        self._shares += shares_delta
        self._cash += cash_delta

        # Track trade timing
        if shares_delta > 0 and self._shares == shares_delta:
            # Opened a new position
            self._entry_price = price
            self._trade_start_step = self._step
        elif self._shares <= 0:
            # Closed position
            self._shares = 0.0
            self._entry_price = 0.0
            self._trade_start_step = self._step

        # Current portfolio value
        portfolio_value = self._cash + self._shares * price

        # Step return
        step_return = (
            (portfolio_value - prev_value) / prev_value if prev_value > 0 else 0.0
        )

        # Reward
        reward = self.reward_calc.calculate(portfolio_value, step_return)

        self._step += 1
        terminated = self._step >= self.feature_engine.num_steps

        # Force liquidate at end of day
        if terminated and self._shares > 0:
            liquidation_value = self._shares * price
            cost = self.friction.calculate_cost(liquidation_value)
            self._cash += liquidation_value - cost
            self._shares = 0.0
            portfolio_value = self._cash

        info = {"portfolio_value": portfolio_value}

        if terminated:
            obs = np.zeros(FeatureEngine.OBS_DIM, dtype=np.float32)
        else:
            obs = self._get_obs()

        return obs, float(reward), terminated, False, info

    def _get_obs(self) -> np.ndarray:
        price = self.feature_engine.get_close_price(self._step)
        total_value = self._cash + self._shares * price

        position = 1.0 if self._shares > 0 else 0.0
        unrealized_pnl_pct = (
            (price - self._entry_price) / self._entry_price
            if self._entry_price > 0
            else 0.0
        )
        cash_ratio = self._cash / total_value if total_value > 0 else 1.0
        trade_duration = (
            (self._step - self._trade_start_step) / self.feature_engine.num_steps
            if self._shares > 0
            else 0.0
        )

        return self.feature_engine.build_observation(
            self._step, position, unrealized_pnl_pct, cash_ratio, trade_duration
        )
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_env.py -v
```

Expected: 9 passed

**Step 5: Commit**

```bash
git add src/trading_model/env/trading_env.py tests/test_env.py
git commit -m "feat: add TradingEnv composing all environment components"
```

---

### Task 8: Package Exports & Full Test Suite

**Files:**
- Modify: `src/trading_model/__init__.py`
- Modify: `src/trading_model/env/__init__.py`
- Modify: `src/trading_model/data/__init__.py`

**Step 1: Add public exports**

```python
# src/trading_model/__init__.py
from trading_model.env.trading_env import TradingEnv

__all__ = ["TradingEnv"]
```

```python
# src/trading_model/env/__init__.py
from trading_model.env.actions import Action, ActionMapper
from trading_model.env.features import FeatureEngine
from trading_model.env.frictions import FrictionModel
from trading_model.env.rewards import RewardCalculator
from trading_model.env.trading_env import TradingEnv

__all__ = [
    "Action",
    "ActionMapper",
    "FeatureEngine",
    "FrictionModel",
    "RewardCalculator",
    "TradingEnv",
]
```

```python
# src/trading_model/data/__init__.py
from trading_model.data.loader import load_days_from_dataframe

__all__ = ["load_days_from_dataframe"]
```

**Step 2: Run full test suite**

```bash
uv run pytest -v
```

Expected: All tests pass (26+ tests across 5 test files)

**Step 3: Commit**

```bash
git add src/trading_model/__init__.py src/trading_model/env/__init__.py src/trading_model/data/__init__.py
git commit -m "feat: add public package exports"
```

---

### Task 9: Integration Smoke Test

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Write integration test**

```python
# tests/test_integration.py
import numpy as np
import pytest

from trading_model import TradingEnv
from trading_model.env import FeatureEngine


class TestIntegrationFullEpisode:
    def test_random_agent_completes_episode(self, synthetic_ohlcv):
        """A random agent should complete a full episode without errors."""
        env = TradingEnv(data_days=[synthetic_ohlcv], shuffle=False)
        obs, _ = env.reset(seed=42)

        total_reward = 0.0
        steps = 0
        terminated = False

        while not terminated:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            assert not truncated

        expected_steps = 400 - FeatureEngine.WARMUP_PERIOD
        assert steps == expected_steps
        assert "portfolio_value" in info
        assert info["portfolio_value"] > 0  # shouldn't go bankrupt in 1 day

    def test_multi_day_episode_cycling(self, synthetic_ohlcv):
        """Env should cycle through multiple days."""
        # Create 3 slightly different days
        days = []
        for i in range(3):
            day = synthetic_ohlcv.copy()
            day["close"] = day["close"] * (1 + i * 0.01)
            days.append(day)

        env = TradingEnv(data_days=days, shuffle=False)

        for episode in range(3):
            obs, _ = env.reset(seed=42)
            terminated = False
            while not terminated:
                _, _, terminated, _, _ = env.step(0)

    def test_buy_and_hold_beats_zero(self, synthetic_ohlcv):
        """Buy-and-hold on trending data should yield non-negative value."""
        # Make data trend upward
        trending = synthetic_ohlcv.copy()
        n = len(trending)
        trend = np.linspace(1.0, 1.1, n)
        for col in ["open", "high", "low", "close"]:
            trending[col] = trending[col] * trend

        env = TradingEnv(data_days=[trending], shuffle=False, fee_rate=0.0001)
        env.reset(seed=42)

        # Buy 100% on first step
        env.step(3)

        # Hold until end
        terminated = False
        info = {}
        while not terminated:
            _, _, terminated, _, info = env.step(0)

        # With a 10% uptrend and minimal fees, portfolio should grow
        assert info["portfolio_value"] > 100_000.0
```

**Step 2: Run integration tests**

```bash
uv run pytest tests/test_integration.py -v
```

Expected: 3 passed

**Step 3: Run full suite one final time**

```bash
uv run pytest -v
```

Expected: All tests pass

**Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add integration smoke tests for full episode lifecycle"
```
