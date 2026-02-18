# Rolling Daily Context Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the 8-element daily context vector update at every intraday step by synthesizing a virtual daily bar from today's bars and recomputing indicators.

**Architecture:** `FeatureEngine.precompute()` stores the raw daily window DataFrame. A new `update_context(step)` method builds a synthetic bar from intraday bars `[0..step]`, appends it to the daily window, and calls `DailyContextFeatureEngine.compute_context()`. `TradingEnv.step()` calls `update_context()` before `_get_obs()`. When no daily window is provided, the method is a no-op for backward compatibility.

**Tech Stack:** Python, pandas, numpy, ta (MACD/RSI/BB), gymnasium, pytest

---

### Task 1: Add `update_context()` to FeatureEngine — tests

**Files:**
- Modify: `tests/conftest.py` (add daily window fixture)
- Modify: `tests/test_features.py` (add rolling context tests)

**Step 1: Add `synthetic_daily_window` fixture to conftest.py**

Add after the existing `synthetic_ohlcv` fixture in `tests/conftest.py`:

```python
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
```

**Step 2: Add tests for `update_context()` in `tests/test_features.py`**

Add a new test class at the end of the file:

```python
from trading_model.env.features import DailyContextFeatureEngine


class TestRollingDailyContext:
    def test_update_context_changes_context_at_different_steps(
        self, synthetic_ohlcv, synthetic_daily_window
    ):
        """Context should differ between early and late steps as intraday data evolves."""
        fe = FeatureEngine()
        fe.precompute(synthetic_ohlcv, daily_window=synthetic_daily_window)

        fe.update_context(30)
        context_early = fe._context.copy()

        fe.update_context(300)
        context_late = fe._context.copy()

        assert not np.array_equal(context_early, context_late), (
            "Context should change as intraday data evolves"
        )

    def test_update_context_produces_valid_shape(
        self, synthetic_ohlcv, synthetic_daily_window
    ):
        fe = FeatureEngine()
        fe.precompute(synthetic_ohlcv, daily_window=synthetic_daily_window)
        fe.update_context(50)
        assert fe._context.shape == (DailyContextFeatureEngine.CONTEXT_DIM,)
        assert fe._context.dtype == np.float32

    def test_update_context_no_nans(
        self, synthetic_ohlcv, synthetic_daily_window
    ):
        fe = FeatureEngine()
        fe.precompute(synthetic_ohlcv, daily_window=synthetic_daily_window)
        for step in range(FeatureEngine.WARMUP_PERIOD, 400, 50):
            fe.update_context(step)
            assert not np.any(np.isnan(fe._context)), f"NaN at step {step}"

    def test_update_context_noop_without_daily_window(self, synthetic_ohlcv):
        """Without daily_window, update_context should leave context as zeros."""
        fe = FeatureEngine()
        fe.precompute(synthetic_ohlcv)
        zeros = fe._context.copy()
        fe.update_context(50)
        np.testing.assert_array_equal(fe._context, zeros)

    def test_synthetic_bar_correctness(
        self, synthetic_ohlcv, synthetic_daily_window
    ):
        """The synthetic bar should have correct OHLCV from intraday slice."""
        fe = FeatureEngine()
        fe.precompute(synthetic_ohlcv, daily_window=synthetic_daily_window)

        step = 100
        bars = fe._features.iloc[: step + 1]
        expected_open = bars.iloc[0]["open"]
        expected_high = bars["high"].max()
        expected_low = bars["low"].min()
        expected_close = bars.iloc[step]["close"]
        expected_volume = bars["volume"].sum()

        synthetic_bar = fe._build_synthetic_bar(step)
        assert synthetic_bar["open"] == pytest.approx(expected_open)
        assert synthetic_bar["high"] == pytest.approx(expected_high)
        assert synthetic_bar["low"] == pytest.approx(expected_low)
        assert synthetic_bar["close"] == pytest.approx(expected_close)
        assert synthetic_bar["volume"] == pytest.approx(expected_volume)
```

**Step 3: Run tests to verify they fail**

Run: `pytest tests/test_features.py::TestRollingDailyContext -v`
Expected: FAIL — `precompute()` doesn't accept `daily_window` param yet, `update_context()` and `_build_synthetic_bar()` don't exist.

**Step 4: Commit**

```bash
git add tests/conftest.py tests/test_features.py
git commit -m "test: add failing tests for rolling daily context"
```

---

### Task 2: Implement `update_context()` and modify `precompute()` in FeatureEngine

**Files:**
- Modify: `src/trading_model/env/features.py:56-108` (FeatureEngine class)

**Step 1: Modify `FeatureEngine.__init__` (line 60-62)**

Replace:
```python
    def __init__(self):
        self._features: pd.DataFrame | None = None
        self._context: np.ndarray | None = None
```

With:
```python
    def __init__(self):
        self._features: pd.DataFrame | None = None
        self._context: np.ndarray | None = None
        self._daily_window: pd.DataFrame | None = None
        self._context_engine = DailyContextFeatureEngine()
```

**Step 2: Modify `precompute()` signature and body (line 64-108)**

Replace the signature and the last line that sets `self._context`:

Change `def precompute(self, ohlcv: pd.DataFrame, context: np.ndarray | None = None) -> None:` to:
```python
    def precompute(self, ohlcv: pd.DataFrame, daily_window: pd.DataFrame | None = None) -> None:
```

Update the docstring `context` param to:
```
            daily_window: Optional daily OHLCV DataFrame for rolling context recomputation.
```

Replace line 108:
```python
        self._context = context if context is not None else np.zeros(DailyContextFeatureEngine.CONTEXT_DIM, dtype=np.float32)
```

With:
```python
        self._daily_window = daily_window
        if daily_window is not None:
            self._context = self._context_engine.compute_context(daily_window)
        else:
            self._context = np.zeros(DailyContextFeatureEngine.CONTEXT_DIM, dtype=np.float32)
```

**Step 3: Add `_build_synthetic_bar()` method**

Add after `precompute()`, before `num_steps` property:

```python
    def _build_synthetic_bar(self, step: int) -> dict:
        """Build a synthetic daily bar from intraday bars [0..step]."""
        bars = self._features.iloc[: step + 1]
        return {
            "open": bars.iloc[0]["open"],
            "high": bars["high"].max(),
            "low": bars["low"].min(),
            "close": bars.iloc[step]["close"],
            "volume": bars["volume"].sum(),
        }
```

**Step 4: Add `update_context()` method**

Add after `_build_synthetic_bar()`:

```python
    def update_context(self, step: int) -> None:
        """Recompute daily context by appending a synthetic bar for today's intraday data."""
        if self._daily_window is None:
            return

        synthetic = self._build_synthetic_bar(step)
        appended = pd.concat(
            [self._daily_window, pd.DataFrame([synthetic])],
            ignore_index=True,
        )
        self._context = self._context_engine.compute_context(appended)
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/test_features.py -v`
Expected: ALL PASS (both existing and new tests)

**Step 6: Commit**

```bash
git add src/trading_model/env/features.py
git commit -m "feat: add rolling daily context recomputation to FeatureEngine"
```

---

### Task 3: Wire `update_context()` into TradingEnv

**Files:**
- Modify: `src/trading_model/env/trading_env.py:52-74` (reset) and `76-124` (step)

**Step 1: Write a failing test for dynamic context in env**

Add to `tests/test_env.py`:

```python
class TestRollingContextInEnv:
    def test_context_updates_during_episode(self, synthetic_ohlcv):
        """Daily context indices [14:22] should change across steps when daily_window is provided."""
        import pandas as pd

        # Build a synthetic daily window
        np.random.seed(123)
        n = 90
        base = 95.0
        returns = np.random.normal(0.001, 0.015, n)
        close = base * np.exp(np.cumsum(returns))
        high = close * (1 + np.abs(np.random.normal(0, 0.01, n)))
        low = close * (1 - np.abs(np.random.normal(0, 0.01, n)))
        open_ = low + (high - low) * np.random.uniform(0.3, 0.7, n)
        volume = np.random.randint(100_000, 5_000_000, n).astype(float)
        daily_window = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume}
        )

        env = TradingEnv()
        obs_first, _ = env.reset(
            seed=42,
            options={"intraday_data": synthetic_ohlcv, "daily_window": daily_window},
        )
        context_first = obs_first[14:22].copy()

        # Step through 100 bars
        for _ in range(100):
            obs, _, _, _, _ = env.step(0)

        context_later = obs[14:22].copy()
        assert not np.array_equal(context_first, context_later), (
            "Daily context should evolve during episode"
        )
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_env.py::TestRollingContextInEnv -v`
Expected: FAIL — `reset()` doesn't recognize `daily_window` option yet.

**Step 3: Update `TradingEnv.reset()` (trading_env.py lines 52-74)**

Replace lines 63-65:
```python
        daily_context = options.get("daily_context")

        self.feature_engine.precompute(self._intraday_data, context=daily_context)
```

With:
```python
        daily_window = options.get("daily_window")

        self.feature_engine.precompute(self._intraday_data, daily_window=daily_window)
```

Update the docstring (line 55) from `'daily_context'` to `'daily_window'`.

**Step 4: Update `TradingEnv.step()` — add update_context call**

In `step()`, after line 110 (`self._step += 1`) and before line 111 (`terminated = ...`), add:

```python
        self.feature_engine.update_context(self._step - 1)
```

Note: we pass `self._step - 1` because `self._step` was just incremented, and we want the context computed up to the bar we just processed. However, since `_get_obs()` reads `self._step` (the next bar), and the context should reflect all bars seen so far, `self._step - 1` is the last bar the agent acted on.

Wait — let me reconsider. The step flow is:
1. Get price at `self._step` (current bar)
2. Execute trade
3. `self._step += 1`
4. Check terminated
5. `_get_obs()` reads bar at `self._step` (next bar)

The context should reflect intraday bars `[0..self._step-1]` — everything up to and including the bar that was just traded on. So `update_context(self._step - 1)` is correct.

Actually simpler: since `_get_obs()` is called after the increment, and we want context to reflect "all bars seen so far including the one just acted on", we call `update_context(self._step - 1)` right after the increment.

Insert after `self._step += 1` (line 110):
```python
        self.feature_engine.update_context(self._step - 1)
```

**Step 5: Run all tests**

Run: `pytest tests/ -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add src/trading_model/env/trading_env.py tests/test_env.py
git commit -m "feat: wire rolling daily context into TradingEnv step loop"
```

---

### Task 4: Update train.py to pass daily_window instead of daily_context

**Files:**
- Modify: `src/trading_model/train.py:117-167` (walkthrough_train function)

**Step 1: Update `walkthrough_train()` — training portion**

Remove line 138:
```python
    context_engine = DailyContextFeatureEngine()
```

Replace lines 144-145:
```python
        daily_window = load_daily_window(ticker, train_day)
        daily_context = context_engine.compute_context(daily_window)
```

With:
```python
        daily_window = load_daily_window(ticker, train_day)
```

Replace line 150:
```python
        patch_env(train_env, {"intraday_data": intraday_train, "daily_context": daily_context})
```

With:
```python
        patch_env(train_env, {"intraday_data": intraday_train, "daily_window": daily_window})
```

**Step 2: Update `walkthrough_train()` — eval portion**

Replace lines 155-157:
```python
            daily_window_eval = load_daily_window(ticker, eval_day)
            daily_context_eval = context_engine.compute_context(daily_window_eval)
            patch_env(train_env, {"intraday_data": intraday_eval, "daily_context": daily_context_eval})
```

With:
```python
            daily_window_eval = load_daily_window(ticker, eval_day)
            patch_env(train_env, {"intraday_data": intraday_eval, "daily_window": daily_window_eval})
```

**Step 3: Remove unused import**

Remove `DailyContextFeatureEngine` from the import on line 13 since it's no longer used in train.py:
```python
from trading_model.env.features import DailyContextFeatureEngine
```

**Step 4: Run all tests**

Run: `pytest tests/ -v`
Expected: ALL PASS (train.py changes don't break any tests since tests don't import train.py)

**Step 5: Commit**

```bash
git add src/trading_model/train.py
git commit -m "refactor: pass daily_window DataFrame instead of precomputed context in walkthrough_train"
```

---

### Task 5: Add integration test for rolling context with full episode

**Files:**
- Modify: `tests/test_integration.py`

**Step 1: Add integration test**

Add to `tests/test_integration.py`:

```python
class TestRollingContextIntegration:
    def test_full_episode_with_rolling_context(self, synthetic_ohlcv, synthetic_daily_window):
        """Full episode with rolling context should complete without errors."""
        env = TradingEnv()
        obs, _ = env.reset(
            seed=42,
            options={"intraday_data": synthetic_ohlcv, "daily_window": synthetic_daily_window},
        )

        steps = 0
        terminated = False
        while not terminated:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            assert not truncated
            assert not np.any(np.isnan(obs[:22])), f"NaN in obs at step {steps}"

        expected_steps = 400 - FeatureEngine.WARMUP_PERIOD
        assert steps == expected_steps
        assert info["portfolio_value"] > 0
```

**Step 2: Run all tests**

Run: `pytest tests/ -v`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add integration test for rolling daily context"
```

---

### Task 6: Verify existing tests still pass (regression check)

**Step 1: Run full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: ALL PASS, no regressions.

Key things to verify:
- `test_obs_dim_is_26` still passes (OBS_DIM unchanged)
- `test_build_observation_shape` still passes (shape=26)
- `test_episode_terminates_at_end_of_day` still passes (374 steps)
- `test_update_context_noop_without_daily_window` passes (backward compat)

**Step 2: Commit (if any fixes were needed)**

No commit expected if all tests pass.
