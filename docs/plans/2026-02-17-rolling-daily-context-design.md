# Rolling Daily Context Recomputation

## Problem

The 8-element daily context vector (`self._context`) is computed once before an episode starts and stays frozen across all ~364 intraday steps. The agent cannot react to how today's intraday movement changes the daily technical picture (e.g., today's rally pushing daily RSI from neutral to overbought).

## Solution

At every step, synthesize a "virtual daily bar" from intraday bars `[0..step]`, append it to the historical daily window, and recompute `DailyContextFeatureEngine.compute_context()`. The `self._context` field updates so `build_observation()` picks it up naturally.

## Architecture Changes

### FeatureEngine (features.py)

New state stored at `precompute()` time:
- `self._daily_window`: historical daily OHLCV DataFrame (stored for reuse each step)
- `self._context_engine`: internal `DailyContextFeatureEngine` instance

New method `update_context(step)`:
- Builds synthetic bar: `open=bar[0].open, high=max(bars[0:step+1].high), low=min(...), close=bar[step].close, volume=sum(...)`
- Appends synthetic bar as new row to a copy of `self._daily_window`
- Calls `self._context_engine.compute_context(appended_df)` -> `self._context`

`precompute()` signature change:
- New param: `daily_window: pd.DataFrame | None = None` (raw daily OHLCV)
- Replaces current `context: np.ndarray | None` param
- Computes initial context from daily window before any intraday bars

### TradingEnv (trading_env.py)

- `step()` calls `self.feature_engine.update_context(self._step)` before `_get_obs()`
- `reset()` accepts `daily_window` (DataFrame) instead of `daily_context` (ndarray)

### train.py

- `walkthrough_train()`: passes `daily_window` DataFrame instead of pre-computed `daily_context` ndarray
- `standard_reset()` / `tune_reset()`: unchanged — no daily_window means update_context is a no-op

## Observation Vector

Shape stays `(26,)`. Indices `[14:22]` daily context now updates every step. No downstream changes to the RL model.

## Data Flow Per Step

```
step(action)
  -> execute trade logic
  -> increment self._step
  -> feature_engine.update_context(self._step)
       -> synthesize virtual bar from intraday[0:step+1]
       -> append to daily_window copy
       -> recompute context -> self._context updated
  -> _get_obs()
       -> build_observation() reads self._context (now fresh)
```

## Backward Compatibility

When `daily_window=None` (standard_reset, tune), `update_context()` is a no-op. Identical behavior to current code.

## Performance

~0.5-1ms per step for recomputing MACD/RSI/BB on ~91 daily rows using `ta` library. ~180-360ms per episode. Negligible vs PPO forward pass.

## Test Plan

Existing tests pass unchanged (no daily_window = no-op). New tests verify:
- `update_context()` produces different context at different steps
- Synthetic bar construction is correct (open/high/low/close/volume)
- Context matches expected values for known inputs
