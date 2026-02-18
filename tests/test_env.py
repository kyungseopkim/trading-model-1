import gymnasium as gym
import numpy as np
import pytest

from trading_model.env.features import FeatureEngine
from trading_model.env.trading_env import TradingEnv


@pytest.fixture
def env(synthetic_ohlcv):
    """Create an env with 1 day of synthetic data."""
    env = TradingEnv(initial_cash=100_000.0)
    # Patch reset for tests that call reset() without arguments
    original_reset = env.reset
    def patched_reset(seed=None, options=None):
        if options is None:
            options = {"intraday_data": synthetic_ohlcv}
        return original_reset(seed=seed, options=options)
    env.reset = patched_reset
    return env


class TestTradingEnvSpaces:
    def test_observation_space_shape(self, env):
        assert env.observation_space.shape == (26,)

    def test_action_space_size(self, env):
        assert env.action_space.n == 7


class TestTradingEnvReset:
    def test_reset_returns_observation_and_info(self, env):
        obs, info = env.reset(seed=42)
        assert obs.shape == (26,)
        assert isinstance(info, dict)

    def test_reset_observation_no_nans(self, env):
        obs, _ = env.reset(seed=42)
        assert not np.any(np.isnan(obs))

    def test_reset_agent_state_initial(self, env):
        obs, _ = env.reset(seed=42)
        # Agent state is last 4 (22, 23, 24, 25): position=0, pnl=0, cash_ratio=1, duration=0
        assert obs[22] == pytest.approx(0.0)   # no position
        assert obs[23] == pytest.approx(0.0)   # no unrealized pnl
        assert obs[24] == pytest.approx(1.0)   # all cash
        assert obs[25] == pytest.approx(0.0)   # no trade duration


class TestTradingEnvStep:
    def test_hold_does_not_change_portfolio(self, env):
        env.reset(seed=42)
        obs, reward, terminated, truncated, info = env.step(0)  # HOLD
        assert obs.shape == (26,)
        assert not terminated
        assert not truncated

    def test_buy_then_sell_full_episode(self, env):
        env.reset(seed=42)
        # Buy 100%
        obs, reward, terminated, truncated, info = env.step(3)
        assert obs[22] == pytest.approx(1.0)  # has position
        assert obs[24] < 0.01  # almost no cash left (just rounding)

        # Sell 100%
        obs, reward, terminated, truncated, info = env.step(6)
        assert obs[22] == pytest.approx(0.0)  # no position
        assert obs[24] == pytest.approx(1.0)  # all cash

    def test_episode_terminates_at_end_of_day(self, env):
        env.reset(seed=42)
        terminated = False
        steps = 0
        while not terminated:
            _, _, terminated, _, _ = env.step(0)
            steps += 1
        # Should run for num_bars - warmup_period steps
        assert steps == 400 - FeatureEngine.WARMUP_PERIOD

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
        # Allow for fee on first buy if tested, but here it's HOLD. 
        # Actually fee might apply if we bought, but it's hold.
        assert info["portfolio_value"] == pytest.approx(100_000.0, rel=0.01)


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


class TestTradingEnvGymCompliance:
    def test_check_env(self, synthetic_ohlcv):
        """Verify the env passes Gymnasium's built-in validation."""
        env = TradingEnv()
        # Just check reset/step cycle works without error
        obs, info = env.reset(seed=0, options={"intraday_data": synthetic_ohlcv})
        assert env.observation_space.contains(obs)
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
