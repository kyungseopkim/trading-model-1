import numpy as np
import pytest

from trading_model import TradingEnv
from trading_model.env import FeatureEngine


class TestIntegrationFullEpisode:
    def test_random_agent_completes_episode(self, synthetic_ohlcv):
        """A random agent should complete a full episode without errors."""
        env = TradingEnv()
        obs, _ = env.reset(seed=42, options={"intraday_data": synthetic_ohlcv})

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
        """Env should work with multiple days by providing different data in reset."""
        # Create 3 slightly different days
        days = []
        for i in range(3):
            day = synthetic_ohlcv.copy()
            day["close"] = day["close"] * (1 + i * 0.01)
            days.append(day)

        env = TradingEnv()

        for episode in range(3):
            obs, _ = env.reset(seed=42, options={"intraday_data": days[episode]})
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

        env = TradingEnv(fee_rate=0.0001)
        env.reset(seed=42, options={"intraday_data": trending})

        # Buy 100% on first step
        env.step(3)

        # Hold until end
        terminated = False
        info = {}
        while not terminated:
            _, _, terminated, _, info = env.step(0)

        # With a 10% uptrend and minimal fees, portfolio should grow
        assert info["portfolio_value"] > 100_000.0
