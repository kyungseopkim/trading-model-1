import pytest
from trading_model.env.rewards import RewardCalculator


class TestRewardCalculator:
    def test_first_step_returns_scaled_return(self):
        rc = RewardCalculator(eta=0.01)
        rc.reset(100_000.0)
        # First step: no EMA history, falls back to scaled return
        reward = rc.calculate(100_100.0, step_return=0.001)
        assert reward == pytest.approx(0.001 * 100.0, abs=1e-6)

    def test_positive_returns_give_positive_reward(self):
        rc = RewardCalculator(eta=0.01)
        rc.reset(100_000.0)
        # Build some history
        value = 100_000.0
        for r in [0.001, 0.002, 0.001, 0.002]:
            value *= (1 + r)
            rc.calculate(value, step_return=r)
        # Next positive return should be positive
        value *= 1.001
        reward = rc.calculate(value, step_return=0.001)
        assert reward > 0

    def test_negative_return_after_positives_gives_negative(self):
        rc = RewardCalculator(eta=0.01)
        rc.reset(100_000.0)
        value = 100_000.0
        # Build positive history
        for r in [0.001, 0.002, 0.001, 0.003]:
            value *= (1 + r)
            rc.calculate(value, step_return=r)
        # Negative return should yield negative DSR
        value *= (1 - 0.005)
        reward = rc.calculate(value, step_return=-0.005)
        assert reward < 0

    def test_reward_is_clamped(self):
        rc = RewardCalculator(eta=0.5)
        rc.reset(100_000.0)
        # Build some variance so DSR denominator is nonzero
        for r in [0.01, -0.01, 0.01, -0.01]:
            rc.calculate(100_000.0, step_return=r)
        # Even with extreme return, reward should be clamped
        reward = rc.calculate(200_000.0, step_return=1.0)
        assert -2.0 <= reward <= 2.0

    def test_reset_clears_state(self):
        rc = RewardCalculator(eta=0.01)
        rc.reset(100_000.0)
        rc.calculate(101_000.0, step_return=0.01)
        rc.reset(50_000.0)
        # After reset, EMAs are zero — should fall back to scaled return
        reward = rc.calculate(50_100.0, step_return=0.002)
        assert reward == pytest.approx(0.002 * 100.0, abs=1e-6)

    def test_default_eta(self):
        rc = RewardCalculator()
        assert rc.eta == 0.01

    def test_ema_updates(self):
        rc = RewardCalculator(eta=0.1)
        rc.reset(100_000.0)
        # After one step, EMA should be updated
        rc.calculate(100_100.0, step_return=0.001)
        assert rc._A == pytest.approx(0.1 * 0.001, abs=1e-10)
        assert rc._B == pytest.approx(0.1 * 0.001**2, abs=1e-10)
