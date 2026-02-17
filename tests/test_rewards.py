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
        # Feed 5 varying positive returns to build rolling stats with nonzero sigma
        returns = [0.001, 0.002, 0.001, 0.003, 0.001]
        for r in returns:
            value *= (1 + r)
            rc.calculate(value, step_return=r)
        # 6th step: risk-adjusted component should be nonzero (positive mu, positive sigma)
        value *= 1.002
        reward = rc.calculate(value, step_return=0.002)
        # With positive mean returns, mu/sigma > 0, so risk component adds to reward
        assert reward > 0.6 * 0.002  # risk component adds to reward

    def test_reset_clears_state(self):
        rc = RewardCalculator(alpha=0.6, beta=0.5, window=60)
        rc.reset(100_000.0)
        rc.calculate(101_000.0, step_return=0.01)
        rc.reset(50_000.0)
        # After reset, peak should be 50_000 not 101_000
        reward = rc.calculate(50_100.0, step_return=0.002)
        assert reward == pytest.approx(0.6 * 0.002, abs=1e-6)
