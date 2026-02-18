import pytest
from trading_model.env.rewards import RewardCalculator


class TestRewardCalculator:
    def test_first_step_returns_scaled_weighted_return(self):
        rc = RewardCalculator(alpha=0.6, beta=0.2, window=60)
        rc.reset(100_000.0)
        # First step: risk_adjusted = 0 (not enough data), no drawdown
        reward = rc.calculate(100_100.0, step_return=0.001)
        assert reward == pytest.approx(0.6 * 0.001 * 100.0, abs=1e-6)

    def test_drawdown_penalty_applied(self):
        # Use beta=0 as baseline, then verify beta>0 gives lower reward
        rc_no_penalty = RewardCalculator(alpha=0.6, beta=0.0, window=60)
        rc_no_penalty.reset(100_000.0)
        rc_no_penalty.calculate(101_000.0, step_return=0.01)
        reward_no_penalty = rc_no_penalty.calculate(100_000.0, step_return=-0.0099)

        rc = RewardCalculator(alpha=0.6, beta=0.2, window=60)
        rc.reset(100_000.0)
        rc.calculate(101_000.0, step_return=0.01)
        reward = rc.calculate(100_000.0, step_return=-0.0099)

        assert reward < reward_no_penalty  # drawdown penalty makes it more negative

    def test_no_drawdown_penalty_when_value_increases(self):
        rc = RewardCalculator(alpha=0.6, beta=0.2, window=60)
        rc.reset(100_000.0)
        rc.calculate(101_000.0, step_return=0.01)
        reward = rc.calculate(102_000.0, step_return=0.0099)
        # No drawdown increase — penalty term is 0
        assert reward > 0

    def test_risk_adjusted_component_after_enough_steps(self):
        rc = RewardCalculator(alpha=0.6, beta=0.2, window=5)
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
        assert reward > 0.6 * 0.002 * 100.0  # risk component adds to reward

    def test_reset_clears_state(self):
        rc = RewardCalculator(alpha=0.6, beta=0.2, window=60)
        rc.reset(100_000.0)
        rc.calculate(101_000.0, step_return=0.01)
        rc.reset(50_000.0)
        # After reset, peak should be 50_000 not 101_000
        reward = rc.calculate(50_100.0, step_return=0.002)
        assert reward == pytest.approx(0.6 * 0.002 * 100.0, abs=1e-6)

    def test_risk_adjusted_clamped_to_bounds(self):
        rc = RewardCalculator(alpha=0.0, beta=0.0, window=5)
        rc.reset(100_000.0)
        # Feed returns with very high Sharpe (all identical positive -> huge mu/sigma)
        for _ in range(3):
            rc.calculate(100_100.0, step_return=0.001)
        # Even with extreme Sharpe ratio, risk_adjusted should be clamped to 1.0
        reward = rc.calculate(100_100.0, step_return=0.001)
        assert reward <= 1.0

    def test_return_scale_constant(self):
        assert RewardCalculator.RETURN_SCALE == 100.0

    def test_default_beta_is_0_2(self):
        rc = RewardCalculator()
        assert rc.beta == 0.2
