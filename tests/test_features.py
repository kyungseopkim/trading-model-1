import numpy as np
import pytest

from trading_model.env.features import DailyContextFeatureEngine, FeatureEngine


class TestFeatureEngine:
    def test_precompute_sets_num_steps(self, synthetic_ohlcv):
        fe = FeatureEngine()
        fe.precompute(synthetic_ohlcv)
        assert fe.num_steps == 400

    def test_warmup_period_is_60(self):
        assert FeatureEngine.WARMUP_PERIOD == 60

    def test_obs_dim_is_26(self):
        assert FeatureEngine.OBS_DIM == 26

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
        assert obs.shape == (26,)
        assert obs.dtype == np.float32

    def test_build_observation_includes_agent_state(self, synthetic_ohlcv):
        fe = FeatureEngine()
        fe.precompute(synthetic_ohlcv)
        obs = fe.build_observation(
            step=30, position=1.0, unrealized_pnl_pct=0.05,
            cash_ratio=0.5, trade_duration=0.1,
        )
        # Last 4 elements are agent state (indices 22, 23, 24, 25)
        assert obs[22] == pytest.approx(1.0)   # position
        assert obs[23] == pytest.approx(0.05)  # unrealized_pnl_pct
        assert obs[24] == pytest.approx(0.5)   # cash_ratio
        assert obs[25] == pytest.approx(0.1)   # trade_duration

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
