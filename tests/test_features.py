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
