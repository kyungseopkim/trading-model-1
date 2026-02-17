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
