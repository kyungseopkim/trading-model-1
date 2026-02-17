from trading_model.env.actions import ActionMapper, Action


class TestAction:
    def test_action_values(self):
        assert Action.HOLD == 0
        assert Action.BUY_25 == 1
        assert Action.BUY_50 == 2
        assert Action.BUY_100 == 3
        assert Action.SELL_25 == 4
        assert Action.SELL_50 == 5
        assert Action.SELL_100 == 6


class TestActionMapper:
    def setup_method(self):
        self.mapper = ActionMapper()

    def test_hold_returns_zero_deltas(self):
        shares_d, cash_d = self.mapper.map_action(0, cash=50_000, shares=10, price=100)
        assert shares_d == 0.0
        assert cash_d == 0.0

    def test_buy_25_spends_quarter_of_cash(self):
        shares_d, cash_d = self.mapper.map_action(1, cash=40_000, shares=0, price=100)
        assert shares_d == 100.0  # 10_000 / 100
        assert cash_d == -10_000.0

    def test_buy_50_spends_half_of_cash(self):
        shares_d, cash_d = self.mapper.map_action(2, cash=40_000, shares=0, price=200)
        assert shares_d == 100.0  # 20_000 / 200
        assert cash_d == -20_000.0

    def test_buy_100_spends_all_cash(self):
        shares_d, cash_d = self.mapper.map_action(3, cash=10_000, shares=0, price=50)
        assert shares_d == 200.0  # 10_000 / 50
        assert cash_d == -10_000.0

    def test_sell_25_sells_quarter_of_shares(self):
        shares_d, cash_d = self.mapper.map_action(4, cash=0, shares=100, price=50)
        assert shares_d == -25.0
        assert cash_d == 1_250.0  # 25 * 50

    def test_sell_50_sells_half_of_shares(self):
        shares_d, cash_d = self.mapper.map_action(5, cash=0, shares=100, price=50)
        assert shares_d == -50.0
        assert cash_d == 2_500.0

    def test_sell_100_sells_all_shares(self):
        shares_d, cash_d = self.mapper.map_action(6, cash=0, shares=100, price=50)
        assert shares_d == -100.0
        assert cash_d == 5_000.0

    def test_buy_with_no_cash_is_hold(self):
        shares_d, cash_d = self.mapper.map_action(1, cash=0, shares=0, price=100)
        assert shares_d == 0.0
        assert cash_d == 0.0

    def test_sell_with_no_shares_is_hold(self):
        shares_d, cash_d = self.mapper.map_action(4, cash=1000, shares=0, price=100)
        assert shares_d == 0.0
        assert cash_d == 0.0
