from enum import IntEnum


class Action(IntEnum):
    HOLD = 0
    BUY_25 = 1
    BUY_50 = 2
    BUY_100 = 3
    SELL_25 = 4
    SELL_50 = 5
    SELL_100 = 6


_BUY_FRACTIONS = {Action.BUY_25: 0.25, Action.BUY_50: 0.50, Action.BUY_100: 1.0}
_SELL_FRACTIONS = {Action.SELL_25: 0.25, Action.SELL_50: 0.50, Action.SELL_100: 1.0}


class ActionMapper:
    def map_action(
        self, action: int, cash: float, shares: float, price: float
    ) -> tuple[float, float]:
        """Map a discrete action to (shares_delta, cash_delta).

        Positive shares_delta = buying, negative = selling.
        Cash delta is the inverse (spend cash to buy, receive cash on sell).
        Invalid actions (buy with no cash, sell with no shares) return (0, 0).
        """
        act = Action(action)

        if act == Action.HOLD:
            return 0.0, 0.0

        if act in _BUY_FRACTIONS:
            if cash <= 0 or price <= 0:
                return 0.0, 0.0
            cash_to_spend = cash * _BUY_FRACTIONS[act]
            shares_to_buy = cash_to_spend / price
            return shares_to_buy, -cash_to_spend

        if act in _SELL_FRACTIONS:
            if shares <= 0:
                return 0.0, 0.0
            shares_to_sell = shares * _SELL_FRACTIONS[act]
            cash_received = shares_to_sell * price
            return -shares_to_sell, cash_received

        return 0.0, 0.0
