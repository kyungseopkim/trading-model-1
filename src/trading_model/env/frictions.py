class FrictionModel:
    def __init__(self, fee_rate: float = 0.001):
        self.fee_rate = fee_rate

    def calculate_cost(self, trade_value: float) -> float:
        return abs(trade_value) * self.fee_rate
