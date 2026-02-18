import numpy as np


class RewardCalculator:
    """Differential Sharpe Ratio (DSR) reward.

    Based on Moody & Saffell (2001). Uses exponential moving averages
    of returns to compute per-step risk-adjusted reward with proper
    temporal credit assignment.
    """

    def __init__(self, eta: float = 0.01):
        self.eta = eta
        self._A = 0.0  # EMA of returns
        self._B = 0.0  # EMA of squared returns

    def reset(self, initial_value: float) -> None:
        self._A = 0.0
        self._B = 0.0

    def calculate(self, portfolio_value: float, step_return: float) -> float:
        """Compute DSR reward for a single step.

        DSR_t = (B_{t-1} * dA - 0.5 * A_{t-1} * dB) / (B_{t-1} - A_{t-1}^2)^{3/2}
        """
        r = step_return
        dA = r - self._A
        dB = r * r - self._B

        variance = self._B - self._A * self._A
        if variance > 1e-12:
            dsr = (self._B * dA - 0.5 * self._A * dB) / (variance ** 1.5)
            dsr = float(np.clip(dsr, -2.0, 2.0))
        else:
            # Not enough history: fall back to scaled return
            dsr = r * 100.0

        # Update EMAs
        self._A += self.eta * dA
        self._B += self.eta * dB

        return dsr
