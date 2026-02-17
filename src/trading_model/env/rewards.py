from collections import deque

import numpy as np


class RewardCalculator:
    def __init__(self, alpha: float = 0.6, beta: float = 0.5, window: int = 60):
        self.alpha = alpha
        self.beta = beta
        self.window = window
        self._returns: deque[float] = deque(maxlen=window)
        self._peak_value = 0.0
        self._prev_drawdown = 0.0

    def reset(self, initial_value: float) -> None:
        self._returns.clear()
        self._peak_value = initial_value
        self._prev_drawdown = 0.0

    def calculate(self, portfolio_value: float, step_return: float) -> float:
        """Compute Sharpe-hybrid reward with drawdown penalty.

        R_t = alpha * r_t + (1 - alpha) * (mu / sigma) - beta * max(0, DD_t - DD_{t-1})
        """
        self._returns.append(step_return)

        # Drawdown tracking
        self._peak_value = max(self._peak_value, portfolio_value)
        current_dd = (self._peak_value - portfolio_value) / self._peak_value
        dd_increase = max(0.0, current_dd - self._prev_drawdown)
        self._prev_drawdown = current_dd

        # Risk-adjusted component (rolling Sharpe)
        if len(self._returns) < 2:
            risk_adjusted = 0.0
        else:
            arr = np.array(self._returns)
            mu = arr.mean()
            sigma = arr.std()
            risk_adjusted = mu / sigma if sigma > 1e-8 else 0.0

        return self.alpha * step_return + (1 - self.alpha) * risk_adjusted - self.beta * dd_increase
