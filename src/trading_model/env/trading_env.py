import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from trading_model.env.actions import ActionMapper
from trading_model.env.features import FeatureEngine
from trading_model.env.frictions import FrictionModel
from trading_model.env.rewards import RewardCalculator


class TradingEnv(gym.Env):
    """Single-stock intraday trading environment.

    One episode = one trading day of minute-level OHLCV data.
    The agent trades with discrete buy/hold/sell actions with position sizing.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        data_days: list[pd.DataFrame],
        initial_cash: float = 100_000.0,
        fee_rate: float = 0.001,
        shuffle: bool = True,
        reward_alpha: float = 0.6,
        reward_beta: float = 0.5,
        reward_window: int = 60,
    ):
        super().__init__()

        self.data_days = data_days
        self.initial_cash = initial_cash
        self.shuffle = shuffle

        self.feature_engine = FeatureEngine()
        self.action_mapper = ActionMapper()
        self.reward_calc = RewardCalculator(
            alpha=reward_alpha, beta=reward_beta, window=reward_window
        )
        self.friction = FrictionModel(fee_rate=fee_rate)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(FeatureEngine.OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(7)

        # Episode state
        self._day_index = 0
        self._day_order: list[int] = list(range(len(data_days)))
        self._step = 0
        self._cash = initial_cash
        self._shares = 0.0
        self._entry_price = 0.0
        self._trade_start_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self._day_index >= len(self.data_days):
            self._day_index = 0
            if self.shuffle:
                self.np_random.shuffle(self._day_order)

        day_idx = self._day_order[self._day_index]
        self._day_index += 1

        self.feature_engine.precompute(self.data_days[day_idx])
        self._step = FeatureEngine.WARMUP_PERIOD
        self._cash = self.initial_cash
        self._shares = 0.0
        self._entry_price = 0.0
        self._trade_start_step = self._step

        self.reward_calc.reset(self._cash)

        return self._get_obs(), {}

    def step(self, action: int):
        price = self.feature_engine.get_close_price(self._step)

        # Map action to trade
        shares_delta, cash_delta = self.action_mapper.map_action(
            action, self._cash, self._shares, price
        )

        # Compute portfolio value before trade (for return calculation)
        prev_value = self._cash + self._shares * price

        # Apply friction on trade
        if shares_delta != 0:
            trade_value = abs(shares_delta * price)
            cost = self.friction.calculate_cost(trade_value)
            self._cash -= cost

        # Update position
        self._shares += shares_delta
        self._cash += cash_delta

        # Track trade timing
        if shares_delta > 0 and self._shares == shares_delta:
            # Opened a new position
            self._entry_price = price
            self._trade_start_step = self._step
        elif self._shares <= 0:
            # Closed position
            self._shares = 0.0
            self._entry_price = 0.0
            self._trade_start_step = self._step

        # Current portfolio value
        portfolio_value = self._cash + self._shares * price

        # Step return
        step_return = (
            (portfolio_value - prev_value) / prev_value if prev_value > 0 else 0.0
        )

        # Reward
        reward = self.reward_calc.calculate(portfolio_value, step_return)

        self._step += 1
        terminated = self._step >= self.feature_engine.num_steps

        # Force liquidate at end of day
        if terminated and self._shares > 0:
            liquidation_value = self._shares * price
            cost = self.friction.calculate_cost(liquidation_value)
            self._cash += liquidation_value - cost
            self._shares = 0.0
            portfolio_value = self._cash

        info = {"portfolio_value": portfolio_value}

        if terminated:
            obs = np.zeros(FeatureEngine.OBS_DIM, dtype=np.float32)
        else:
            obs = self._get_obs()

        return obs, float(reward), terminated, False, info

    def _get_obs(self) -> np.ndarray:
        price = self.feature_engine.get_close_price(self._step)
        total_value = self._cash + self._shares * price

        position = 1.0 if self._shares > 0 else 0.0
        unrealized_pnl_pct = (
            (price - self._entry_price) / self._entry_price
            if self._entry_price > 0
            else 0.0
        )
        cash_ratio = self._cash / total_value if total_value > 0 else 1.0
        trade_duration = (
            (self._step - self._trade_start_step) / self.feature_engine.num_steps
            if self._shares > 0
            else 0.0
        )

        return self.feature_engine.build_observation(
            self._step, position, unrealized_pnl_pct, cash_ratio, trade_duration
        )
