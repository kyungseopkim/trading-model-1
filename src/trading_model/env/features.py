import numpy as np
import pandas as pd
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands


class DailyContextFeatureEngine:
    """Computes features from daily OHLCV data to serve as context for the agent."""
    CONTEXT_DIM = 8

    def compute_context(self, daily_ohlcv: pd.DataFrame) -> np.ndarray:
        """Compute technical indicators for the last day of the provided window.

        Args:
            daily_ohlcv: DataFrame with columns: datetime, open, high, low, close, volume.

        Returns:
            A 1D numpy array of indicators.
        """
        if len(daily_ohlcv) < 26: # Minimum for indicators like MACD
            return np.zeros(self.CONTEXT_DIM, dtype=np.float32)

        df = daily_ohlcv[["open", "high", "low", "close", "volume"]].copy()

        # Last day's indicators
        macd_ind = MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
        macd = macd_ind.macd().iloc[-1]
        macd_signal = macd_ind.macd_signal().iloc[-1]
        
        rsi_ind = RSIIndicator(df["close"], window=14)
        rsi = rsi_ind.rsi().iloc[-1] / 100.0

        bb_ind = BollingerBands(df["close"], window=20)
        close = df["close"].iloc[-1]
        bb_upper_pct = (bb_ind.bollinger_hband().iloc[-1] - close) / close
        bb_lower_pct = (bb_ind.bollinger_lband().iloc[-1] - close) / close
        bb_width = bb_ind.bollinger_wband().iloc[-1]

        # Daily performance
        prev_close = df["close"].iloc[-2]
        daily_return = (close - prev_close) / prev_close
        
        # Log volume trend
        avg_vol = df["volume"].iloc[-20:].mean()
        vol_ratio = df["volume"].iloc[-1] / avg_vol if avg_vol > 0 else 1.0

        context = np.array([
            macd, macd_signal, rsi, bb_upper_pct, bb_lower_pct, bb_width, 
            daily_return, np.log1p(vol_ratio)
        ], dtype=np.float32)
        
        return np.nan_to_num(context)


class FeatureEngine:
    WARMUP_PERIOD = 26
    OBS_DIM = 18 + DailyContextFeatureEngine.CONTEXT_DIM

    def __init__(self):
        self._features: pd.DataFrame | None = None
        self._context: np.ndarray | None = None

    def precompute(self, ohlcv: pd.DataFrame, context: np.ndarray | None = None) -> None:
        """Precompute all technical indicators for a day's OHLCV data.

        Args:
            ohlcv: DataFrame with columns: open, high, low, close, volume.
                   Rows are minute bars in chronological order.
            context: Optional daily context vector.
        """
        df = ohlcv[["open", "high", "low", "close", "volume"]].copy()
        n = len(df)

        # Price features: normalize by previous close
        prev_close = df["close"].shift(1)
        df["norm_open"] = df["open"] / prev_close - 1
        df["norm_high"] = df["high"] / prev_close - 1
        df["norm_low"] = df["low"] / prev_close - 1
        df["norm_close"] = df["close"] / prev_close - 1
        df["log_volume"] = np.log1p(df["volume"])

        # MACD (12, 26, 9)
        macd_ind = MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
        df["macd"] = macd_ind.macd()
        df["macd_signal"] = macd_ind.macd_signal()
        df["macd_hist"] = macd_ind.macd_diff()

        # RSI (14), scaled to [0, 1]
        rsi_ind = RSIIndicator(df["close"], window=14)
        df["rsi"] = rsi_ind.rsi() / 100.0

        # Bollinger Bands (20)
        bb_ind = BollingerBands(df["close"], window=20)
        df["bb_upper_pct"] = (bb_ind.bollinger_hband() - df["close"]) / df["close"]
        df["bb_lower_pct"] = (bb_ind.bollinger_lband() - df["close"]) / df["close"]
        df["bb_width"] = bb_ind.bollinger_wband()

        # Time embeddings
        t = np.arange(n, dtype=np.float64) / n
        df["time_sin"] = np.sin(2 * np.pi * t)
        df["time_cos"] = np.cos(2 * np.pi * t)

        # Fill NaN from warmup with 0
        df = df.fillna(0.0)

        self._features = df
        self._context = context if context is not None else np.zeros(DailyContextFeatureEngine.CONTEXT_DIM, dtype=np.float32)

    @property
    def num_steps(self) -> int:
        if self._features is None:
            raise RuntimeError("Call precompute() first")
        return len(self._features)

    def get_close_price(self, step: int) -> float:
        return float(self._features.iloc[step]["close"])

    def get_market_features(self, step: int) -> np.ndarray:
        """Return the 14 market features at the given step."""
        row = self._features.iloc[step]
        return np.array(
            [
                row["norm_open"],
                row["norm_high"],
                row["norm_low"],
                row["norm_close"],
                row["log_volume"],
                row["macd"],
                row["macd_signal"],
                row["macd_hist"],
                row["rsi"],
                row["bb_upper_pct"],
                row["bb_lower_pct"],
                row["bb_width"],
                row["time_sin"],
                row["time_cos"],
            ],
            dtype=np.float32,
        )

    def build_observation(
        self,
        step: int,
        position: float,
        unrealized_pnl_pct: float,
        cash_ratio: float,
        trade_duration: float,
    ) -> np.ndarray:
        """Build the full observation vector including daily context."""
        market = self.get_market_features(step)
        agent_state = np.array(
            [position, unrealized_pnl_pct, cash_ratio, trade_duration],
            dtype=np.float32,
        )
        return np.concatenate([market, self._context, agent_state])
