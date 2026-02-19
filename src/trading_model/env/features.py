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

        # Last day's indicators (daily scale — keep standard windows)
        macd_ind = MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
        close = df["close"].iloc[-1]
        macd = macd_ind.macd().iloc[-1] / close if close > 0 else 0.0
        macd_signal = macd_ind.macd_signal().iloc[-1] / close if close > 0 else 0.0

        rsi_ind = RSIIndicator(df["close"], window=14)
        rsi = rsi_ind.rsi().iloc[-1] / 100.0

        bb_ind = BollingerBands(df["close"], window=20)
        bb_upper_pct = (bb_ind.bollinger_hband().iloc[-1] - close) / close
        bb_lower_pct = (bb_ind.bollinger_lband().iloc[-1] - close) / close
        bb_width = bb_ind.bollinger_wband().iloc[-1]

        # Daily performance
        prev_close = df["close"].iloc[-2]
        daily_return = (close - prev_close) / prev_close

        # Volume ratio (relative)
        avg_vol = df["volume"].iloc[-20:].mean()
        vol_ratio = df["volume"].iloc[-1] / avg_vol if avg_vol > 0 else 1.0

        context = np.array([
            macd, macd_signal, rsi, bb_upper_pct, bb_lower_pct, bb_width,
            daily_return, np.log1p(vol_ratio)
        ], dtype=np.float32)

        return np.nan_to_num(context)


# Intraday indicator windows (scaled for minute bars)
_MACD_FAST = 26
_MACD_SLOW = 60
_MACD_SIGNAL = 18
_RSI_WINDOW = 60
_BB_WINDOW = 60
_VOL_WINDOW = 60
_CONTEXT_UPDATE_INTERVAL = 10


_MARKET_FEATURE_COLS = [
    "norm_open", "norm_high", "norm_low", "norm_close", "rel_volume",
    "macd", "macd_signal", "macd_hist", "rsi",
    "bb_upper_pct", "bb_lower_pct", "bb_width",
    "time_sin", "time_cos",
]


class FeatureEngine:
    WARMUP_PERIOD = _MACD_SLOW
    OBS_DIM = 18 + DailyContextFeatureEngine.CONTEXT_DIM

    def __init__(self):
        self._market_features: np.ndarray | None = None  # (n, 14) float32
        self._close_prices: np.ndarray | None = None     # (n,) float64
        self._raw_ohlcv: np.ndarray | None = None        # (n, 5) float64
        self._n_steps: int = 0
        self._context: np.ndarray | None = None
        self._daily_window: pd.DataFrame | None = None
        self._context_engine = DailyContextFeatureEngine()
        self._last_context_step: int = -_CONTEXT_UPDATE_INTERVAL

    def precompute(self, ohlcv: pd.DataFrame, daily_window: pd.DataFrame | None = None) -> None:
        """Precompute all technical indicators for a day's OHLCV data.

        Args:
            ohlcv: DataFrame with columns: open, high, low, close, volume.
                   Rows are minute bars in chronological order.
            daily_window: Optional daily OHLCV DataFrame for rolling context recomputation.
        """
        df = ohlcv[["open", "high", "low", "close", "volume"]].copy()
        n = len(df)

        # Price features: normalize by previous close
        prev_close = df["close"].shift(1)
        df["norm_open"] = df["open"] / prev_close - 1
        df["norm_high"] = df["high"] / prev_close - 1
        df["norm_low"] = df["low"] / prev_close - 1
        df["norm_close"] = df["close"] / prev_close - 1

        # Relative volume (ratio to rolling mean, scale-invariant)
        vol_ma = df["volume"].rolling(window=_VOL_WINDOW, min_periods=1).mean()
        df["rel_volume"] = df["volume"] / vol_ma.clip(lower=1.0)

        # MACD (minute-scale windows), normalized by price
        macd_ind = MACD(df["close"], window_slow=_MACD_SLOW, window_fast=_MACD_FAST, window_sign=_MACD_SIGNAL)
        df["macd"] = macd_ind.macd() / df["close"]
        df["macd_signal"] = macd_ind.macd_signal() / df["close"]
        df["macd_hist"] = macd_ind.macd_diff() / df["close"]

        # RSI (minute-scale), scaled to [0, 1]
        rsi_ind = RSIIndicator(df["close"], window=_RSI_WINDOW)
        df["rsi"] = rsi_ind.rsi() / 100.0

        # Bollinger Bands (minute-scale)
        bb_ind = BollingerBands(df["close"], window=_BB_WINDOW)
        df["bb_upper_pct"] = (bb_ind.bollinger_hband() - df["close"]) / df["close"]
        df["bb_lower_pct"] = (bb_ind.bollinger_lband() - df["close"]) / df["close"]
        df["bb_width"] = bb_ind.bollinger_wband()

        # Time embeddings
        t = np.arange(n, dtype=np.float64) / n
        df["time_sin"] = np.sin(2 * np.pi * t)
        df["time_cos"] = np.cos(2 * np.pi * t)

        # Fill NaN from warmup with 0
        df = df.fillna(0.0)

        # Convert to numpy arrays for fast per-step access
        self._close_prices = df["close"].to_numpy(dtype=np.float64)
        self._raw_ohlcv = df[["open", "high", "low", "close", "volume"]].to_numpy(dtype=np.float64)
        self._market_features = df[_MARKET_FEATURE_COLS].to_numpy(dtype=np.float32)
        self._n_steps = n

        self._daily_window = daily_window
        self._last_context_step = -_CONTEXT_UPDATE_INTERVAL
        if daily_window is not None:
            self._context = self._context_engine.compute_context(daily_window)
        else:
            self._context = np.zeros(DailyContextFeatureEngine.CONTEXT_DIM, dtype=np.float32)

    def _build_synthetic_bar(self, step: int) -> dict:
        """Build a synthetic daily bar from intraday bars [0..step]."""
        raw = self._raw_ohlcv
        return {
            "open": raw[0, 0],
            "high": raw[:step + 1, 1].max(),
            "low": raw[:step + 1, 2].min(),
            "close": raw[step, 3],
            "volume": raw[:step + 1, 4].sum(),
        }

    def update_context(self, step: int) -> None:
        """Recompute daily context by appending a synthetic bar for today's intraday data."""
        if self._daily_window is None:
            return
        if step - self._last_context_step < _CONTEXT_UPDATE_INTERVAL:
            return

        self._last_context_step = step
        synthetic = self._build_synthetic_bar(step)
        appended = pd.concat(
            [self._daily_window, pd.DataFrame([synthetic])],
            ignore_index=True,
        )
        self._context = self._context_engine.compute_context(appended)

    @property
    def num_steps(self) -> int:
        if self._market_features is None:
            raise RuntimeError("Call precompute() first")
        return self._n_steps

    def get_close_price(self, step: int) -> float:
        return float(self._close_prices[step])

    def get_market_features(self, step: int) -> np.ndarray:
        """Return the 14 market features at the given step."""
        return self._market_features[step].copy()

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
