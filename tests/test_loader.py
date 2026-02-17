import pandas as pd
import pytest

from trading_model.data.loader import load_days_from_dataframe


class TestLoadDaysFromDataFrame:
    def test_splits_by_date(self):
        df = pd.DataFrame({
            "datetime": pd.date_range("2025-01-02 09:30", periods=800, freq="min"),
            "open": range(800),
            "high": range(800),
            "low": range(800),
            "close": range(800),
            "volume": range(800),
        })
        days = load_days_from_dataframe(df)
        assert len(days) >= 1
        for day in days:
            assert list(day.columns) == ["open", "high", "low", "close", "volume"]

    def test_skips_short_days(self):
        # Day 1: 400 bars, Day 2: 10 bars (should be skipped)
        dt1 = pd.date_range("2025-01-02 09:30", periods=400, freq="min")
        dt2 = pd.date_range("2025-01-03 09:30", periods=10, freq="min")
        df = pd.DataFrame({
            "datetime": dt1.tolist() + dt2.tolist(),
            "open": range(410),
            "high": range(410),
            "low": range(410),
            "close": range(410),
            "volume": range(410),
        })
        days = load_days_from_dataframe(df, min_bars=30)
        assert len(days) == 1

    def test_returns_sorted_within_day(self):
        # Create data out of order using pd.date_range then shuffling
        timestamps = pd.date_range("2025-01-02 09:30", periods=50, freq="min")
        # Swap first two to create out-of-order data
        shuffled = [timestamps[2], timestamps[0], timestamps[1]] + list(timestamps[3:])
        df = pd.DataFrame({
            "datetime": shuffled,
            "open": range(50),
            "high": range(50),
            "low": range(50),
            "close": range(50),
            "volume": range(50),
        })
        days = load_days_from_dataframe(df, min_bars=1)
        assert len(days) == 1
        # First row should be 09:30 (was index 1 in original)
        assert days[0].iloc[0]["open"] == 1
