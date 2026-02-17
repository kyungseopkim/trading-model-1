import pandas as pd


def load_days_from_dataframe(
    df: pd.DataFrame, min_bars: int = 30
) -> list[pd.DataFrame]:
    """Split a multi-day OHLCV DataFrame into a list of per-day DataFrames.

    Args:
        df: Must contain columns: datetime, open, high, low, close, volume.
        min_bars: Days with fewer bars than this are skipped.

    Returns:
        List of DataFrames, each with columns [open, high, low, close, volume],
        sorted chronologically within each day, index reset.
    """
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["_date"] = df["datetime"].dt.date

    days = []
    for _, group in df.groupby("_date"):
        if len(group) < min_bars:
            continue
        day = (
            group.sort_values("datetime")
            .reset_index(drop=True)[["open", "high", "low", "close", "volume"]]
        )
        days.append(day)

    return days
