import pandas as pd
from sqlalchemy import create_engine, text

# SQLAlchemy connection string: mysql+pymysql://user:password@host:port/database
DB_URL = "mysql+pymysql://root:comeOn#200@10.0.0.205:3306/stockdb"
engine = create_engine(DB_URL)


def load_from_db(ticker: str, data_type: str = "historical") -> pd.DataFrame:
    """Load OHLCV data for a given ticker from the database.

    Args:
        ticker: Ticker symbol (e.g., 'NVDA').
        data_type: Dataset type ('historical' or 'daily').

    Returns:
        DataFrame with columns: datetime, open, high, low, close, volume.
    """
    table_name = f"{ticker.lower()}_{data_type}"
    print(f"Loading {data_type} data for {ticker} from table {table_name}...")
    
    time_col = "timestamp" if data_type == "historical" else "date"
    query = f"SELECT {time_col} as datetime, open, high, low, close, volume FROM `{table_name}` ORDER BY {time_col}"
    
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn)
    return df


def load_daily_window(ticker: str, end_date: pd.Timestamp, window_size: int = 90) -> pd.DataFrame:
    """Load a window of daily data ending at (but not including) end_date.

    Args:
        ticker: Ticker symbol.
        end_date: The date to end the window at.
        window_size: Number of days to include.

    Returns:
        DataFrame with columns: datetime, open, high, low, close, volume.
    """
    table_name = f"{ticker.lower()}_daily"
    query = f"""
        SELECT date as datetime, open, high, low, close, volume 
        FROM `{table_name}` 
        WHERE date < :end_date 
        ORDER BY date DESC 
        LIMIT :window_size
    """
    
    with engine.connect() as conn:
        df = pd.read_sql(
            text(query), 
            conn, 
            params={"end_date": end_date.strftime("%Y-%m-%d"), "window_size": window_size}
        )
    return df.sort_values("datetime").reset_index(drop=True)


def get_trading_days(ticker: str, start_date: str, end_date: str) -> list[pd.Timestamp]:
    """Get a list of available trading dates from the historical table.

    Args:
        ticker: Ticker symbol.
        start_date: Start date string (YYYY-MM-DD).
        end_date: End date string (YYYY-MM-DD).

    Returns:
        List of Timestamp objects.
    """
    table_name = f"{ticker.lower()}_historical"
    query = f"""
        SELECT DISTINCT DATE(timestamp) as trade_date 
        FROM `{table_name}` 
        WHERE timestamp >= :start_date AND timestamp <= :end_date 
        ORDER BY trade_date
    """
    
    with engine.connect() as conn:
        df = pd.read_sql(
            text(query), 
            conn, 
            params={"start_date": start_date, "end_date": end_date}
        )
    return pd.to_datetime(df["trade_date"]).tolist()


def load_specific_day(ticker: str, day: pd.Timestamp) -> pd.DataFrame:
    """Load intraday data for a specific day.

    Args:
        ticker: Ticker symbol.
        day: The date to load.

    Returns:
        DataFrame with columns: datetime, open, high, low, close, volume.
    """
    table_name = f"{ticker.lower()}_historical"
    query = f"""
        SELECT timestamp as datetime, open, high, low, close, volume 
        FROM `{table_name}` 
        WHERE DATE(timestamp) = :day 
        ORDER BY timestamp
    """
    
    with engine.connect() as conn:
        df = pd.read_sql(
            text(query), 
            conn, 
            params={"day": day.strftime("%Y-%m-%d")}
        )
    return df


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
