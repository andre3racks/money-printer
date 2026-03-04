import os
import datetime
from pathlib import Path
import vectorbt as vbt
import pandas as pd

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def fetch_data(ticker: str = "BTC-USD", interval: str = "1h") -> pd.DataFrame:
    """
    Fetches historical data for a given ticker from Yahoo Finance using vectorbt.
    Caches the data to disk using Parquet. Only refetches if data is older than 1 week.
    """
    cache_file = DATA_DIR / f"{ticker}_{interval}.parquet"
    
    # Check cache freshness (1 week)
    if cache_file.exists():
        file_mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(cache_file))
        if datetime.datetime.now() - file_mod_time < datetime.timedelta(days=7):
            print(f"Loading {ticker} data from cache...")
            return pd.read_parquet(cache_file)
    
    print(f"Fetching fresh data for {ticker}...")
    
    # 2 years of data (slightly less than 730 days to avoid YF limits)
    start_date = (datetime.datetime.now() - datetime.timedelta(days=720)).strftime("%Y-%m-%d")
    
    try:
        data = vbt.YFData.download(
            ticker, 
            start=start_date, 
            interval=interval
        ).get()
        
        # Save to cache
        data.to_parquet(cache_file)
        return data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        if cache_file.exists():
            print("Falling back to older cache.")
            return pd.read_parquet(cache_file)
        raise e

if __name__ == "__main__":
    df = fetch_data()
    print(df.head())
    print(df.shape)
