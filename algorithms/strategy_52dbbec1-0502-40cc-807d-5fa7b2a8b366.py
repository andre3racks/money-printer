import pandas as pd
import vectorbt as vbt

HYPERPARAMETERS = {
    "macd_fast": [10, 12, 14, 16],
    "macd_slow": [21, 26, 30, 34],
    "macd_signal": [7, 9, 11, 13],
    "sma_window": [50, 100, 150, 200]
}

def run_strategy(
    data: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_window: int = 200
) -> tuple[pd.Series, pd.Series]:
    
    # Extract the Close price series
    close = data['Close']
    
    # Calculate MACD indicator
    macd = vbt.MACD.run(
        close,
        fast_window=macd_fast,
        slow_window=macd_slow,
        signal_window=macd_signal
    )
    
    # Calculate Simple Moving Average for trend filtering
    sma = vbt.MA.run(close, window=sma_window)
    
    # Extract MACD lines
    macd_line = macd.macd
    signal_line = macd.signal
    
    # Generate crossover signals using vectorbt accessor
    macd_cross_up = macd_line.vbt.crossed_above(signal_line)
    macd_cross_down = macd_line.vbt.crossed_below(signal_line)
    
    # Define trend conditions based on SMA
    uptrend = close > sma.ma
    downtrend = close < sma.ma
    
    # Combine conditions for entries and exits
    # Entry: MACD crosses above Signal AND Price is above SMA (Uptrend)
    entries = macd_cross_up & uptrend
    
    # Exit: MACD crosses below Signal OR Price is below SMA (Downtrend)
    exits = macd_cross_down | downtrend
    
    # Ensure outputs are explicitly pandas Series (in case of single-column DataFrame operations)
    if isinstance(entries, pd.DataFrame):
        entries = entries.squeeze()
    if isinstance(exits, pd.DataFrame):
        exits = exits.squeeze()
        
    return entries, exits