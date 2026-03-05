import pandas as pd
import vectorbt as vbt

HYPERPARAMETERS = {
    "macd_fast": [10, 12, 14],
    "macd_slow": [21, 26, 30],
    "macd_signal": [7, 9, 11],
    "rsi_window": [10, 14, 20],
    "rsi_entry_threshold": [40, 50, 60],
    "rsi_exit_threshold": [70, 75, 80]
}

def run_strategy(
    data: pd.DataFrame, 
    macd_fast: int = 12, 
    macd_slow: int = 26, 
    macd_signal: int = 9, 
    rsi_window: int = 14, 
    rsi_entry_threshold: int = 50, 
    rsi_exit_threshold: int = 70
) -> tuple[pd.Series, pd.Series]:
    
    # Extract Close prices
    close = data['Close']
    
    # Calculate MACD
    macd_ind = vbt.MACD.run(
        close, 
        fast_window=macd_fast, 
        slow_window=macd_slow, 
        signal_window=macd_signal
    )
    
    # Calculate RSI
    rsi_ind = vbt.RSI.run(close, window=rsi_window)
    
    # Extract indicator series
    macd_line = macd_ind.macd
    signal_line = macd_ind.signal
    rsi_line = rsi_ind.rsi
    
    # Generate MACD crossovers
    macd_cross_above = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
    macd_cross_below = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))
    
    # Entry Logic: MACD crosses above Signal line AND RSI shows bullish momentum (above entry threshold)
    entries = macd_cross_above & (rsi_line > rsi_entry_threshold)
    
    # Exit Logic: MACD crosses below Signal line OR RSI reaches overbought levels
    exits = macd_cross_below | (rsi_line > rsi_exit_threshold)
    
    # Clean up NaN values created by shifts or indicator windows
    entries = entries.fillna(False)
    exits = exits.fillna(False)
    
    return entries, exits