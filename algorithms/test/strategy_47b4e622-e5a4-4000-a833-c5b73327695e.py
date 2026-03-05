import pandas as pd
import numpy as np
import vectorbt as vbt

HYPERPARAMETERS = {
    "macd_fast": [10, 12, 14],
    "macd_slow": [21, 26, 30],
    "macd_signal": [7, 9, 11],
    "rsi_window": [10, 14, 21],
    "rsi_entry": [30, 40, 50],
    "rsi_exit": [60, 70, 80]
}

def run_strategy(
    data: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    rsi_window: int = 14,
    rsi_entry: int = 40,
    rsi_exit: int = 70
) -> tuple[pd.Series, pd.Series]:
    """
    A MACD and RSI combined strategy.
    
    Entries: MACD line crosses above the Signal line AND RSI is below the entry threshold (identifying bullish momentum from a low/oversold base).
    Exits: MACD line crosses below the Signal line OR RSI exceeds the exit threshold (identifying bearish crossover or overbought conditions).
    """
    close = data['Close']
    
    # Calculate MACD
    macd_ind = vbt.MACD.run(
        close, 
        fast_window=macd_fast, 
        slow_window=macd_slow, 
        signal_window=macd_signal
    )
    macd_line = macd_ind.macd
    signal_line = macd_ind.signal
    
    # Calculate RSI
    rsi_ind = vbt.RSI.run(close, window=rsi_window)
    rsi_line = rsi_ind.rsi
    
    # Generate MACD crossover signals
    macd_cross_up = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
    macd_cross_down = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))
    
    # Determine entries and exits based on conditions
    entries = macd_cross_up & (rsi_line < rsi_entry)
    exits = macd_cross_down | (rsi_line > rsi_exit)
    
    # Ensure standard boolean series format and handle NaNs
    entries = entries.fillna(False).astype(bool)
    exits = exits.fillna(False).astype(bool)
    
    return entries, exits