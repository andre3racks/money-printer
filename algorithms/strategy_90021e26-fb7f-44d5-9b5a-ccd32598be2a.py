import pandas as pd
import vectorbt as vbt

def run_strategy(data: pd.DataFrame, 
                 fast_window: int = 10, 
                 slow_window: int = 50, 
                 rsi_window: int = 14, 
                 rsi_entry_lower_bound: float = 30, 
                 rsi_exit_upper_bound: float = 70
                ) -> tuple[pd.Series, pd.Series]:
    """
    A trading strategy using Dual Moving Average Crossover filtered by RSI.
    
    Entry Signal:
    - Fast MA crosses above Slow MA.
    - RSI is above a 'not too low' threshold (e.g., 30).
    - RSI is below an 'overbought' threshold (e.g., 70) to prevent buying at peaks.
    
    Exit Signal:
    - Fast MA crosses below Slow MA.
    - OR RSI crosses above an 'overbought' threshold (e.g., 70).
    
    Args:
        data (pd.DataFrame): DataFrame with OHLCV data. Expects at least 'Close' column.
        fast_window (int): Window size for the fast moving average.
        slow_window (int): Window size for the slow moving average.
        rsi_window (int): Window size for the RSI indicator.
        rsi_entry_lower_bound (float): Minimum RSI value required for an entry signal.
                                       Ensures some momentum and avoids buying into very weak conditions.
        rsi_exit_upper_bound (float): Maximum RSI value allowed for an entry signal, 
                                      and a level that triggers an exit signal if exceeded.
                                      Helps avoid buying into overbought conditions and signals exits when overbought.
        
    Returns:
        tuple[pd.Series, pd.Series]: A tuple of (entries, exits) boolean Series.
                                     `entries` is True for long entry signals.
                                     `exits` is True for long exit signals.
    """
    
    price = data['Close']
    
    # Calculate Moving Averages using vectorbt
    fast_ma = vbt.MA.run(price, window=fast_window, short_name='fast_ma').ma
    slow_ma = vbt.MA.run(price, window=slow_window, short_name='slow_ma').ma
    
    # Calculate RSI using vectorbt
    rsi = vbt.RSI.run(price, window=rsi_window).rsi
    
    # --- Entry Conditions ---
    # 1. Fast MA crosses above Slow MA
    ma_crossover_long = fast_ma.vbt.crossed_above(slow_ma)
    
    # 2. RSI is above the lower bound (to ensure some momentum)
    rsi_above_entry_lower = rsi > rsi_entry_lower_bound
    
    # 3. RSI is below the upper bound (to avoid buying into immediately overbought conditions)
    rsi_below_entry_upper = rsi < rsi_exit_upper_bound
    
    # Combine all entry conditions
    entries = ma_crossover_long & rsi_above_entry_lower & rsi_below_entry_upper
    
    # --- Exit Conditions ---
    # 1. Fast MA crosses below Slow MA
    ma_crossover_short = fast_ma.vbt.crossed_below(slow_ma)
    
    # 2. OR RSI crosses above the upper bound (indicating overbought conditions)
    rsi_overbought_exit = rsi > rsi_exit_upper_bound
    
    # Combine all exit conditions
    exits = ma_crossover_short | rsi_overbought_exit
    
    # Ensure entries and exits are boolean Series with the same index as price
    entries = entries.reindex(price.index, fill_value=False)
    exits = exits.reindex(price.index, fill_value=False)
    
    return entries, exits