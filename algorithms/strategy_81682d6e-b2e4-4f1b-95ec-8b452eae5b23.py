import pandas as pd
import vectorbt as vbt

def run_strategy(
    data: pd.DataFrame, 
    rsi_window: int = 14, 
    rsi_lower: float = 30.0, 
    rsi_upper: float = 70.0, 
    ma_window: int = 200
) -> tuple[pd.Series, pd.Series]:
    
    # Extract close prices
    close = data['Close']
    
    # Calculate indicators using vectorbt
    rsi = vbt.RSI.run(close, window=rsi_window)
    ma = vbt.MA.run(close, window=ma_window)
    
    rsi_series = rsi.rsi
    ma_series = ma.ma
    
    # Determine crossovers
    rsi_crossed_below = (rsi_series < rsi_lower) & (rsi_series.shift(1) >= rsi_lower)
    rsi_crossed_above = (rsi_series > rsi_upper) & (rsi_series.shift(1) <= rsi_upper)
    
    # Entry condition: RSI crosses below oversold threshold while price is above long-term MA (uptrend)
    entries = rsi_crossed_below & (close > ma_series)
    
    # Exit condition: RSI crosses above overbought threshold
    exits = rsi_crossed_above
    
    return entries, exits