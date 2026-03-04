import pandas as pd
import vectorbt as vbt

def run_strategy(
    data: pd.DataFrame, 
    rsi_window: int = 14, 
    rsi_lower: float = 40.0, 
    rsi_upper: float = 65.0, 
    ma_window: int = 200,
    bb_window: int = 20,
    bb_alpha: float = 2.0
) -> tuple[pd.Series, pd.Series]:
    
    # Extract close prices
    close = data['Close']
    
    # Calculate indicators using vectorbt
    rsi = vbt.RSI.run(close, window=rsi_window).rsi
    ma = vbt.MA.run(close, window=ma_window).ma
    bbands = vbt.BBANDS.run(close, window=bb_window, alpha=bb_alpha)
    
    lower_band = bbands.lower
    upper_band = bbands.upper
    
    # Entry condition: Deep pullback in an uptrend
    # Price is below the long-term MA, RSI is in the lower bounds, and price drops below the lower Bollinger Band
    entries = (rsi < rsi_lower) & (close < lower_band) & (close > ma)
    
    # Exit condition: Mean reversion profit-taking or trend-break stop-loss
    # RSI reaches overbought levels OR price stretches above upper Bollinger Band OR the long-term uptrend breaks
    exits = (rsi > rsi_upper) | (close > upper_band) | (close < ma)
    
    return entries, exits