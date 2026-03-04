import pandas as pd
import vectorbt as vbt

def run_strategy(
    data: pd.DataFrame, 
    rsi_window: int = 10, 
    rsi_lower: float = 35.0, 
    rsi_upper: float = 70.0, 
    ma_window: int = 200,
    bb_window: int = 20,
    bb_alpha: float = 2.0
) -> tuple[pd.Series, pd.Series]:
    
    # Extract close prices
    close = data['Close']
    
    # Calculate indicators using vectorbt
    rsi = vbt.RSI.run(close, window=rsi_window).rsi
    ma = vbt.MA.run(close, window=ma_window).ma
    bb = vbt.BBANDS.run(close, window=bb_window, alpha=bb_alpha)
    
    # Long-term uptrend filter
    uptrend = close > ma
    
    # Mean-reversion conditions
    rsi_oversold = rsi < rsi_lower
    price_dip = close < bb.lower
    
    # Wait for momentum to tick upwards to avoid catching a falling knife
    rsi_turning_up = rsi > rsi.shift(1)
    
    # Entry condition: In an uptrend, price is dipped (BB lower or RSI oversold), and RSI is recovering
    entries = uptrend & (rsi_oversold | price_dip) & rsi_turning_up
    
    # Exit condition: Momentum overbought or price reaches upper Bollinger Band
    rsi_overbought = rsi > rsi_upper
    price_peak = close > bb.upper
    
    exits = rsi_overbought | price_peak
    
    return entries, exits