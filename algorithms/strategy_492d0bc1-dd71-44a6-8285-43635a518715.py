import pandas as pd
import vectorbt as vbt

def run_strategy(
    data: pd.DataFrame, 
    rsi_window: int = 14, 
    rsi_lower: float = 35.0, 
    rsi_upper: float = 65.0, 
    ma_fast_window: int = 50,
    ma_slow_window: int = 200,
    bb_window: int = 20,
    bb_alpha: float = 2.0,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9
) -> tuple[pd.Series, pd.Series]:
    
    # Extract closing prices
    close = data['Close']
    
    # Calculate indicators using vectorbt
    rsi = vbt.RSI.run(close, window=rsi_window).rsi
    ma_fast = vbt.MA.run(close, window=ma_fast_window).ma
    ma_slow = vbt.MA.run(close, window=ma_slow_window).ma
    bb = vbt.BBANDS.run(close, window=bb_window, alpha=bb_alpha)
    macd = vbt.MACD.run(close, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)
    
    # Long-term strong uptrend filter
    uptrend = (close > ma_slow) & (ma_fast > ma_slow)
    
    # Mean-reversion (pullback) conditions
    rsi_oversold = rsi < rsi_lower
    price_dip = close < bb.lower
    
    # Wait for momentum to tick upwards to avoid catching a falling knife
    rsi_turning_up = rsi > rsi.shift(1)
    macd_improving = macd.hist > macd.hist.shift(1)
    
    # Entry condition: In an uptrend, price is dipped, and multiple momentums are recovering
    entries = uptrend & (rsi_oversold | price_dip) & rsi_turning_up & macd_improving
    
    # Exit conditions
    rsi_overbought = rsi > rsi_upper
    price_peak = close > bb.upper
    trend_failure = close < ma_slow  # Hard stop if the structural uptrend is broken
    
    exits = rsi_overbought | price_peak | trend_failure
    
    return entries, exits