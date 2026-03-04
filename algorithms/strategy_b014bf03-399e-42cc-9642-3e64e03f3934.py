import pandas as pd
import vectorbt as vbt

def run_strategy(
    data: pd.DataFrame, 
    rsi_window: int = 14, 
    rsi_lower: float = 35.0, 
    rsi_upper: float = 70.0, 
    fast_ma_window: int = 50,
    slow_ma_window: int = 200,
    bb_window: int = 20,
    bb_alpha: float = 2.25,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9
) -> tuple[pd.Series, pd.Series]:
    
    # Extract price components
    close = data['Close']
    open_price = data['Open']
    
    # Moving Averages for robust trend filtering
    fast_ma = vbt.MA.run(close, window=fast_ma_window).ma
    slow_ma = vbt.MA.run(close, window=slow_ma_window).ma
    
    # Trend Condition: Ensure strong, established uptrend
    uptrend = (fast_ma > slow_ma) & (close > slow_ma)
    
    # Calculate Oscillators and Bands
    rsi = vbt.RSI.run(close, window=rsi_window).rsi
    bb = vbt.BBANDS.run(close, window=bb_window, alpha=bb_alpha)
    macd = vbt.MACD.run(close, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)
    
    # Mean-reversion conditions
    rsi_oversold = rsi < rsi_lower
    price_dip = close < bb.lower
    
    # Momentum and Price Action Confirmation
    macd_improving = macd.hist > macd.hist.shift(1)
    bullish_candle = close > open_price
    
    # Entry logic: In an uptrend, price is dipped/oversold, momentum is turning up, with a bullish close
    entries = uptrend & (rsi_oversold | price_dip) & macd_improving & bullish_candle
    
    # Exit conditions
    rsi_overbought = rsi > rsi_upper
    price_peak = close > bb.upper
    
    # Dynamic momentum exit: MACD crosses below signal line
    macd_bearish_cross = (macd.macd < macd.signal) & (macd.macd.shift(1) >= macd.signal.shift(1))
    
    # Combine exit logic
    exits = rsi_overbought | price_peak | macd_bearish_cross
    
    return entries, exits