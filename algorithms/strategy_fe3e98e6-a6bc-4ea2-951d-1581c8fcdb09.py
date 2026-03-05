import pandas as pd
import vectorbt as vbt

HYPERPARAMETERS = {
    "bb_window": [14, 20, 30],
    "bb_alpha": [1.5, 2.0, 2.5],
    "rsi_window": [10, 14, 21],
    "rsi_oversold": [25, 30, 35],
    "rsi_overbought": [65, 70, 75]
}

def run_strategy(
    data: pd.DataFrame, 
    bb_window: int = 20, 
    bb_alpha: float = 2.0, 
    rsi_window: int = 14, 
    rsi_oversold: int = 30, 
    rsi_overbought: int = 70
) -> tuple[pd.Series, pd.Series]:
    
    close = data['Close']
    
    # Calculate indicators
    bb = vbt.BBANDS.run(close, window=bb_window, alpha=bb_alpha)
    rsi = vbt.RSI.run(close, window=rsi_window)
    
    # Generate Entry Signals: Price drops below Lower Bollinger Band AND RSI is Oversold
    entries = (close < bb.lower) & (rsi.rsi < rsi_oversold)
    
    # Generate Exit Signals: Price breaks above Upper Bollinger Band OR RSI is Overbought
    exits = (close > bb.upper) | (rsi.rsi > rsi_overbought)
    
    return entries, exits