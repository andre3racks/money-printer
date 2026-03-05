import pandas as pd
import vectorbt as vbt

HYPERPARAMETERS = {
    "bb_window": [20, 25, 30],
    "bb_alpha": [2.0, 2.5, 3.0],
    "rsi_window": [2, 3, 4],
    "rsi_oversold": [10, 15, 20],
    "rsi_overbought": [70, 80, 90],
    "sma_window": [100, 150, 200]
}

def run_strategy(
    data: pd.DataFrame, 
    bb_window: int = 20, 
    bb_alpha: float = 2.0, 
    rsi_window: int = 3, 
    rsi_oversold: int = 15, 
    rsi_overbought: int = 80,
    sma_window: int = 200
) -> tuple[pd.Series, pd.Series]:
    
    close = data['Close']
    
    # Calculate indicators
    bb = vbt.BBANDS.run(close, window=bb_window, alpha=bb_alpha)
    rsi = vbt.RSI.run(close, window=rsi_window)
    sma = vbt.MA.run(close, window=sma_window)
    
    # Generate Entry Signals: 
    # 1. Price is stretched to the downside (below Lower Bollinger Band)
    # 2. Short-term momentum is extremely oversold
    # 3. Trend filter: Ensure we are in a macro uptrend (Price > Long-term SMA)
    entries = (close < bb.lower) & (rsi.rsi < rsi_oversold) & (close > sma.ma)
    
    # Generate Exit Signals: 
    # 1. Price reverted to the mean (crossed above Middle Bollinger Band / SMA)
    # 2. Short-term momentum is extremely overbought
    exits = (close > bb.middle) | (rsi.rsi > rsi_overbought)
    
    return entries, exits