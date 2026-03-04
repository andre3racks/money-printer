import pandas as pd
import vectorbt as vbt

def run_strategy(data: pd.DataFrame,
                 # MACD Parameters
                 fast_window_macd: int = 12,
                 slow_window_macd: int = 26,
                 signal_window_macd: int = 9,
                 # RSI Parameters
                 rsi_window: int = 14,
                 rsi_entry_lower_bound: float = 30, # RSI value for entry when crossed above (momentum building)
                 rsi_exit_upper_bound: float = 70,  # RSI value that triggers an exit signal if crossed (overbought)
                 # Bollinger Bands Parameters (retained for specific entry confirmation)
                 bb_window: int = 20,
                 bb_std: float = 2.0,
                 # NEW: Moving Average Trend Filter for Entry
                 ma_trend_filter_window: int = 50, # Longer-term EMA for overall trend filtering (e.g., 50-period EMA)
                 # NEW: Moving Average for Quicker Exit
                 fast_exit_ma_window: int = 20 # Faster EMA for dynamic exit trigger (e.g., 20-period EMA)
                ) -> tuple[pd.Series, pd.Series]:
    """
    An evolved trading strategy combining MACD, RSI, Bollinger Bands, and Exponential Moving Averages
    for robust trend following and momentum-based entries/exits.

    Entry Signal (All conditions must be met):
    - Overall uptrend confirmed: Close price is above a long-term EMA (ma_trend_filter_window).
    - MACD line crosses above Signal line (bullish momentum confirmation).
    - RSI crosses above rsi_entry_lower_bound (momentum building from a neutral/oversold state).
    - Close price is above the Middle Bollinger Band (confirms short-term uptrend).

    Exit Signal (Any of the following conditions will trigger an exit):
    - MACD line crosses below Signal line (loss of bullish momentum).
    - RSI crosses above rsi_exit_upper_bound (indicating overbought conditions and potential reversal).
    - Close price crosses below a faster EMA (fast_exit_ma_window) (quicker trend reversal indication for exit).

    Args:
        data (pd.DataFrame): DataFrame with OHLCV data. Expects at least 'Close' column.
        fast_window_macd (int): Window size for the fast EMA in MACD.
        slow_window_macd (int): Window size for the slow EMA in MACD.
        signal_window_macd (int): Window size for the signal line EMA in MACD.
        rsi_window (int): Window size for the RSI indicator.
        rsi_entry_lower_bound (float): RSI value that triggers an entry signal if crossed above.
                                       Ensures momentum is picking up from a neutral/oversold state.
        rsi_exit_upper_bound (float): RSI value that triggers an exit signal if exceeded.
                                      Signals exits when overbought.
        bb_window (int): Window size for the Bollinger Bands (used for entry trend confirmation).
        bb_std (float): Standard deviation multiplier for the Bollinger Bands.
        ma_trend_filter_window (int): Window size for the longer-term EMA used as a primary trend filter.
        fast_exit_ma_window (int): Window size for a faster EMA used as a dynamic exit trigger.

    Returns:
        tuple[pd.Series, pd.Series]: A tuple of (entries, exits) boolean Series.
                                     `entries` is True for long entry signals.
                                     `exits` is True for long exit signals.
    """

    price = data['Close']

    # Calculate MACD using vectorbt
    macd_indicator = vbt.MACD.run(price,
                                  fast_window=fast_window_macd,
                                  slow_window=slow_window_macd,
                                  signal_window=signal_window_macd)
    macd_line = macd_indicator.macd
    signal_line = macd_indicator.signal

    # Calculate RSI using vectorbt
    rsi = vbt.RSI.run(price, window=rsi_window).rsi

    # Calculate Bollinger Bands using vectorbt (for entry confirmation)
    bbands = vbt.BBANDS.run(price, window=bb_window, alpha=bb_std)
    middle_band = bbands.middle

    # Calculate Exponential Moving Averages for trend filtering and quick exits
    ma_trend_filter = vbt.MA.run(price, window=ma_trend_filter_window, ewm=True).ma
    fast_exit_ma = vbt.MA.run(price, window=fast_exit_ma_window, ewm=True).ma

    # --- Entry Conditions (All conditions must be met for an entry) ---
    # 1. Overall Uptrend Filter: Close price is above the long-term EMA
    trend_filter_long = price > ma_trend_filter

    # 2. MACD Bullish Crossover: MACD line crosses above its signal line
    macd_long_signal = macd_line.vbt.crossed_above(signal_line)

    # 3. RSI Momentum Pick-up: RSI crosses above its lower bound, indicating strength from neutral/oversold
    rsi_momentum_entry = rsi.vbt.crossed_above(rsi_entry_lower_bound)

    # 4. Bollinger Bands Trend Confirmation: Close price is above the middle band (inherited from ancestor)
    bb_trend_long = price > middle_band

    # Combine all entry conditions using AND logic for more selective entries
    entries = trend_filter_long & macd_long_signal & rsi_momentum_entry & bb_trend_long

    # --- Exit Conditions (Any of these conditions will trigger an exit) ---
    # 1. MACD Bearish Crossover: MACD line crosses below its signal line (loss of bullish momentum)
    macd_short_signal = macd_line.vbt.crossed_below(signal_line)

    # 2. RSI Overbought Exit: RSI crosses above the exit upper bound (indicating potential reversal from overbought)
    rsi_overbought_exit = rsi.vbt.crossed_above(rsi_exit_upper_bound)

    # 3. Fast EMA Trend Reversal: Close price crosses below the fast exit EMA
    # This provides a dynamic, quicker exit than the BB middle band cross used in the ancestor,
    # improving responsiveness to short-term trend breakdowns.
    fast_ma_exit = price.vbt.crossed_below(fast_exit_ma)

    # Combine all exit conditions using OR logic for comprehensive risk management
    exits = macd_short_signal | rsi_overbought_exit | fast_ma_exit

    # Ensure entries and exits are boolean Series with the same index as price, filling missing values with False
    entries = entries.reindex(price.index, fill_value=False)
    exits = exits.reindex(price.index, fill_value=False)

    return entries, exits