import pandas as pd
import vectorbt as vbt

def run_strategy(data: pd.DataFrame,
                 # MACD Parameters
                 fast_window_macd: int = 12,
                 slow_window_macd: int = 26,
                 signal_window_macd: int = 9,
                 # RSI Parameters
                 rsi_window: int = 14,
                 rsi_entry_lower_bound: float = 35, # Minimum RSI for entry, ensuring some momentum
                 rsi_entry_upper_bound: float = 65, # Maximum RSI for entry, avoiding buying into overbought conditions
                 rsi_exit_upper_bound: float = 70,  # RSI value that triggers an exit signal if crossed
                 # Bollinger Bands Parameters
                 bb_window: int = 20,
                 bb_std: float = 2.0
                ) -> tuple[pd.Series, pd.Series]:
    """
    An evolved trading strategy combining MACD, RSI, and Bollinger Bands for trend following and momentum confirmation.

    Entry Signal:
    - MACD line crosses above Signal line (bullish momentum).
    - RSI is within a specified range (above rsi_entry_lower_bound and below rsi_entry_upper_bound)
      to confirm momentum but avoid extremely overbought conditions at entry.
    - Close price is above the Middle Bollinger Band (confirms uptrend).

    Exit Signal:
    - MACD line crosses below Signal line (loss of bullish momentum).
    - OR RSI crosses above rsi_exit_upper_bound (indicating overbought conditions and potential reversal).
    - OR Close price crosses below the Middle Bollinger Band (trend reversal indication).

    Args:
        data (pd.DataFrame): DataFrame with OHLCV data. Expects at least 'Close' column.
        fast_window_macd (int): Window size for the fast EMA in MACD.
        slow_window_macd (int): Window size for the slow EMA in MACD.
        signal_window_macd (int): Window size for the signal line EMA in MACD.
        rsi_window (int): Window size for the RSI indicator.
        rsi_entry_lower_bound (float): Minimum RSI value required for an entry signal.
                                       Ensures some momentum and avoids buying into very weak conditions.
        rsi_entry_upper_bound (float): Maximum RSI value allowed for an entry signal.
                                       Helps avoid buying into overbought conditions.
        rsi_exit_upper_bound (float): RSI value that triggers an exit signal if exceeded.
                                      Signals exits when overbought.
        bb_window (int): Window size for the Bollinger Bands.
        bb_std (float): Standard deviation multiplier for the Bollinger Bands.

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

    # Calculate Bollinger Bands using vectorbt
    bbands = vbt.BBANDS.run(price, window=bb_window, alpha=bb_std)
    middle_band = bbands.middle

    # --- Entry Conditions ---
    # 1. MACD Bullish Crossover: MACD line crosses above its signal line
    macd_long_signal = macd_line.vbt.crossed_above(signal_line)

    # 2. RSI Confirmation: RSI is within the desired range for entry
    rsi_entry_ok = (rsi > rsi_entry_lower_bound) & (rsi < rsi_entry_upper_bound)

    # 3. Bollinger Bands Trend Confirmation: Close price is above the middle band
    bb_trend_long = price > middle_band

    # Combine all entry conditions
    entries = macd_long_signal & rsi_entry_ok & bb_trend_long

    # --- Exit Conditions ---
    # 1. MACD Bearish Crossover: MACD line crosses below its signal line
    macd_short_signal = macd_line.vbt.crossed_below(signal_line)

    # 2. RSI Overbought Exit: RSI crosses above the exit upper bound
    rsi_overbought_exit = rsi.vbt.crossed_above(rsi_exit_upper_bound)

    # 3. Bollinger Bands Trend Reversal: Close price crosses below the middle band
    bb_trend_short = price.vbt.crossed_below(middle_band)

    # Combine all exit conditions using OR
    exits = macd_short_signal | rsi_overbought_exit | bb_trend_short

    # Ensure entries and exits are boolean Series with the same index as price
    entries = entries.reindex(price.index, fill_value=False)
    exits = exits.reindex(price.index, fill_value=False)

    return entries, exits