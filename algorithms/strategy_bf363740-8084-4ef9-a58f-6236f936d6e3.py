import pandas as pd
import vectorbt as vbt

def run_strategy(data: pd.DataFrame,
                 # MACD Parameters
                 fast_window_macd: int = 12,
                 slow_window_macd: int = 26,
                 signal_window_macd: int = 9,
                 # RSI Parameters
                 rsi_window: int = 14,
                 rsi_entry_lower_bound: float = 40, # Minimum RSI for entry, ensuring stronger momentum
                 rsi_entry_upper_bound: float = 70, # Maximum RSI for entry, allows entry in strong trends
                 rsi_exit_upper_bound: float = 75,  # RSI value that triggers an exit signal if crossed (overbought)
                 rsi_exit_lower_bound: float = 25,  # RSI value that triggers an exit signal if crossed (oversold/stop-loss)
                 # Bollinger Bands Parameters
                 bb_window: int = 20,
                 bb_std: float = 2.0,
                 # ADX Parameters
                 adx_window: int = 14,
                 adx_threshold: float = 25.0
                ) -> tuple[pd.Series, pd.Series]:
    """
    An evolved trading strategy combining MACD, RSI, Bollinger Bands, and ADX for trend following and momentum confirmation.

    Entry Signal:
    - MACD line crosses above Signal line (bullish momentum).
    - RSI is within a specified range (above rsi_entry_lower_bound and below rsi_entry_upper_bound)
      to confirm momentum but avoid extremely overbought conditions at entry.
    - Close price is above the Middle Bollinger Band (confirms uptrend).
    - ADX is above a threshold, indicating a strong trend is present.

    Exit Signal:
    - MACD line crosses below Signal line (loss of bullish momentum).
    - OR RSI crosses above rsi_exit_upper_bound (indicating overbought conditions and potential reversal).
    - OR RSI crosses below rsi_exit_lower_bound (indicating oversold conditions, potentially acting as a momentum-based stop-loss).
    - OR Close price crosses below the Middle Bollinger Band (trend reversal indication).

    Args:
        data (pd.DataFrame): DataFrame with OHLCV data. Expects at least 'Open', 'High', 'Low', 'Close' columns.
        fast_window_macd (int): Window size for the fast EMA in MACD.
        slow_window_macd (int): Window size for the slow EMA in MACD.
        signal_window_macd (int): Window size for the signal line EMA in MACD.
        rsi_window (int): Window size for the RSI indicator.
        rsi_entry_lower_bound (float): Minimum RSI value required for an entry signal.
                                       Ensures stronger momentum and avoids buying into very weak conditions.
        rsi_entry_upper_bound (float): Maximum RSI value allowed for an entry signal.
                                       Helps avoid buying into excessively overbought conditions while allowing strong trends.
        rsi_exit_upper_bound (float): RSI value that triggers an exit signal if exceeded.
                                      Signals exits when overbought.
        rsi_exit_lower_bound (float): RSI value that triggers an exit signal if crossed below.
                                      Signals exits when oversold, acting as a momentum-based stop-loss.
        bb_window (int): Window size for the Bollinger Bands.
        bb_std (float): Standard deviation multiplier for the Bollinger Bands.
        adx_window (int): Window size for the ADX indicator.
        adx_threshold (float): Minimum ADX value required to confirm a strong trend for entry.

    Returns:
        tuple[pd.Series, pd.Series]: A tuple of (entries, exits) boolean Series.
                                     `entries` is True for long entry signals.
                                     `exits` is True for long exit signals.
    """

    price = data['Close']
    high = data['High']
    low = data['Low']

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

    # Calculate ADX using vectorbt
    adx_indicator = vbt.ADX.run(high, low, price, window=adx_window)
    adx = adx_indicator.adx

    # --- Entry Conditions ---
    # 1. MACD Bullish Crossover: MACD line crosses above its signal line
    macd_long_signal = macd_line.vbt.crossed_above(signal_line)

    # 2. RSI Confirmation: RSI is within the desired range for entry
    rsi_entry_ok = (rsi > rsi_entry_lower_bound) & (rsi < rsi_entry_upper_bound)

    # 3. Bollinger Bands Trend Confirmation: Close price is above the middle band
    bb_trend_long = price > middle_band

    # 4. ADX Trend Strength Confirmation: ADX is above the specified threshold
    adx_strong_trend = adx > adx_threshold

    # Combine all entry conditions
    entries = macd_long_signal & rsi_entry_ok & bb_trend_long & adx_strong_trend

    # --- Exit Conditions ---
    # 1. MACD Bearish Crossover: MACD line crosses below its signal line
    macd_short_signal = macd_line.vbt.crossed_below(signal_line)

    # 2. RSI Overbought Exit: RSI crosses above the exit upper bound
    rsi_overbought_exit = rsi.vbt.crossed_above(rsi_exit_upper_bound)

    # 3. RSI Oversold Exit: RSI crosses below the exit lower bound (momentum-based stop-loss)
    rsi_oversold_exit = rsi.vbt.crossed_below(rsi_exit_lower_bound)

    # 4. Bollinger Bands Trend Reversal: Close price crosses below the middle band
    bb_trend_short = price.vbt.crossed_below(middle_band)

    # Combine all exit conditions using OR
    exits = macd_short_signal | rsi_overbought_exit | rsi_oversold_exit | bb_trend_short

    # Ensure entries and exits are boolean Series with the same index as price
    entries = entries.reindex(price.index, fill_value=False)
    exits = exits.reindex(price.index, fill_value=False)

    return entries, exits