import pandas as pd
import vectorbt as vbt

def run_strategy(data: pd.DataFrame,
                 # SMA Trend Filter Parameters (New)
                 fast_ma_window: int = 20,
                 slow_ma_window: int = 50,

                 # MACD Parameters
                 fast_window_macd: int = 12,
                 slow_window_macd: int = 26,
                 signal_window_macd: int = 9,
                 # RSI Parameters
                 rsi_window: int = 14,
                 rsi_entry_lower_bound: float = 35,
                 rsi_entry_upper_bound: float = 65,
                 rsi_exit_upper_bound: float = 70,
                 # Bollinger Bands Parameters
                 bb_window: int = 20,
                 bb_std: float = 2.0
                ) -> tuple[pd.Series, pd.Series]:
    """
    An evolved trading strategy combining MACD, RSI, Bollinger Bands, and a Simple Moving Average (SMA) Crossover
    for enhanced trend identification, momentum confirmation, and earlier trend reversal exits.

    Entry Signal:
    - **Primary Trend Confirmation (New):** A faster SMA is currently above a slower SMA, indicating an overall uptrend.
    - **MACD Bullish Momentum Trigger:** MACD line crosses above its Signal line.
    - **MACD Trend Context (New):** The MACD line itself is positive (above zero), confirming strong bullish momentum.
    - **RSI Range Filter:** RSI is within a specified range (above rsi_entry_lower_bound and below rsi_entry_upper_bound)
      to confirm momentum but avoid extremely overbought conditions at entry.
    - **Bollinger Band Confirmation:** Close price is above the Middle Bollinger Band, confirming short-term uptrend.

    Exit Signal:
    - **Primary Trend Reversal (New):** The faster SMA crosses below the slower SMA, signaling a significant shift in the overall trend.
    - OR **MACD Bearish Reversal:** MACD line crosses below its Signal line, indicating loss of bullish momentum.
    - OR **RSI Overbought Exit:** RSI crosses above rsi_exit_upper_bound, signaling overbought conditions and potential reversal.
    - OR **Bollinger Band Breakdown:** Close price crosses below the Middle Bollinger Band, indicating a short-term trend reversal.

    Args:
        data (pd.DataFrame): DataFrame with OHLCV data. Expects at least 'Close' column.
        fast_ma_window (int): Window size for the fast Simple Moving Average (SMA) used for primary trend filtering.
        slow_ma_window (int): Window size for the slow Simple Moving Average (SMA) used for primary trend filtering.
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

    # --- Calculate Indicators ---

    # 1. Simple Moving Averages for overall trend
    fast_ma = vbt.MA.run(price, window=fast_ma_window).ma
    slow_ma = vbt.MA.run(price, window=slow_ma_window).ma

    # 2. MACD
    macd_indicator = vbt.MACD.run(price,
                                  fast_window=fast_window_macd,
                                  slow_window=slow_window_macd,
                                  signal_window=signal_window_macd)
    macd_line = macd_indicator.macd
    signal_line = macd_indicator.signal

    # 3. RSI
    rsi = vbt.RSI.run(price, window=rsi_window).rsi

    # 4. Bollinger Bands
    bbands = vbt.BBANDS.run(price, window=bb_window, alpha=bb_std)
    middle_band = bbands.middle

    # --- Entry Conditions ---

    # 1. SMA Trend Filter: Fast MA must be above Slow MA (ensures overall uptrend)
    ma_trend_is_up = fast_ma > slow_ma

    # 2. MACD Bullish Crossover: MACD line crosses above its signal line (momentum trigger)
    macd_long_signal_trigger = macd_line.vbt.crossed_above(signal_line)

    # 3. MACD Positive Context: MACD line itself is positive (above zero), indicating strong bullish momentum context
    macd_positive_context = macd_line > 0

    # 4. RSI Confirmation: RSI is within the desired range for entry
    rsi_entry_ok = (rsi > rsi_entry_lower_bound) & (rsi < rsi_entry_upper_bound)

    # 5. Bollinger Bands Trend Confirmation: Close price is above the middle band
    bb_price_above_middle = price > middle_band

    # Combine all entry conditions using AND logic for strict filtering
    entries = (ma_trend_is_up &
               macd_long_signal_trigger &
               macd_positive_context &
               rsi_entry_ok &
               bb_price_above_middle)

    # --- Exit Conditions ---

    # 1. SMA Trend Reversal: Fast MA crosses below Slow MA (strong trend reversal signal)
    ma_short_signal = fast_ma.vbt.crossed_below(slow_ma)

    # 2. MACD Bearish Crossover: MACD line crosses below its signal line (loss of bullish momentum)
    macd_short_signal = macd_line.vbt.crossed_below(signal_line)

    # 3. RSI Overbought Exit: RSI crosses above the exit upper bound (overbought conditions)
    rsi_overbought_exit = rsi.vbt.crossed_above(rsi_exit_upper_bound)

    # 4. Bollinger Bands Trend Reversal: Close price crosses below the middle band
    bb_price_below_middle = price.vbt.crossed_below(middle_band)

    # Combine all exit conditions using OR logic for quick exits
    exits = (ma_short_signal |
             macd_short_signal |
             rsi_overbought_exit |
             bb_price_below_middle)

    # Ensure entries and exits are boolean Series with the same index as price
    entries = entries.reindex(price.index, fill_value=False)
    exits = exits.reindex(price.index, fill_value=False)

    return entries, exits