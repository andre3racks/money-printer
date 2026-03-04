import pandas as pd
import vectorbt as vbt

def run_strategy(data: pd.DataFrame,
                 # MACD Parameters
                 fast_window_macd: int = 12,
                 slow_window_macd: int = 26,
                 signal_window_macd: int = 9,
                 # RSI Parameters
                 rsi_window: int = 14,
                 rsi_entry_lower_bound: float = 30, # Lowered from 35, allows entry when RSI is lower but not oversold
                 rsi_entry_upper_bound: float = 70, # Increased from 65, allows entry in stronger momentum phases
                 rsi_exit_upper_bound: float = 75,  # Increased from 70, allows positions to run longer before overbought exit
                 # Bollinger Bands Parameters
                 bb_window: int = 20,
                 bb_std: float = 2.0,
                 # NEW: EMA Crossover Parameters for trend confirmation
                 short_ema_window: int = 10,
                 long_ema_window: int = 50
                ) -> tuple[pd.Series, pd.Series]:
    """
    An evolved trading strategy combining MACD, RSI, Bollinger Bands, and EMA Crossover
    for trend following and momentum confirmation, with refined entry/exit logic.

    Entry Signal (All conditions must be met, triggered by MACD cross, confirmed by states):
    - MACD line crosses above Signal line (bullish momentum trigger).
    - RSI is within a specified range (above rsi_entry_lower_bound and below rsi_entry_upper_bound)
      to confirm momentum but avoid extremely overbought/oversold conditions at entry. (State)
    - Close price is above the Middle Bollinger Band (confirms existing uptrend). (State - like ancestor)
    - Short EMA is above Long EMA AND Close price is above Short EMA (confirms strong underlying uptrend and price position). (State)

    Exit Signal (Any of the following conditions triggers an exit):
    - MACD line crosses below Signal line (loss of bullish momentum).
    - RSI crosses above rsi_exit_upper_bound (indicating overbought conditions and potential reversal).
    - Close price crosses below the Middle Bollinger Band (trend reversal indication).
    - Short EMA crosses below Long EMA (confirms a strong underlying downtrend/reversal).

    Args:
        data (pd.DataFrame): DataFrame with OHLCV data. Expects at least 'Close' column.
        fast_window_macd (int): Window size for the fast EMA in MACD.
        slow_window_macd (int): Window size for the slow EMA in MACD.
        signal_window_macd (int): Window size for the signal line EMA in MACD.
        rsi_window (int): Window size for the RSI indicator.
        rsi_entry_lower_bound (float): Minimum RSI value required for an entry signal.
                                       Ensures some momentum and avoids buying into very weak conditions.
        rsi_entry_upper_bound (float): Maximum RSI value allowed for an entry signal.
                                       Helps avoid buying into extremely overbought conditions.
        rsi_exit_upper_bound (float): RSI value that triggers an exit signal if exceeded.
                                      Signals exits when potentially overbought.
        bb_window (int): Window size for the Bollinger Bands.
        bb_std (float): Standard deviation multiplier for the Bollinger Bands.
        short_ema_window (int): Window size for the shorter EMA in the crossover.
        long_ema_window (int): Window size for the longer EMA in the crossover.

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

    # Calculate EMAs using vectorbt (using ewms=True for EMA)
    short_ema = vbt.MA.run(price, window=short_ema_window, ewms=True).ma
    long_ema = vbt.MA.run(price, window=long_ema_window, ewms=True).ma

    # --- Entry Conditions ---
    # 1. MACD Bullish Crossover: MACD line crosses above its signal line (Trigger)
    macd_long_signal = macd_line.vbt.crossed_above(signal_line)

    # 2. RSI Confirmation: RSI is within the desired range for entry (State)
    rsi_entry_ok = (rsi > rsi_entry_lower_bound) & (rsi < rsi_entry_upper_bound)

    # 3. Bollinger Bands Uptrend Confirmation: Close price is above the middle band (State)
    bb_uptrend_state = price > middle_band

    # 4. EMA Trend Confirmation: Short EMA is above Long EMA AND price is above the Short EMA (State)
    ema_uptrend_state = (short_ema > long_ema) & (price > short_ema)

    # Combine all entry conditions using AND
    entries = macd_long_signal & rsi_entry_ok & bb_uptrend_state & ema_uptrend_state

    # --- Exit Conditions ---
    # 1. MACD Bearish Crossover: MACD line crosses below its signal line (Trigger)
    macd_short_signal = macd_line.vbt.crossed_below(signal_line)

    # 2. RSI Overbought Exit: RSI crosses above the exit upper bound (Trigger)
    rsi_overbought_exit = rsi.vbt.crossed_above(rsi_exit_upper_bound)

    # 3. Bollinger Bands Trend Reversal: Close price crosses below the Middle Bollinger Band (Trigger)
    bb_trend_short = price.vbt.crossed_below(middle_band)

    # 4. EMA Death Cross: Short EMA crosses below Long EMA (Trigger)
    ema_death_cross = short_ema.vbt.crossed_below(long_ema)

    # Combine all exit conditions using OR
    exits = macd_short_signal | rsi_overbought_exit | bb_trend_short | ema_death_cross

    # Ensure entries and exits are boolean Series with the same index as price
    # fill_value=False is crucial for periods where indicators might not have enough data yet
    entries = entries.reindex(price.index, fill_value=False)
    exits = exits.reindex(price.index, fill_value=False)

    return entries, exits