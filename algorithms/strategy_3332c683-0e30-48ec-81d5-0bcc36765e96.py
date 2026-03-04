import pandas as pd
import vectorbt as vbt

def run_strategy(
    data: pd.DataFrame,
    fast_window: int = 10,
    slow_window: int = 50,
    rsi_window: int = 14,
    rsi_entry_threshold: float = 50.0,
    rsi_exit_threshold: float = 50.0
) -> vbt.Portfolio:
    """
    VectorBT trading strategy combining Moving Averages and RSI.

    Entry Logic:
    - A buy signal is generated when the Fast Moving Average crosses above the Slow Moving Average
      AND the RSI crosses above a specified entry threshold.

    Exit Logic:
    - A sell signal is generated when the Fast Moving Average crosses below the Slow Moving Average
      OR the RSI crosses below a specified exit threshold.

    Args:
        data (pd.DataFrame): OHLCV data with at least a 'Close' column.
        fast_window (int): The window size for the fast moving average.
        slow_window (int): The window size for the slow moving average.
        rsi_window (int): The window size for the Relative Strength Index.
        rsi_entry_threshold (float): The RSI value that must be crossed above for an entry signal.
        rsi_exit_threshold (float): The RSI value that must be crossed below for an exit signal.

    Returns:
        vbt.Portfolio: A vectorbt Portfolio object representing the backtest results.
    """
    close = data['Close']

    # Calculate Moving Averages using vectorbt's built-in indicator
    fast_ma = vbt.MA.run(close, window=fast_window, short_name='fast_ma')
    slow_ma = vbt.MA.run(close, window=slow_window, short_name='slow_ma')

    # Calculate RSI using vectorbt's built-in indicator
    rsi = vbt.RSI.run(close, window=rsi_window)

    # Generate Entry Signals
    # Condition 1: Fast MA crosses above Slow MA (bullish crossover)
    ma_crossover_up = fast_ma.ma_crossed_above(slow_ma.ma)
    # Condition 2: RSI crosses above the entry threshold (momentum confirmation)
    rsi_entry_signal = rsi.rsi_crossed_above(rsi_entry_threshold)
    # Combined entry: both conditions must be true for a long entry
    entries = ma_crossover_up & rsi_entry_signal

    # Generate Exit Signals
    # Condition 1: Fast MA crosses below Slow MA (bearish crossover)
    ma_crossover_down = fast_ma.ma_crossed_below(slow_ma.ma)
    # Condition 2: RSI crosses below the exit threshold (loss of momentum or potential overbought)
    rsi_exit_signal = rsi.rsi_crossed_below(rsi_exit_threshold)
    # Combined exit: either condition triggers an exit from a long position
    exits = ma_crossover_down | rsi_exit_signal

    # Create and return the vectorbt Portfolio object
    portfolio = vbt.Portfolio.from_signals(
        close,
        entries,
        exits,
        # Initial capital for the portfolio
        init_cash=100000.0,
        # Transaction fees as a percentage per trade (e.g., 0.1%)
        fees=0.001,
        # Slippage as a percentage per trade (e.g., 0.1%)
        slippage=0.001,
        # Whether to share cash across columns/symbols (True for single symbol, or multi-symbol portfolio)
        cash_sharing=True,
        # Use close price for order execution (default)
        call_w_price=close
    )

    return portfolio