"""
Backtesting Script for TTM Squeeze Strategy

This script performs a backtest of a trading strategy based on the TTM Squeeze indicator
for a single stock ('NSE:HINDCOPPER').

---
Strategy Logic:
1.  **Indicator Calculation**: The script calculates:
    -   Bollinger Bands (BB): 20-period SMA with 2 standard deviations.
    -   Keltner Channels (KC): 20-period EMA with ATR(14) bands multiplied by 2.0.
    -   Average True Range (ATR): 14-period, calculated using EMA for smoothing.

2.  **Squeeze Detection**: A "squeeze" is identified when the Bollinger Bands are
    fully contained within the Keltner Channels. This signals a period of low
    volatility and consolidation.

3.  **Trade Signal (Fired Event)**: A trade signal occurs when the squeeze condition ends.
    -   The script specifically looks for a "bullish fired" event, which is defined as:
        1.  The squeeze was on in the previous period (`squeeze_on[i-1]`).
        2.  The squeeze is off in the current period (`!squeeze_on[i]`).
        3.  The closing price breaks out above the upper Bollinger Band, which is also
            above the upper Keltner Channel.

4.  **Trade Execution**:
    -   **Entry**: A long position is entered at the opening price of the day *after*
      the bullish fired signal.
    -   **Exit**: The position is held for a fixed period of 10 trading days and then
      exited at the opening price.

---
Assumptions:
-   **No Slippage or Commissions**: The backtest assumes trades are executed at the exact
    opening price without any transaction costs.
-   **Full Capital Allocation**: The profit/loss is calculated on a per-share basis and
    is not compounded. The initial capital is only used for drawdown calculation.
-   **Fixed Holding Period**: The exit strategy is not dynamic; it is based solely on a
    fixed 10-day holding period.
-   **Data Accuracy**: The historical data from `tvdatafeed` is assumed to be accurate.
-   **No Lookahead Bias**: The script is designed to avoid lookahead bias by making trading
    decisions based only on data available up to the point of the decision.
"""

import pandas as pd
from tvDatafeed import TvDatafeed, Interval

def fetch_data(symbol, exchange, interval, n_bars):
    """
    Fetches historical OHLCV data for a given symbol using tvdatafeed.

    Args:
        symbol (str): The ticker symbol.
        exchange (str): The exchange where the symbol is traded.
        interval (Interval): The timeframe for the data (e.g., Interval.in_daily).
        n_bars (int): The number of historical bars to fetch.

    Returns:
        pd.DataFrame: A DataFrame containing the OHLCV data, or None if fetching fails.
    """
    try:
        tv = TvDatafeed()
        print("Fetching data from tvdatafeed...")
        data = tv.get_hist(symbol=symbol, exchange=exchange, interval=interval, n_bars=n_bars)
        return data
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

def calculate_indicators(df, atr_period=14, bb_period=20, kc_period=20, kc_multiplier=2.0):
    """
    Calculates ATR, Bollinger Bands, and Keltner Channels for the given DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with OHLCV data.
        atr_period (int): The period for ATR calculation.
        bb_period (int): The period for Bollinger Bands calculation.
        kc_period (int): The period for Keltner Channels calculation.
        kc_multiplier (float): The multiplier for the Keltner Channels ATR bands.

    Returns:
        pd.DataFrame: The DataFrame with added indicator columns.
    """
    # ATR Calculation
    df['high_low'] = df['high'] - df['low']
    df['high_prev_close'] = abs(df['high'] - df['close'].shift(1))
    df['low_prev_close'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['high_low', 'high_prev_close', 'low_prev_close']].max(axis=1)
    # Using Exponential Moving Average (EMA) for ATR as is standard
    df['atr'] = df['tr'].ewm(alpha=1/atr_period, adjust=False).mean()

    # Bollinger Bands Calculation (20-period SMA, 2 std devs)
    df['bb_sma'] = df['close'].rolling(window=bb_period).mean()
    df['bb_std'] = df['close'].rolling(window=bb_period).std()
    df['bb_upper'] = df['bb_sma'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_sma'] - (df['bb_std'] * 2)

    # Keltner Channels Calculation (20-period SMA, 2.0 * ATR)
    df['kc_sma'] = df['close'].rolling(window=kc_period).mean()
    df['kc_upper'] = df['kc_sma'] + (df['atr'] * kc_multiplier)
    df['kc_lower'] = df['kc_sma'] - (df['atr'] * kc_multiplier)

    # Clean up intermediate columns used for calculation
    df.drop(['high_low', 'high_prev_close', 'low_prev_close', 'tr'], axis=1, inplace=True)

    return df

def run_backtest(df, hold_period=10):
    """
    Runs the backtest based on the TTM Squeeze "fired" signal.

    Args:
        df (pd.DataFrame): DataFrame with indicator data.
        hold_period (int): The number of days to hold a position after entry.

    Returns:
        pd.DataFrame: A DataFrame containing the details of all simulated trades.
    """
    trades = []
    in_position = False
    entry_date = None
    hold_counter = 0

    # Squeeze condition: True when Bollinger Bands are inside Keltner Channels
    df['squeeze_on'] = (df['bb_lower'] > df['kc_lower']) & (df['bb_upper'] < df['kc_upper'])

    # Iterate through the data to find trade signals
    for i in range(1, len(df)):
        # --- Exit Logic ---
        # If in a position, check if the holding period is over
        if in_position:
            hold_counter += 1
            if hold_counter >= hold_period:
                exit_price = df['open'].iloc[i]
                profit_loss = exit_price - df.loc[entry_date, 'entry_price']

                # Find the trade and update its exit details
                for trade in trades:
                    if trade['entry_date'] == entry_date and trade['exit_date'] is None:
                        trade['exit_date'] = df.index[i]
                        trade['exit_price'] = exit_price
                        trade['profit_loss'] = profit_loss
                        break

                in_position = False
                hold_counter = 0

        # --- Entry Logic ---
        # Only look for entry signals if not already in a position
        if not in_position:
            # A "squeeze fired" event happens the day the squeeze turns off
            squeeze_fired = df['squeeze_on'].iloc[i-1] and not df['squeeze_on'].iloc[i]

            if squeeze_fired:
                # Bullish breakout: close is above upper BB, and upper BB is above upper KC
                is_bullish_breakout = df['close'].iloc[i] > df['bb_upper'].iloc[i] and df['bb_upper'].iloc[i] > df['kc_upper'].iloc[i]

                if is_bullish_breakout:
                    # Enter on the next day's open to avoid lookahead bias
                    if i + 1 < len(df):
                        entry_price = df['open'].iloc[i+1]
                        entry_date = df.index[i+1]
                        in_position = True
                        hold_counter = 0

                        # Record the entry of the trade
                        trades.append({
                            'entry_date': entry_date,
                            'entry_price': entry_price,
                            'exit_date': None,
                            'exit_price': None,
                            'profit_loss': None
                        })
                        # Store entry price in the main dataframe for easier P/L calculation
                        df.loc[entry_date, 'entry_price'] = entry_price

    return pd.DataFrame(trades)

def calculate_performance_metrics(trades, initial_capital=100000.0):
    """
    Calculates key performance metrics from a series of trades.

    Args:
        trades (pd.DataFrame): DataFrame of trades with 'profit_loss'.
        initial_capital (float): The starting capital for the backtest.

    Returns:
        dict: A dictionary containing the calculated performance metrics.
    """
    if trades.empty or trades['profit_loss'].isnull().any():
        return {
            'Total Trades': 0,
            'Win Rate (%)': 0,
            'Total Profit/Loss ($)': 0,
            'Average Profit per Trade': 0,
            'Maximum Drawdown (%)': 0
        }

    total_trades = len(trades)
    winning_trades = trades[trades['profit_loss'] > 0]
    win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
    total_profit_loss = trades['profit_loss'].sum()
    average_profit_per_trade = total_profit_loss / total_trades if total_trades > 0 else 0

    # Calculate Maximum Drawdown
    trades['cumulative_profit'] = trades['profit_loss'].cumsum()
    trades['equity_curve'] = initial_capital + trades['cumulative_profit']
    trades['running_max'] = trades['equity_curve'].cummax()
    trades['drawdown'] = trades['running_max'] - trades['equity_curve']
    max_drawdown_value = trades['drawdown'].max()
    max_drawdown_percent = (max_drawdown_value / trades['running_max'].max()) * 100 if trades['running_max'].max() > 0 else 0

    return {
        'Total Trades': total_trades,
        'Win Rate (%)': win_rate,
        'Total Profit/Loss ($)': total_profit_loss,
        'Average Profit per Trade': average_profit_per_trade,
        'Maximum Drawdown (%)': max_drawdown_percent
    }

def generate_report(metrics):
    """
    Prints a formatted performance report.

    Args:
        metrics (dict): A dictionary of performance metrics.
    """
    print("\n--- Backtest Performance Report ---")
    print(f"Symbol: {SYMBOL}")
    print(f"Total Trades: {metrics['Total Trades']}")
    print(f"Win Rate: {metrics['Win Rate (%)']:.2f}%")
    print(f"Total Profit/Loss: ${metrics['Total Profit/Loss ($)']:.2f}")
    print(f"Average Profit per Trade: ${metrics['Average Profit per Trade']:.2f}")
    print(f"Maximum Drawdown: {metrics['Maximum Drawdown (%)']:.2f}%")
    print("-----------------------------------")


if __name__ == "__main__":
    # --- Parameters ---
    SYMBOL = "HINDCOPPER"
    EXCHANGE = "NSE"
    INTERVAL = Interval.in_daily
    N_BARS = 5000 # Number of historical bars to fetch

    # --- Load Data ---
    # Try to load data from a local CSV file first to speed up subsequent runs.
    try:
        df = pd.read_csv(f"{SYMBOL}_data.csv", index_col='datetime', parse_dates=True)
        print(f"Successfully loaded data for {SYMBOL} from CSV.")
    except FileNotFoundError:
        print("CSV file not found. Fetching data from the API...")
        df = fetch_data(SYMBOL, EXCHANGE, INTERVAL, N_BARS)
        if df is not None:
            print(f"Successfully fetched {len(df)} bars for {SYMBOL}")
            df.to_csv(f"{SYMBOL}_data.csv")
            print(f"Data saved to {SYMBOL}_data.csv")
        else:
            print("Failed to fetch data. Exiting.")
            exit()

    # --- Calculate Indicators ---
    df = calculate_indicators(df)
    df.dropna(inplace=True) # Remove rows with NaN values from indicator calculations

    # --- Run Backtest ---
    trades = run_backtest(df)

    if not trades.empty:
        print("\nBacktest Trades:")
        # Set display options for better readability
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_columns', 10)
        print(trades.to_string())

        # --- Calculate and Report Performance ---
        performance_metrics = calculate_performance_metrics(trades)
        generate_report(performance_metrics)
    else:
        print("\nNo trades were executed in this backtest.")