"""
Backtesting Script for TTM Squeeze Strategy with RVOL and Risk Management

This script performs a backtest of an advanced trading strategy based on the
TTM Squeeze indicator for a single stock ('NSE:HINDCOPPER').

---
Strategy Logic:
1.  **Indicator Calculation**: The script calculates:
    -   Bollinger Bands (BB): 20-period SMA with 2 standard deviations.
    -   Keltner Channels (KC): 20-period EMA with ATR(14) bands multiplied by 2.0.
    -   Average True Range (ATR): 14-period, calculated using EMA for smoothing.
    -   Relative Volume (RVOL): Current volume divided by its 20-day simple moving average.

2.  **Squeeze Detection**: A "squeeze" is identified when the Bollinger Bands are
    fully contained within the Keltner Channels.

3.  **Trade Signal (Fired Event)**: A trade signal occurs when the squeeze ends,
    but only if the breakout has high volume.
    -   **Entry Condition**: A trade is only considered if the RVOL on the breakout
      day is greater than 1.
    -   **Bullish Breakout**: Close is above the upper BB, which is above the upper KC.
    -   **Bearish Breakout**: Close is below the lower BB, which is below the lower KC.

4.  **Trade Execution and Risk Management**:
    -   **Entry**: A position is entered at the opening price of the day *after* the
      signal day to avoid lookahead bias.
    -   **Risk (R)**: Risk is defined as the distance from the entry price to the
      opposite Bollinger Band on the signal day.
    -   **Stop-Loss**: Set at 1R from the entry price (e.g., for a long trade,
      stop-loss = entry_price - R).
    -   **Take-Profit**: Set at 2R from the entry price (e.g., for a long trade,
      take-profit = entry_price + 2*R).
    -   **Exit**: The trade is exited if the daily high/low touches either the
      stop-loss or take-profit level.

---
Assumptions:
-   **No Slippage or Commissions**: Trades are executed at the exact price levels.
-   **Full Capital Allocation**: Profit/loss is calculated on a per-share basis.
-   **Exit Priority**: On any given day, the take-profit level is checked before the
    stop-loss level, assuming a more favorable exit.
-   **Data Accuracy**: The historical data is assumed to be accurate.
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
    Calculates ATR, Bollinger Bands, Keltner Channels, and RVOL for the given DataFrame.

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
    df['atr'] = df['tr'].ewm(alpha=1/atr_period, adjust=False).mean()

    # Bollinger Bands Calculation
    df['bb_sma'] = df['close'].rolling(window=bb_period).mean()
    df['bb_std'] = df['close'].rolling(window=bb_period).std()
    df['bb_upper'] = df['bb_sma'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_sma'] - (df['bb_std'] * 2)

    # Keltner Channels Calculation
    df['kc_sma'] = df['close'].rolling(window=kc_period).mean()
    df['kc_upper'] = df['kc_sma'] + (df['atr'] * kc_multiplier)
    df['kc_lower'] = df['kc_sma'] - (df['atr'] * kc_multiplier)

    # Relative Volume (RVOL) Calculation
    df['avg_volume'] = df['volume'].rolling(window=20).mean()
    df['rvol'] = df['volume'] / df['avg_volume']

    df.drop(['high_low', 'high_prev_close', 'low_prev_close', 'tr', 'avg_volume'], axis=1, inplace=True)

    return df

def run_backtest(df):
    """
    Runs the backtest based on the TTM Squeeze "fired" signal, incorporating RVOL
    and a dynamic 1R/2R stop-loss/take-profit exit strategy.

    Args:
        df (pd.DataFrame): DataFrame with indicator data.

    Returns:
        pd.DataFrame: A DataFrame containing the details of all simulated trades.
    """
    trades = []
    in_position = False
    trade_details = {}

    df['squeeze_on'] = (df['bb_lower'] > df['kc_lower']) & (df['bb_upper'] < df['kc_upper'])

    for i in range(1, len(df)):
        # --- Exit Logic ---
        if in_position:
            current_low = df['low'].iloc[i]
            current_high = df['high'].iloc[i]
            exit_price = None

            if trade_details['direction'] == 'long':
                if current_high >= trade_details['take_profit']:
                    exit_price = trade_details['take_profit']
                elif current_low <= trade_details['stop_loss']:
                    exit_price = trade_details['stop_loss']

            elif trade_details['direction'] == 'short':
                if current_low <= trade_details['take_profit']:
                    exit_price = trade_details['take_profit']
                elif current_high >= trade_details['stop_loss']:
                    exit_price = trade_details['stop_loss']

            if exit_price is not None:
                if trade_details['direction'] == 'long':
                    profit_loss = exit_price - trade_details['entry_price']
                else:
                    profit_loss = trade_details['entry_price'] - exit_price

                trade_details.update({
                    'exit_date': df.index[i],
                    'exit_price': exit_price,
                    'profit_loss': profit_loss
                })
                trades.append(trade_details.copy())

                in_position = False
                trade_details = {}

        # --- Entry Logic ---
        if not in_position:
            squeeze_fired = df['squeeze_on'].iloc[i-1] and not df['squeeze_on'].iloc[i]

            if squeeze_fired and df['rvol'].iloc[i] > 1:
                is_bullish = df['close'].iloc[i] > df['bb_upper'].iloc[i] and df['bb_upper'].iloc[i] > df['kc_upper'].iloc[i]
                is_bearish = df['close'].iloc[i] < df['bb_lower'].iloc[i] and df['bb_lower'].iloc[i] < df['kc_lower'].iloc[i]

                direction = 'long' if is_bullish else 'short' if is_bearish else None

                if direction and i + 1 < len(df):
                    entry_price = df['open'].iloc[i+1]

                    if direction == 'long':
                        risk = entry_price - df['bb_lower'].iloc[i]
                        stop_loss = entry_price - risk
                        take_profit = entry_price + (2 * risk)
                    else: # short
                        risk = df['bb_upper'].iloc[i] - entry_price
                        stop_loss = entry_price + risk
                        take_profit = entry_price - (2 * risk)

                    if risk <= 0: continue

                    in_position = True
                    trade_details = {
                        'entry_date': df.index[i+1],
                        'entry_price': entry_price,
                        'direction': direction,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'risk': risk,
                        'exit_date': None, 'exit_price': None, 'profit_loss': None
                    }

    return pd.DataFrame([t for t in trades if t.get('exit_date') is not None])

def calculate_performance_metrics(trades, initial_capital=100000.0):
    """
    Calculates key performance metrics for all trades, and for long and short trades separately.

    Args:
        trades (pd.DataFrame): DataFrame of trades with 'profit_loss' and 'direction'.
        initial_capital (float): The starting capital for the backtest.

    Returns:
        dict: A dictionary containing the calculated performance metrics.
    """
    if trades.empty or trades['profit_loss'].isnull().any():
        return {'Overall': {}, 'Long': {}, 'Short': {}}

    results = {}

    def _calculate_subset_metrics(trade_subset):
        if trade_subset.empty: return {'Total Trades': 0, 'Win Rate (%)': 0, 'Total Profit/Loss ($)': 0}
        total = len(trade_subset)
        wins = trade_subset[trade_subset['profit_loss'] > 0]
        return {
            'Total Trades': total,
            'Win Rate (%)': (len(wins) / total) * 100,
            'Total Profit/Loss ($)': trade_subset['profit_loss'].sum()
        }

    results['Overall'] = _calculate_subset_metrics(trades)
    results['Long'] = _calculate_subset_metrics(trades[trades['direction'] == 'long'])
    results['Short'] = _calculate_subset_metrics(trades[trades['direction'] == 'short'])

    results['Overall']['Average Profit per Trade'] = trades['profit_loss'].mean()

    trades['equity_curve'] = initial_capital + trades['profit_loss'].cumsum()
    trades['running_max'] = trades['equity_curve'].cummax()
    trades['drawdown'] = trades['running_max'] - trades['equity_curve']
    max_drawdown_value = trades['drawdown'].max()
    results['Overall']['Maximum Drawdown (%)'] = (max_drawdown_value / trades['running_max'].max()) * 100 if trades['running_max'].max() > 0 else 0

    return results

def generate_report(metrics):
    """
    Prints a formatted performance report with a breakdown for long and short trades.

    Args:
        metrics (dict): A nested dictionary of performance metrics.
    """
    print("\n--- Backtest Performance Report ---")
    print(f"Symbol: {SYMBOL}")
    print("-----------------------------------")

    def _print_subset(name, data):
        print(f"\n{name} Performance:")
        if data and data['Total Trades'] > 0:
            print(f"  Total Trades: {data['Total Trades']}")
            print(f"  Win Rate: {data['Win Rate (%)']:.2f}%")
            print(f"  Total Profit/Loss: ${data['Total Profit/Loss ($)']:.2f}")
            if 'Average Profit per Trade' in data:
                print(f"  Average Profit per Trade: ${data['Average Profit per Trade']:.2f}")
            if 'Maximum Drawdown (%)' in data:
                print(f"  Maximum Drawdown: {data['Maximum Drawdown (%)']:.2f}%")
        else:
            print("  No trades to analyze.")

    _print_subset("Overall", metrics.get('Overall', {}))
    _print_subset("Long", metrics.get('Long', {}))
    _print_subset("Short", metrics.get('Short', {}))

    print("-----------------------------------")


if __name__ == "__main__":
    # --- Parameters ---
    SYMBOL = "RPOWER"
    EXCHANGE = "NSE"
    INTERVAL = Interval.in_15_minute
    N_BARS = 5000

    # --- Load Data ---
    try:
        df = pd.read_csv(f"{SYMBOL}_data.csv", index_col='datetime', parse_dates=True)
        print(f"Successfully loaded data for {SYMBOL} from CSV.")
    except FileNotFoundError:
        df = fetch_data(SYMBOL, EXCHANGE, INTERVAL, N_BARS)
        if df is not None:
            df.to_csv(f"{SYMBOL}_data.csv")
        else:
            exit()

    # --- Main Execution ---
    df = calculate_indicators(df)
    df.dropna(inplace=True)

    trades = run_backtest(df)

    if not trades.empty:
        print("\nBacktest Trades:")
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_columns', 10)
        print(trades.to_string())

        performance_metrics = calculate_performance_metrics(trades)
        generate_report(performance_metrics)
    else:
        print("\nNo trades were executed in this backtest.")