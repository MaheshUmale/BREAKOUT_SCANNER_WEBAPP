# Multi-Timeframe TTM Squeeze Backtester

## 1. Project Overview

This script is a sophisticated backtesting engine designed to test a trading strategy based on the **TTM Squeeze** indicator. It is built to be flexible, allowing for analysis across multiple intraday timeframes, and incorporates a robust risk management system.

The primary goal of the script is to simulate historical trades based on a specific set of rules and provide detailed performance metrics and visual charts to evaluate the strategy's effectiveness.

## 2. Core Features

- **Multi-Timeframe Analysis**: The script is designed to run the same backtest logic across multiple timeframes (e.g., 5-minute, 15-minute, 30-minute) in a single execution, allowing for comprehensive strategy analysis.
- **Advanced Entry Logic**: Trade entries are not based on a simple breakout. They are qualified by a **Relative Volume (RVOL)** filter, ensuring that breakouts are supported by significant market interest.
- **Bidirectional Trading**: The engine can simulate both **long** (bullish) and **short** (bearish) trades, providing a complete picture of the strategy's performance in different market conditions.
- **Dynamic Risk Management**: Instead of a fixed exit, the script uses a dynamic **1R/2R risk-reward system**:
    - **Stop-Loss**: A stop-loss is automatically set at 1R (one unit of risk).
    - **Take-Profit**: A take-profit target is set at 2R (two units of risk).
- **Automated Data Caching**: The script automatically fetches historical data using `tvdatafeed` and caches it in local CSV files. This significantly speeds up subsequent backtesting runs.
- **Detailed Performance Reporting**: For each timeframe, a comprehensive performance report is printed to the console, with a breakdown of metrics for overall, long, and short trades.
- **Trade Visualization**: The script generates and saves a PNG chart for each timeframe, visually plotting the price action along with entry/exit points and the corresponding stop-loss and take-profit levels for the last 15 trades.

## 3. Setup and Installation

1.  **Install Dependencies**: Install the required Python libraries.
    ```bash
    pip install pandas tvdatafeed matplotlib mplfinance
    ```
    *Note: `tvdatafeed` is installed directly from its GitHub repository in the setup steps.*

## 4. How to Run the Backtester

1.  Configure the parameters in the `if __name__ == "__main__"` block at the bottom of `backtest.py`:
    -   `SYMBOL`: The stock ticker to test (e.g., "HINDCOPPER").
    -   `EXCHANGE`: The exchange where the stock is traded (e.g., "NSE").
    -   `timeframes_to_test`: A list of timeframes and the number of bars to fetch for each.

2.  Run the script from your terminal:
    ```bash
    python backtest.py
    ```

## 5. Backtesting Logic: A Step-by-Step Explanation

The script's logic is encapsulated in the `run_backtest_for_timeframe` function, which executes the following steps for each timeframe.

### Step 1: Data Handling

-   The script first attempts to load historical data from a local CSV file (e.g., `HINDCOPPER_5_minute_data.csv`).
-   If the file doesn't exist, it calls the `fetch_data` function to download the data from `tvdatafeed` and saves it to the CSV for future use.

### Step 2: Indicator Calculation

The `calculate_indicators` function computes the following on the raw OHLCV data:

1.  **Bollinger Bands (BB)**: 20-period Simple Moving Average with 2 standard deviations.
2.  **Keltner Channels (KC)**: 20-period Exponential Moving Average with bands based on the Average True Range (ATR).
3.  **Average True Range (ATR)**: 14-period, used for the KC calculation.
4.  **Relative Volume (RVOL)**: The volume of the current bar divided by the 20-period average volume. An RVOL of 2.5 means the current volume is 2.5 times the recent average.

### Step 3: The TTM Squeeze State

-   A `squeeze_on` boolean column is created. It is `True` for any period where the **Bollinger Bands are inside the Keltner Channels**.

### Step 4: Trade Entry Logic

The script iterates through the data, bar by bar, looking for entry signals. A trade is only initiated if the script is not already in a position.

1.  **Squeeze Fired Condition**: The first condition is a "squeeze fire," which occurs on the bar immediately after the squeeze ends.
    ```python
    squeeze_fired = df['squeeze_on'].iloc[i-1] and not df['squeeze_on'].iloc[i]
    ```

2.  **Volume Confirmation**: The breakout must be confirmed by high volume.
    ```python
    df['rvol'].iloc[i] > 1
    ```

3.  **Breakout Direction**: The script checks for the direction of the breakout.
    -   **Bullish**: The close is above the upper Bollinger Band, which is itself above the upper Keltner Channel.
    -   **Bearish**: The close is below the lower Bollinger Band, which is itself below the lower Keltner Channel.

4.  **Trade Entry**: If all conditions are met, a trade is entered on the **opening price of the next bar**. This is crucial to prevent lookahead bias.

### Step 5: Risk Management and Exit Logic

Once a trade is entered, the exit strategy is purely based on the predefined risk parameters.

1.  **Calculate Risk (R)**: As soon as a trade is entered, the initial risk (R) is calculated.
    -   For a **long trade**, R = `entry_price - lower_bollinger_band` (of the signal bar).
    -   For a **short trade**, R = `upper_bollinger_band - entry_price` (of the signal bar).

2.  **Set Stop-Loss and Take-Profit**:
    -   **Stop-Loss** is set at **1R** away from the entry price against the trade's direction.
    -   **Take-Profit** is set at **2R** away from the entry price in the trade's favor.

3.  **Monitor for Exit**: For every subsequent bar while in a position, the script checks:
    -   If the bar's high/low has hit the take-profit level.
    -   If the bar's high/low has hit the stop-loss level.
    -   The take-profit is checked first, assuming a more favorable exit if both are hit on the same bar.
    -   When an exit is triggered, the trade details (exit price, exit date, P/L) are recorded.

### Step 6: Reporting and Visualization

After the backtest is complete:

1.  **Performance Report**: The `calculate_performance_metrics` and `generate_report` functions produce a detailed summary, which is printed to the console. The report includes metrics for overall performance as well as a breakdown for long and short trades.
2.  **Trade Plot**: The `plot_trades` function is called to generate a candlestick chart using `mplfinance`. It overlays markers for entries and exits, and draws the stop-loss and take-profit lines for the duration of each of the last 15 trades, saving the result as a PNG file.