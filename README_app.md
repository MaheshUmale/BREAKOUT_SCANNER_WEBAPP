# Real-Time TTM Squeeze Scanner

## 1. Project Overview

This application is a Flask-based web server designed to perform real-time scans of the stock market for a specific technical analysis pattern known as the **TTM Squeeze**. It identifies stocks that are currently in a period of low volatility ("in a squeeze"), stocks that have just entered a squeeze ("formed"), and stocks that have just broken out of a squeeze ("fired").

The results are persisted in a local database and can be viewed through a web-based dashboard that includes a heatmap visualization.

## 2. Core Features

- **Multi-Timeframe Scanning**: Scans for squeezes across 10 different timeframes, from 1 minute to monthly.
- **Event-Based Detection**: Tracks the state of squeezes between scans to identify key events:
    - **In Squeeze**: Stocks currently in a volatility compression phase.
    - **Formed**: Stocks that have just entered a squeeze since the last scan.
    - **Fired**: Stocks that have just broken out of a squeeze, indicating a potential for a significant price move.
- **Data Enrichment**: Calculates additional metrics to help qualify signals, including:
    - **Relative Volume (RVOL)**: To gauge if a breakout is supported by volume.
    - **Squeeze Strength**: To measure the intensity of the volatility compression.
    - **Momentum**: Uses the MACD histogram to determine bullish or bearish momentum.
    - **Heatmap Score**: A composite score for ranking squeeze opportunities.
- **Persistent History**: Uses a SQLite database to store historical squeeze data, which is essential for detecting "fired" and "formed" events.
- **Web-Based UI**: Provides several Flask-based API endpoints to serve data to a frontend dashboard (HTML templates are included in the `templates/` directory).
- **Automated Background Scanning**: Runs scans automatically every 2 minutes in a background thread.

## 3. Setup and Installation

1.  **Install Dependencies**: Install the required Python libraries from the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

2.  **Install `rookiepy`**: This application uses `rookiepy` to access browser cookies, which are required to authenticate with TradingView's private screeners. This library needs to be installed separately.
    ```bash
    pip install rookiepy
    ```

3.  **Browser Requirement**: You must have a Chromium-based browser (like Google Chrome or Brave) installed and be logged into your TradingView account. The script is configured to use Brave by default. To change this, modify the `rookiepy.brave` call in `app.py`.

4.  **Initialize the Database**: The first time you run the application, it will automatically create the `squeeze_history.db` file.

## 4. How It Works: Step-by-Step Logic

The core logic of the scanner is executed in the `run_scan()` function.

### Step 1: Define the Squeeze Condition

A TTM Squeeze is defined by a specific relationship between two standard indicators: Bollinger Bands® (BB) and Keltner Channels (KC).

-   **The Condition**: A stock is "in a squeeze" when its **Bollinger Bands are completely inside its Keltner Channels**.
-   **The Implication**: This indicates a period of low volatility and price consolidation. When the price breaks out of this consolidation (i.e., the Bollinger Bands move back outside the Keltner Channels), it is called a "squeeze fire," which often precedes a strong price move.

### Step 2: Load Previous State

At the beginning of each scan, the application queries the `squeeze_history.db` database to get the list of all stocks that were "in a squeeze" during the *previous* scan. This is critical for identifying state changes (new squeezes or breakouts).

```python
# Fetches the list of (ticker, timeframe) pairs from the last scan
prev_squeeze_pairs = load_previous_squeeze_list_from_db()
```

### Step 3: Scan for Stocks Currently in a Squeeze

The application uses the `tradingview_screener` library to find all stocks on the NSE exchange that currently meet the squeeze condition on at least one of the 10 specified timeframes.

-   The `squeeze_conditions` list dynamically creates the filter: `BB.upper < KltChnl.upper AND BB.lower > KltChnl.lower`.
-   Additional filters are applied for liquidity, price, and exchange.

```python
# Builds and executes the query to find all stocks in a squeeze right now
_, df_in_squeeze = query_in_squeeze.get_scanner_data(cookies=cookies)
```

### Step 4: Process and Enrich Squeeze Data

The raw data from the screener is processed to calculate additional metrics:

-   **Squeeze Count**: The number of timeframes on which a stock is simultaneously in a squeeze.
-   **Highest Timeframe**: The longest timeframe on which a squeeze is active.
-   **Squeeze Strength**: A measure of how tight the squeeze is, calculated as `KC_width / BB_width`. A higher value indicates a stronger squeeze.
-   **Relative Volume (RVOL)**: Calculated as `current_volume / average_volume`.
-   **Heatmap Score**: A proprietary score calculated from RVOL, momentum, and volatility.

### Step 5: Detect "Formed" and "Fired" Events

By comparing the current list of squeezed stocks with the list from the previous scan, the application can identify two key events:

-   **Newly Formed Squeezes**: These are stocks that are in a squeeze *now* but were *not* in the previous scan.
    ```python
    # Set difference: current squeezes minus previous squeezes
    formed_pairs = current_squeeze_set - prev_squeeze_set
    ```

-   **Newly Fired Squeezes (Breakouts)**: These are stocks that were in a squeeze in the previous scan but are *not* anymore. This indicates a breakout has just occurred.
    ```python
    # Set difference: previous squeezes minus current squeezes
    fired_pairs = prev_squeeze_set - current_squeeze_set
    ```

### Step 6: Process Fired Events

For stocks that have just "fired," the application performs additional analysis:

-   It re-queries the data for these specific tickers to get the most up-to-date price information.
-   It compares the `current_volatility` to the `previous_volatility` to ensure the breakout is accompanied by an expansion in volatility.
-   It determines the breakout direction (Bullish or Bearish) based on the closing price relative to the bands.
-   The results are saved to the `fired_squeeze_events` table in the database and appended to a daily CSV log.

### Step 7: Save Current State and Serve Data

-   The list of stocks currently in a squeeze is saved to the database, becoming the "previous state" for the next scan.
-   The processed DataFrames for "in_squeeze", "formed", and "fired" events are cached in a global variable.
-   This data is served to the frontend via the Flask API endpoints.

## 5. Application Structure

-   **`app.py`**: The main application file containing all the logic for scanning, data processing, database interaction, and the Flask web server.
-   **`templates/`**: This directory contains the HTML files for the web dashboard (e.g., `SqueezeHeatmap.html`, `Fired.html`).
-   **`squeeze_history.db`**: A SQLite database file that stores the history of squeeze states and fired events.
-   **`requirements.txt`**: A list of Python dependencies.
-   **`BBSCAN_FIRED_*.csv`**: Daily log files that record all "fired" events.

## 6. API Endpoints

The Flask application exposes several endpoints:

-   `/`: Renders the main "In Squeeze" heatmap dashboard.
-   `/fired`: Renders the dashboard for "Fired" events.
-   `/formed`: Renders the dashboard for "Formed" events.
-   `/get_latest_data`: An API endpoint that returns the latest cached scan results in JSON format. This is what the frontend polls to get live data.
-   `/scan`: A POST endpoint that triggers a manual scan.
-   `/toggle_scan`: A POST endpoint to enable or disable the background scanning thread.

## 7. How to Run the Application

1.  Ensure all setup steps are complete.
2.  Run the following command in your terminal:
    ```bash
    python app.py
    ```
3.  Open your web browser and navigate to `http://127.0.0.1:5001` to view the dashboard. The application will perform its first scan on startup.