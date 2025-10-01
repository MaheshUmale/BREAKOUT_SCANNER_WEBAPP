# Real-Time Multi-Timeframe Squeeze Scanner Web Application

This project is an interactive web application that identifies stocks in a TTM Squeeze across multiple timeframes. It's designed to help traders spot potential volatility breakouts by automatically using their local browser's TradingView session for authenticated requests.

The results are visualized through a dynamic, single-page dashboard optimized for high-density information display.

## Key Features

-   **Optimized UI**: The interface is designed to be compact, allowing for the maximum number of stock symbols to be displayed on a single screen without sacrificing readability.
-   **Automatic Authentication**: Seamlessly uses your existing TradingView login session from your local browser (via `rookiepy`) to run authenticated scans. No more manual cookie entry.
-   **Interactive Dashboard**: A clean, single-page Flask application for visualizing squeeze data.
-   **Automated Scanning**: The app runs scans automatically in the background every two minutes.
-   **On-Demand Refresh**: Manually trigger a new scan at any time directly from the UI.
-   **Multi-Timeframe Analysis**: Monitors squeezes across timeframes from 1 minute to 1 month.
-   **RVOL Filter**: Filter the results to show only stocks with a specific Relative Volume (RVOL) threshold (e.g., >1, >1.5, >2).
-   **Intelligent Event Detection**:
    -   **In Squeeze**: Shows stocks currently in a "STRONG" or "VERY STRONG" squeeze.
    -   **Fired Squeezes**: Highlights stocks that have recently fired out of a squeeze with a verifiable *increase in volatility*.
-   **Detailed Information**: Hover over any stock to see key data points like momentum, RVOL, and squeeze strength.

## How It Works

The application runs locally and uses your browser's cookies to authenticate with TradingView.

1.  **Frontend Interface**: The main interface is a single-page application built with HTML, TailwindCSS, and D3.js. It features a compact design to display a large amount of data efficiently.
2.  **Flask Backend**: The backend is a Python Flask application (`app.py`) that handles the scanning logic. It automatically finds your TradingView session cookies to make authenticated requests.
3.  **Core Scanning Logic**:
    -   **The Squeeze Condition**: Identifies stocks where the **Bollinger Bands (BB)** are inside the **Keltner Channels (KC)**.
    -   **Filtering**: Applies baseline filters for price, volume, and traded value.
    -   **Squeeze Strength**: Calculates squeeze strength and filters for only **"STRONG"** and **"VERY STRONG"** squeezes.
    -   **Event Detection**: Compares the current scan against the previous scan (stored in a local SQLite database) to identify "Newly Formed" and "Recently Fired" events.
    -   **Fired Squeeze Analysis**: Validates fired squeezes by confirming an increase in volatility and determining the breakout direction.
    -   **Dynamic RVOL**: Relative Volume is calculated dynamically based on the most relevant timeframe for each event.
4.  **Data Visualization**: The backend returns a JSON object with the processed data, which the frontend uses to render the interactive heatmaps.

## Setup and Usage

### 1. Install Dependencies

Ensure you have Python installed. Then, open a terminal in the project directory and install the required packages:

```bash
pip install -r requirements.txt
```
*Note: The application uses `rookiepy` to automatically find browser cookies. If it fails, ensure you are logged into TradingView in a supported browser (like Chrome, Firefox, or Brave).*

### 2. Run the Web Application

To start the application, run the following command in your terminal:

```bash
python app.py
```

The application will start, and you can access it by opening your web browser and navigating to:

**http://127.0.0.1:5001**

### 3. How to Use the Dashboard

The dashboard is designed for simplicity:

-   **Auto-Scan**: The application automatically fetches the latest data every two minutes. You can toggle this feature on or off with the "Auto-Scan" checkbox.
-   **RVOL Filter**: Use the **RVOL** dropdown to filter the results based on relative volume. The view will update automatically when you change the selection.
-   **Manual Refresh**: Click the **Refresh** button to trigger an immediate new scan.
-   **View Details**: Hover your mouse over any stock in the heatmap to view more detailed information.
-   **TradingView Link**: Click on any stock to open its chart directly on TradingView.
