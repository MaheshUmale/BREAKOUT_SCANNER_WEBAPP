# Real-Time Squeeze Scanner Dashboard

This interactive web application identifies and visualizes stocks undergoing a TTM Squeeze across multiple timeframes, helping traders spot potential volatility breakouts in real-time.

## Key Features

-   **Interactive Dashboard**: A clean, single-page Flask application for visualizing squeeze data.
-   **Automated Scanning**: The app runs scans automatically in the background every two minutes.
-   **On-Demand Refresh**: Manually trigger a new scan at any time directly from the UI.
-   **Multi-Timeframe Analysis**: Monitors squeezes across timeframes from 1 minute to 1 month.
-   **RVOL Filter**: Filter the results to show only stocks with a specific Relative Volume (RVOL) threshold (e.g., >1, >1.5, >2).
-   **Intelligent Event Detection**:
    -   **In Squeeze**: Shows stocks currently in a "STRONG" or "VERY STRONG" squeeze.
    -   **Fired Squeezes**: Highlights stocks that have recently fired out of a squeeze with a verifiable *increase in volatility*.
-   **Detailed Information**: Hover over any stock to see key data points like momentum, RVOL, and squeeze strength.

## Setup and Usage

### 1. Install Dependencies

Ensure you have Python installed. Then, open a terminal in the project directory and install the required packages:

```bash
pip install -r requirements.txt
```

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