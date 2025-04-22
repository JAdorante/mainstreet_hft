"""
Configuration settings for the HFT bot.
"""

# Interactive Brokers Connection Settings
IB_HOST = "127.0.0.1"  # TWS/IB Gateway host
IB_PORT = 7497       # Port for paper trading (7496 for live trading)
IB_CLIENT_ID = 6969       # Client ID for IB API connection

# Trading Settings
SYMBOLS = ["AAPL", "MSFT", "AMZN", "GOOGL"]  # Symbols to trade
DEFAULT_POSITION_SIZE = 100                   # Default position size
UPDATE_INTERVAL = 1.0                         # Strategy update interval in seconds


# Strategy Parameters
# Moving Average Crossover
MA_FAST_PERIOD = 10      # Fast moving average period
MA_SLOW_PERIOD = 30      # Slow moving average period

# Statistical Arbitrage
STAT_ARB_LOOKBACK = 100  # Lookback period for calculating statistics
STAT_ARB_ENTRY = 2.0     # Z-score threshold for entry
STAT_ARB_EXIT = 0.0      # Z-score threshold for exit

# Risk Management
MAX_POSITION_SIZE = 1000      # Maximum position size per symbol
MAX_DAILY_LOSS_PCT = 2.0      # Maximum daily loss as percentage of equity
MAX_DRAWDOWN_PCT = 5.0        # Maximum drawdown as percentage of equity
MAX_TRADES_PER_DAY = 100      # Maximum number of trades per day

# Logging
LOG_LEVEL = "INFO"       # Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_FILE = "mainstreet_hft.log" # Log file name