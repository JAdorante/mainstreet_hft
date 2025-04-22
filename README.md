# Mainstreet HFT

A high-frequency trading bot using Interactive Brokers' paper trading API, with integrated stock screening capabilities.

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/mainstreet_hft.git
cd mainstreet_hft
```

2. Install dependencies:
```bash
pip install ibapi pandas numpy matplotlib requests tabulate
```

## Setting Up Interactive Brokers

1. Download and install TWS (Trader Workstation) or IB Gateway from the [Interactive Brokers website](https://www.interactivebrokers.com)
2. Set up a paper trading account if you don't have one
3. Launch TWS or IB Gateway and log in with your paper trading credentials
4. Enable API connections:
   - In TWS: Go to Edit > Global Configuration > API > Settings
   - Enable "Allow ActiveX and Socket Clients"
   - Set the socket port (default is 7497 for paper trading)
   - Allow connections from localhost

## Stock Screening Capabilities

Mainstreet HFT includes a stock screening module that can help you find trading opportunities. To use this functionality, you'll need an API key from one of the supported data providers:

- [Alpha Vantage](https://www.alphavantage.co/) (Free tier available)
- [Finnhub](https://finnhub.io/) (Free tier available)
- [IEX Cloud](https://iexcloud.io/) (Free tier available)

### Using the Stock Screener CLI

The stock screener can be used as a standalone command-line tool:

```bash
# Search for symbols
python stock_screener_cli.py --api-key YOUR_API_KEY --provider alphavantage search --keywords "tech"

# Get top gainers
python stock_screener_cli.py --api-key YOUR_API_KEY --provider iex gainers --limit 5

# Get historical data
python stock_screener_cli.py --api-key YOUR_API_KEY --provider alphavantage history --symbol AAPL --interval daily --period 1month

# Find correlated pairs for statistical arbitrage
python stock_screener_cli.py --api-key YOUR_API_KEY --provider alphavantage correlate --symbols AAPL,MSFT,GOOGL,AMZN --min-correlation 0.7
```

### Integrating Stock Screening with Trading

You can run the stock screener before starting your trading bot to find optimal trading candidates:

```bash
# Find the best correlated pairs and start trading them
python main.py --api-key YOUR_API_KEY --api-provider alphavantage --strategy stat_arb --screen --correlate
```

## Running the Trading Bot

Run the bot with your desired strategy and symbols:

```bash
# Moving Average Crossover Strategy
python main.py --symbols AAPL,MSFT --strategy ma_cross

# Statistical Arbitrage Strategy
python main.py --symbols SPY,QQQ --strategy stat_arb --update-interval 0.5
```

### Command Line Options

```
--host              TWS/IB Gateway host (default: 127.0.0.1)
--port              TWS/IB Gateway port (default: 7497 for paper trading)
--client-id         Client ID for IB API connection (default: 1)
--symbols           Comma-separated list of symbols to trade (default: AAPL)
--strategy          Trading strategy to use (ma_cross or stat_arb)
--update-interval   Strategy update interval in seconds (default: 1.0)
--api-key           API key for stock data provider
--api-provider      Stock data provider (alphavantage, finnhub, or iex)
--screen            Run stock screening before trading
--correlate         Find correlated pairs for statistical arbitrage
```

## Trading Strategies

### Moving Average Crossover

This strategy trades based on the crossing of two moving averages (fast and slow). It generates buy signals when the fast moving average crosses above the slow moving average, and sell signals when it crosses below.

### Statistical Arbitrage

This strategy looks for pairs of correlated securities and trades based on deviations from their historical relationship. When the spread between the pair widens beyond a threshold, the strategy shorts the outperforming security and buys the underperforming one, expecting the spread to revert to its mean.

## Risk Management

The built-in risk manager monitors your trading activity and can enforce various risk limits:

- Maximum position size per symbol
- Maximum daily loss percentage
- Maximum drawdown percentage 
- Maximum number of trades per day

If any of these limits are breached, the risk manager will disable trading temporarily.

## Disclaimer

This software is for educational and research purposes only. It is not intended for use with real money. Always use paper trading for testing. The authors accept no responsibility for any financial losses incurred from using this software.