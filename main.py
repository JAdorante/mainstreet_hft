import logging
import time
import argparse
import signal
import sys
import json
import os
from datetime import datetime, timedelta
from alpaca_data_provider import AlpacaDataProvider
from strategy import HybridMovingAverageCrossStrategy, HybridRSIStrategy, HybridBollingerBandsStrategy, HybridMACDStrategy, HybridStatisticalArbitrageStrategy
from ib_connection import IBConnection
from strategy import MovingAverageCrossStrategy, StatisticalArbitrageStrategy, RSIStrategy, BollingerBandsStrategy, MACDStrategy
from risk_manager import RiskManager
from stock_screener import StockScreener
from execution import OrderExecutor, SmartOrderRouter
from data_processor import MarketDataProcessor

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("mainstreet_hft.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Mainstreet HFT for Interactive Brokers Paper Trading")
    
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="TWS/IB Gateway host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=7497,
                        help="TWS/IB Gateway port (default: 7497 for paper trading)")
    parser.add_argument("--client-id", type=int, default=6969,
                        help="Client ID for IB API connection (default: 6969)")
    parser.add_argument("--no-defaults", action="store_true",
                        help="Disable all default values")
    parser.add_argument("--symbols", type=str, default="AAPL",
                        help="Comma-separated list of symbols to trade (default: AAPL)")
    parser.add_argument("--strategy", type=str, default="ma_cross",
                        choices=["ma_cross", "stat_arb", "rsi", "bollinger", "macd"],
                        help="Trading strategy to use (default: ma_cross)")
    parser.add_argument("--update-interval", type=float, default=1.0,
                        help="Strategy update interval in seconds (default: 1.0)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key for stock data provider")
    parser.add_argument("--api-provider", type=str, default="alphavantage",
                        choices=["alphavantage", "finnhub", "iex"],
                        help="Stock data provider (default: alphavantage)")
    parser.add_argument("--screen", action="store_true",
                        help="Run stock screening before trading")
    parser.add_argument("--correlate", action="store_true",
                        help="Find correlated pairs for statistical arbitrage")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to configuration file (JSON)")
    parser.add_argument("--backtest", action="store_true",
                        help="Run in backtest mode instead of live trading")
    parser.add_argument("--execution-algo", type=str, default="market",
                        choices=["market", "limit", "twap", "vwap", "iceberg", "sniper"],
                        help="Order execution algorithm (default: market)")
    parser.add_argument("--smart-routing", action="store_true",
                        help="Use smart order routing")
    parser.add_argument("--risk-limit", type=float, default=2.0,
                        help="Maximum daily loss percentage (default: 2.0)")
    parser.add_argument("--position-size", type=int, default=100,
                        help="Default position size (default: 100)")
    parser.add_argument("--stop-loss", type=float, default=None,
                        help="Stop loss percentage (optional)")
    parser.add_argument("--take-profit", type=float, default=None,
                        help="Take profit percentage (optional)")
    parser.add_argument("--fixed-take-profit", type=float, default=None,
                        help="Fixed take profit price (optional)")
    parser.add_argument("--trailing-profit", type=float, default=None,
                        help="Trailing take profit distance percentage (optional)")
    parser.add_argument("--strategy-params", type=str, default=None,
                        help="JSON string with strategy-specific parameters")
    
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from JSON file"""
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        return {}
        
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return {}

def merge_configs(args, file_config):
    """Merge command line arguments with file configuration"""
    if args.no_defaults:
        config = {}
    else:
    # Start with file config as base
        config = file_config.copy()
    
    # Override with command line arguments if provided
    if args.host is not None:
        config["host"] = args.host
    if args.port is not None:
        config["port"] = args.port
    if args.client_id is not None:
        config["client_id"] = args.client_id
    if args.symbols is not None:
        config["symbols"] = args.symbols
    if args.strategy is not None:
        config["strategy"] = args.strategy
    if args.update_interval is not None:
        config["update_interval"] = args.update_interval
    if args.api_key is not None:
        config["api_key"] = args.api_key
    if args.api_provider is not None:
        config["api_provider"] = args.api_provider
    if args.execution_algo is not None:
        config["execution_algo"] = args.execution_algo
    if args.smart_routing:
        config["smart_routing"] = True
    if args.risk_limit is not None:
        config["risk_limit"] = args.risk_limit
    if args.position_size is not None:
        config["position_size"] = args.position_size
    if args.stop_loss is not None:
        config["stop_loss"] = args.stop_loss
    if args.take_profit is not None:
        config["take_profit"] = args.take_profit
    if args.fixed_take_profit is not None:
        config["fixed_take_profit"] = args.fixed_take_profit
    if args.trailing_profit is not None:
        config["trailing_profit"] = args.trailing_profit
    if args.screen:
        config["screen"] = True
    if args.correlate:
        config["correlate"] = True
    if args.backtest:
        config["backtest"] = True
    if args.strategy_params is not None:
        try:
            params = json.loads(args.strategy_params)
            config["strategy_params"] = params
        except json.JSONDecodeError:
            logger.error("Invalid JSON for strategy parameters")
    
    return config

def signal_handler(sig, frame):
    """Handle Ctrl+C and other termination signals"""
    logger.info("Termination signal received. Shutting down...")
    global running
    running = False

def create_strategy(strategy_name, symbols, ib_connection, alpaca_provider, params=None):
    """Create strategy instance based on strategy name
    
    Args:
        strategy_name: The strategy to create ('ma_cross', 'rsi', 'bollinger', 'macd', 'stat_arb')
        symbols: List of symbols to trade
        ib_connection: Interactive Brokers connection for order execution
        alpaca_provider: Alpaca data provider for real-time market data
        params: Dictionary of strategy parameters
        
    Returns:
        List of strategy instances
    """
    if params is None:
        params = {}
        
    # Set default position size if not provided
    if "position_size" not in params:
        params["position_size"] = 100
        
    # Extract take-profit parameters
    take_profit_pct = params.get("take_profit", None)
    fixed_take_profit = params.get("fixed_take_profit", None)
    trailing_profit_pct = params.get("trailing_profit", None)
        
    strategies = []
    
    if strategy_name == "ma_cross":
        # Extract strategy-specific parameters
        fast_period = params.get("fast_period", 10)
        slow_period = params.get("slow_period", 30)
        position_size = params.get("position_size", 100)
        
        # Create one strategy instance per symbol
        for symbol in symbols:
            strategy = HybridMovingAverageCrossStrategy(
                symbol, 
                ib_connection, 
                alpaca_provider,
                fast_period=fast_period,
                slow_period=slow_period,
                position_size=position_size,
                take_profit_pct=take_profit_pct,
                fixed_take_profit=fixed_take_profit,
                trailing_profit_pct=trailing_profit_pct
            )
            strategies.append(strategy)
            
    elif strategy_name == "stat_arb":
        # Statistical arbitrage requires pairs of symbols
        if len(symbols) < 2:
            logger.error("Statistical arbitrage strategy requires at least 2 symbols")
            return []
            
        # Extract strategy-specific parameters
        lookback_period = params.get("lookback_period", 100)
        entry_threshold = params.get("entry_threshold", 2.0)
        exit_threshold = params.get("exit_threshold", 0.0)
        position_size = params.get("position_size", 100)
        
        # Create strategy instances for pairs of symbols
        for i in range(0, len(symbols) - 1, 2):
            symbol1 = symbols[i]
            symbol2 = symbols[i+1]
            
            strategy = HybridStatisticalArbitrageStrategy(
                symbol1, 
                symbol2, 
                ib_connection,
                alpaca_provider,
                lookback_period=lookback_period,
                entry_threshold=entry_threshold,
                exit_threshold=exit_threshold,
                position_size=position_size,
                take_profit_pct=take_profit_pct,
                fixed_take_profit=fixed_take_profit,
                trailing_profit_pct=trailing_profit_pct
            )
            strategies.append(strategy)
            
    elif strategy_name == "rsi":
        # Extract strategy-specific parameters
        period = params.get("period", 14)
        overbought = params.get("overbought", 70)
        oversold = params.get("oversold", 30)
        position_size = params.get("position_size", 100)
        
        # Create one strategy instance per symbol
        for symbol in symbols:
            strategy = HybridRSIStrategy(
                symbol, 
                ib_connection,
                alpaca_provider,
                period=period,
                overbought=overbought,
                oversold=oversold,
                position_size=position_size,
                take_profit_pct=take_profit_pct,
                fixed_take_profit=fixed_take_profit,
                trailing_profit_pct=trailing_profit_pct
            )
            strategies.append(strategy)
            
    elif strategy_name == "bollinger":
        # Extract strategy-specific parameters
        period = params.get("period", 20)
        std_dev = params.get("std_dev", 2)
        position_size = params.get("position_size", 100)
        
        # Create one strategy instance per symbol
        for symbol in symbols:
            strategy = HybridBollingerBandsStrategy(
                symbol, 
                ib_connection,
                alpaca_provider,
                period=period,
                std_dev=std_dev,
                position_size=position_size,
                take_profit_pct=take_profit_pct,
                fixed_take_profit=fixed_take_profit,
                trailing_profit_pct=trailing_profit_pct
            )
            strategies.append(strategy)
            
    elif strategy_name == "macd":
        # Extract strategy-specific parameters
        fast_period = params.get("fast_period", 12)
        slow_period = params.get("slow_period", 26)
        signal_period = params.get("signal_period", 9)
        position_size = params.get("position_size", 100)
        
        # Create one strategy instance per symbol
        for symbol in symbols:
            strategy = HybridMACDStrategy(
                symbol, 
                ib_connection,
                alpaca_provider,
                fast_period=fast_period,
                slow_period=slow_period,
                signal_period=signal_period,
                position_size=position_size,
                take_profit_pct=take_profit_pct,
                fixed_take_profit=fixed_take_profit,
                trailing_profit_pct=trailing_profit_pct
            )
            strategies.append(strategy)
            
    else:
        logger.error(f"Unknown strategy: {strategy_name}")
        return []
        
    logger.info(f"Created {len(strategies)} {strategy_name} strategy instances")
    return strategies

def run_backtest(config):
    """Run backtesting mode"""
    from backtest import BacktestEngine, BacktestRunner
    
    logger.info("Starting backtest mode")
    
    # Extract symbols and strategy
    symbols = [s.strip() for s in config.get("symbols", "AAPL").split(",")]
    strategy_name = config.get("strategy", "ma_cross")
    strategy_params = config.get("strategy_params", {})
    
    # Create backtest runner
    runner = BacktestRunner(initial_capital=config.get("initial_capital", 100000.0),
                           commission=config.get("commission", 0.001))
    
    # Load historical data for symbols
    data = {}
    for symbol in symbols:
        # Check if historical data file exists
        data_file = f"data/{symbol}.csv"
        if os.path.exists(data_file):
            # Load from file
            df = runner.load_data(symbol, data_path=data_file)
            data[symbol] = df
        else:
            # If API key is provided, download data
            if "api_key" in config and "api_provider" in config:
                logger.info(f"Downloading historical data for {symbol}")
                screener = StockScreener(api_key=config["api_key"], provider=config["api_provider"])
                df = screener.get_historical_data(symbol, interval="daily", period="1year")
                if df is not None:
                    data[symbol] = df
                    
                    # Save data to file for future use
                    os.makedirs("data", exist_ok=True)
                    df.to_csv(data_file, index=False)
                else:
                    logger.error(f"Failed to download data for {symbol}")
            else:
                logger.error(f"No historical data found for {symbol} and no API key provided")
    
    if not data:
        logger.error("No historical data available for backtesting")
        return 1
    
    # Combine data from all symbols
    combined_data = pd.concat(data.values())
    
    # Get the appropriate strategy class
    if strategy_name == "ma_cross":
        strategy_class = MovingAverageCrossStrategy
    elif strategy_name == "stat_arb":
        strategy_class = StatisticalArbitrageStrategy
    elif strategy_name == "rsi":
        strategy_class = RSIStrategy
    elif strategy_name == "bollinger":
        strategy_class = BollingerBandsStrategy
    elif strategy_name == "macd":
        strategy_class = MACDStrategy
    else:
        logger.error(f"Unknown strategy: {strategy_name}")
        return 1
    
    # Run backtest with parameters
    if "param_grid" in config:
        # Parameter sweep
        logger.info("Running parameter sweep")
        results = runner.run_parameter_sweep(strategy_class, combined_data, config["param_grid"])
        runner.print_top_results(5)
        
        # Run backtest with best parameters
        logger.info("Running backtest with best parameters")
        best_metrics = runner.run_backtest_with_best_params(strategy_class, combined_data)
    else:
        # Run backtest with provided parameters
        logger.info(f"Running backtest with parameters: {strategy_params}")
        metrics = runner.engine.backtest_strategy(strategy_class, combined_data, strategy_params)
        runner.engine.print_results()
        runner.engine.plot_results()
    
    logger.info("Backtest completed")
    return 0

def connect_to_interactive_brokers(config):
    """Connect to Interactive Brokers TWS or Gateway"""
    host = config.get("host", "127.0.0.1")
    port = config.get("port", 7497)
    client_id = config.get("client_id", 6969)
    
    logger.info(f"Connecting to Interactive Brokers at {host}:{port} (client ID: {client_id})")
    
    # Create IB connection
    ib_connection = IBConnection(host=host, port=port, clientId=client_id)
    
    # Connect to TWS/IB Gateway
    if not ib_connection.connect():
        logger.error("Failed to connect to Interactive Brokers. Exiting.")
        return None
        
    logger.info("Successfully connected to Interactive Brokers")
    return ib_connection

def main():
    """Main entry point for the HFT bot"""
    global running
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Load configuration file if provided
    file_config = {}
    if args.config:
        file_config = load_config(args.config)
        
    # Merge configurations
    config = merge_configs(args, file_config)
    
    # Log configuration
    logger.info(f"Configuration: {json.dumps(config, indent=2)}")
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Check if we should run in backtest mode
    if config.get("backtest", False):
        return run_backtest(config)
    
    # Initialize Alpaca data provider with hard-coded credentials
    alpaca_provider = AlpacaDataProvider(
        api_key="PKQG8ZIMLBXOZ03NI6PI",
        api_secret="COLiyxHcteDEDh4zIAH3OSssnNYMd2bkKD2xwe6P"
    )
    logger.info("Initialized Alpaca data provider for real-time market data")
    
    # Connect to Interactive Brokers for paper trading
    ib_connection = connect_to_interactive_brokers(config)
    if ib_connection is None:
        return 1
    
    # Create data processor
    data_processor = MarketDataProcessor()
    logger.info("Initialized market data processor")
    
    # Create risk manager
    risk_manager = RiskManager(ib_connection, 
                              max_position_size=config.get("max_position_size", 1000),
                              max_daily_loss_pct=config.get("risk_limit", 2.0),
                              max_drawdown_pct=config.get("max_drawdown", 5.0),
                              max_trades_per_day=config.get("max_trades_per_day", 100))
    risk_manager.initialize()
    logger.info("Initialized risk manager")
    
    # Create order executor
    if config.get("smart_routing", False):
        # Use smart order routing
        router = SmartOrderRouter(ib_connection)
        logger.info("Initialized smart order router")
        executor = router.order_executor
    else:
        # Use regular order executor
        executor = OrderExecutor(ib_connection)
    logger.info(f"Initialized order executor with {config.get('execution_algo', 'market')} algorithm")
    
    # Parse symbols
    symbols = [s.strip() for s in config.get("symbols", "AAPL").split(",")]
    logger.info(f"Trading symbols: {symbols}")
    
    # Create strategies with IB for execution and Alpaca for data
    strategy_name = config.get("strategy", "ma_cross")
    
    # Build strategy params including take-profit settings
    strategy_params = config.get("strategy_params", {})
    
    # Add take-profit parameters if provided
    if config.get("take_profit") is not None:
        strategy_params["take_profit"] = config.get("take_profit")
    if config.get("fixed_take_profit") is not None:
        strategy_params["fixed_take_profit"] = config.get("fixed_take_profit")
    if config.get("trailing_profit") is not None:
        strategy_params["trailing_profit"] = config.get("trailing_profit")
    
    # Create strategy instances
    strategies = create_strategy(strategy_name, symbols, ib_connection, alpaca_provider, strategy_params)
    
    if not strategies:
        logger.error("No valid strategies created. Exiting.")
        ib_connection.disconnect()
        return 1
    
    # Start all strategies
    for strategy in strategies:
        strategy.start()
    
    # Main loop
    running = True
    update_interval = config.get("update_interval", 1.0)
    
    try:
        while running:
            # Update risk manager
            risk_manager.update()
            
            # If trading is not allowed, check if we need to close positions
            if not risk_manager.trading_allowed:
                logger.warning("Trading is currently disabled due to risk limits")
                # Optionally close positions if needed
                # risk_manager.emergency_close_all_positions()
            
            # Update all strategies
            for strategy in strategies:
                strategy.update()
            
            # Sleep for the specified interval
            time.sleep(update_interval)
            
    except Exception as e:
        logger.exception(f"Error in main loop: {e}")
    finally:
        # Stop all strategies
        for strategy in strategies:
            strategy.stop()
        
        # Disconnect from IB
        ib_connection.disconnect()
    
    logger.info("Mainstreet HFT shutdown complete")
    return 0

if __name__ == "__main__":
    # Import pandas for backtest mode
    import pandas as pd
    sys.exit(main())