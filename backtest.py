import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
from strategy import MovingAverageCrossStrategy, StatisticalArbitrageStrategy, RSIStrategy, BollingerBandsStrategy, MACDStrategy
from collections import deque

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BacktestEngine:
    """Backtesting engine for trading strategies"""
    def __init__(self, initial_capital=100000.0, commission=0.0):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.commission = commission
        self.positions = {}  # Current positions {symbol: quantity}
        self.position_values = {}  # Current position values {symbol: value}
        self.trades = []  # Trade history
        self.equity_curve = []  # Historical equity values
        
        # Performance metrics
        self.metrics = {
            'total_return': 0.0,
            'annualized_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'total_trades': 0
        }
        
    def reset(self):
        """Reset the backtester to initial state"""
        self.capital = self.initial_capital
        self.positions = {}
        self.position_values = {}
        self.trades = []
        self.equity_curve = []
        self.metrics = {
            'total_return': 0.0,
            'annualized_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'total_trades': 0
        }
        
    def load_data(self, symbol, data_path=None, dataframe=None, start_date=None, end_date=None):
        """Load historical data for backtesting
        
        Args:
            symbol: Symbol for the data
            data_path: Path to CSV file with historical data
            dataframe: Directly provide a pandas DataFrame
            start_date: Start date for filtering data
            end_date: End date for filtering data
            
        Returns:
            DataFrame with historical data
        """
        if dataframe is not None:
            # Use provided DataFrame
            df = dataframe.copy()
        elif data_path is not None:
            # Load from CSV file
            if not os.path.exists(data_path):
                logger.error(f"Data file not found: {data_path}")
                return None
                
            df = pd.read_csv(data_path)
        else:
            logger.error("Either data_path or dataframe must be provided")
            return None
            
        # Make sure we have required columns
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        # Try to map common column names
        column_mapping = {}
        for col in df.columns:
            col_lower = col.lower()
            if 'date' in col_lower or 'time' in col_lower:
                column_mapping[col] = 'timestamp'
            elif col_lower == 'open' or col_lower == 'o':
                column_mapping[col] = 'open'
            elif col_lower == 'high' or col_lower == 'h':
                column_mapping[col] = 'high'
            elif col_lower == 'low' or col_lower == 'l':
                column_mapping[col] = 'low'
            elif col_lower == 'close' or col_lower == 'c':
                column_mapping[col] = 'close'
            elif col_lower == 'volume' or col_lower == 'vol' or col_lower == 'v':
                column_mapping[col] = 'volume'
                
        # Apply column mapping if needed
        if column_mapping:
            df = df.rename(columns=column_mapping)
            
        # Check if we have all required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return None
            
        # Convert timestamp to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Filter by date range if specified
        if start_date:
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            df = df[df['timestamp'] >= start_date]
            
        if end_date:
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)
            df = df[df['timestamp'] <= end_date]
            
        # Add symbol column if not present
        if 'symbol' not in df.columns:
            df['symbol'] = symbol
            
        logger.info(f"Loaded {len(df)} data points for {symbol}")
        return df
        
    def execute_trade(self, symbol, action, quantity, price, timestamp):
        """Execute a trade in the backtesting environment
        
        Args:
            symbol: Symbol to trade
            action: 'BUY' or 'SELL'
            quantity: Number of shares to trade
            price: Execution price
            timestamp: Timestamp of the trade
        """
        # Calculate trade cost
        trade_value = quantity * price
        commission_cost = trade_value * self.commission
        
        # Update positions and capital
        if action == 'BUY':
            # Add to position
            if symbol in self.positions:
                self.positions[symbol] += quantity
            else:
                self.positions[symbol] = quantity
                
            # Update position value
            self.position_values[symbol] = self.positions[symbol] * price
            
            # Deduct from capital
            self.capital -= (trade_value + commission_cost)
            
        elif action == 'SELL':
            # Reduce position
            if symbol in self.positions:
                self.positions[symbol] -= quantity
            else:
                self.positions[symbol] = -quantity
                
            # Update position value
            self.position_values[symbol] = self.positions[symbol] * price
            
            # Add to capital
            self.capital += (trade_value - commission_cost)
            
        # Record the trade
        trade = {
            'timestamp': timestamp,
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'price': price,
            'value': trade_value,
            'commission': commission_cost,
            'capital': self.capital
        }
        self.trades.append(trade)
        
        # Log the trade
        logger.debug(f"Trade: {action} {quantity} {symbol} @ {price:.2f}")
        
    def update_equity(self, current_prices, timestamp):
        """Update equity curve with current positions and prices
        
        Args:
            current_prices: Dictionary of current prices {symbol: price}
            timestamp: Current timestamp
        """
        # Calculate current position values
        for symbol, quantity in self.positions.items():
            if symbol in current_prices:
                self.position_values[symbol] = quantity * current_prices[symbol]
                
        # Calculate total equity (capital + position values)
        total_equity = self.capital + sum(self.position_values.values())
        
        # Record equity value
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': total_equity
        })
        
    def calculate_metrics(self):
        """Calculate performance metrics based on equity curve and trades"""
        if not self.equity_curve:
            logger.warning("No equity curve data available")
            return
            
        # Convert equity curve to DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df['returns'] = equity_df['equity'].pct_change()
        
        # Calculate total return
        initial_equity = self.equity_curve[0]['equity']
        final_equity = self.equity_curve[-1]['equity']
        total_return = (final_equity - initial_equity) / initial_equity * 100
        
        # Calculate annualized return
        days = (equity_df['timestamp'].max() - equity_df['timestamp'].min()).days
        if days > 0:
            annualized_return = ((1 + total_return / 100) ** (365 / days) - 1) * 100
        else:
            annualized_return = 0
            
        # Calculate Sharpe ratio (assuming risk-free rate of 0%)
        if len(equity_df) > 1:
            daily_returns = equity_df['returns'].dropna()
            if len(daily_returns) > 0 and daily_returns.std() > 0:
                sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0
            
        # Calculate max drawdown
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak'] * 100
        max_drawdown = abs(equity_df['drawdown'].min())
        
        # Calculate trade metrics
        trades_df = pd.DataFrame(self.trades)
        if not trades_df.empty:
            # Calculate P&L for each trade
            trades_df['pnl'] = 0.0
            
            # Group trades by symbol and calculate P&L
            symbols = trades_df['symbol'].unique()
            for symbol in symbols:
                symbol_trades = trades_df[trades_df['symbol'] == symbol].copy()
                symbol_trades = symbol_trades.sort_values('timestamp')
                
                # Simulate tracking of position and P&L
                position = 0
                cost_basis = 0
                for idx, trade in symbol_trades.iterrows():
                    if trade['action'] == 'BUY':
                        # Update cost basis for buys
                        new_quantity = position + trade['quantity']
                        if new_quantity > 0:
                            new_cost = cost_basis + trade['value']
                            cost_basis = new_cost
                        position += trade['quantity']
                    else:  # SELL
                        # Calculate P&L for sells
                        if position != 0:
                            avg_price = cost_basis / position if position > 0 else cost_basis / abs(position)
                            trade_pnl = (trade['price'] - avg_price) * trade['quantity']
                            trades_df.at[idx, 'pnl'] = trade_pnl
                        position -= trade['quantity']
                        if position == 0:
                            cost_basis = 0
            
            # Calculate win rate and profit factor
            winning_trades = trades_df[trades_df['pnl'] > 0]
            losing_trades = trades_df[trades_df['pnl'] < 0]
            
            total_trades = len(trades_df)
            win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
            
            gross_profit = winning_trades['pnl'].sum() if not winning_trades.empty else 0
            gross_loss = abs(losing_trades['pnl'].sum()) if not losing_trades.empty else 0
            
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
        else:
            win_rate = 0
            profit_factor = 0
            total_trades = 0
            
        # Store metrics
        self.metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': total_trades
        }
        
        return self.metrics
        
    def print_results(self):
        """Print backtesting results"""
        print("\n=== Backtesting Results ===")
        print(f"Initial Capital: ${self.initial_capital:.2f}")
        print(f"Final Capital: ${self.capital:.2f}")
        
        if self.position_values:
            position_value = sum(self.position_values.values())
            print(f"Final Position Value: ${position_value:.2f}")
            print(f"Total Equity: ${(self.capital + position_value):.2f}")
            
        print(f"\nTotal Return: {self.metrics['total_return']:.2f}%")
        print(f"Annualized Return: {self.metrics['annualized_return']:.2f}%")
        print(f"Sharpe Ratio: {self.metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {self.metrics['max_drawdown']:.2f}%")
        
        if self.trades:
            print(f"\nTotal Trades: {self.metrics['total_trades']}")
            print(f"Win Rate: {self.metrics['win_rate']:.2f}%")
            print(f"Profit Factor: {self.metrics['profit_factor']:.2f}")
            
    def plot_results(self, save_path=None):
        """Plot equity curve and drawdowns
        
        Args:
            save_path: Path to save the plot (if None, display only)
        """
        if not self.equity_curve:
            logger.warning("No equity curve data available")
            return
            
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak'] * 100
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot equity curve
        ax1.plot(equity_df['timestamp'], equity_df['equity'])
        ax1.set_title('Equity Curve')
        ax1.set_ylabel('Equity ($)')
        ax1.grid(True)
        
        # Plot drawdowns
        ax2.fill_between(equity_df['timestamp'], equity_df['drawdown'], 0, color='red', alpha=0.3)
        ax2.set_title('Drawdowns')
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_xlabel('Date')
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()
            
    def backtest_strategy(self, strategy_class, data, strategy_params=None):
        """Backtest a strategy using historical data
        
        Args:
            strategy_class: The strategy class to backtest
            data: DataFrame with historical data
            strategy_params: Dictionary of strategy parameters
            
        Returns:
            Dictionary of performance metrics
        """
        # Reset the backtester
        self.reset()
        
        if strategy_params is None:
            strategy_params = {}
            
        # Create a mock IB connection class for backtesting
        class MockIBConnection:
            def __init__(self, backtest_engine):
                self.backtest_engine = backtest_engine
                self.data = {}
                self.positions = {}
                self.current_data = None
                self.current_positions = {}
                
            def get_data(self, req_id):
                return self.current_data
                
            def get_positions(self):
                return self.current_positions
                
            def create_contract(self, symbol):
                # Mock contract object
                class MockContract:
                    def __init__(self, symbol):
                        self.symbol = symbol
                        
                return MockContract(symbol)
                
            def create_order(self, action, quantity):
                # Mock order object
                class MockOrder:
                    def __init__(self, action, quantity):
                        self.action = action
                        self.totalQuantity = quantity
                        
                return MockOrder(action, quantity)
                
            def place_order(self, contract, order):
                # Execute the order in the backtester
                self.backtest_engine.execute_trade(
                    contract.symbol,
                    order.action,
                    order.totalQuantity,
                    self.current_price,
                    self.current_timestamp
                )
                
                # Update positions
                symbol = contract.symbol
                if order.action == 'BUY':
                    if symbol in self.current_positions:
                        self.current_positions[symbol]['position'] += order.totalQuantity
                    else:
                        self.current_positions[symbol] = {'position': order.totalQuantity, 'avgCost': self.current_price}
                else:  # SELL
                    if symbol in self.current_positions:
                        self.current_positions[symbol]['position'] -= order.totalQuantity
                    else:
                        self.current_positions[symbol] = {'position': -order.totalQuantity, 'avgCost': self.current_price}
                        
                return len(self.backtest_engine.trades)
                
        # Create mock IB connection
        mock_ib = MockIBConnection(self)
        
        # Group data by symbol
        symbols = data['symbol'].unique()
        
        # Create strategy instances
        strategies = {}
        for symbol in symbols:
            if strategy_class == StatisticalArbitrageStrategy and len(symbols) >= 2:
                # For stat arb, we need pairs of symbols
                # Let's use the first two symbols
                symbol1 = symbols[0]
                symbol2 = symbols[1]
                strategies[f"{symbol1}_{symbol2}"] = strategy_class(symbol1, symbol2, mock_ib, **strategy_params)
                break  # Only create one instance for the pair
            else:
                # For other strategies, create one instance per symbol
                strategies[symbol] = strategy_class(symbol, mock_ib, **strategy_params)
                
        # Set strategy running state
        for strategy in strategies.values():
            strategy.is_running = True
            
        # Group data by date to simulate the passage of time
        data = data.sort_values('timestamp')
        grouped_by_date = data.groupby(pd.Grouper(key='timestamp', freq='D'))
        
        # Run the backtest
        current_prices = {symbol: 0 for symbol in symbols}
        
        for date, group in grouped_by_date:
            # Process each bar within the day
            for idx, row in group.iterrows():
                symbol = row['symbol']
                timestamp = row['timestamp']
                current_price = row['close']
                
                # Update current price
                current_prices[symbol] = current_price
                
                # Prepare mock data for the strategy
                tick_types = {4: current_price}  # Tick type 4 is last price
                mock_ib.current_data = tick_types
                mock_ib.current_price = current_price
                mock_ib.current_timestamp = timestamp
                
                # Update the strategy
                if symbol in strategies:
                    strategies[symbol].update()
                elif strategy_class == StatisticalArbitrageStrategy and len(symbols) >= 2:
                    # For stat arb, update with both symbols' data
                    strat_key = f"{symbols[0]}_{symbols[1]}"
                    if strat_key in strategies and row['symbol'] == symbols[0]:
                        # For the first symbol, store the data
                        tick_types1 = {4: current_price}
                        mock_ib.data[1] = tick_types1
                    elif strat_key in strategies and row['symbol'] == symbols[1]:
                        # For the second symbol, run the update
                        tick_types2 = {4: current_price}
                        mock_ib.data[2] = tick_types2
                        
                        # Get both prices
                        price1 = next(iter(mock_ib.data.get(1, {}).values())) if 1 in mock_ib.data else None
                        price2 = next(iter(mock_ib.data.get(2, {}).values())) if 2 in mock_ib.data else None
                        
                        if price1 and price2:
                            strategies[strat_key].generate_signals(
                                {4: price1},
                                {4: price2}
                            )
                            
                # Update equity curve
                self.update_equity(current_prices, timestamp)
                
        # Calculate performance metrics
        metrics = self.calculate_metrics()
        
        return metrics

class BacktestRunner:
    """Class to run multiple backtests with different parameters"""
    def __init__(self, data_path=None, dataframe=None, initial_capital=100000.0, commission=0.0):
        self.data_path = data_path
        self.dataframe = dataframe
        self.initial_capital = initial_capital
        self.commission = commission
        self.engine = BacktestEngine(initial_capital, commission)
        self.results = []
        
    def load_data(self, symbol, start_date=None, end_date=None):
        """Load historical data"""
        return self.engine.load_data(symbol, self.data_path, self.dataframe, start_date, end_date)
        
    def run_parameter_sweep(self, strategy_class, data, param_grid):
        """Run multiple backtests with different parameter combinations
        
        Args:
            strategy_class: The strategy class to backtest
            data: DataFrame with historical data
            param_grid: Dictionary of parameter lists to sweep through
            
        Returns:
            List of results dictionaries
        """
        import itertools
        
        # Generate all parameter combinations
        param_names = sorted(param_grid.keys())
        param_values = [param_grid[name] for name in param_names]
        param_combinations = list(itertools.product(*param_values))
        
        # Run backtest for each combination
        results = []
        for combo in param_combinations:
            # Create parameter dictionary
            params = {name: value for name, value in zip(param_names, combo)}
            
            # Run backtest
            metrics = self.engine.backtest_strategy(strategy_class, data, params)
            
            # Store results
            result = {
                'params': params,
                'metrics': metrics
            }
            results.append(result)
            
            # Log results
            param_str = ', '.join(f"{k}={v}" for k, v in params.items())
            logger.info(f"Parameters: {param_str}, Return: {metrics['total_return']:.2f}%, Sharpe: {metrics['sharpe_ratio']:.2f}")
            
        # Sort results by total return
        results.sort(key=lambda x: x['metrics']['total_return'], reverse=True)
        
        self.results = results
        return results
        
    def print_top_results(self, n=5):
        """Print the top n results"""
        if not self.results:
            logger.warning("No results available")
            return
            
        print(f"\n=== Top {n} Results ===")
        for i, result in enumerate(self.results[:n]):
            params = result['params']
            metrics = result['metrics']
            
            print(f"\nRank {i+1}:")
            print(f"Parameters: {params}")
            print(f"Total Return: {metrics['total_return']:.2f}%")
            print(f"Annualized Return: {metrics['annualized_return']:.2f}%")
            print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
            print(f"Win Rate: {metrics['win_rate']:.2f}%")
            print(f"Profit Factor: {metrics['profit_factor']:.2f}")
            print(f"Total Trades: {metrics['total_trades']}")
            
    def plot_parameter_impact(self, param_name, metric='total_return'):
        """Plot the impact of a parameter on a performance metric
        
        Args:
            param_name: Name of the parameter to analyze
            metric: Metric to plot ('total_return', 'sharpe_ratio', etc.)
        """
        if not self.results:
            logger.warning("No results available")
            return
            
        # Extract parameter values and metrics
        param_values = []
        metric_values = []
        
        for result in self.results:
            if param_name in result['params']:
                param_values.append(result['params'][param_name])
                metric_values.append(result['metrics'][metric])
                
        if not param_values:
            logger.warning(f"Parameter {param_name} not found in results")
            return
            
        # Group by parameter value and calculate mean metric
        df = pd.DataFrame({
            'param': param_values,
            'metric': metric_values
        })
        
        grouped = df.groupby('param').mean().reset_index()
        
        # Plot the results
        plt.figure(figsize=(10, 6))
        plt.plot(grouped['param'], grouped['metric'], marker='o')
        plt.title(f'Impact of {param_name} on {metric}')
        plt.xlabel(param_name)
        plt.ylabel(metric)
        plt.grid(True)
        plt.show()
        
    def run_backtest_with_best_params(self, strategy_class, data):
        """Run a backtest with the best parameters found
        
        Args:
            strategy_class: The strategy class to backtest
            data: DataFrame with historical data
            
        Returns:
            Dictionary of performance metrics
        """
        if not self.results:
            logger.warning("No parameter sweep results available")
            return None
            
        # Get best parameters (highest total return)
        best_params = self.results[0]['params']
        
        # Run backtest with best parameters
        metrics = self.engine.backtest_strategy(strategy_class, data, best_params)
        
        # Print and plot results
        self.engine.print_results()
        self.engine.plot_results()
        
        return metrics


# Example usage
if __name__ == "__main__":
    # Create a backtest runner
    runner = BacktestRunner(initial_capital=100000.0, commission=0.001)
    
    # Load sample data (this would be your historical data)
    import yfinance as yf
    
    # Download sample data
    aapl = yf.download("AAPL", start="2022-01-01", end="2023-01-01")
    aapl['symbol'] = 'AAPL'
    aapl.reset_index(inplace=True)
    aapl.rename(columns={'Date': 'timestamp', 'Open': 'open', 'High': 'high', 
                          'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, 
                inplace=True)
    
    # Run parameter sweep for Moving Average Crossover strategy
    param_grid = {
        'fast_period': [5, 10, 15, 20],
        'slow_period': [20, 30, 40, 50],
        'position_size': [100]
    }
    
    results = runner.run_parameter_sweep(MovingAverageCrossStrategy, aapl, param_grid)
    
    # Print top results
    runner.print_top_results(5)
    
    # Plot parameter impact
    runner.plot_parameter_impact('fast_period')
    runner.plot_parameter_impact('slow_period')
    
    # Run a backtest with the best parameters
    best_metrics = runner.run_backtest_with_best_params(MovingAverageCrossStrategy, aapl)