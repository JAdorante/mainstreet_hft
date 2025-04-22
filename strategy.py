import logging
import time
import pandas as pd
import numpy as np
from collections import deque

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BaseStrategy:
    """Base Strategy class that all strategies should inherit from"""
    def __init__(self, symbol, ib_connection, take_profit_pct=None, fixed_take_profit=None, trailing_profit_pct=None, commission=0.0):
        self.commission = commission
        self.symbol = symbol
        self.ib_connection = ib_connection
        self.contract = ib_connection.create_contract(symbol)
        self.market_data_id = None
        self.is_running = False
        self.positions = {}
        
        # Take profit parameters
        self.take_profit_pct = take_profit_pct
        self.fixed_take_profit = fixed_take_profit
        self.trailing_profit_pct = trailing_profit_pct
        
        # Entry tracking for take-profit calculations
        self.entry_prices = {}
        self.highest_since_entry = {}
        self.lowest_since_entry = {}
        
    def start(self):
        """Start the strategy"""
        if self.is_running:
            logger.warning("Strategy is already running")
            return
            
        # Request market data
        self.market_data_id = self.ib_connection.request_market_data(self.contract)
        if self.market_data_id is None:
            logger.error("Failed to request market data")
            return
            
        self.is_running = True
        logger.info(f"Strategy started for {self.symbol}")
        
    def stop(self):
        """Stop the strategy"""
        if not self.is_running:
            logger.warning("Strategy is not running")
            return
            
        # Cancel market data subscription
        if self.market_data_id is not None:
            self.ib_connection.app.cancelMktData(self.market_data_id)
            
        self.is_running = False
        logger.info(f"Strategy stopped for {self.symbol}")
        
    def update(self):
        """Update the strategy based on new market data"""
        if not self.is_running:
            return
            
        # Get the latest market data
        market_data = self.ib_connection.get_data(self.market_data_id)
        
        # Update positions
        self.positions = self.ib_connection.get_positions()
        
        # Check take-profit conditions if enabled
        if self.take_profit_pct is not None or self.fixed_take_profit is not None or self.trailing_profit_pct is not None:
            self.check_take_profit_conditions(market_data)
            
        # Update highest/lowest prices for trailing take-profit
        if self.trailing_profit_pct is not None and 4 in market_data:
            self.update_price_extremes(market_data)
        
        # Process the market data and generate signals
        signals = self.generate_signals(market_data)
        
        # Execute trades based on signals
        self.execute_trades(signals)
    
    def update_price_extremes(self, market_data):
        """Update highest and lowest prices since entry for trailing take-profit"""
        if 4 not in market_data:  # Tick type 4 is last price
            return
            
        current_price = market_data[4]
        current_position = 0
        
        if self.symbol in self.positions:
            current_position = self.positions[self.symbol].get('position', 0)
        
        # For long positions, track highest price
        if self.symbol in self.entry_prices and current_position > 0:
            if self.symbol not in self.highest_since_entry:
                self.highest_since_entry[self.symbol] = current_price
            else:
                self.highest_since_entry[self.symbol] = max(
                    self.highest_since_entry[self.symbol], current_price)
        
        # For short positions, track lowest price
        if self.symbol in self.entry_prices and current_position < 0:
            if self.symbol not in self.lowest_since_entry:
                self.lowest_since_entry[self.symbol] = current_price
            else:
                self.lowest_since_entry[self.symbol] = min(
                    self.lowest_since_entry[self.symbol], current_price)
    
    def check_take_profit_conditions(self, market_data):
        """Check if take-profit conditions are met and exit if necessary"""
        if 4 not in market_data:  # Tick type 4 is last price
            return
            
        current_price = market_data[4]
        
        # Check if we have a position
        current_position = 0
        if self.symbol in self.positions:
            current_position = self.positions[self.symbol].get('position', 0)
        
        # If no position, nothing to do
        if current_position == 0:
            return
        
        # If we have no entry price and need it, nothing to do
        if self.take_profit_pct is not None and self.symbol not in self.entry_prices:
            return
        
        # Check percentage-based take-profit
        if self.take_profit_pct is not None and self.symbol in self.entry_prices:
            entry_price = self.entry_prices[self.symbol]
            
            # For long positions
            if current_position > 0:
                target_price = entry_price * (1 + self.take_profit_pct / 100)
                if current_price >= target_price:
                    logger.info(f"PERCENTAGE TAKE PROFIT triggered: Price ({current_price:.2f}) reached target ({target_price:.2f}, {self.take_profit_pct}%)")
                    
                    # Create sell order to close position
                    order = self.ib_connection.create_order('SELL', abs(current_position))
                    self.ib_connection.place_order(self.contract, order)
                    
                    # Clear tracking data
                    self.clear_position_tracking()
                    return  # Exit after taking profit
            
            # For short positions
            elif current_position < 0:
                target_price = entry_price * (1 - self.take_profit_pct / 100)
                if current_price <= target_price:
                    logger.info(f"PERCENTAGE TAKE PROFIT triggered: Price ({current_price:.2f}) reached target ({target_price:.2f}, {self.take_profit_pct}%)")
                    
                    # Create buy order to close position
                    order = self.ib_connection.create_order('BUY', abs(current_position))
                    self.ib_connection.place_order(self.contract, order)
                    
                    # Clear tracking data
                    self.clear_position_tracking()
                    return  # Exit after taking profit
        
        # Check fixed price take-profit
        if self.fixed_take_profit is not None:
            # For long positions
            if current_position > 0 and current_price >= self.fixed_take_profit:
                logger.info(f"FIXED TAKE PROFIT triggered: Price ({current_price:.2f}) reached target ({self.fixed_take_profit:.2f})")
                
                # Create sell order to close position
                order = self.ib_connection.create_order('SELL', abs(current_position))
                self.ib_connection.place_order(self.contract, order)
                
                # Clear tracking data
                self.clear_position_tracking()
                return  # Exit after taking profit
            
            # For short positions
            elif current_position < 0 and current_price <= self.fixed_take_profit:
                logger.info(f"FIXED TAKE PROFIT triggered: Price ({current_price:.2f}) reached target ({self.fixed_take_profit:.2f})")
                
                # Create buy order to close position
                order = self.ib_connection.create_order('BUY', abs(current_position))
                self.ib_connection.place_order(self.contract, order)
                
                # Clear tracking data
                self.clear_position_tracking()
                return  # Exit after taking profit
        
        # Check trailing take-profit
        if self.trailing_profit_pct is not None:
            # For long positions
            if current_position > 0 and self.symbol in self.highest_since_entry:
                highest = self.highest_since_entry[self.symbol]
                trail_level = highest * (1 - self.trailing_profit_pct / 100)
                
                if current_price <= trail_level:
                    logger.info(f"TRAILING TAKE PROFIT triggered: Price ({current_price:.2f}) fell below trail level ({trail_level:.2f}, {self.trailing_profit_pct}% from high of {highest:.2f})")
                    
                    # Create sell order to close position
                    order = self.ib_connection.create_order('SELL', abs(current_position))
                    self.ib_connection.place_order(self.contract, order)
                    
                    # Clear tracking data
                    self.clear_position_tracking()
                    return  # Exit after taking profit
            
            # For short positions
            elif current_position < 0 and self.symbol in self.lowest_since_entry:
                lowest = self.lowest_since_entry[self.symbol]
                trail_level = lowest * (1 + self.trailing_profit_pct / 100)
                
                if current_price >= trail_level:
                    logger.info(f"TRAILING TAKE PROFIT triggered: Price ({current_price:.2f}) rose above trail level ({trail_level:.2f}, {self.trailing_profit_pct}% from low of {lowest:.2f})")
                    
                    # Create buy order to close position
                    order = self.ib_connection.create_order('BUY', abs(current_position))
                    self.ib_connection.place_order(self.contract, order)
                    
                    # Clear tracking data
                    self.clear_position_tracking()
                    return  # Exit after taking profit
    
    def clear_position_tracking(self):
        """Clear tracking data after closing a position"""
        if self.symbol in self.entry_prices:
            del self.entry_prices[self.symbol]
        if self.symbol in self.highest_since_entry:
            del self.highest_since_entry[self.symbol]
        if self.symbol in self.lowest_since_entry:
            del self.lowest_since_entry[self.symbol]
    
    def generate_signals(self, market_data):
        """Generate trading signals based on market data"""
        # This should be implemented by the subclasses
        pass
        
    def execute_trades(self, signals):
        """Execute trades based on signals"""
        # This should be implemented by the subclasses
        pass


class MovingAverageCrossStrategy(BaseStrategy):
    """Moving Average Crossover Strategy"""
    def __init__(self, symbol, ib_connection, fast_period=10, slow_period=30, position_size=100, 
                 take_profit_pct=None, fixed_take_profit=None, trailing_profit_pct=None):
        super().__init__(symbol, ib_connection, take_profit_pct, fixed_take_profit, trailing_profit_pct)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.position_size = position_size
        
        # Store price history
        self.price_history = deque(maxlen=slow_period+10)
        self.last_signal = None
        
    def generate_signals(self, market_data):
        """Generate trading signals based on moving average crossover"""
        # We need tick type 4 (last price)
        if 4 not in market_data:
            return None
            
        last_price = market_data[4]
        self.price_history.append(last_price)
        
        # Need enough data to calculate moving averages
        if len(self.price_history) < self.slow_period:
            return None
            
        # Calculate moving averages
        prices = list(self.price_history)
        fast_ma = sum(prices[-self.fast_period:]) / self.fast_period
        slow_ma = sum(prices[-self.slow_period:]) / self.slow_period
        
        # Generate signals
        if fast_ma > slow_ma and (self.last_signal is None or self.last_signal != 'BUY'):
            logger.info(f"BUY signal: Fast MA ({fast_ma:.2f}) crossed above Slow MA ({slow_ma:.2f})")
            self.last_signal = 'BUY'
            return {'action': 'BUY', 'quantity': self.position_size}
        elif fast_ma < slow_ma and (self.last_signal is None or self.last_signal != 'SELL'):
            logger.info(f"SELL signal: Fast MA ({fast_ma:.2f}) crossed below Slow MA ({slow_ma:.2f})")
            self.last_signal = 'SELL'
            return {'action': 'SELL', 'quantity': self.position_size}
            
        return None
        
    def execute_trades(self, signals):
        """Execute trades based on signals"""
        if signals is None:
            return
            
        # Check current position
        current_position = 0
        if self.symbol in self.positions:
            current_position = self.positions[self.symbol].get('position', 0)
            
        action = signals['action']
        quantity = signals['quantity']
        
        # Determine order type based on action and current position
        if action == 'BUY' and current_position <= 0:
            # If we have no position or a short position, buy
            order = self.ib_connection.create_order('BUY', quantity)
            self.ib_connection.place_order(self.contract, order)
            
            # Track entry price for new position
            market_data = self.ib_connection.get_data(self.market_data_id)
            if 4 in market_data:  # Tick type 4 is last price
                self.entry_prices[self.symbol] = market_data[4]
                # Initialize highest price for trailing take-profit
                if self.trailing_profit_pct is not None:
                    self.highest_since_entry[self.symbol] = market_data[4]
                    
        elif action == 'SELL' and current_position >= 0:
            # If we have no position or a long position, sell
            order = self.ib_connection.create_order('SELL', quantity)
            self.ib_connection.place_order(self.contract, order)
            
            # Track entry price for new position
            market_data = self.ib_connection.get_data(self.market_data_id)
            if 4 in market_data:  # Tick type 4 is last price
                self.entry_prices[self.symbol] = market_data[4]
                # Initialize lowest price for trailing take-profit
                if self.trailing_profit_pct is not None:
                    self.lowest_since_entry[self.symbol] = market_data[4]


class StatisticalArbitrageStrategy(BaseStrategy):
    """Statistical Arbitrage Strategy between two correlated assets"""
    def __init__(self, symbol1, symbol2, ib_connection, lookback_period=100, 
                 entry_threshold=2.0, exit_threshold=0.0, position_size=100,
                 take_profit_pct=None, fixed_take_profit=None, trailing_profit_pct=None):
        super().__init__(symbol1, ib_connection, take_profit_pct, fixed_take_profit, trailing_profit_pct)
        self.symbol1 = symbol1
        self.symbol2 = symbol2
        self.contract1 = ib_connection.create_contract(symbol1)
        self.contract2 = ib_connection.create_contract(symbol2)
        self.market_data_id1 = None
        self.market_data_id2 = None
        self.lookback_period = lookback_period
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.position_size = position_size
        
        # Store price history
        self.price_history1 = deque(maxlen=lookback_period+10)
        self.price_history2 = deque(maxlen=lookback_period+10)
        self.spread_history = deque(maxlen=lookback_period+10)
        self.in_position = False
        self.position_direction = None  # 'LONG_SPREAD' or 'SHORT_SPREAD'
        
    def start(self):
        """Start the strategy"""
        if self.is_running:
            logger.warning("Strategy is already running")
            return
            
        # Request market data for both symbols
        self.market_data_id1 = self.ib_connection.request_market_data(self.contract1)
        self.market_data_id2 = self.ib_connection.request_market_data(self.contract2)
        
        if self.market_data_id1 is None or self.market_data_id2 is None:
            logger.error("Failed to request market data")
            if self.market_data_id1 is not None:
                self.ib_connection.app.cancelMktData(self.market_data_id1)
            if self.market_data_id2 is not None:
                self.ib_connection.app.cancelMktData(self.market_data_id2)
            return
            
        self.is_running = True
        logger.info(f"Statistical Arbitrage strategy started for {self.symbol1} and {self.symbol2}")
        
    def stop(self):
        """Stop the strategy"""
        if not self.is_running:
            logger.warning("Strategy is not running")
            return
            
        # Cancel market data subscriptions
        if self.market_data_id1 is not None:
            self.ib_connection.app.cancelMktData(self.market_data_id1)
        if self.market_data_id2 is not None:
            self.ib_connection.app.cancelMktData(self.market_data_id2)
            
        self.is_running = False
        logger.info(f"Strategy stopped for {self.symbol1} and {self.symbol2}")
        
    def update(self):
        """Update the strategy based on new market data"""
        if not self.is_running:
            return
            
        # Get the latest market data
        market_data1 = self.ib_connection.get_data(self.market_data_id1)
        market_data2 = self.ib_connection.get_data(self.market_data_id2)
        
        # Update positions
        self.positions = self.ib_connection.get_positions()
        
        # Check take-profit conditions if enabled
        if self.take_profit_pct is not None or self.fixed_take_profit is not None or self.trailing_profit_pct is not None:
            # For stat arb, we'd need to handle take-profit differently
            # This is simplified; in a real implementation, you'd check both legs of the trade
            self.check_take_profit_conditions(market_data1)
        
        # Process the market data and generate signals
        signals = self.generate_signals(market_data1, market_data2)
        
        # Execute trades based on signals
        self.execute_trades(signals)
        
    def generate_signals(self, market_data1, market_data2):
        """Generate trading signals based on pairs trading logic"""
        # We need tick type 4 (last price) for both assets
        if 4 not in market_data1 or 4 not in market_data2:
            return None
            
        price1 = market_data1[4]
        price2 = market_data2[4]
        
        self.price_history1.append(price1)
        self.price_history2.append(price2)
        
        # Need enough data
        if len(self.price_history1) < self.lookback_period or len(self.price_history2) < self.lookback_period:
            return None
            
        # Calculate the spread (can be more sophisticated in real implementations)
        spread = price1 - price2
        self.spread_history.append(spread)
        
        # Calculate z-score
        mean_spread = sum(self.spread_history) / len(self.spread_history)
        std_spread = np.std(list(self.spread_history))
        
        if std_spread == 0:
            return None
            
        z_score = (spread - mean_spread) / std_spread
        
        # Generate signals based on z-score
        if not self.in_position:
            if z_score > self.entry_threshold:
                # Spread is too high, short the spread (sell symbol1, buy symbol2)
                logger.info(f"SHORT_SPREAD signal: Z-score ({z_score:.2f}) > threshold ({self.entry_threshold})")
                return {'action': 'SHORT_SPREAD', 'quantity': self.position_size}
            elif z_score < -self.entry_threshold:
                # Spread is too low, long the spread (buy symbol1, sell symbol2)
                logger.info(f"LONG_SPREAD signal: Z-score ({z_score:.2f}) < -threshold ({-self.entry_threshold})")
                return {'action': 'LONG_SPREAD', 'quantity': self.position_size}
        else:
            # Check for exit signals
            if (self.position_direction == 'LONG_SPREAD' and z_score >= self.exit_threshold) or \
               (self.position_direction == 'SHORT_SPREAD' and z_score <= -self.exit_threshold):
                logger.info(f"EXIT signal: Z-score ({z_score:.2f}) crossed exit threshold ({self.exit_threshold})")
                return {'action': 'EXIT', 'quantity': self.position_size}
                
        return None
        
    def execute_trades(self, signals):
        """Execute trades based on signals"""
        if signals is None:
            return
            
        action = signals['action']
        quantity = signals['quantity']
        
        if action == 'LONG_SPREAD':
            # Buy symbol1, sell symbol2
            order1 = self.ib_connection.create_order('BUY', quantity)
            order2 = self.ib_connection.create_order('SELL', quantity)
            self.ib_connection.place_order(self.contract1, order1)
            self.ib_connection.place_order(self.contract2, order2)
            self.in_position = True
            self.position_direction = 'LONG_SPREAD'
            
            # Track entry prices
            market_data1 = self.ib_connection.get_data(self.market_data_id1)
            market_data2 = self.ib_connection.get_data(self.market_data_id2)
            if 4 in market_data1 and 4 in market_data2:
                self.entry_prices[self.symbol1] = market_data1[4]
                self.entry_prices[self.symbol2] = market_data2[4]
            
        elif action == 'SHORT_SPREAD':
            # Sell symbol1, buy symbol2
            order1 = self.ib_connection.create_order('SELL', quantity)
            order2 = self.ib_connection.create_order('BUY', quantity)
            self.ib_connection.place_order(self.contract1, order1)
            self.ib_connection.place_order(self.contract2, order2)
            self.in_position = True
            self.position_direction = 'SHORT_SPREAD'
            
            # Track entry prices
            market_data1 = self.ib_connection.get_data(self.market_data_id1)
            market_data2 = self.ib_connection.get_data(self.market_data_id2)
            if 4 in market_data1 and 4 in market_data2:
                self.entry_prices[self.symbol1] = market_data1[4]
                self.entry_prices[self.symbol2] = market_data2[4]
            
        elif action == 'EXIT':
            # Close positions
            if self.position_direction == 'LONG_SPREAD':
                # Sell symbol1, buy symbol2
                order1 = self.ib_connection.create_order('SELL', quantity)
                order2 = self.ib_connection.create_order('BUY', quantity)
                self.ib_connection.place_order(self.contract1, order1)
                self.ib_connection.place_order(self.contract2, order2)
            else:  # SHORT_SPREAD
                # Buy symbol1, sell symbol2
                order1 = self.ib_connection.create_order('BUY', quantity)
                order2 = self.ib_connection.create_order('SELL', quantity)
                self.ib_connection.place_order(self.contract1, order1)
                self.ib_connection.place_order(self.contract2, order2)
                
            self.in_position = False
            self.position_direction = None
            
            # Clear entry prices
            self.clear_position_tracking()


class RSIStrategy(BaseStrategy):
    """Relative Strength Index (RSI) Strategy"""
    def __init__(self, symbol, ib_connection, period=14, overbought=70, oversold=30, position_size=100,
                 take_profit_pct=None, fixed_take_profit=None, trailing_profit_pct=None):
        super().__init__(symbol, ib_connection, take_profit_pct, fixed_take_profit, trailing_profit_pct)
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
        self.position_size = position_size
        
        # Store price history and gain/loss history
        self.price_history = deque(maxlen=period+50)
        self.gain_history = deque(maxlen=period+1)
        self.loss_history = deque(maxlen=period+1)
        self.last_rsi = None
        self.last_price = None
        self.last_signal = None
        
    def calculate_rsi(self):
        """Calculate the Relative Strength Index"""
        if len(self.price_history) <= self.period:
            return None
            
        # Calculate price changes
        changes = []
        prices = list(self.price_history)
        for i in range(1, len(prices)):
            changes.append(prices[i] - prices[i-1])
            
        # Separate gains and losses
        gains = [max(0, change) for change in changes]
        losses = [max(0, -change) for change in changes]
        
        # Calculate average gain and average loss
        avg_gain = sum(gains[-self.period:]) / self.period
        avg_loss = sum(losses[-self.period:]) / self.period
        
        # Store for future calculations
        self.gain_history.append(avg_gain)
        self.loss_history.append(avg_loss)
        
        if avg_loss == 0:
            # No losses, RSI is 100
            return 100
            
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
        
    def generate_signals(self, market_data):
        """Generate trading signals based on RSI"""
        # We need tick type 4 (last price)
        if 4 not in market_data:
            return None
            
        price = market_data[4]
        
        # Skip if price hasn't changed
        if self.last_price == price:
            return None
            
        self.last_price = price
        self.price_history.append(price)
        
        # Calculate RSI
        rsi = self.calculate_rsi()
        if rsi is None:
            return None
            
        # Track RSI changes
        rsi_changed = self.last_rsi != rsi
        self.last_rsi = rsi
        
        # Generate signals based on RSI thresholds
        if rsi <= self.oversold and (self.last_signal != 'BUY' or rsi_changed):
            logger.info(f"BUY signal: RSI ({rsi:.2f}) below oversold threshold ({self.oversold})")
            self.last_signal = 'BUY'
            return {'action': 'BUY', 'quantity': self.position_size}
        elif rsi >= self.overbought and (self.last_signal != 'SELL' or rsi_changed):
            logger.info(f"SELL signal: RSI ({rsi:.2f}) above overbought threshold ({self.overbought})")
            self.last_signal = 'SELL'
            return {'action': 'SELL', 'quantity': self.position_size}
            
        return None
        
    def execute_trades(self, signals):
        """Execute trades based on signals"""
        if signals is None:
            return
            
        # Check current position
        current_position = 0
        if self.symbol in self.positions:
            current_position = self.positions[self.symbol].get('position', 0)
            
        action = signals['action']
        quantity = signals['quantity']
        
        # Determine order type based on action and current position
        if action == 'BUY' and current_position <= 0:
            # If we have no position or a short position, buy
            order = self.ib_connection.create_order('BUY', quantity)
            self.ib_connection.place_order(self.contract, order)
            
            # Track entry price for new position
            market_data = self.ib_connection.get_data(self.market_data_id)
            if 4 in market_data:  # Tick type 4 is last price
                self.entry_prices[self.symbol] = market_data[4]
                # Initialize highest price for trailing take-profit
                if self.trailing_profit_pct is not None:
                    self.highest_since_entry[self.symbol] = market_data[4]
                    
        elif action == 'SELL' and current_position >= 0:
            # If we have no position or a long position, sell
            order = self.ib_connection.create_order('SELL', quantity)
            self.ib_connection.place_order(self.contract, order)
            
            # Track entry price for new position
            market_data = self.ib_connection.get_data(self.market_data_id)
            if 4 in market_data:  # Tick type 4 is last price
                self.entry_prices[self.symbol] = market_data[4]
                # Initialize lowest price for trailing take-profit
                if self.trailing_profit_pct is not None:
                    self.lowest_since_entry[self.symbol] = market_data[4]


class BollingerBandsStrategy(BaseStrategy):
    """Bollinger Bands Strategy"""
    def __init__(self, symbol, ib_connection, period=20, std_dev=2, position_size=100,
                 take_profit_pct=None, fixed_take_profit=None, trailing_profit_pct=None):
        super().__init__(symbol, ib_connection, take_profit_pct, fixed_take_profit, trailing_profit_pct)
        self.period = period
        self.std_dev = std_dev
        self.position_size = position_size
        
        # Store price history
        self.price_history = deque(maxlen=period+50)
        self.last_signal = None
        
    def calculate_bollinger_bands(self):
        """Calculate Bollinger Bands"""
        if len(self.price_history) < self.period:
            return None
            
        # Calculate simple moving average
        prices = list(self.price_history)
        sma = sum(prices[-self.period:]) / self.period
        
        # Calculate standard deviation
        std = np.std(prices[-self.period:])
        
        # Calculate upper and lower bands
        upper_band = sma + (self.std_dev * std)
        lower_band = sma - (self.std_dev * std)
        
        return {
            'sma': sma,
            'upper': upper_band,
            'lower': lower_band
        }
        
    def generate_signals(self, market_data):
        """Generate trading signals based on Bollinger Bands"""
        # We need tick type 4 (last price)
        if 4 not in market_data:
            return None
            
        price = market_data[4]
        self.price_history.append(price)
        
        # Calculate Bollinger Bands
        bands = self.calculate_bollinger_bands()
        if bands is None:
            return None
            
        # Generate signals based on price crossing Bollinger Bands
        if price <= bands['lower'] and self.last_signal != 'BUY':
            logger.info(f"BUY signal: Price ({price:.2f}) crossed below lower band ({bands['lower']:.2f})")
            self.last_signal = 'BUY'
            return {'action': 'BUY', 'quantity': self.position_size}
        elif price >= bands['upper'] and self.last_signal != 'SELL':
            logger.info(f"SELL signal: Price ({price:.2f}) crossed above upper band ({bands['upper']:.2f})")
            self.last_signal = 'SELL'
            return {'action': 'SELL', 'quantity': self.position_size}
        elif price > bands['sma'] and price < bands['upper'] and self.last_signal == 'SELL':
            logger.info(f"EXIT SELL signal: Price ({price:.2f}) crossed back below upper band")
            self.last_signal = None
            return {'action': 'BUY', 'quantity': self.position_size}
        elif price < bands['sma'] and price > bands['lower'] and self.last_signal == 'BUY':
            logger.info(f"EXIT BUY signal: Price ({price:.2f}) crossed back above lower band")
            self.last_signal = None
            return {'action': 'SELL', 'quantity': self.position_size}
            
        return None
        
    def execute_trades(self, signals):
        """Execute trades based on signals"""
        if signals is None:
            return
            
        # Check current position
        current_position = 0
        if self.symbol in self.positions:
            current_position = self.positions[self.symbol].get('position', 0)
            
        action = signals['action']
        quantity = signals['quantity']
        
        # Determine order type based on action and current position
        if action == 'BUY' and current_position <= 0:
            # If we have no position or a short position, buy
            order = self.ib_connection.create_order('BUY', quantity)
            self.ib_connection.place_order(self.contract, order)
            
            # Track entry price for new position
            market_data = self.ib_connection.get_data(self.market_data_id)
            if 4 in market_data:  # Tick type 4 is last price
                self.entry_prices[self.symbol] = market_data[4]
                # Initialize highest price for trailing take-profit
                if self.trailing_profit_pct is not None:
                    self.highest_since_entry[self.symbol] = market_data[4]
                    
        elif action == 'SELL' and current_position >= 0:
            # If we have no position or a long position, sell
            order = self.ib_connection.create_order('SELL', quantity)
            self.ib_connection.place_order(self.contract, order)
            
            # Track entry price for new position
            market_data = self.ib_connection.get_data(self.market_data_id)
            if 4 in market_data:  # Tick type 4 is last price
                self.entry_prices[self.symbol] = market_data[4]
                # Initialize lowest price for trailing take-profit
                if self.trailing_profit_pct is not None:
                    self.lowest_since_entry[self.symbol] = market_data[4]


class MACDStrategy(BaseStrategy):
    """Moving Average Convergence Divergence (MACD) Strategy"""
    def __init__(self, symbol, ib_connection, fast_period=12, slow_period=26, signal_period=9, position_size=100,
                 take_profit_pct=None, fixed_take_profit=None, trailing_profit_pct=None):
        super().__init__(symbol, ib_connection, take_profit_pct, fixed_take_profit, trailing_profit_pct)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.position_size = position_size
        
        # Store price history and MACD history
        self.price_history = deque(maxlen=slow_period+signal_period+50)
        self.macd_history = deque(maxlen=signal_period+50)
        self.last_signal = None
        
    def calculate_ema(self, period, prices):
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return None
            
        # Simple case - use SMA for first EMA value
        ema = sum(prices[:period]) / period
        
        # Calculate multiplier
        multiplier = 2 / (period + 1)
        
        # Calculate EMA for remaining prices
        for i in range(period, len(prices)):
            ema = (prices[i] - ema) * multiplier + ema
            
        return ema
        
    def calculate_macd(self):
        """Calculate MACD and signal line"""
        if len(self.price_history) < self.slow_period + self.signal_period:
            return None
            
        prices = list(self.price_history)
        
        # Calculate fast and slow EMAs
        fast_ema = self.calculate_ema(self.fast_period, prices)
        slow_ema = self.calculate_ema(self.slow_period, prices)
        
        if fast_ema is None or slow_ema is None:
            return None
            
        # Calculate MACD line (fast EMA - slow EMA)
        macd_line = fast_ema - slow_ema
        
        # Store MACD history for signal line calculation
        self.macd_history.append(macd_line)
        
        if len(self.macd_history) < self.signal_period:
            return {
                'macd': macd_line,
                'signal': None,
                'histogram': None
            }
            
        # Calculate signal line (EMA of MACD line)
        signal_line = self.calculate_ema(self.signal_period, list(self.macd_history))
        
        # Calculate histogram (MACD line - signal line)
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
        
    def generate_signals(self, market_data):
        """Generate trading signals based on MACD"""
        # We need tick type 4 (last price)
        if 4 not in market_data:
            return None
            
        price = market_data[4]
        self.price_history.append(price)
        
        # Calculate MACD
        macd_data = self.calculate_macd()
        if macd_data is None or macd_data['signal'] is None:
            return None
            
        macd_line = macd_data['macd']
        signal_line = macd_data['signal']
        histogram = macd_data['histogram']
        
        # Generate signals based on MACD line crossing signal line
        if macd_line > signal_line and self.last_signal != 'BUY':
            logger.info(f"BUY signal: MACD ({macd_line:.2f}) crossed above signal line ({signal_line:.2f})")
            self.last_signal = 'BUY'
            return {'action': 'BUY', 'quantity': self.position_size}
        elif macd_line < signal_line and self.last_signal != 'SELL':
            logger.info(f"SELL signal: MACD ({macd_line:.2f}) crossed below signal line ({signal_line:.2f})")
            self.last_signal = 'SELL'
            return {'action': 'SELL', 'quantity': self.position_size}
            
        return None
        
    def execute_trades(self, signals):
        """Execute trades based on signals"""
        if signals is None:
            return
            
        # Check current position
        current_position = 0
        if self.symbol in self.positions:
            current_position = self.positions[self.symbol].get('position', 0)
            
        action = signals['action']
        quantity = signals['quantity']
        
        # Determine order type based on action and current position
        if action == 'BUY' and current_position <= 0:
            # If we have no position or a short position, buy
            order = self.ib_connection.create_order('BUY', quantity)
            self.ib_connection.place_order(self.contract, order)
            
            # Track entry price for new position
            market_data = self.ib_connection.get_data(self.market_data_id)
            if 4 in market_data:  # Tick type 4 is last price
                self.entry_prices[self.symbol] = market_data[4]
                # Initialize highest price for trailing take-profit
                if self.trailing_profit_pct is not None:
                    self.highest_since_entry[self.symbol] = market_data[4]
                    
        elif action == 'SELL' and current_position >= 0:
            # If we have no position or a long position, sell
            order = self.ib_connection.create_order('SELL', quantity)
            self.ib_connection.place_order(self.contract, order)
            
            # Track entry price for new position
            market_data = self.ib_connection.get_data(self.market_data_id)
            if 4 in market_data:  # Tick type 4 is last price
                self.entry_prices[self.symbol] = market_data[4]
                # Initialize lowest price for trailing take-profit
                if self.trailing_profit_pct is not None:
                    self.lowest_since_entry[self.symbol] = market_data[4]

# New Hybrid Strategy classes
class HybridBaseStrategy(BaseStrategy):
    """Base Strategy class that uses Alpaca for data and IB for execution"""
    def __init__(self, symbol, ib_connection, alpaca_provider, take_profit_pct=None, fixed_take_profit=None, trailing_profit_pct=None):
        super().__init__(symbol, ib_connection, take_profit_pct, fixed_take_profit, trailing_profit_pct)
        self.alpaca = alpaca_provider
        
    def start(self):
        """Start the strategy"""
        if self.is_running:
            logger.warning("Strategy is already running")
            return
            
        # We don't need IB market data, using Alpaca instead
        self.is_running = True
        logger.info(f"Hybrid strategy started for {self.symbol}")
        
    def stop(self):
        """Stop the strategy"""
        if not self.is_running:
            logger.warning("Strategy is not running")
            return
            
        self.is_running = False
        logger.info(f"Hybrid strategy stopped for {self.symbol}")
        
    def update(self):
        """Update the strategy based on Alpaca market data"""
        if not self.is_running:
            return
            
        # Get the latest market data from Alpaca
        market_data = self.alpaca.get_market_data_dict(self.symbol)
        
        # Update positions from IB
        self.positions = self.ib_connection.get_positions()
        
        # Check take-profit conditions if enabled
        if self.take_profit_pct is not None or self.fixed_take_profit is not None or self.trailing_profit_pct is not None:
            self.check_take_profit_conditions(market_data)
            
        # Update highest/lowest prices for trailing take-profit
        if self.trailing_profit_pct is not None and 4 in market_data:
            self.update_price_extremes(market_data)
        
        # Process the market data and generate signals
        signals = self.generate_signals(market_data)
        
        # Execute trades based on signals through IB
        self.execute_trades(signals)


class HybridMovingAverageCrossStrategy(HybridBaseStrategy):
    """Moving Average Crossover Strategy using Alpaca data and IB execution"""
    def __init__(self, symbol, ib_connection, alpaca_provider, fast_period=10, slow_period=30, position_size=100,
                 take_profit_pct=None, fixed_take_profit=None, trailing_profit_pct=None):
        super().__init__(symbol, ib_connection, alpaca_provider, take_profit_pct, fixed_take_profit, trailing_profit_pct)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.position_size = position_size
        
        # Store price history
        self.price_history = deque(maxlen=slow_period+10)
        self.last_signal = None
        
    def generate_signals(self, market_data):
        """Generate trading signals based on moving average crossover"""
        # We need tick type 4 (last price)
        if 4 not in market_data:
            return None
            
        last_price = market_data[4]
        self.price_history.append(last_price)
        
        # Need enough data to calculate moving averages
        if len(self.price_history) < self.slow_period:
            return None
            
        # Calculate moving averages
        prices = list(self.price_history)
        fast_ma = sum(prices[-self.fast_period:]) / self.fast_period
        slow_ma = sum(prices[-self.slow_period:]) / self.slow_period
        
        # Generate signals
        if fast_ma > slow_ma and (self.last_signal is None or self.last_signal != 'BUY'):
            logger.info(f"BUY signal: Fast MA ({fast_ma:.2f}) crossed above Slow MA ({slow_ma:.2f})")
            self.last_signal = 'BUY'
            return {'action': 'BUY', 'quantity': self.position_size}
        elif fast_ma < slow_ma and (self.last_signal is None or self.last_signal != 'SELL'):
            logger.info(f"SELL signal: Fast MA ({fast_ma:.2f}) crossed below Slow MA ({slow_ma:.2f})")
            self.last_signal = 'SELL'
            return {'action': 'SELL', 'quantity': self.position_size}
           
        return None
       
    def execute_trades(self, signals):
       """Execute trades based on signals"""
       if signals is None:
           return
           
       # Check current position
       current_position = 0
       if self.symbol in self.positions:
           current_position = self.positions[self.symbol].get('position', 0)
           
       action = signals['action']
       quantity = signals['quantity']
       
       # Determine order type based on action and current position
       if action == 'BUY' and current_position <= 0:
           # If we have no position or a short position, buy
           order = self.ib_connection.create_order('BUY', quantity)
           self.ib_connection.place_order(self.contract, order)
           
           # Track entry price for new position
           market_data = self.alpaca.get_market_data_dict(self.symbol)
           if 4 in market_data:  # Tick type 4 is last price
               self.entry_prices[self.symbol] = market_data[4]
               # Initialize highest price for trailing take-profit
               if self.trailing_profit_pct is not None:
                   self.highest_since_entry[self.symbol] = market_data[4]
                   
       elif action == 'SELL' and current_position >= 0:
           # If we have no position or a long position, sell
           order = self.ib_connection.create_order('SELL', quantity)
           self.ib_connection.place_order(self.contract, order)
           
           # Track entry price for new position
           market_data = self.alpaca.get_market_data_dict(self.symbol)
           if 4 in market_data:  # Tick type 4 is last price
               self.entry_prices[self.symbol] = market_data[4]
               # Initialize lowest price for trailing take-profit
               if self.trailing_profit_pct is not None:
                   self.lowest_since_entry[self.symbol] = market_data[4]


class HybridStatisticalArbitrageStrategy(HybridBaseStrategy):
   """Statistical Arbitrage Strategy between two correlated assets using Alpaca data and IB execution"""
   def __init__(self, symbol1, symbol2, ib_connection, alpaca_provider, lookback_period=100, 
                entry_threshold=2.0, exit_threshold=0.0, position_size=100,
                take_profit_pct=None, fixed_take_profit=None, trailing_profit_pct=None):
       super().__init__(symbol1, ib_connection, alpaca_provider, take_profit_pct, fixed_take_profit, trailing_profit_pct)
       self.symbol1 = symbol1
       self.symbol2 = symbol2
       self.contract1 = ib_connection.create_contract(symbol1)
       self.contract2 = ib_connection.create_contract(symbol2)
       self.lookback_period = lookback_period
       self.entry_threshold = entry_threshold
       self.exit_threshold = exit_threshold
       self.position_size = position_size
       
       # Store price history
       self.price_history1 = deque(maxlen=lookback_period+10)
       self.price_history2 = deque(maxlen=lookback_period+10)
       self.spread_history = deque(maxlen=lookback_period+10)
       self.in_position = False
       self.position_direction = None  # 'LONG_SPREAD' or 'SHORT_SPREAD'
       
   def update(self):
       """Update the strategy based on Alpaca market data"""
       if not self.is_running:
           return
           
       # Get the latest market data from Alpaca for both symbols
       price1 = self.alpaca.get_latest_price(self.symbol1)
       price2 = self.alpaca.get_latest_price(self.symbol2)
       
       if price1 is None or price2 is None:
           return
       
       # Update price histories
       self.price_history1.append(price1)
       self.price_history2.append(price2)
       
       # Update positions from IB
       self.positions = self.ib_connection.get_positions()
       
       # Check take-profit conditions if enabled
       # For simplicity, we'll skip the take-profit implementation for pairs trading here
       
       # Process the market data and generate signals
       signals = self.generate_signals(price1, price2)
       
       # Execute trades based on signals through IB
       self.execute_trades(signals)
       
   def generate_signals(self, price1, price2):
       """Generate trading signals based on pairs trading logic"""
       # Need enough data
       if len(self.price_history1) < self.lookback_period or len(self.price_history2) < self.lookback_period:
           return None
           
       # Calculate the spread (can be more sophisticated in real implementations)
       spread = price1 - price2
       self.spread_history.append(spread)
       
       # Calculate z-score
       mean_spread = sum(self.spread_history) / len(self.spread_history)
       std_spread = np.std(list(self.spread_history))
       
       if std_spread == 0:
           return None
           
       z_score = (spread - mean_spread) / std_spread
       
       # Generate signals based on z-score
       if not self.in_position:
           if z_score > self.entry_threshold:
               # Spread is too high, short the spread (sell symbol1, buy symbol2)
               logger.info(f"SHORT_SPREAD signal: Z-score ({z_score:.2f}) > threshold ({self.entry_threshold})")
               return {'action': 'SHORT_SPREAD', 'quantity': self.position_size}
           elif z_score < -self.entry_threshold:
               # Spread is too low, long the spread (buy symbol1, sell symbol2)
               logger.info(f"LONG_SPREAD signal: Z-score ({z_score:.2f}) < -threshold ({-self.entry_threshold})")
               return {'action': 'LONG_SPREAD', 'quantity': self.position_size}
       else:
           # Check for exit signals
           if (self.position_direction == 'LONG_SPREAD' and z_score >= self.exit_threshold) or \
              (self.position_direction == 'SHORT_SPREAD' and z_score <= -self.exit_threshold):
               logger.info(f"EXIT signal: Z-score ({z_score:.2f}) crossed exit threshold ({self.exit_threshold})")
               return {'action': 'EXIT', 'quantity': self.position_size}
               
       return None
       
   def execute_trades(self, signals):
       """Execute trades based on signals"""
       if signals is None:
           return
           
       action = signals['action']
       quantity = signals['quantity']
       
       if action == 'LONG_SPREAD':
           # Buy symbol1, sell symbol2
           order1 = self.ib_connection.create_order('BUY', quantity)
           order2 = self.ib_connection.create_order('SELL', quantity)
           self.ib_connection.place_order(self.contract1, order1)
           self.ib_connection.place_order(self.contract2, order2)
           self.in_position = True
           self.position_direction = 'LONG_SPREAD'
           
           # Track entry prices
           price1 = self.alpaca.get_latest_price(self.symbol1)
           price2 = self.alpaca.get_latest_price(self.symbol2)
           if price1 is not None and price2 is not None:
               self.entry_prices[self.symbol1] = price1
               self.entry_prices[self.symbol2] = price2
           
       elif action == 'SHORT_SPREAD':
           # Sell symbol1, buy symbol2
           order1 = self.ib_connection.create_order('SELL', quantity)
           order2 = self.ib_connection.create_order('BUY', quantity)
           self.ib_connection.place_order(self.contract1, order1)
           self.ib_connection.place_order(self.contract2, order2)
           self.in_position = True
           self.position_direction = 'SHORT_SPREAD'
           
           # Track entry prices
           price1 = self.alpaca.get_latest_price(self.symbol1)
           price2 = self.alpaca.get_latest_price(self.symbol2)
           if price1 is not None and price2 is not None:
               self.entry_prices[self.symbol1] = price1
               self.entry_prices[self.symbol2] = price2
           
       elif action == 'EXIT':
           # Close positions
           if self.position_direction == 'LONG_SPREAD':
               # Sell symbol1, buy symbol2
               order1 = self.ib_connection.create_order('SELL', quantity)
               order2 = self.ib_connection.create_order('BUY', quantity)
               self.ib_connection.place_order(self.contract1, order1)
               self.ib_connection.place_order(self.contract2, order2)
           else:  # SHORT_SPREAD
               # Buy symbol1, sell symbol2
               order1 = self.ib_connection.create_order('BUY', quantity)
               order2 = self.ib_connection.create_order('SELL', quantity)
               self.ib_connection.place_order(self.contract1, order1)
               self.ib_connection.place_order(self.contract2, order2)
               
           self.in_position = False
           self.position_direction = None
           
           # Clear entry prices
           self.clear_position_tracking()
class HybridRSIStrategy(HybridBaseStrategy):
   """RSI Strategy using Alpaca data and IB execution"""
   def __init__(self, symbol, ib_connection, alpaca_provider, period=14, overbought=70, oversold=30, position_size=100,
                take_profit_pct=None, fixed_take_profit=None, trailing_profit_pct=None):
       super().__init__(symbol, ib_connection, alpaca_provider, take_profit_pct, fixed_take_profit, trailing_profit_pct)
       self.period = period
       self.overbought = overbought
       self.oversold = oversold
       self.position_size = position_size
       
       # Store price history and gain/loss history
       self.price_history = deque(maxlen=period+50)
       self.gain_history = deque(maxlen=period+1)
       self.loss_history = deque(maxlen=period+1)
       self.last_rsi = None
       self.last_price = None
       self.last_signal = None
       
   def calculate_rsi(self):
       """Calculate the Relative Strength Index"""
       if len(self.price_history) <= self.period:
           return None
           
       # Calculate price changes
       changes = []
       prices = list(self.price_history)
       for i in range(1, len(prices)):
           changes.append(prices[i] - prices[i-1])
           
       # Separate gains and losses
       gains = [max(0, change) for change in changes]
       losses = [max(0, -change) for change in changes]
       
       # Calculate average gain and average loss
       avg_gain = sum(gains[-self.period:]) / self.period
       avg_loss = sum(losses[-self.period:]) / self.period
       
       # Store for future calculations
       self.gain_history.append(avg_gain)
       self.loss_history.append(avg_loss)
       
       if avg_loss == 0:
           # No losses, RSI is 100
           return 100
           
       # Calculate RS and RSI
       rs = avg_gain / avg_loss
       rsi = 100 - (100 / (1 + rs))
       
       return rsi
       
   def generate_signals(self, market_data):
       """Generate trading signals based on RSI"""
       # We need tick type 4 (last price)
       if 4 not in market_data:
           return None
           
       price = market_data[4]
       
       # Skip if price hasn't changed
       if self.last_price == price:
           return None
           
       self.last_price = price
       self.price_history.append(price)
       
       # Calculate RSI
       rsi = self.calculate_rsi()
       if rsi is None:
           return None
           
       # Track RSI changes
       rsi_changed = self.last_rsi != rsi
       self.last_rsi = rsi
       
       # Generate signals based on RSI thresholds
       if rsi <= self.oversold and (self.last_signal != 'BUY' or rsi_changed):
           logger.info(f"BUY signal: RSI ({rsi:.2f}) below oversold threshold ({self.oversold})")
           self.last_signal = 'BUY'
           return {'action': 'BUY', 'quantity': self.position_size}
       elif rsi >= self.overbought and (self.last_signal != 'SELL' or rsi_changed):
           logger.info(f"SELL signal: RSI ({rsi:.2f}) above overbought threshold ({self.overbought})")
           self.last_signal = 'SELL'
           return {'action': 'SELL', 'quantity': self.position_size}
           
       return None
       
   def execute_trades(self, signals):
       """Execute trades based on signals"""
       if signals is None:
           return
           
       # Check current position
       current_position = 0
       if self.symbol in self.positions:
           current_position = self.positions[self.symbol].get('position', 0)
           
       action = signals['action']
       quantity = signals['quantity']
       
       # Determine order type based on action and current position
       if action == 'BUY' and current_position <= 0:
           # If we have no position or a short position, buy
           order = self.ib_connection.create_order('BUY', quantity)
           self.ib_connection.place_order(self.contract, order)
           
           # Track entry price for new position
           market_data = self.alpaca.get_market_data_dict(self.symbol)
           if 4 in market_data:  # Tick type 4 is last price
               self.entry_prices[self.symbol] = market_data[4]
               # Initialize highest price for trailing take-profit
               if self.trailing_profit_pct is not None:
                   self.highest_since_entry[self.symbol] = market_data[4]
                   
       elif action == 'SELL' and current_position >= 0:
           # If we have no position or a long position, sell
           order = self.ib_connection.create_order('SELL', quantity)
           self.ib_connection.place_order(self.contract, order)
           
           # Track entry price for new position
           market_data = self.alpaca.get_market_data_dict(self.symbol)
           if 4 in market_data:  # Tick type 4 is last price
               self.entry_prices[self.symbol] = market_data[4]
               # Initialize lowest price for trailing take-profit
               if self.trailing_profit_pct is not None:
                   self.lowest_since_entry[self.symbol] = market_data[4]


class HybridBollingerBandsStrategy(HybridBaseStrategy):
   """Bollinger Bands Strategy using Alpaca data and IB execution"""
   def __init__(self, symbol, ib_connection, alpaca_provider, period=20, std_dev=2, position_size=100,
                take_profit_pct=None, fixed_take_profit=None, trailing_profit_pct=None):
       super().__init__(symbol, ib_connection, alpaca_provider, take_profit_pct, fixed_take_profit, trailing_profit_pct)
       self.period = period
       self.std_dev = std_dev
       self.position_size = position_size
       
       # Store price history
       self.price_history = deque(maxlen=period+50)
       self.last_signal = None
       
   def calculate_bollinger_bands(self):
       """Calculate Bollinger Bands"""
       if len(self.price_history) < self.period:
           return None
           
       # Calculate simple moving average
       prices = list(self.price_history)
       sma = sum(prices[-self.period:]) / self.period
       
       # Calculate standard deviation
       std = np.std(prices[-self.period:])
       
       # Calculate upper and lower bands
       upper_band = sma + (self.std_dev * std)
       lower_band = sma - (self.std_dev * std)
       
       return {
           'sma': sma,
           'upper': upper_band,
           'lower': lower_band
       }
       
   def generate_signals(self, market_data):
       """Generate trading signals based on Bollinger Bands"""
       # We need tick type 4 (last price)
       if 4 not in market_data:
           return None
           
       price = market_data[4]
       self.price_history.append(price)
       
       # Calculate Bollinger Bands
       bands = self.calculate_bollinger_bands()
       if bands is None:
           return None
           
       # Generate signals based on price crossing Bollinger Bands
       if price <= bands['lower'] and self.last_signal != 'BUY':
           logger.info(f"BUY signal: Price ({price:.2f}) crossed below lower band ({bands['lower']:.2f})")
           self.last_signal = 'BUY'
           return {'action': 'BUY', 'quantity': self.position_size}
       elif price >= bands['upper'] and self.last_signal != 'SELL':
           logger.info(f"SELL signal: Price ({price:.2f}) crossed above upper band ({bands['upper']:.2f})")
           self.last_signal = 'SELL'
           return {'action': 'SELL', 'quantity': self.position_size}
       elif price > bands['sma'] and price < bands['upper'] and self.last_signal == 'SELL':
           logger.info(f"EXIT SELL signal: Price ({price:.2f}) crossed back below upper band")
           self.last_signal = None
           return {'action': 'BUY', 'quantity': self.position_size}
       elif price < bands['sma'] and price > bands['lower'] and self.last_signal == 'BUY':
           logger.info(f"EXIT BUY signal: Price ({price:.2f}) crossed back above lower band")
           self.last_signal = None
           return {'action': 'SELL', 'quantity': self.position_size}
           
       return None
       
   def execute_trades(self, signals):
       """Execute trades based on signals"""
       if signals is None:
           return
           
       # Check current position
       current_position = 0
       if self.symbol in self.positions:
           current_position = self.positions[self.symbol].get('position', 0)
           
       action = signals['action']
       quantity = signals['quantity']
       
       # Determine order type based on action and current position
       if action == 'BUY' and current_position <= 0:
           # If we have no position or a short position, buy
           order = self.ib_connection.create_order('BUY', quantity)
           self.ib_connection.place_order(self.contract, order)
           
           # Track entry price for new position
           market_data = self.alpaca.get_market_data_dict(self.symbol)
           if 4 in market_data:
               self.entry_prices[self.symbol] = market_data[4]
               # Initialize highest price for trailing take-profit
               if self.trailing_profit_pct is not None:
                   self.highest_since_entry[self.symbol] = market_data[4]
                   
       elif action == 'SELL' and current_position >= 0:
           # If we have no position or a long position, sell
           order = self.ib_connection.create_order('SELL', quantity)
           self.ib_connection.place_order(self.contract, order)
           
           # Track entry price for new position
           market_data = self.alpaca.get_market_data_dict(self.symbol)
           if 4 in market_data:
               self.entry_prices[self.symbol] = market_data[4]
               # Initialize lowest price for trailing take-profit
               if self.trailing_profit_pct is not None:
                   self.lowest_since_entry[self.symbol] = market_data[4]


class HybridMACDStrategy(HybridBaseStrategy):
   """MACD Strategy using Alpaca data and IB execution"""
   def __init__(self, symbol, ib_connection, alpaca_provider, fast_period=12, slow_period=26, signal_period=9, position_size=100,
                take_profit_pct=None, fixed_take_profit=None, trailing_profit_pct=None):
       super().__init__(symbol, ib_connection, alpaca_provider, take_profit_pct, fixed_take_profit, trailing_profit_pct)
       self.fast_period = fast_period
       self.slow_period = slow_period
       self.signal_period = signal_period
       self.position_size = position_size
       
       # Store price history and MACD history
       self.price_history = deque(maxlen=slow_period+signal_period+50)
       self.macd_history = deque(maxlen=signal_period+50)
       self.last_signal = None
       
   def calculate_ema(self, period, prices):
       """Calculate Exponential Moving Average"""
       if len(prices) < period:
           return None
           
       # Simple case - use SMA for first EMA value
       ema = sum(prices[:period]) / period
       
       # Calculate multiplier
       multiplier = 2 / (period + 1)
       
       # Calculate EMA for remaining prices
       for i in range(period, len(prices)):
           ema = (prices[i] - ema) * multiplier + ema
           
       return ema
       
   def calculate_macd(self):
       """Calculate MACD and signal line"""
       if len(self.price_history) < self.slow_period + self.signal_period:
           return None
           
       prices = list(self.price_history)
       
       # Calculate fast and slow EMAs
       fast_ema = self.calculate_ema(self.fast_period, prices)
       slow_ema = self.calculate_ema(self.slow_period, prices)
       
       if fast_ema is None or slow_ema is None:
           return None
           
       # Calculate MACD line (fast EMA - slow EMA)
       macd_line = fast_ema - slow_ema
       
       # Store MACD history for signal line calculation
       self.macd_history.append(macd_line)
       
       if len(self.macd_history) < self.signal_period:
           return {
               'macd': macd_line,
               'signal': None,
               'histogram': None
           }
           
       # Calculate signal line (EMA of MACD line)
       signal_line = self.calculate_ema(self.signal_period, list(self.macd_history))
       
       # Calculate histogram (MACD line - signal line)
       histogram = macd_line - signal_line
       
       return {
           'macd': macd_line,
           'signal': signal_line,
           'histogram': histogram
       }
       
   def generate_signals(self, market_data):
       """Generate trading signals based on MACD"""
       # We need tick type 4 (last price)
       if 4 not in market_data:
           return None
           
       price = market_data[4]
       self.price_history.append(price)
       
       # Calculate MACD
       macd_data = self.calculate_macd()
       if macd_data is None or macd_data['signal'] is None:
           return None
           
       macd_line = macd_data['macd']
       signal_line = macd_data['signal']
       histogram = macd_data['histogram']
       
       # Generate signals based on MACD line crossing signal line
       if macd_line > signal_line and self.last_signal != 'BUY':
           logger.info(f"BUY signal: MACD ({macd_line:.2f}) crossed above signal line ({signal_line:.2f})")
           self.last_signal = 'BUY'
           return {'action': 'BUY', 'quantity': self.position_size}
       elif macd_line < signal_line and self.last_signal != 'SELL':
           logger.info(f"SELL signal: MACD ({macd_line:.2f}) crossed below signal line ({signal_line:.2f})")
           self.last_signal = 'SELL'
           return {'action': 'SELL', 'quantity': self.position_size}
           
       return None
       
   def execute_trades(self, signals):
       """Execute trades based on signals"""
       if signals is None:
           return
           
       # Check current position
       current_position = 0
       if self.symbol in self.positions:
           current_position = self.positions[self.symbol].get('position', 0)
           
       action = signals['action']
       quantity = signals['quantity']
       
       # Determine order type based on action and current position
       if action == 'BUY' and current_position <= 0:
           # If we have no position or a short position, buy
           order = self.ib_connection.create_order('BUY', quantity)
           self.ib_connection.place_order(self.contract, order)
           
           # Track entry price for new position
           market_data = self.alpaca.get_market_data_dict(self.symbol)
           if 4 in market_data:
               self.entry_prices[self.symbol] = market_data[4]
               # Initialize highest price for trailing take-profit
               if self.trailing_profit_pct is not None:
                   self.highest_since_entry[self.symbol] = market_data[4]
                   
       elif action == 'SELL' and current_position >= 0:
           # If we have no position or a long position, sell
           order = self.ib_connection.create_order('SELL', quantity)
           self.ib_connection.place_order(self.contract, order)
           
           # Track entry price for new position
           market_data = self.alpaca.get_market_data_dict(self.symbol)
           if 4 in market_data:
               self.entry_prices[self.symbol] = market_data[4]
               # Initialize lowest price for trailing take-profit
               if self.trailing_profit_pct is not None:
                   self.lowest_since_entry[self.symbol] = market_data[4]


# Example usage
if __name__ == "__main__":
   # This is a simple test to demonstrate how to use the strategies
   # In a real application, you would import this module and use it with IBConnection
   
   import time
   from ib_connection import IBConnection
   
   # Create a connection
   ib = IBConnection()
   
   # Connect to TWS/IB Gateway
   if ib.connect():
       try:
           # Create a strategy for AAPL with take-profit
           strategy = MovingAverageCrossStrategy("AAPL", ib, 
                                              take_profit_pct=5.0,  # Take 5% profit
                                              trailing_profit_pct=2.0)  # Or trail by 2%
           
           # Start the strategy
           strategy.start()
           
           # Run for 60 seconds
           for _ in range(60):
               strategy.update()
               time.sleep(1)
               
       finally:
           # Stop the strategy
           strategy.stop()
           
           # Disconnect
           ib.disconnect()
   else:
       print("Failed to connect to Interactive Brokers")