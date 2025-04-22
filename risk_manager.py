import logging
import time
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RiskManager:
    """Risk management component to control trading risk"""
    def __init__(self, ib_connection, max_position_size=1000, max_daily_loss_pct=2.0,
                 max_drawdown_pct=5.0, max_trades_per_day=100):
        self.ib_connection = ib_connection
        self.max_position_size = max_position_size
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.max_trades_per_day = max_trades_per_day
        
        # Trading metrics
        self.initial_equity = 0
        self.current_equity = 0
        self.peak_equity = 0
        self.daily_start_equity = 0
        self.trade_count_today = 0
        self.last_reset_day = None
        self.trading_allowed = True
        
        # Position tracking
        self.position_limits = {}  # Per-symbol position limits
        self.symbol_exposures = {}  # Current exposure per symbol
        
        # Order tracking
        self.open_orders = {}  # Track open orders
        self.order_history = []  # Historical order record
        
    def initialize(self):
        """Initialize risk management with current account values"""
        # In a real implementation, you would request account summary
        # For paper trading, we'll use a placeholder value
        self.initial_equity = 100000  # $100,000 starting capital
        self.current_equity = self.initial_equity
        self.peak_equity = self.initial_equity
        self.daily_start_equity = self.initial_equity
        self.last_reset_day = datetime.now().date()
        self.trading_allowed = True
        
        # Request current positions to initialize position tracking
        self.ib_connection.request_positions()
        
        # Allow time for position data to be received
        time.sleep(0.01)
        
        # Initialize position exposures
        positions = self.ib_connection.get_positions()
        for symbol, position_data in positions.items():
            self.symbol_exposures[symbol] = abs(position_data.get('position', 0))
        
        logger.info(f"Risk manager initialized with ${self.initial_equity:.2f} equity")
        
    def update(self):
        """Update risk metrics and check risk limits"""
        # Reset daily metrics if needed
        current_date = datetime.now().date()
        if self.last_reset_day != current_date:
            self.reset_daily_metrics()
            
        # In a real implementation, you would get current equity from account updates
        # For this example, we'll estimate current equity based on positions and prices
        
        # Get current positions and their values
        positions = self.ib_connection.get_positions()
        
        # Update position exposures
        for symbol, position_data in positions.items():
            self.symbol_exposures[symbol] = abs(position_data.get('position', 0))
        
        # Update order status
        self._update_order_status()
        
        # Calculate current equity (simplified)
        # In reality, you would use account updates from IB
        
        # Check risk limits
        self.check_risk_limits()
        
    def reset_daily_metrics(self):
        """Reset daily trading metrics"""
        self.daily_start_equity = self.current_equity
        self.trade_count_today = 0
        self.last_reset_day = datetime.now().date()
        self.trading_allowed = True
        
        logger.info(f"Daily metrics reset. Starting equity: ${self.daily_start_equity:.2f}")
        
    def check_risk_limits(self):
        """Check all risk limits and disable trading if necessary"""
        if not self.trading_allowed:
            return False
            
        # Check maximum drawdown
        if self.current_equity < self.peak_equity * (1 - self.max_drawdown_pct / 100):
            logger.warning(f"Maximum drawdown limit reached. Trading disabled.")
            self.trading_allowed = False
            return False
            
        # Check daily loss limit
        if self.current_equity < self.daily_start_equity * (1 - self.max_daily_loss_pct / 100):
            logger.warning(f"Maximum daily loss limit reached. Trading disabled for today.")
            self.trading_allowed = False
            return False
            
        # Check daily trade count limit
        if self.trade_count_today >= self.max_trades_per_day:
            logger.warning(f"Maximum daily trades limit reached. Trading disabled for today.")
            self.trading_allowed = False
            return False
            
        return True
        
    def check_position_size(self, symbol, quantity):
        """Check if a new position would exceed position size limits"""
        if not self.trading_allowed:
            return False
            
        # Get current position
        positions = self.ib_connection.get_positions()
        current_position = 0
        if symbol in positions:
            current_position = abs(positions[symbol].get('position', 0))
            
        # Get symbol-specific limit if it exists, otherwise use default
        max_size = self.position_limits.get(symbol, self.max_position_size)
        
        # Check if new position would exceed limits
        if current_position + quantity > max_size:
            logger.warning(f"Position size limit would be exceeded for {symbol}. Trade rejected.")
            return False
            
        return True
        
    def set_position_limit(self, symbol, max_size):
        """Set a position size limit for a specific symbol"""
        self.position_limits[symbol] = max_size
        logger.info(f"Position limit set for {symbol}: {max_size}")
        
    def record_trade(self, symbol, action, quantity, price, order_id):
        """Record a completed trade"""
        self.trade_count_today += 1
        
        # Record in order history
        trade_record = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'price': price,
            'order_id': order_id
        }
        self.order_history.append(trade_record)
        
        logger.info(f"Trade recorded: {action} {quantity} {symbol} @ {price}. "
                  f"Total trades today: {self.trade_count_today}")
        
    def emergency_close_all_positions(self):
        """Emergency function to close all positions"""
        logger.warning("EMERGENCY: Closing all positions")
        
        # Get all current positions
        positions = self.ib_connection.get_positions()
        
        for symbol, position_data in positions.items():
            position = position_data.get('position', 0)
            if position == 0:
                continue
                
            # Create a contract for this symbol
            contract = self.ib_connection.create_contract(symbol)
            
            # Create an order to close the position
            action = 'SELL' if position > 0 else 'BUY'
            quantity = abs(position)
            order = self.ib_connection.create_order(action, quantity)
            
            # Place the order
            self.ib_connection.place_order(contract, order)
            
            logger.info(f"Emergency order placed: {action} {quantity} {symbol}")
            
        logger.warning("Emergency close all positions completed")
        
    def cancel_all_orders(self):
        """Cancel all open orders"""
        logger.warning("Cancelling all open orders")
        
        # Loop through all open orders
        for order_id in list(self.open_orders.keys()):
            self.ib_connection.cancel_order(order_id)
            
        logger.warning("Cancel all orders completed")
        
    def _update_order_status(self):
        """Update status of open orders"""
        # Loop through open orders and check their status
        for order_id in list(self.open_orders.keys()):
            status = self.ib_connection.get_order_status(order_id)
            
            # If order is filled or cancelled, remove from open orders
            if status.get('status') in ['Filled', 'Cancelled', 'ApiCancelled']:
                # If filled, record the trade
                if status.get('status') == 'Filled':
                    order_info = self.open_orders[order_id]
                    self.record_trade(
                        order_info['symbol'],
                        order_info['action'],
                        status.get('filled', 0),
                        status.get('avgFillPrice', 0),
                        order_id
                    )
                
                # Remove from open orders
                del self.open_orders[order_id]
                
    def track_order(self, order_id, symbol, action, quantity):
        """Track a new order"""
        self.open_orders[order_id] = {
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'timestamp': datetime.now()
        }
        
    def get_trading_status(self):
        """Get current trading status and limits"""
        return {
            'trading_allowed': self.trading_allowed,
            'current_equity': self.current_equity,
            'initial_equity': self.initial_equity,
            'peak_equity': self.peak_equity,
            'daily_start_equity': self.daily_start_equity,
            'trade_count_today': self.trade_count_today,
            'max_trades_per_day': self.max_trades_per_day,
            'max_daily_loss_pct': self.max_daily_loss_pct,
            'max_drawdown_pct': self.max_drawdown_pct
        }
        
    def get_position_exposure(self, symbol=None):
        """Get current position exposure"""
        if symbol:
            return self.symbol_exposures.get(symbol, 0)
        return self.symbol_exposures
        
    def calculate_portfolio_exposure(self):
        """Calculate total portfolio exposure as percentage of equity"""
        total_exposure = sum(self.symbol_exposures.values())
        if self.current_equity == 0:
            return 0
        return (total_exposure / self.current_equity) * 100
        
    def check_portfolio_exposure(self, max_exposure_pct=100):
        """Check if portfolio exposure is below the maximum allowed"""
        exposure_pct = self.calculate_portfolio_exposure()
        if exposure_pct > max_exposure_pct:
            logger.warning(f"Portfolio exposure ({exposure_pct:.2f}%) exceeds maximum ({max_exposure_pct}%)")
            return False
        return True

    def adjust_position_limits_by_volatility(self, symbol, lookback_days=20):
        """Adjust position limits based on historical volatility"""
        # Get historical data
        contract = self.ib_connection.create_contract(symbol)
        req_id = self.ib_connection.request_historical_data(
            contract, 
            duration=f"{lookback_days} D", 
            bar_size="1 day", 
            what_to_show="TRADES"
        )
        
        # Wait for data to arrive
        time.sleep(0.01)
        
        # Get the data
        hist_data = self.ib_connection.get_data(req_id)
        
        if "bars" not in hist_data or len(hist_data["bars"]) < lookback_days/2:
            logger.warning(f"Insufficient historical data for {symbol}")
            return
            
        # Calculate daily returns
        returns = []
        bars = hist_data["bars"]
        for i in range(1, len(bars)):
            prev_close = bars[i-1]["close"]
            curr_close = bars[i]["close"]
            returns.append((curr_close - prev_close) / prev_close)
            
        # Calculate volatility (standard deviation of returns)
        import numpy as np
        volatility = np.std(returns) * 100  # Convert to percentage
        
        # Adjust position limit based on volatility
        # Lower volatility = higher position limit
        base_limit = self.max_position_size
        volatility_factor = 20 / volatility if volatility > 0 else 2  # Cap at 2x for low volatility
        adjusted_limit = int(base_limit * min(volatility_factor, 2))
        
        # Set the new limit
        self.set_position_limit(symbol, adjusted_limit)
        
        logger.info(f"Adjusted position limit for {symbol} based on volatility ({volatility:.2f}%): {adjusted_limit}")
        
    def generate_risk_report(self):
        """Generate a comprehensive risk report"""
        # Calculate drawdown
        current_drawdown_pct = 0
        if self.peak_equity > 0:
            current_drawdown_pct = ((self.peak_equity - self.current_equity) / self.peak_equity) * 100
            
        # Calculate daily P&L
        daily_pnl = self.current_equity - self.daily_start_equity
        daily_pnl_pct = (daily_pnl / self.daily_start_equity) * 100 if self.daily_start_equity > 0 else 0
        
        # Calculate overall P&L
        total_pnl = self.current_equity - self.initial_equity
        total_pnl_pct = (total_pnl / self.initial_equity) * 100 if self.initial_equity > 0 else 0
        
        # Calculate exposure statistics
        portfolio_exposure = self.calculate_portfolio_exposure()
        max_symbol_exposure = max(self.symbol_exposures.values()) if self.symbol_exposures else 0
        most_exposed_symbol = max(self.symbol_exposures.items(), key=lambda x: x[1])[0] if self.symbol_exposures else 'None'
        
        # Generate report
        report = {
            'timestamp': datetime.now(),
            'trading_status': {
                'trading_allowed': self.trading_allowed,
                'trade_count_today': self.trade_count_today,
                'max_trades_per_day': self.max_trades_per_day
            },
            'equity': {
                'current_equity': self.current_equity,
                'initial_equity': self.initial_equity,
                'peak_equity': self.peak_equity,
                'daily_start_equity': self.daily_start_equity
            },
            'performance': {
                'current_drawdown_pct': current_drawdown_pct,
                'max_drawdown_pct': self.max_drawdown_pct,
                'daily_pnl': daily_pnl,
                'daily_pnl_pct': daily_pnl_pct,
                'max_daily_loss_pct': self.max_daily_loss_pct,
                'total_pnl': total_pnl,
                'total_pnl_pct': total_pnl_pct
            },
            'exposure': {
                'portfolio_exposure_pct': portfolio_exposure,
                'max_symbol_exposure': max_symbol_exposure,
                'most_exposed_symbol': most_exposed_symbol,
                'symbol_exposures': self.symbol_exposures.copy()
            },
            'limits': {
                'position_limits': self.position_limits.copy()
            }
        }
        
        # Log summary
        logger.info(f"Risk Report Summary:")
        logger.info(f"  Trading Allowed: {self.trading_allowed}")
        logger.info(f"  Current Equity: ${self.current_equity:.2f}")
        logger.info(f"  Daily P&L: ${daily_pnl:.2f} ({daily_pnl_pct:.2f}%)")
        logger.info(f"  Current Drawdown: {current_drawdown_pct:.2f}%")
        logger.info(f"  Portfolio Exposure: {portfolio_exposure:.2f}%")
        logger.info(f"  Trades Today: {self.trade_count_today} / {self.max_trades_per_day}")
        
        return report