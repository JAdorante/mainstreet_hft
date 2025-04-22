import logging
import time
import random
from datetime import datetime, timedelta
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OrderExecutor:
    """Handles the execution of orders with various algorithms"""
    def __init__(self, ib_connection):
        self.ib_connection = ib_connection
        self.open_orders = {}  # Track open orders by order_id
        self.execution_algos = {
            'market': self.execute_market_order,
            'limit': self.execute_limit_order,
            'twap': self.execute_twap,
            'vwap': self.execute_vwap,
            'iceberg': self.execute_iceberg,
            'sniper': self.execute_sniper
        }
        
    def execute_order(self, contract, action, quantity, algo='market', algo_params=None):
        """Execute an order using the specified algorithm
        
        Args:
            contract: Contract object for the security
            action: 'BUY' or 'SELL'
            quantity: Number of shares to trade
            algo: Execution algorithm ('market', 'limit', 'twap', 'vwap', 'iceberg', 'sniper')
            algo_params: Dictionary of algorithm-specific parameters
            
        Returns:
            order_id: ID of the executed order
        """
        if algo not in self.execution_algos:
            logger.error(f"Unknown execution algorithm: {algo}")
            return None
            
        # Call the appropriate execution algorithm
        return self.execution_algos[algo](contract, action, quantity, algo_params)
        
    def execute_market_order(self, contract, action, quantity, params=None):
        """Execute a simple market order
        
        Args:
            contract: Contract object for the security
            action: 'BUY' or 'SELL'
            quantity: Number of shares to trade
            params: Not used for market orders
            
        Returns:
            order_id: ID of the market order
        """
        # Create a market order
        order = self.ib_connection.create_order(action, quantity)
        
        # Place the order
        order_id = self.ib_connection.place_order(contract, order)
        
        # Track the order
        self.open_orders[order_id] = {
            'contract': contract,
            'action': action,
            'quantity': quantity,
            'algo': 'market',
            'status': 'submitted',
            'timestamp': datetime.now()
        }
        
        logger.info(f"Executed market order: {action} {quantity} {contract.symbol}, order_id: {order_id}")
        
        return order_id
        
    def execute_limit_order(self, contract, action, quantity, params=None):
        """Execute a limit order
        
        Args:
            contract: Contract object for the security
            action: 'BUY' or 'SELL'
            quantity: Number of shares to trade
            params: Dictionary with parameters:
                - 'limit_price': Limit price for the order
                - 'tif': Time in force ('DAY', 'GTC', etc.)
            
        Returns:
            order_id: ID of the limit order
        """
        if params is None or 'limit_price' not in params:
            logger.error("Limit price is required for limit orders")
            return None
            
        limit_price = params['limit_price']
        tif = params.get('tif', 'DAY')
        
        # Create a limit order
        order = self.ib_connection.create_order(action, quantity, 'LMT', limit_price, None, tif)
        
        # Place the order
        order_id = self.ib_connection.place_order(contract, order)
        
        # Track the order
        self.open_orders[order_id] = {
            'contract': contract,
            'action': action,
            'quantity': quantity,
            'algo': 'limit',
            'params': params,
            'status': 'submitted',
            'timestamp': datetime.now()
        }
        
        logger.info(f"Executed limit order: {action} {quantity} {contract.symbol} @ {limit_price}, order_id: {order_id}")
        
        return order_id
        
    def execute_twap(self, contract, action, quantity, params=None):
        """Execute a Time-Weighted Average Price (TWAP) order
        
        Args:
            contract: Contract object for the security
            action: 'BUY' or 'SELL'
            quantity: Number of shares to trade
            params: Dictionary with parameters:
                - 'duration': Duration in minutes for the TWAP execution
                - 'num_slices': Number of slices to divide the order into
            
        Returns:
            List of order IDs for the TWAP child orders
        """
        if params is None:
            params = {}
            
        duration = params.get('duration', 30)  # Default 30 minutes
        num_slices = params.get('num_slices', 10)  # Default 10 slices
        
        # Calculate slice size (round to nearest integer)
        slice_size = max(1, int(quantity / num_slices))
        
        # Adjust last slice size to match total quantity
        slices = [slice_size] * (num_slices - 1)
        slices.append(quantity - sum(slices))
        
        # Calculate time interval between slices
        interval_seconds = (duration * 60) / num_slices
        
        # Track all child order IDs
        order_ids = []
        
        # Create parent order ID to track all child orders
        parent_id = int(time.time())
        
        # Execute each slice in a separate thread
        import threading
        
        def execute_slice(i):
            # Wait until the scheduled time for this slice
            wait_time = i * interval_seconds
            time.sleep(wait_time)
            
            # Execute the slice as a market order
            slice_quantity = slices[i]
            if slice_quantity <= 0:
                return
                
            order = self.ib_connection.create_order(action, slice_quantity)
            order_id = self.ib_connection.place_order(contract, order)
            
            # Track the order
            self.open_orders[order_id] = {
                'contract': contract,
                'action': action,
                'quantity': slice_quantity,
                'algo': 'twap_slice',
                'parent_id': parent_id,
                'status': 'submitted',
                'timestamp': datetime.now()
            }
            
            order_ids.append(order_id)
            
            logger.info(f"TWAP slice {i+1}/{num_slices}: {action} {slice_quantity} {contract.symbol}, order_id: {order_id}")
            
        # Start a thread for each slice
        threads = []
        for i in range(num_slices):
            thread = threading.Thread(target=execute_slice, args=(i,))
            thread.daemon = True
            thread.start()
            threads.append(thread)
            
        # Track the TWAP parent order
        self.open_orders[parent_id] = {
            'contract': contract,
            'action': action,
            'quantity': quantity,
            'algo': 'twap',
            'params': params,
            'child_orders': [],
            'status': 'submitted',
            'timestamp': datetime.now()
        }
        
        logger.info(f"Executed TWAP order: {action} {quantity} {contract.symbol} over {duration} minutes in {num_slices} slices")
        
        return parent_id
        
    def execute_vwap(self, contract, action, quantity, params=None):
        """Execute a Volume-Weighted Average Price (VWAP) order
        
        This is a simplified VWAP implementation that uses historical volume profiles.
        
        Args:
            contract: Contract object for the security
            action: 'BUY' or 'SELL'
            quantity: Number of shares to trade
            params: Dictionary with parameters:
                - 'duration': Duration in minutes for the VWAP execution
                - 'num_slices': Number of slices to divide the order into
                - 'volume_profile': Optional list of volume percentages for each slice
            
        Returns:
            Parent order ID that tracks all VWAP child orders
        """
        if params is None:
            params = {}
            
        duration = params.get('duration', 60)  # Default 60 minutes
        num_slices = params.get('num_slices', 12)  # Default 12 slices (5-minute intervals)
        
        # Use provided volume profile or generate a typical one
        volume_profile = params.get('volume_profile', None)
        if volume_profile is None:
            # Typical U-shaped volume profile (higher at open and close)
            volume_profile = [0.12, 0.10, 0.08, 0.07, 0.06, 0.06, 0.06, 0.06, 0.07, 0.08, 0.10, 0.14]
            
            # If number of slices doesn't match profile length, interpolate
            if len(volume_profile) != num_slices:
                from scipy.interpolate import interp1d
                x = np.linspace(0, 1, len(volume_profile))
                y = volume_profile
                f = interp1d(x, y, kind='cubic')
                x_new = np.linspace(0, 1, num_slices)
                volume_profile = f(x_new)
                
                # Normalize to sum to 1
                volume_profile = volume_profile / sum(volume_profile)
        
        # Calculate slice sizes based on volume profile
        slices = [max(1, int(quantity * pct)) for pct in volume_profile]
        
        # Adjust last slice size to match total quantity
        remaining = quantity - sum(slices[:-1])
        slices[-1] = remaining
        
        # Calculate time interval between slices
        interval_seconds = (duration * 60) / num_slices
        
        # Create parent order ID to track all child orders
        parent_id = int(time.time())
        
        # Execute each slice in a separate thread
        import threading
        
        def execute_slice(i):
            # Wait until the scheduled time for this slice
            wait_time = i * interval_seconds
            time.sleep(wait_time)
            
            # Execute the slice as a market order
            slice_quantity = slices[i]
            if slice_quantity <= 0:
                return
                
            order = self.ib_connection.create_order(action, slice_quantity)
            order_id = self.ib_connection.place_order(contract, order)
            
            # Track the order
            self.open_orders[order_id] = {
                'contract': contract,
                'action': action,
                'quantity': slice_quantity,
                'algo': 'vwap_slice',
                'parent_id': parent_id,
                'status': 'submitted',
                'timestamp': datetime.now()
            }
            
            logger.info(f"VWAP slice {i+1}/{num_slices}: {action} {slice_quantity} {contract.symbol}, order_id: {order_id}")
            
        # Start a thread for each slice
        threads = []
        for i in range(num_slices):
            thread = threading.Thread(target=execute_slice, args=(i,))
            thread.daemon = True
            thread.start()
            threads.append(thread)
            
        # Track the VWAP parent order
        self.open_orders[parent_id] = {
            'contract': contract,
            'action': action,
            'quantity': quantity,
            'algo': 'vwap',
            'params': params,
            'child_orders': [],
            'status': 'submitted',
            'timestamp': datetime.now()
        }
        
        logger.info(f"Executed VWAP order: {action} {quantity} {contract.symbol} over {duration} minutes in {num_slices} slices")
        
        return parent_id
        
    def execute_iceberg(self, contract, action, quantity, params=None):
        """Execute an Iceberg order (display only a portion of the total order)
        
        Args:
            contract: Contract object for the security
            action: 'BUY' or 'SELL'
            quantity: Number of shares to trade
            params: Dictionary with parameters:
                - 'display_size': Visible quantity for each slice
                - 'limit_price': Optional limit price
            
        Returns:
            Parent order ID that tracks all Iceberg child orders
        """
        if params is None:
            params = {}
            
        display_size = params.get('display_size', min(100, quantity))  # Default 100 shares or less
        limit_price = params.get('limit_price', None)
        
        # Calculate number of slices
        num_slices = quantity // display_size
        if quantity % display_size > 0:
            num_slices += 1
            
        # Create parent order ID to track all child orders
        parent_id = int(time.time())
        
        # Execute first slice immediately
        order_type = 'LMT' if limit_price is not None else 'MKT'
        order = self.ib_connection.create_order(action, display_size, order_type, limit_price)
        first_order_id = self.ib_connection.place_order(contract, order)
        
        # Track the order
        self.open_orders[first_order_id] = {
            'contract': contract,
            'action': action,
            'quantity': display_size,
            'algo': 'iceberg_slice',
            'parent_id': parent_id,
            'status': 'submitted',
            'timestamp': datetime.now()
        }
        
        logger.info(f"Iceberg slice 1/{num_slices}: {action} {display_size} {contract.symbol}, order_id: {first_order_id}")
        
        # Track the Iceberg parent order
        self.open_orders[parent_id] = {
            'contract': contract,
            'action': action,
            'quantity': quantity,
            'algo': 'iceberg',
            'params': params,
            'remaining': quantity - display_size,
            'child_orders': [first_order_id],
            'status': 'submitted',
            'timestamp': datetime.now()
        }
        
        # Create a thread to monitor and replace filled slices
        import threading
        
        def monitor_and_replace():
            remaining = quantity - display_size
            slice_count = 1
            
            while remaining > 0:
                # Check if the previous order was filled
                for order_id in list(self.open_orders[parent_id]['child_orders']):
                    order_status = self.ib_connection.get_order_status(order_id)
                    
                    if order_status.get('status') == 'Filled':
                        # Submit the next slice
                        slice_count += 1
                        next_quantity = min(display_size, remaining)
                        
                        order = self.ib_connection.create_order(action, next_quantity, order_type, limit_price)
                        next_order_id = self.ib_connection.place_order(contract, order)
                        
                        # Track the order
                        self.open_orders[next_order_id] = {
                            'contract': contract,
                            'action': action,
                            'quantity': next_quantity,
                            'algo': 'iceberg_slice',
                            'parent_id': parent_id,
                            'status': 'submitted',
                            'timestamp': datetime.now()
                        }
                        
                        # Update parent order
                        self.open_orders[parent_id]['child_orders'].append(next_order_id)
                        self.open_orders[parent_id]['remaining'] = remaining - next_quantity
                        
                        remaining -= next_quantity
                        
                        logger.info(f"Iceberg slice {slice_count}/{num_slices}: {action} {next_quantity} {contract.symbol}, order_id: {next_order_id}")
                
                # Sleep before checking again
                time.sleep (0.01)
        
        # Start the monitoring thread
        monitor_thread = threading.Thread(target=monitor_and_replace)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        logger.info(f"Executed Iceberg order: {action} {quantity} {contract.symbol} with display size {display_size}")
        
        return parent_id
        
    def execute_sniper(self, contract, action, quantity, params=None):
        """Execute a Sniper order (wait for price to hit target, then execute quickly)
        
        Args:
            contract: Contract object for the security
            action: 'BUY' or 'SELL'
            quantity: Number of shares to trade
            params: Dictionary with parameters:
                - 'target_price': Target price to execute at
                - 'time_limit': Optional time limit in seconds (default: 300)
                - 'price_type': 'exact', 'better', or 'touch' (default: 'better')
            
        Returns:
            order_id: ID of the sniper order
        """
        if params is None or 'target_price' not in params:
            logger.error("Target price is required for sniper orders")
            return None
            
        target_price = params['target_price']
        time_limit = params.get('time_limit', 300)  # Default 5 minutes
        price_type = params.get('price_type', 'better')  # Default 'better'
        
        # Create an order ID
        order_id = int(time.time())
        
        # Track the order
        self.open_orders[order_id] = {
            'contract': contract,
            'action': action,
            'quantity': quantity,
            'algo': 'sniper',
            'params': params,
            'status': 'watching',
            'timestamp': datetime.now()
        }
        
        # Create a thread to monitor prices and execute when target is hit
        import threading
        
        def monitor_and_execute():
            start_time = time.time()
            executed = False
            
            while time.time() - start_time < time_limit and not executed:
                # Request market data
                market_data_id = self.ib_connection.request_market_data(contract)
                
                # Wait for data to arrive
                time.sleep(0.01)
                
                # Get the latest data
                market_data = self.ib_connection.get_data(market_data_id)
                
                # Cancel the market data subscription
                self.ib_connection.cancel_market_data(market_data_id)
                
                # Check if we have price data
                if 4 in market_data:  # Tick type 4 is last price
                    current_price = market_data[4]
                    
                    # Check if price condition is met
                    condition_met = False
                    
                    if price_type == 'exact':
                        # Execute only at exact price
                        condition_met = current_price == target_price
                    elif price_type == 'better':
                        # Execute at target price or better
                        if action == 'BUY':
                            condition_met = current_price <= target_price
                        else:  # SELL
                            condition_met = current_price >= target_price
                    elif price_type == 'touch':
                        # Execute when price touches or crosses target
                        if action == 'BUY':
                            condition_met = current_price <= target_price
                        else:  # SELL
                            condition_met = current_price >= target_price
                    
                    if condition_met:
                        # Execute as market order for immediate fill
                        order = self.ib_connection.create_order(action, quantity)
                        child_order_id = self.ib_connection.place_order(contract, order)
                        
                        # Update order status
                        self.open_orders[order_id]['status'] = 'executed'
                        self.open_orders[order_id]['execution_price'] = current_price
                        self.open_orders[order_id]['child_order_id'] = child_order_id
                        
                        logger.info(f"Sniper order executed: {action} {quantity} {contract.symbol} @ {current_price}")
                        
                        executed = True
                        break
                
                # Sleep before checking again
                time.sleep (0.01)
                
            if not executed:
                # Time limit reached without execution
                self.open_orders[order_id]['status'] = 'expired'
                logger.info(f"Sniper order expired: {action} {quantity} {contract.symbol}, target price not reached")
                
        # Start the monitoring thread
        monitor_thread = threading.Thread(target=monitor_and_execute)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        logger.info(f"Submitted Sniper order: {action} {quantity} {contract.symbol} @ target {target_price}")
        
        return order_id
        
    def update_order_status(self):
        """Update status of all open orders"""
        for order_id in list(self.open_orders.keys()):
            order_info = self.open_orders[order_id]
            
            # Skip parent orders of algos
            if order_info['algo'] in ['twap', 'vwap', 'iceberg']:
                continue
                
            # Get current status from IB
            status = self.ib_connection.get_order_status(order_id)
            
            if status:
                order_info['status'] = status.get('status', order_info['status'])
                
                # If order is filled, calculate metrics
                if status.get('status') == 'Filled' and 'filled_time' not in order_info:
                    order_info['filled_time'] = datetime.now()
                    order_info['fill_price'] = status.get('avgFillPrice', 0)
                    
                    # Calculate execution time
                    if 'timestamp' in order_info:
                        execution_time = (order_info['filled_time'] - order_info['timestamp']).total_seconds()
                        order_info['execution_time'] = execution_time
                        
                    logger.info(f"Order {order_id} filled: {order_info['action']} {order_info['quantity']} "
                               f"{order_info['contract'].symbol} @ {order_info['fill_price']}")
                    
    def cancel_order(self, order_id):
        """Cancel an open order
        
        Args:
            order_id: ID of the order to cancel
            
        Returns:
            bool: True if cancel was successful, False otherwise
        """
        if order_id not in self.open_orders:
            logger.warning(f"Order {order_id} not found")
            return False
            
        order_info = self.open_orders[order_id]
        
        # Check if this is a parent algo order
        if order_info['algo'] in ['twap', 'vwap', 'iceberg', 'sniper']:
            # For algo orders, cancel all child orders
            if 'child_orders' in order_info:
                for child_id in order_info['child_orders']:
                    self.ib_connection.cancel_order(child_id)
                    
            # Update status
            order_info['status'] = 'cancelled'
            logger.info(f"Cancelled algo order {order_id}: {order_info['action']} {order_info['quantity']} {order_info['contract'].symbol}")
            
            return True
            
        else:
            # For regular orders, just cancel
            self.ib_connection.cancel_order(order_id)
            
            # Update status
            order_info['status'] = 'pending_cancel'
            logger.info(f"Cancelling order {order_id}: {order_info['action']} {order_info['quantity']} {order_info['contract'].symbol}")
            
            return True
            
    def cancel_all_orders(self):
        """Cancel all open orders"""
        cancelled_count = 0
        
        for order_id in list(self.open_orders.keys()):
            order_info = self.open_orders[order_id]
            
            # Skip orders that are already filled or cancelled
            if order_info['status'] in ['Filled', 'Cancelled', 'cancelled']:
                continue
                
            # Cancel the order
            if self.cancel_order(order_id):
                cancelled_count += 1
                
        logger.info(f"Cancelled {cancelled_count} orders")
        
        return cancelled_count
        
    def get_order_info(self, order_id):
        """Get information about an order
        
        Args:
            order_id: ID of the order
            
        Returns:
            dict: Order information
        """
        if order_id not in self.open_orders:
            return None
            
        return self.open_orders[order_id].copy()
        
    def get_all_orders(self, status=None):
        """Get all orders, optionally filtered by status
        
        Args:
            status: Optional status filter ('submitted', 'Filled', 'Cancelled', etc.)
            
        Returns:
            dict: Dictionary of order information
        """
        if status is None:
            return self.open_orders.copy()
            
        return {order_id: info for order_id, info in self.open_orders.items() 
                if info['status'] == status}


class SmartOrderRouter:
    """Routes orders to different venues based on execution quality"""
    def __init__(self, ib_connection):
        self.ib_connection = ib_connection
        self.order_executor = OrderExecutor(ib_connection)
        self.venue_stats = {}  # Statistics for each venue
        
    def route_order(self, contract, action, quantity, algo='market', algo_params=None):
        """Route an order to the best venue based on historical execution quality
        
        Args:
            contract: Contract object for the security
            action: 'BUY' or 'SELL'
            quantity: Number of shares to trade
            algo: Execution algorithm
            algo_params: Dictionary of algorithm-specific parameters
            
        Returns:
            order_id: ID of the executed order
        """
        # Get best venue for this security
        venue = self.select_best_venue(contract.symbol)
        
        # Set the exchange in the contract
        original_exchange = contract.exchange
        contract.exchange = venue
        
        # Execute the order
        order_id = self.order_executor.execute_order(contract, action, quantity, algo, algo_params)
        
        # Restore original exchange setting
        contract.exchange = original_exchange
        
        logger.info(f"Order routed to venue {venue}: {action} {quantity} {contract.symbol}")
        
        return order_id
        
    def select_best_venue(self, symbol):
        """Select the best venue for a security based on historical execution quality
        
        Args:
            symbol: Symbol of the security
            
        Returns:
            string: Exchange code for the best venue
        """
        # Default to SMART routing if no statistics available
        if symbol not in self.venue_stats or not self.venue_stats[symbol]:
            return "SMART"
            
        # Calculate a score for each venue based on fill rate, speed, and price improvement
        venues = self.venue_stats[symbol].keys()
        venue_scores = {}
        
        for venue in venues:
            stats = self.venue_stats[symbol][venue]
            
            # Calculate score components
            fill_rate_score = stats['fill_rate'] * 0.4  # 40% weight
            speed_score = (1.0 / (1.0 + stats['avg_execution_time'])) * 0.3  # 30% weight
            price_score = stats['price_improvement'] * 0.3  # 30% weight
            
            # Combined score
            venue_scores[venue] = fill_rate_score + speed_score + price_score
            
        # Select venue with highest score
        best_venue = max(venue_scores.items(), key=lambda x: x[1])[0]
        
        return best_venue
        
    def update_venue_stats(self, symbol, venue, execution_info):
        """Update execution statistics for a venue
        
        Args:
            symbol: Symbol of the security
            venue: Exchange code for the venue
            execution_info: Dictionary with execution information:
                - 'filled': Whether the order was filled (bool)
                - 'execution_time': Time to execution in seconds
                - 'price_improvement': Price improvement from NBBO in percentage
        """
        if symbol not in self.venue_stats:
            self.venue_stats[symbol] = {}
            
        if venue not in self.venue_stats[symbol]:
            self.venue_stats[symbol][venue] = {
                'fill_count': 0,
                'total_orders': 0,
                'fill_rate': 0.0,
                'total_execution_time': 0.0,
                'avg_execution_time': 0.0,
                'price_improvement': 0.0
            }
            
        stats = self.venue_stats[symbol][venue]
        
        # Update fill statistics
        stats['total_orders'] += 1
        if execution_info.get('filled', False):
            stats['fill_count'] += 1
            
        stats['fill_rate'] = stats['fill_count'] / stats['total_orders']
        
        # Update execution time statistics
        if 'execution_time' in execution_info:
            stats['total_execution_time'] += execution_info['execution_time']
            stats['avg_execution_time'] = stats['total_execution_time'] / stats['fill_count']
            
        # Update price improvement statistics (exponential moving average)
        if 'price_improvement' in execution_info:
            alpha = 0.2  # Smoothing factor for EMA
            stats['price_improvement'] = (alpha * execution_info['price_improvement'] + 
                                         (1 - alpha) * stats['price_improvement'])
                                         
    def get_venue_stats(self, symbol=None):
        """Get execution statistics for venues
        
        Args:
            symbol: Optional symbol filter
            
        Returns:
            dict: Dictionary of venue statistics
        """
        if symbol is not None:
            return self.venue_stats.get(symbol, {})
        return self.venue_stats
        
    def reset_venue_stats(self, symbol=None):
        """Reset execution statistics
        
        Args:
            symbol: Optional symbol to reset (if None, reset all)
        """
        if symbol is not None:
            if symbol in self.venue_stats:
                del self.venue_stats[symbol]
        else:
            self.venue_stats = {}
            
    def simulate_venue_stats(self, symbol, num_venues=3, num_executions=100):
        """Simulate venue statistics for testing
        
        Args:
            symbol: Symbol to simulate statistics for
            num_venues: Number of venues to simulate
            num_executions: Number of executions to simulate per venue
        """
        venues = ["SMART", "ARCA", "ISLAND", "NYSE", "NSDQ"][:num_venues]
        
        if symbol not in self.venue_stats:
            self.venue_stats[symbol] = {}
            
        for venue in venues:
            # Simulate different characteristics for each venue
            if venue == "SMART":
                base_fill_rate = 0.98
                base_execution_time = 0.3
                base_price_improvement = 0.02
            elif venue == "ARCA":
                base_fill_rate = 0.95
                base_execution_time = 0.5
                base_price_improvement = 0.03
            elif venue == "ISLAND":
                base_fill_rate = 0.97
                base_execution_time = 0.4
                base_price_improvement = 0.01
            elif venue == "NYSE":
                base_fill_rate = 0.99
                base_execution_time = 0.8
                base_price_improvement = 0.02
            else:  # NSDQ
                base_fill_rate = 0.96
                base_execution_time = 0.6
                base_price_improvement = 0.025
                
            # Initialize venue stats
            self.venue_stats[symbol][venue] = {
                'fill_count': 0,
                'total_orders': 0,
                'fill_rate': 0.0,
                'total_execution_time': 0.0,
                'avg_execution_time': 0.0,
                'price_improvement': 0.0
            }
            
            # Simulate executions
            for _ in range(num_executions):
                # Randomize execution characteristics
                filled = random.random() < base_fill_rate
                execution_time = max(0.1, random.normalvariate(base_execution_time, 0.2))
                price_improvement = max(0.0, random.normalvariate(base_price_improvement, 0.01))
                
                # Update stats
                self.update_venue_stats(symbol, venue, {
                    'filled': filled,
                    'execution_time': execution_time,
                    'price_improvement': price_improvement
                })
                
        logger.info(f"Simulated venue statistics for {symbol} across {num_venues} venues")


# Example usage
if __name__ == "__main__":
    from ib_connection import IBConnection
    
    # Create a connection
    ib = IBConnection()
    
    # Connect to TWS/IB Gateway
    if ib.connect():
        try:
            # Create an order executor
            executor = OrderExecutor(ib)
            
            # Create a contract for Apple stock
            contract = ib.create_contract("AAPL")
            
            # Execute a simple market order
            order_id = executor.execute_order(contract, "BUY", 100, "market")
            print(f"Executed market order with ID: {order_id}")
            
            # Wait for the order to be processed
            time.sleep(0.01)
            
            # Check order status
            executor.update_order_status()
            order_info = executor.get_order_info(order_id)
            print(f"Order status: {order_info['status']}")
            
            # Create a smart order router
            router = SmartOrderRouter(ib)
            
            # Simulate venue statistics
            router.simulate_venue_stats("AAPL")
            
            # Print venue statistics
            stats = router.get_venue_stats("AAPL")
            for venue, venue_stats in stats.items():
                print(f"Venue: {venue}")
                print(f"  Fill Rate: {venue_stats['fill_rate']:.2f}")
                print(f"  Avg Execution Time: {venue_stats['avg_execution_time']:.2f} sec")
                print(f"  Price Improvement: {venue_stats['price_improvement']:.4f}")
                
            # Route an order to the best venue
            routed_order_id = router.route_order(contract, "BUY", 100)
            print(f"Routed order with ID: {routed_order_id}")
            
        finally:
            # Disconnect
            ib.disconnect()
    else:
        print("Failed to connect to Interactive Brokers")