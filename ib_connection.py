from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
import threading
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IBapi(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.data = {}  # Dictionary to store market data
        self.positions = {}  # Dictionary to store positions
        self.orders = {}  # Dictionary to store orders
        self.nextValidOrderId = None
        self.connected = False
        
    def nextValidId(self, orderId: int):
        """Callback when the next valid order ID is received"""
        super().nextValidId(orderId)
        self.nextValidOrderId = orderId
        logger.info(f"Next Valid Order ID: {orderId}")
        self.connected = True
        
    def error(self, reqId, errorCode, errorString):
        """Callback for error messages"""
        logger.error(f"Error {errorCode}: {errorString}")
        
    def tickPrice(self, reqId, tickType, price, attrib):
        """Callback for price updates"""
        super().tickPrice(reqId, tickType, price, attrib)
        if reqId not in self.data:
            self.data[reqId] = {}
        self.data[reqId][tickType] = price
        
    def position(self, account, contract, position, avgCost):
        """Callback for position updates"""
        super().position(account, contract, position, avgCost)
        key = contract.symbol
        self.positions[key] = {"position": position, "avgCost": avgCost}
        logger.info(f"Position: {key}, Qty: {position}, AvgCost: {avgCost}")
    
    def orderStatus(self, orderId, status, filled, remaining, avgFillPrice, 
                    permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice):
        """Callback for order status updates"""
        super().orderStatus(orderId, status, filled, remaining, avgFillPrice,
                            permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice)
        self.orders[orderId] = {
            "status": status,
            "filled": filled,
            "remaining": remaining,
            "avgFillPrice": avgFillPrice
        }
        logger.info(f"Order {orderId} status: {status}, Filled: {filled}")
        
    def execDetails(self, reqId, contract, execution):
        """Callback for execution details"""
        super().execDetails(reqId, contract, execution)
        logger.info(f"Execution: {execution.execId}, Symbol: {contract.symbol}, "
                   f"Side: {execution.side}, Shares: {execution.shares}, Price: {execution.price}")
        
    def accountSummary(self, reqId, account, tag, value, currency):
        """Callback for account summary"""
        super().accountSummary(reqId, account, tag, value, currency)
        logger.info(f"Account Summary: {account}, {tag}: {value} {currency}")
        
    def updateAccountValue(self, key, val, currency, accountName):
        """Callback for account value updates"""
        super().updateAccountValue(key, val, currency, accountName)
        logger.debug(f"Account Update: {accountName}, {key}: {val} {currency}")
        
    def updatePortfolio(self, contract, position, marketPrice, marketValue, 
                        averageCost, unrealizedPNL, realizedPNL, accountName):
        """Callback for portfolio updates"""
        super().updatePortfolio(contract, position, marketPrice, marketValue,
                               averageCost, unrealizedPNL, realizedPNL, accountName)
        logger.info(f"Portfolio Update: {contract.symbol}, Position: {position}, "
                   f"Market Value: {marketValue}, Unrealized P&L: {unrealizedPNL}")
        
    def historicalData(self, reqId, bar):
        """Callback for historical data bars"""
        if reqId not in self.data:
            self.data[reqId] = {"bars": []}
        
        self.data[reqId]["bars"].append({
            "time": bar.date,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume
        })
        
    def historicalDataEnd(self, reqId, start, end):
        """Callback for end of historical data"""
        logger.info(f"Historical data end: ReqId {reqId}, {start} to {end}")
        
    def contractDetails(self, reqId, contractDetails):
        """Callback for contract details"""
        if reqId not in self.data:
            self.data[reqId] = {"contracts": []}
            
        self.data[reqId]["contracts"].append(contractDetails)
        
    def contractDetailsEnd(self, reqId):
        """Callback for end of contract details"""
        logger.info(f"Contract details end: ReqId {reqId}")
        
    def tickSize(self, reqId, tickType, size):
        """Callback for size updates (e.g., volume)"""
        super().tickSize(reqId, tickType, size)
        if reqId not in self.data:
            self.data[reqId] = {}
        self.data[reqId][tickType] = size

class IBConnection:
    def __init__(self, host='127.0.0.1', port=7497, clientId=6969):
        self.host = host
        self.port = port
        self.clientId = clientId
        self.app = IBapi()
        self.connection_thread = None
        
    def connect(self):
        """Connect to TWS/IB Gateway"""
        # Connect to IB
        self.app.connect(self.host, self.port, self.clientId)
        
        # Start the socket in a thread
        self.connection_thread = threading.Thread(target=self.run_loop, daemon=True)
        self.connection_thread.start()
        
        # Wait for connection to complete
        timeout = 10
        start_time = time.time()
        while not self.app.connected and time.time() - start_time < timeout:
            time.sleep(0.1)
            
        if not self.app.connected:
            logger.error("Failed to connect to TWS")
            return False
            
        logger.info("Connected to TWS/IB Gateway")
        return True
    
    def disconnect(self):
        """Disconnect from TWS/IB Gateway"""
        if self.app.isConnected():
            self.app.disconnect()
            logger.info("Disconnected from TWS/IB Gateway")
            
    def run_loop(self):
        """Run the client message loop"""
        self.app.run()
        
    def create_contract(self, symbol, secType='STK', exchange='SMART', currency='USD', expiry='', strike=0.0, right='', multiplier='', localSymbol=''):
        """Create a contract object"""
        contract = Contract()
        contract.symbol = symbol
        contract.secType = secType
        contract.exchange = exchange
        contract.currency = currency
        
        # Optional parameters for futures, options, etc.
        if expiry:
            contract.lastTradeDateOrContractMonth = expiry
        if strike:
            contract.strike = strike
        if right:
            contract.right = right
        if multiplier:
            contract.multiplier = multiplier
        if localSymbol:
            contract.localSymbol = localSymbol
            
        return contract
    
    def create_order(self, action, quantity, order_type='MKT', limit_price=None, stop_price=None, tif='DAY'):
        """
            Create an order object with simplified parameters
    
        Args:
            action: 'BUY' or 'SELL'
            quantity: Number of shares to trade
            order_type: Order type ('MKT' for market, 'LMT' for limit, 'STP' for stop)
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders
            tif: Time in force ('DAY', 'GTC', 'IOC', etc.)
    
            Returns:
                Order object
        """
        order = Order()
    
        # Validate inputs
        action = action.upper()
        if action not in ['BUY', 'SELL']:
            raise ValueError(f"Invalid action: {action}. Must be 'BUY' or 'SELL'.")
    
        if quantity <= 0:
            raise ValueError(f"Invalid quantity: {quantity}. Must be positive.")
    
        # Set core order parameters
        order.action = action
        order.eTradeOnly = False
        order.firmQuoteOnly = False
        order.optOutSmartRouting = False
        order.totalQuantity = quantity
        order.orderType = order_type
        order.tif = tif

        # Set additional price parameters based on order type
        if order_type == 'LMT':
            if limit_price is None:
                raise ValueError("Limit price must be specified for limit orders")
            order.lmtPrice = limit_price
    
        elif order_type == 'STP':
            if stop_price is None:
                raise ValueError("Stop price must be specified for stop orders")
            order.auxPrice = stop_price
    
    # Clear any potentially problematic attributes
        order.algoStrategy = ""
        order.whatIf = False
    
        return order
    def create_bracket_order(self, action, quantity, limit_price=None, stop_loss_price=None, take_profit_price=None):
        """
        Create a bracket order (entry, take-profit, stop-loss)

        Args:
            action: 'BUY' or 'SELL'
            quantity: Number of shares
            limit_price: Entry price for limit order; None for market order
            stop_loss_price: Price for stop-loss order
            take_profit_price: Price for take-profit order

        Returns:
            List of Order objects (parent, take-profit, stop-loss)
        """
        if not self.app.isConnected():
            logger.error("Not connected to TWS")
            return None

        try:
            action = action.upper()
            if action not in ['BUY', 'SELL']:
                raise ValueError(f"Invalid action: {action}")
            if quantity <= 0:
                raise ValueError(f"Invalid quantity: {quantity}")
            if stop_loss_price is None or take_profit_price is None:
                raise ValueError("Stop-loss and take-profit prices must be specified")

            parent_order_id = self.app.nextValidOrderId
            self.app.nextValidOrderId += 3  # Reserve IDs for parent, TP, SL

            # Parent order (entry)
            parent = Order()
            parent.orderId = parent_order_id
            parent.action = action
            parent.totalQuantity = quantity
            parent.orderType = 'MKT' if limit_price is None else 'LMT'
            if limit_price is not None:
                parent.lmtPrice = limit_price
            parent.transmit = False

            # Take-profit order
            take_profit = Order()
            take_profit.orderId = parent_order_id + 1
            take_profit.action = 'SELL' if action == 'BUY' else 'BUY'
            take_profit.totalQuantity = quantity
            take_profit.orderType = 'LMT'
            take_profit.lmtPrice = take_profit_price
            take_profit.parentId = parent_order_id
            take_profit.transmit = False

            # Stop-loss order
            stop_loss = Order()
            stop_loss.orderId = parent_order_id + 2
            stop_loss.action = 'SELL' if action == 'BUY' else 'BUY'
            stop_loss.totalQuantity = quantity
            stop_loss.orderType = 'STP'
            stop_loss.auxPrice = stop_loss_price
            stop_loss.parentId = parent_order_id
            stop_loss.transmit = True  # Transmit entire bracket

            logger.info(
                f"Created bracket order: {action} {quantity} "
                f"{'MKT' if limit_price is None else f'LMT at {limit_price}'}, "
                f"TP: {take_profit_price}, SL: {stop_loss_price}"
            )
            return [parent, take_profit, stop_loss]

        except Exception as e:
            logger.error(f"Error creating bracket order: {str(e)}")
            return None
    def request_market_data(self, contract, tick_list="233"):
        """Request market data for a specific contract"""
        if not self.app.isConnected():
            logger.error("Not connected to TWS")
            return None
            
        req_id = self.app.nextValidOrderId
        self.app.nextValidOrderId += 1
        
        # Request market data
        self.app.reqMktData(req_id, contract, tick_list, False, False, [])
        logger.info(f"Requested market data for {contract.symbol}, req_id: {req_id}")
        
        return req_id
    
    def place_order(self, contract, order):
        """Place an order"""
        if not self.app.isConnected():
            logger.error("Not connected to TWS")
            return None
            
        order_id = self.app.nextValidOrderId
        self.app.nextValidOrderId += 1
        
        # Place the order
        self.app.placeOrder(order_id, contract, order)
        logger.info(f"Placed order: {order_id}, {order.action} {order.totalQuantity} {contract.symbol}")
        
        return order_id
    
    def request_positions(self):
        """Request current positions"""
        if self.app.isConnected():
            self.app.reqPositions()
            logger.info("Requested positions")
        else:
            logger.error("Not connected to TWS")
            
    def cancel_order(self, order_id):
        """Cancel an order"""
        if self.app.isConnected():
            self.app.cancelOrder(order_id)
            logger.info(f"Cancelled order: {order_id}")
        else:
            logger.error("Not connected to TWS")
            
    def get_data(self, req_id):
        """Get market data for a specific request ID"""
        return self.app.data.get(req_id, {})
    
    def get_positions(self):
        """Get current positions"""
        return self.app.positions
    
    def get_order_status(self, order_id):
        """Get status of a specific order"""
        return self.app.orders.get(order_id, {})
    
    def request_account_updates(self, account_code=""):
        """Request account updates"""
        if self.app.isConnected():
            self.app.reqAccountUpdates(True, account_code)
            logger.info(f"Requested account updates for {account_code}")
        else:
            logger.error("Not connected to TWS")
            
    def request_historical_data(self, contract, duration="1 D", bar_size="1 min", what_to_show="TRADES", use_rth=True):
        """Request historical data for a contract"""
        if not self.app.isConnected():
            logger.error("Not connected to TWS")
            return None
            
        req_id = self.app.nextValidOrderId
        self.app.nextValidOrderId += 1
        
        # Current time as end time
        end_time = ""  # Empty string means "now"
        
        # Request historical data
        self.app.reqHistoricalData(
            req_id,
            contract,
            end_time,
            duration,
            bar_size,
            what_to_show,
            use_rth,
            1,  # formatDate: 1 for yyyyMMdd format
            False,  # keep up to date
            []  # chart options
        )
        
        logger.info(f"Requested historical data for {contract.symbol}, req_id: {req_id}")
        
        return req_id
    
    def request_contract_details(self, contract):
        """Request contract details"""
        if not self.app.isConnected():
            logger.error("Not connected to TWS")
            return None
            
        req_id = self.app.nextValidOrderId
        self.app.nextValidOrderId += 1
        
        # Request contract details
        self.app.reqContractDetails(req_id, contract)
        logger.info(f"Requested contract details for {contract.symbol}, req_id: {req_id}")
        
        return req_id
    
    def place_bracket_order(self, contract, action, quantity, limit_price, take_profit_price, stop_loss_price):
        """Place a bracket order (entry, take profit, and stop loss)"""
        if not self.app.isConnected():
            logger.error("Not connected to TWS")
            return None
            
        # Get the next valid order ID
        parent_order_id = self.app.nextValidOrderId
        self.app.nextValidOrderId += 3  # We need 3 IDs: parent, take profit, and stop loss
        
        # Create parent order
        parent = Order()
        parent.orderId = parent_order_id
        parent.action = action
        parent.totalQuantity = quantity
        parent.orderType = "LMT"
        parent.lmtPrice = limit_price
        parent.transmit = False  # Don't transmit until we've created child orders
        
        # Create take profit order
        take_profit = Order()
        take_profit.orderId = parent_order_id + 1
        take_profit.action = "SELL" if action == "BUY" else "BUY"
        take_profit.totalQuantity = quantity
        take_profit.orderType = "LMT"
        take_profit.lmtPrice = take_profit_price
        take_profit.parentId = parent_order_id
        take_profit.transmit = False
        
        # Create stop loss order
        stop_loss = Order()
        stop_loss.orderId = parent_order_id + 2
        stop_loss.action = "SELL" if action == "BUY" else "BUY"
        stop_loss.totalQuantity = quantity
        stop_loss.orderType = "STP"
        stop_loss.auxPrice = stop_loss_price
        stop_loss.parentId = parent_order_id
        stop_loss.transmit = True  # Transmit the entire bracket order
        
        # Place the orders
        self.app.placeOrder(parent.orderId, contract, parent)
        self.app.placeOrder(take_profit.orderId, contract, take_profit)
        self.app.placeOrder(stop_loss.orderId, contract, stop_loss)
        
        logger.info(f"Placed bracket order: {action} {quantity} {contract.symbol} "
                   f"at {limit_price}, TP: {take_profit_price}, SL: {stop_loss_price}")
        
        return parent_order_id
    
    def place_trailing_stop_order(self, contract, action, quantity, trailing_percent=None, trailing_amount=None):
        """Place a trailing stop order"""
        if not self.app.isConnected():
            logger.error("Not connected to TWS")
            return None
            
        order_id = self.app.nextValidOrderId
        self.app.nextValidOrderId += 1
        
        # Create trailing stop order
        order = Order()
        order.action = action
        order.totalQuantity = quantity
        order.orderType = "TRAIL"
        
        if trailing_percent is not None:
            order.trailingPercent = trailing_percent
        elif trailing_amount is not None:
            order.auxPrice = trailing_amount  # Trail amount
        else:
            logger.error("Either trailing_percent or trailing_amount must be specified")
            return None
        
        # Place the order
        self.app.placeOrder(order_id, contract, order)
        
        logger.info(f"Placed trailing stop order: {action} {quantity} {contract.symbol}, "
                   f"Trailing {'percent' if trailing_percent else 'amount'}: "
                   f"{trailing_percent if trailing_percent else trailing_amount}")
        
        return order_id
    
    def cancel_market_data(self, req_id):
        """Cancel market data subscription"""
        if self.app.isConnected():
            self.app.cancelMktData(req_id)
            logger.info(f"Cancelled market data for req_id: {req_id}")
        else:
            logger.error("Not connected to TWS")


# Example usage
if __name__ == "__main__":
    # Create a connection
    ib = IBConnection()
    
    # Connect to TWS/IB Gateway
    if ib.connect():
        try:
            # Create a contract for Apple stock
            contract = ib.create_contract("AAPL")
            
            # Request market data
            req_id = ib.request_market_data(contract)
            
            # Wait for some data to arrive
            time.sleep(5)
            
            # Print the received data
            data = ib.get_data(req_id)
            print(f"Market data for AAPL: {data}")
            
            # Request historical data
            hist_req_id = ib.request_historical_data(contract)
            
            # Wait for historical data
            time.sleep(5)
            
            # Print historical data
            hist_data = ib.get_data(hist_req_id)
            if "bars" in hist_data:
                for bar in hist_data["bars"][:5]:  # Print first 5 bars
                    print(f"Bar: {bar}")
            
            # Request positions
            ib.request_positions()
            
            # Wait for positions data
            time.sleep(2)
            
            # Print positions
            positions = ib.get_positions()
            print(f"Current positions: {positions}")
            
        finally:
            # Disconnect
            ib.disconnect()
    else:
        print("Failed to connect to Interactive Brokers")