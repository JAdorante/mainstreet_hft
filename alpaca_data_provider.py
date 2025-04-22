# alpaca_data_provider.py
import logging
import alpaca_trade_api as tradeapi
from alpaca_trade_api.stream import Stream
from datetime import datetime
import threading
import asyncio
import time

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AlpacaDataProvider:
    def __init__(self, api_key, api_secret, base_url="https://paper-api.alpaca.markets"):
        self.api = tradeapi.REST(api_key, api_secret, base_url)
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.stream = None
        self.ws_thread = None
        self.data_callbacks = {}
        self.latest_prices = {}
        self.last_update_time = {}
        self.connected = False
        
        logger.info("Alpaca Data Provider initialized")
    
    def connect_websocket(self, symbols):
        """Connect to Alpaca WebSocket for real-time data"""
        try:
            # Create a new event loop for the websocket thread
            loop = asyncio.new_event_loop()
            
            # Set up the stream
            self.stream = Stream(
                key_id=self.api_key,
                secret_key=self.api_secret,
                base_url=self.base_url,
                data_stream_url="https://stream.data.alpaca.markets/v2",
                loop=loop
            )
            
            # Define the trade update handler
            async def on_trade(trade):
                symbol = trade.symbol
                self.latest_prices[symbol] = trade.price
                self.last_update_time[symbol] = time.time()
                logger.debug(f"WebSocket update for {symbol}: {trade.price}")
            
            # Subscribe to trade updates for each symbol
            for symbol in symbols:
                self.stream.subscribe_trades(on_trade, symbol)
                logger.info(f"Subscribed to WebSocket trades for {symbol}")
            
            # Define a function to run the event loop
            def run_websocket():
                asyncio.set_event_loop(loop)
                try:
                    self.stream.run()
                except Exception as e:
                    logger.error(f"WebSocket error: {e}")
                finally:
                    self.connected = False
                    logger.warning("WebSocket connection closed")
            
            # Start the WebSocket in a separate thread
            self.ws_thread = threading.Thread(target=run_websocket, daemon=True)
            self.ws_thread.start()
            
            # Give the connection time to establish
            time.sleep(2)
            self.connected = True
            
            logger.info(f"WebSocket connection established for {len(symbols)} symbols")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to Alpaca WebSocket: {e}")
            return False

    def get_latest_price(self, symbol):
        """Get the latest price for a symbol, falling back to REST API if needed"""
        # If WebSocket is connected and we have recent data, use it
        if self.connected and symbol in self.latest_prices:
            # Check if the data is fresh (less than 5 seconds old)
            current_time = time.time()
            if symbol in self.last_update_time and current_time - self.last_update_time.get(symbol, 0) < 5:
                return self.latest_prices[symbol]
        
        # Otherwise, fall back to REST API
        try:
            # Get the latest trade
            trade = self.api.get_latest_trade(symbol)
            price = trade.price
            self.latest_prices[symbol] = price
            self.last_update_time[symbol] = time.time()
            return price
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            # If trade data fails, try getting a quote
            try:
                quote = self.api.get_latest_quote(symbol)
                price = (quote.ask_price + quote.bid_price) / 2  # Midpoint price
                self.latest_prices[symbol] = price
                self.last_update_time[symbol] = time.time()
                return price
            except Exception as quote_e:
                logger.error(f"Error getting quote for {symbol}: {quote_e}")
                # Return the last known price if available
                return self.latest_prices.get(symbol, None)
        
    def get_historical_data(self, symbol, timeframe="1Day", limit=100):
        """Get historical data (still uses REST API)"""
        try:
            bars = self.api.get_bars(symbol, timeframe, limit=limit).df
            # Convert to format expected by MainStreet HFT
            df = bars.reset_index()
            df.rename(columns={
                'timestamp': 'timestamp',
                'open': 'open', 
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            }, inplace=True)
            df['symbol'] = symbol
            return df
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return None
            
    def get_market_data_dict(self, symbol):
        """Get market data in a format compatible with IB's market data"""
        price = self.get_latest_price(symbol)
        if price is None:
            return {}
            
        # Format to match IB's format (tick type 4 is last price)
        return {4: price}
        
    def disconnect(self):
        """Disconnect from the websocket"""
        if self.stream:
            self.stream.stop()
            self.connected = False
            logger.info("WebSocket connection closed")