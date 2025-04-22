import logging
import pandas as pd
import numpy as np
from collections import deque

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MarketDataProcessor:
    """Process and analyze market data"""
    def __init__(self, max_data_points=1000):
        self.max_data_points = max_data_points
        self.data = {}  # Dictionary to store market data by symbol
        
    def add_tick(self, symbol, timestamp, price, volume=None, bid=None, ask=None):
        """Add a new tick of market data"""
        if symbol not in self.data:
            self.data[symbol] = {
                'timestamp': deque(maxlen=self.max_data_points),
                'price': deque(maxlen=self.max_data_points),
                'volume': deque(maxlen=self.max_data_points),
                'bid': deque(maxlen=self.max_data_points),
                'ask': deque(maxlen=self.max_data_points)
            }
            
        self.data[symbol]['timestamp'].append(timestamp)
        self.data[symbol]['price'].append(price)
        self.data[symbol]['volume'].append(volume)
        self.data[symbol]['bid'].append(bid)
        self.data[symbol]['ask'].append(ask)
        
    def get_latest_price(self, symbol):
        """Get the latest price for a symbol"""
        if symbol in self.data and len(self.data[symbol]['price']) > 0:
            return self.data[symbol]['price'][-1]
        return None
        
    def get_price_history(self, symbol, n=None):
        """Get price history for a symbol"""
        if symbol not in self.data:
            return []
            
        prices = list(self.data[symbol]['price'])
        if n is not None:
            return prices[-n:]
        return prices
        
    def calculate_moving_average(self, symbol, period):
        """Calculate moving average for a symbol"""
        prices = self.get_price_history(symbol)
        if len(prices) < period:
            return None
            
        return sum(prices[-period:]) / period
        
    def calculate_exponential_moving_average(self, symbol, period):
        """Calculate exponential moving average for a symbol"""
        prices = self.get_price_history(symbol)
        if len(prices) < period:
            return None
            
        # Convert to pandas Series for easy EMA calculation
        price_series = pd.Series(prices)
        return price_series.ewm(span=period, adjust=False).mean().iloc[-1]
        
    def calculate_rsi(self, symbol, period=14):
        """Calculate Relative Strength Index for a symbol"""
        prices = self.get_price_history(symbol)
        if len(prices) <= period:
            return None
            
        # Calculate price changes
        deltas = np.diff(prices)
        
        # Separate gains and losses
        gains = np.clip(deltas, 0, float('inf'))
        losses = -np.clip(deltas, float('-inf'), 0)
        
        # Calculate average gains and losses
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100
            
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
        
    def calculate_bollinger_bands(self, symbol, period=20, num_std=2):
        """Calculate Bollinger Bands for a symbol"""
        prices = self.get_price_history(symbol)
        if len(prices) < period:
            return None
            
        # Calculate SMA and standard deviation
        sma = sum(prices[-period:]) / period
        std = np.std(prices[-period:])
        
        # Calculate Bollinger Bands
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        
        return {
            'middle': sma,
            'upper': upper_band,
            'lower': lower_band
        }
        
    def calculate_macd(self, symbol, fast_period=12, slow_period=26, signal_period=9):
        """Calculate MACD (Moving Average Convergence Divergence) for a symbol"""
        if len(self.get_price_history(symbol)) < slow_period + signal_period:
            return None
            
        # Calculate fast and slow EMAs
        fast_ema = self.calculate_exponential_moving_average(symbol, fast_period)
        slow_ema = self.calculate_exponential_moving_average(symbol, slow_period)
        
        if fast_ema is None or slow_ema is None:
            return None
            
        # Calculate MACD line
        macd_line = fast_ema - slow_ema
        
        # Calculate MACD signal line (EMA of MACD line)
        # In a real implementation, you would need to store the MACD line history
        # For this example, we'll use a simplified approach
        
        return {
            'macd': macd_line,
            'signal': None,  # Would need MACD history to calculate
            'histogram': None  # Would be MACD - Signal
        }
        
    def calculate_correlation(self, symbol1, symbol2, period=100):
        """Calculate correlation between two symbols"""
        prices1 = self.get_price_history(symbol1, period)
        prices2 = self.get_price_history(symbol2, period)
        
        if len(prices1) < period or len(prices2) < period:
            return None
            
        # Calculate correlation coefficient
        return np.corrcoef(prices1, prices2)[0, 1]
        
    def export_to_dataframe(self, symbol):
        """Export market data for a symbol to a pandas DataFrame"""
        if symbol not in self.data:
            return None
            
        df = pd.DataFrame({
            'timestamp': list(self.data[symbol]['timestamp']),
            'price': list(self.data[symbol]['price']),
            'volume': list(self.data[symbol]['volume']),
            'bid': list(self.data[symbol]['bid']),
            'ask': list(self.data[symbol]['ask'])
        })
        
        return df
        
    def clear_data(self, symbol=None):
        """Clear stored market data"""
        if symbol is None:
            self.data = {}
        elif symbol in self.data:
            del self.data[symbol]
            
    def calculate_vwap(self, symbol):
        """Calculate Volume-Weighted Average Price (VWAP)"""
        if symbol not in self.data or len(self.data[symbol]['price']) == 0:
            return None
            
        # Get price and volume histories
        prices = list(self.data[symbol]['price'])
        volumes = list(self.data[symbol]['volume'])
        
        # Filter out None values from volumes
        price_volume_pairs = [(p, v) for p, v in zip(prices, volumes) if v is not None]
        
        if not price_volume_pairs:
            return None
            
        # Unzip pairs
        filtered_prices, filtered_volumes = zip(*price_volume_pairs)
        
        # Calculate VWAP
        volume_sum = sum(filtered_volumes)
        if volume_sum == 0:
            return None
            
        vwap = sum(p * v for p, v in zip(filtered_prices, filtered_volumes)) / volume_sum
        
        return vwap
        
    def calculate_atr(self, symbol, period=14):
        """Calculate Average True Range (ATR)"""
        if symbol not in self.data or len(self.data[symbol]['price']) < period + 1:
            return None
            
        # Get price history
        prices = list(self.data[symbol]['price'])
        
        # Calculate true ranges
        true_ranges = []
        for i in range(1, len(prices)):
            high = prices[i]
            low = prices[i]
            prev_close = prices[i-1]
            
            # True range is the greatest of:
            # 1. Current high - current low
            # 2. Abs(current high - previous close)
            # 3. Abs(current low - previous close)
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            true_ranges.append(tr)
            
        # Calculate ATR (simple average of true ranges)
        atr = sum(true_ranges[-period:]) / period
        
        return atr
        
    def calculate_support_resistance(self, symbol, period=20, threshold=0.03):
        """Calculate basic support and resistance levels"""
        prices = self.get_price_history(symbol)
        if len(prices) < period:
            return None
            
        # Use recent price history
        recent_prices = prices[-period:]
        
        # Find local minima and maxima
        local_min = float('inf')
        local_max = float('-inf')
        
        for i in range(1, len(recent_prices) - 1):
            # Local minimum
            if recent_prices[i] < recent_prices[i-1] and recent_prices[i] < recent_prices[i+1]:
                local_min = min(local_min, recent_prices[i])
                
            # Local maximum
            if recent_prices[i] > recent_prices[i-1] and recent_prices[i] > recent_prices[i+1]:
                local_max = max(local_max, recent_prices[i])
                
        # If we couldn't find local min/max, use absolute min/max
        if local_min == float('inf'):
            local_min = min(recent_prices)
        if local_max == float('-inf'):
            local_max = max(recent_prices)
            
        return {
            'support': local_min,
            'resistance': local_max
        }
        
    def calculate_momentum(self, symbol, period=14):
        """Calculate momentum indicator"""
        prices = self.get_price_history(symbol)
        if len(prices) <= period:
            return None
            
        # Momentum = Current Price - Price 'n' periods ago
        current_price = prices[-1]
        past_price = prices[-period-1]
        
        momentum = current_price - past_price
        
        return momentum
        
    def calculate_stochastic_oscillator(self, symbol, k_period=14, d_period=3):
        """Calculate Stochastic Oscillator"""
        prices = self.get_price_history(symbol)
        if len(prices) < k_period:
            return None
            
        # Calculate %K
        recent_prices = prices[-k_period:]
        current_price = prices[-1]
        lowest_low = min(recent_prices)
        highest_high = max(recent_prices)
        
        if highest_high == lowest_low:
            k = 50  # Middle value if there's no range
        else:
            k = 100 * (current_price - lowest_low) / (highest_high - lowest_low)
            
        # In a real implementation, you would calculate %D as SMA of %K
        # We would need to keep track of %K history for this
        
        return {
            'k': k,
            'd': None  # Would be SMA of %K values
        }
        
    def detect_breakout(self, symbol, period=20, threshold=0.02):
        """Detect price breakouts above resistance or below support"""
        prices = self.get_price_history(symbol)
        if len(prices) < period:
            return None
            
        # Get support and resistance levels
        levels = self.calculate_support_resistance(symbol, period)
        if levels is None:
            return None
            
        current_price = prices[-1]
        
        # Check for breakouts
        if current_price > levels['resistance'] * (1 + threshold):
            return {
                'type': 'bullish',
                'level': levels['resistance'],
                'price': current_price
            }
        elif current_price < levels['support'] * (1 - threshold):
            return {
                'type': 'bearish',
                'level': levels['support'],
                'price': current_price
            }
            
        return None
        
    def calculate_pivot_points(self, symbol, method='standard'):
        """Calculate pivot points (Standard, Fibonacci, or Woodie)"""
        # Need high, low, close data from the previous period
        # For this example, we'll use the most recent data as a proxy
        if symbol not in self.data:
            return None
            
        # Get recent price data
        prices = list(self.data[symbol]['price'])
        if len(prices) < 3:
            return None
            
        # Use recent prices as proxy for high, low, close
        high = max(prices[-3:])
        low = min(prices[-3:])
        close = prices[-1]
        
        if method == 'standard':
            # Standard Pivot Points
            p = (high + low + close) / 3  # Pivot
            r1 = (2 * p) - low           # Resistance 1
            r2 = p + (high - low)        # Resistance 2
            s1 = (2 * p) - high          # Support 1
            s2 = p - (high - low)        # Support 2
            
        elif method == 'fibonacci':
            # Fibonacci Pivot Points
            p = (high + low + close) / 3
            r1 = p + 0.382 * (high - low)
            r2 = p + 0.618 * (high - low)
            r3 = p + 1.000 * (high - low)
            s1 = p - 0.382 * (high - low)
            s2 = p - 0.618 * (high - low)
            s3 = p - 1.000 * (high - low)
            
            return {
                'pivot': p,
                'r1': r1,
                'r2': r2,
                'r3': r3,
                's1': s1,
                's2': s2,
                's3': s3
            }
            
        elif method == 'woodie':
            # Woodie Pivot Points
            p = (high + low + 2 * close) / 4
            r1 = (2 * p) - low
            r2 = p + (high - low)
            s1 = (2 * p) - high
            s2 = p - (high - low)
            
        return {
            'pivot': p,
            'r1': r1,
            'r2': r2,
            's1': s1,
            's2': s2
        }
        
    def calculate_ichimoku_cloud(self, symbol):
        """Calculate Ichimoku Cloud components"""
        prices = self.get_price_history(symbol)
        if len(prices) < 52:  # Need at least 52 periods for Senkou Span B
            return None
            
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
        period9_high = max(prices[-9:])
        period9_low = min(prices[-9:])
        tenkan = (period9_high + period9_low) / 2
        
        # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
        period26_high = max(prices[-26:])
        period26_low = min(prices[-26:])
        kijun = (period26_high + period26_low) / 2
        
        # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
        senkou_a = (tenkan + kijun) / 2
        
        # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
        period52_high = max(prices[-52:])
        period52_low = min(prices[-52:])
        senkou_b = (period52_high + period52_low) / 2
        
        # Chikou Span (Lagging Span): Current closing price, shifted back 26 periods
        chikou = prices[-1]  # Would be shifted in plotting
        
        return {
            'tenkan': tenkan,
            'kijun': kijun,
            'senkou_a': senkou_a,
            'senkou_b': senkou_b,
            'chikou': chikou
        }
        
    def calculate_volatility(self, symbol, period=20):
        """Calculate historical volatility"""
        prices = self.get_price_history(symbol)
        if len(prices) < period + 1:
            return None
            
        # Calculate returns
        returns = []
        for i in range(1, len(prices)):
            returns.append((prices[i] - prices[i-1]) / prices[i-1])
            
        # Calculate standard deviation of returns
        std_dev = np.std(returns[-period:])
        
        # Annualized volatility (assuming daily data)
        annualized_volatility = std_dev * np.sqrt(252)
        
        return {
            'daily_volatility': std_dev,
            'annualized_volatility': annualized_volatility
        }
        
    def calculate_price_channels(self, symbol, period=20):
        """Calculate price channels (Donchian Channels)"""
        prices = self.get_price_history(symbol)
        if len(prices) < period:
            return None
            
        recent_prices = prices[-period:]
        
        upper_channel = max(recent_prices)
        lower_channel = min(recent_prices)
        middle_channel = (upper_channel + lower_channel) / 2
        
        return {
            'upper': upper_channel,
            'lower': lower_channel,
            'middle': middle_channel
        }
        
    def get_market_statistics(self, symbol):
        """Get various market statistics for a symbol"""
        if symbol not in self.data:
            return None
            
        # Get price history
        prices = list(self.data[symbol]['price'])
        if not prices:
            return None
            
        # Calculate basic statistics
        current_price = prices[-1]
        daily_change = current_price - prices[0] if len(prices) > 1 else 0
        daily_change_pct = (daily_change / prices[0]) * 100 if len(prices) > 1 and prices[0] != 0 else 0
        
        # Calculate technical indicators if enough data
        rsi = self.calculate_rsi(symbol) if len(prices) > 14 else None
        macd = self.calculate_macd(symbol) if len(prices) > 26 else None
        bollinger = self.calculate_bollinger_bands(symbol) if len(prices) > 20 else None
        
        # Get market data
        volumes = list(self.data[symbol]['volume'])
        volume = volumes[-1] if volumes and volumes[-1] is not None else None
        
        # Calculate average volume
        valid_volumes = [v for v in volumes if v is not None]
        avg_volume = sum(valid_volumes) / len(valid_volumes) if valid_volumes else None
        
        return {
            'symbol': symbol,
            'price': current_price,
            'daily_change': daily_change,
            'daily_change_pct': daily_change_pct,
            'volume': volume,
            'avg_volume': avg_volume,
            'rsi': rsi,
            'macd': macd['macd'] if macd else None,
            'bollinger': bollinger
        }


# Example usage
if __name__ == "__main__":
    # Create a market data processor
    processor = MarketDataProcessor()
    
    # Add some fake data for testing
    import datetime
    import random
    
    symbol = "AAPL"
    now = datetime.datetime.now()
    
    # Generate 100 fake data points
    base_price = 150.0
    for i in range(100):
        # Generate a timestamp 1 minute apart
        timestamp = now - datetime.timedelta(minutes=100-i)
        
        # Generate a random price with a slight upward trend
        price = base_price + (i * 0.05) + (random.random() - 0.4)
        
        # Generate a random volume
        volume = int(random.random() * 1000) + 500
        
        # Add the tick
        processor.add_tick(symbol, timestamp, price, volume)
    
    # Calculate and print some indicators
    print(f"Latest price for {symbol}: ${processor.get_latest_price(symbol):.2f}")
    
    ma = processor.calculate_moving_average(symbol, 20)
    print(f"20-period Moving Average: ${ma:.2f}")
    
    rsi = processor.calculate_rsi(symbol)
    print(f"RSI: {rsi:.2f}")
    
    bollinger = processor.calculate_bollinger_bands(symbol)
    print(f"Bollinger Bands:")
    print(f"  Upper: ${bollinger['upper']:.2f}")
    print(f"  Middle: ${bollinger['middle']:.2f}")
    print(f"  Lower: ${bollinger['lower']:.2f}")
    
    # Export to DataFrame
    df = processor.export_to_dataframe(symbol)
    print("\nDataFrame head:")
    print(df.head())