import logging
import requests
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StockScreener:
    """Stock screening and data retrieval component"""
    
    def __init__(self, api_key=None, provider="alphavantage"):
        """Initialize the stock screener with API credentials"""
        self.api_key = api_key
        self.provider = provider.lower()
        self.base_urls = {
            "alphavantage": "https://www.alphavantage.co/query",
            "finnhub": "https://finnhub.io/api/v1",
            "iex": "https://cloud.iexapis.com/stable"
        }
        
        if self.provider not in self.base_urls:
            logger.warning(f"Provider {provider} not recognized. Defaulting to alphavantage.")
            self.provider = "alphavantage"
            
        self.base_url = self.base_urls[self.provider]
        
        # Cache to avoid repeated API calls
        self.cache = {}
        self.cache_expiry = {}
        self.cache_duration = 300  # 5 minutes
        
    def search_symbols(self, keywords):
        """Search for symbols based on keywords"""
        if not self.api_key:
            logger.error("API key is required for symbol search")
            return []
            
        cache_key = f"search_{keywords}"
        if cache_key in self.cache and datetime.now() < self.cache_expiry.get(cache_key, datetime.min):
            logger.info(f"Using cached results for {keywords}")
            return self.cache[cache_key]
            
        try:
            if self.provider == "alphavantage":
                params = {
                    "function": "SYMBOL_SEARCH",
                    "keywords": keywords,
                    "apikey": self.api_key
                }
                response = requests.get(self.base_url, params=params)
                data = response.json()
                
                if "bestMatches" not in data:
                    logger.error(f"API error: {data}")
                    return []
                    
                results = []
                for match in data["bestMatches"]:
                    results.append({
                        "symbol": match["1. symbol"],
                        "name": match["2. name"],
                        "region": match["4. region"],
                        "type": match["3. type"]
                    })
                
                # Cache the results
                self.cache[cache_key] = results
                self.cache_expiry[cache_key] = datetime.now() + timedelta(seconds=self.cache_duration)
                
                return results
                
            elif self.provider == "finnhub":
                params = {
                    "q": keywords,
                    "token": self.api_key
                }
                response = requests.get(f"{self.base_url}/search", params=params)
                data = response.json()
                
                if "result" not in data:
                    logger.error(f"API error: {data}")
                    return []
                    
                results = []
                for match in data["result"]:
                    results.append({
                        "symbol": match["symbol"],
                        "name": match["description"],
                        "type": match["type"]
                    })
                
                # Cache the results
                self.cache[cache_key] = results
                self.cache_expiry[cache_key] = datetime.now() + timedelta(seconds=self.cache_duration)
                
                return results
                
            elif self.provider == "iex":
                params = {
                    "token": self.api_key
                }
                response = requests.get(f"{self.base_url}/search/{keywords}", params=params)
                data = response.json()
                
                if not isinstance(data, list):
                    logger.error(f"API error: {data}")
                    return []
                    
                results = []
                for match in data:
                    results.append({
                        "symbol": match["symbol"],
                        "name": match.get("securityName", ""),
                        "type": match.get("securityType", "")
                    })
                
                # Cache the results
                self.cache[cache_key] = results
                self.cache_expiry[cache_key] = datetime.now() + timedelta(seconds=self.cache_duration)
                
                return results
                
        except Exception as e:
            logger.exception(f"Error searching symbols: {e}")
            return []
    
    def get_top_gainers(self, limit=10):
        """Get top gaining stocks"""
        if not self.api_key:
            logger.error("API key is required for top gainers")
            return []
            
        cache_key = f"gainers_{limit}"
        if cache_key in self.cache and datetime.now() < self.cache_expiry.get(cache_key, datetime.min):
            logger.info(f"Using cached results for top gainers")
            return self.cache[cache_key]
            
        try:
            if self.provider == "iex":
                params = {
                    "token": self.api_key,
                    "listLimit": limit
                }
                response = requests.get(f"{self.base_url}/stock/market/list/gainers", params=params)
                data = response.json()
                
                if not isinstance(data, list):
                    logger.error(f"API error: {data}")
                    return []
                    
                results = []
                for stock in data:
                    results.append({
                        "symbol": stock["symbol"],
                        "name": stock.get("companyName", ""),
                        "price": stock.get("latestPrice", 0),
                        "change_pct": stock.get("changePercent", 0) * 100
                    })
                
                # Cache the results
                self.cache[cache_key] = results
                self.cache_expiry[cache_key] = datetime.now() + timedelta(seconds=self.cache_duration)
                
                return results
            elif self.provider == "finnhub":
                # For Finnhub, we'll get quote data for major indices and sort by performance
                symbols = ["AAPL", "MSFT", "AMZN", "GOOGL", "FB", "TSLA", "NVDA", "JPM", "JNJ", "V", 
                           "PG", "UNH", "HD", "BAC", "MA", "DIS", "ADBE", "CRM", "CMCSA", "XOM",
                           "NFLX", "VZ", "INTC", "T", "PFE", "ABT", "CSCO", "TMO", "NKE", "KO"]
                
                quotes = {}
                for symbol in symbols:
                    params = {
                        "symbol": symbol,
                        "token": self.api_key
                    }
                    response = requests.get(f"{self.base_url}/quote", params=params)
                    quote = response.json()
                    
                    quotes[symbol] = {
                        "symbol": symbol,
                        "price": quote.get("c", 0),
                        "change_pct": quote.get("dp", 0)
                    }
                    
                    # To avoid API rate limits
                    time.sleep(0.01)
                
                # Sort by percent change and get top performers
                sorted_quotes = sorted(quotes.values(), key=lambda x: x["change_pct"], reverse=True)
                results = sorted_quotes[:limit]
                
                # Cache the results
                self.cache[cache_key] = results
                self.cache_expiry[cache_key] = datetime.now() + timedelta(seconds=self.cache_duration)
                
                return results
            else:
                logger.warning(f"Top gainers not implemented for provider {self.provider}")
                return []
                
        except Exception as e:
            logger.exception(f"Error getting top gainers: {e}")
            return []
    
    def get_top_losers(self, limit=10):
        """Get top losing stocks"""
        if not self.api_key:
            logger.error("API key is required for top losers")
            return []
            
        cache_key = f"losers_{limit}"
        if cache_key in self.cache and datetime.now() < self.cache_expiry.get(cache_key, datetime.min):
            logger.info(f"Using cached results for top losers")
            return self.cache[cache_key]
            
        try:
            if self.provider == "iex":
                params = {
                    "token": self.api_key,
                    "listLimit": limit
                }
                response = requests.get(f"{self.base_url}/stock/market/list/losers", params=params)
                data = response.json()
                
                if not isinstance(data, list):
                    logger.error(f"API error: {data}")
                    return []
                    
                results = []
                for stock in data:
                    results.append({
                        "symbol": stock["symbol"],
                        "name": stock.get("companyName", ""),
                        "price": stock.get("latestPrice", 0),
                        "change_pct": stock.get("changePercent", 0) * 100
                    })
                
                # Cache the results
                self.cache[cache_key] = results
                self.cache_expiry[cache_key] = datetime.now() + timedelta(seconds=self.cache_duration)
                
                return results
            elif self.provider == "finnhub":
                # For Finnhub, we'll get quote data for major indices and sort by performance
                symbols = ["AAPL", "MSFT", "AMZN", "GOOGL", "FB", "TSLA", "NVDA", "JPM", "JNJ", "V", 
                           "PG", "UNH", "HD", "BAC", "MA", "DIS", "ADBE", "CRM", "CMCSA", "XOM",
                           "NFLX", "VZ", "INTC", "T", "PFE", "ABT", "CSCO", "TMO", "NKE", "KO"]
                
                quotes = {}
                for symbol in symbols:
                    params = {
                        "symbol": symbol,
                        "token": self.api_key
                    }
                    response = requests.get(f"{self.base_url}/quote", params=params)
                    quote = response.json()
                    
                    quotes[symbol] = {
                        "symbol": symbol,
                        "price": quote.get("c", 0),
                        "change_pct": quote.get("dp", 0)
                    }
                    
                    # To avoid API rate limits
                    time.sleep(0.01)
                
                # Sort by percent change (ascending) and get top losers
                sorted_quotes = sorted(quotes.values(), key=lambda x: x["change_pct"])
                results = sorted_quotes[:limit]
                
                # Cache the results
                self.cache[cache_key] = results
                self.cache_expiry[cache_key] = datetime.now() + timedelta(seconds=self.cache_duration)
                
                return results
            else:
                logger.warning(f"Top losers not implemented for provider {self.provider}")
                return []
                
        except Exception as e:
            logger.exception(f"Error getting top losers: {e}")
            return []
    
    def screen_stocks(self, criteria):
        """Screen stocks based on criteria
        
        Example criteria:
        {
            "market_cap_min": 1000000000,  # $1B minimum market cap
            "pe_ratio_max": 20,            # P/E ratio below 20
            "sector": "Technology",        # Sector filter
            "price_min": 10,               # Minimum price
            "price_max": 100               # Maximum price
        }
        """
        if not self.api_key:
            logger.error("API key is required for stock screening")
            return []
            
        # Convert criteria to a string for cache key
        criteria_str = json.dumps(criteria, sort_keys=True)
        cache_key = f"screen_{criteria_str}"
        
        if cache_key in self.cache and datetime.now() < self.cache_expiry.get(cache_key, datetime.min):
            logger.info(f"Using cached results for screening")
            return self.cache[cache_key]
            
        try:
            if self.provider == "finnhub":
                # Finnhub has a dedicated stock screener endpoint
                params = {
                    "token": self.api_key
                }
                
                # Map our criteria to Finnhub parameters
                if "market_cap_min" in criteria:
                    params["marketCapLowerLimit"] = criteria["market_cap_min"]
                if "market_cap_max" in criteria:
                    params["marketCapUpperLimit"] = criteria["market_cap_max"]
                if "price_min" in criteria:
                    params["priceRangeLowerLimit"] = criteria["price_min"]
                if "price_max" in criteria:
                    params["priceRangeUpperLimit"] = criteria["price_max"]
                if "pe_ratio_min" in criteria:
                    params["peRatioLowerLimit"] = criteria["pe_ratio_min"]
                if "pe_ratio_max" in criteria:
                    params["peRatioUpperLimit"] = criteria["pe_ratio_max"]
                
                response = requests.get(f"{self.base_url}/stock/screener", params=params)
                data = response.json()
                
                if not isinstance(data, list):
                    logger.error(f"API error: {data}")
                    return []
                
                results = []
                for stock in data:
                    results.append({
                        "symbol": stock.get("symbol", ""),
                        "name": stock.get("name", ""),
                        "price": stock.get("price", 0),
                        "market_cap": stock.get("marketCap", 0),
                        "pe_ratio": stock.get("pe", 0)
                    })
                
                # Apply any additional filters not supported by the API
                if "sector" in criteria:
                    # We would need to get sector data for each stock
                    # This is a simplified approach
                    filtered_results = []
                    for stock in results:
                        # Get company profile to check sector
                        profile_params = {
                            "symbol": stock["symbol"],
                            "token": self.api_key
                        }
                        profile_response = requests.get(f"{self.base_url}/stock/profile2", params=profile_params)
                        profile = profile_response.json()
                        
                        if profile.get("sector") == criteria["sector"]:
                            filtered_results.append(stock)
                            
                        # To avoid API rate limits
                        time.sleep(0.01)
                        
                    results = filtered_results
                
                # Cache the results
                self.cache[cache_key] = results
                self.cache_expiry[cache_key] = datetime.now() + timedelta(seconds=self.cache_duration)
                
                return results
            
            elif self.provider == "alphavantage":
                # Alpha Vantage doesn't have a dedicated screener endpoint
                # We'll use a simplified approach with basic filtering
                
                # First, get list of symbols from S&P 500 or other major indices
                # For this example, we'll use a small predefined list
                symbols = ["AAPL", "MSFT", "AMZN", "GOOGL", "FB", "TSLA", "NVDA", "JPM", "JNJ", "V"]
                
                results = []
                for symbol in symbols:
                    # Get overview data for the symbol
                    params = {
                        "function": "OVERVIEW",
                        "symbol": symbol,
                        "apikey": self.api_key
                    }
                    
                    response = requests.get(self.base_url, params=params)
                    data = response.json()
                    
                    # Check if this stock meets the criteria
                    meets_criteria = True
                    
                    if "market_cap_min" in criteria and "MarketCapitalization" in data:
                        market_cap = float(data.get("MarketCapitalization", 0))
                        if market_cap < criteria["market_cap_min"]:
                            meets_criteria = False
                            
                    if "market_cap_max" in criteria and "MarketCapitalization" in data:
                        market_cap = float(data.get("MarketCapitalization", 0))
                        if market_cap > criteria["market_cap_max"]:
                            meets_criteria = False
                            
                    if "pe_ratio_min" in criteria and "PERatio" in data:
                        try:
                            pe_ratio = float(data.get("PERatio", 0))
                            if pe_ratio < criteria["pe_ratio_min"]:
                                meets_criteria = False
                        except ValueError:
                            meets_criteria = False
                            
                    if "pe_ratio_max" in criteria and "PERatio" in data:
                        try:
                            pe_ratio = float(data.get("PERatio", 0))
                            if pe_ratio > criteria["pe_ratio_max"]:
                                meets_criteria = False
                        except ValueError:
                            meets_criteria = False
                            
                    if "sector" in criteria and data.get("Sector") != criteria["sector"]:
                        meets_criteria = False
                        
                    # Check price criteria
                    if "price_min" in criteria or "price_max" in criteria:
                        # Get latest price
                        quote_params = {
                            "function": "GLOBAL_QUOTE",
                            "symbol": symbol,
                            "apikey": self.api_key
                        }
                        
                        quote_response = requests.get(self.base_url, params=quote_params)
                        quote_data = quote_response.json()
                        
                        if "Global Quote" in quote_data:
                            current_price = float(quote_data["Global Quote"].get("05. price", 0))
                            
                            if "price_min" in criteria and current_price < criteria["price_min"]:
                                meets_criteria = False
                                
                            if "price_max" in criteria and current_price > criteria["price_max"]:
                                meets_criteria = False
                        else:
                            meets_criteria = False
                    
                    if meets_criteria:
                        results.append({
                            "symbol": symbol,
                            "name": data.get("Name", ""),
                            "sector": data.get("Sector", ""),
                            "price": float(quote_data["Global Quote"].get("05. price", 0)) if "Global Quote" in quote_data else 0,
                            "market_cap": float(data.get("MarketCapitalization", 0)),
                            "pe_ratio": float(data.get("PERatio", 0)) if data.get("PERatio") else None
                        })
                    
                    # To avoid API rate limits
                    time.sleep(0.01)
                
                # Cache the results
                self.cache[cache_key] = results
                self.cache_expiry[cache_key] = datetime.now() + timedelta(seconds=self.cache_duration)
                
                return results
            
            else:
                logger.warning(f"Stock screening not fully implemented for provider {self.provider}")
                return []
                
        except Exception as e:
            logger.exception(f"Error screening stocks: {e}")
            return []
    
    def get_historical_data(self, symbol, interval="daily", period="1month"):
        """Get historical price data for a symbol
        
        Args:
            symbol: The stock symbol
            interval: Data interval (e.g., "daily", "hourly", "1min")
            period: How far back to go (e.g., "1month", "1year")
        """
        if not self.api_key:
            logger.error("API key is required for historical data")
            return None
            
        cache_key = f"history_{symbol}_{interval}_{period}"
        if cache_key in self.cache and datetime.now() < self.cache_expiry.get(cache_key, datetime.min):
            logger.info(f"Using cached historical data for {symbol}")
            return self.cache[cache_key]
            
        try:
            if self.provider == "alphavantage":
                # Map parameters to Alpha Vantage format
                if interval == "daily":
                    function = "TIME_SERIES_DAILY"
                    key = "Time Series (Daily)"
                elif interval == "hourly":
                    function = "TIME_SERIES_INTRADAY"
                    key = "Time Series (60min)"
                    outputsize = "full"
                    interval = "60min"
                else:
                    function = "TIME_SERIES_INTRADAY"
                    key = f"Time Series ({interval})"
                    outputsize = "full"
                
                params = {
                    "function": function,
                    "symbol": symbol,
                    "apikey": self.api_key
                }
                
                if function == "TIME_SERIES_INTRADAY":
                    params["interval"] = interval
                    params["outputsize"] = outputsize
                
                response = requests.get(self.base_url, params=params)
                data = response.json()
                
                if key not in data:
                    logger.error(f"API error: {data}")
                    return None
                
                # Parse the data
                time_series = data[key]
                df = pd.DataFrame.from_dict(time_series, orient="index")
                
                # Rename columns to more intuitive names
                df.columns = [col.split(". ")[1] for col in df.columns]
                
                # Convert strings to numeric values
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col])
                
                # Add date as a column and sort by date
                df["timestamp"] = pd.to_datetime(df.index)
                df = df.sort_values("timestamp")
                
                # Apply period filter
                if period == "1month":
                    cutoff = datetime.now() - timedelta(days=30)
                elif period == "3months":
                    cutoff = datetime.now() - timedelta(days=90)
                elif period == "6months":
                    cutoff = datetime.now() - timedelta(days=180)
                elif period == "1year":
                    cutoff = datetime.now() - timedelta(days=365)
                else:
                    # Default to 1 month
                    cutoff = datetime.now() - timedelta(days=30)
                    
                df = df[df["timestamp"] > cutoff]
                
                # Cache the results
                self.cache[cache_key] = df
                self.cache_expiry[cache_key] = datetime.now() + timedelta(seconds=self.cache_duration)
                
                return df
                
            elif self.provider == "finnhub":
                # Convert period to seconds
                if period == "1month":
                    seconds = 30 * 24 * 60 * 60
                elif period == "3months":
                    seconds = 90 * 24 * 60 * 60
                elif period == "6months":
                    seconds = 180 * 24 * 60 * 60
                elif period == "1year":
                    seconds = 365 * 24 * 60 * 60
                else:
                    seconds = 30 * 24 * 60 * 60
                
                # Convert interval to Finnhub format
                if interval == "daily":
                    resolution = "D"
                elif interval == "hourly":
                    resolution = "60"
                elif interval == "1min":
                    resolution = "1"
                elif interval == "5min":
                    resolution = "5"
                elif interval == "15min":
                    resolution = "15"
                elif interval == "30min":
                    resolution = "30"
                elif interval == "60min":
                    resolution = "60"
                else:
                    resolution = "D"
                
                # Calculate from and to timestamps
                to_timestamp = int(time.time())
                from_timestamp = to_timestamp - seconds
                
                params = {
                    "symbol": symbol,
                    "resolution": resolution,
                    "from": from_timestamp,
                    "to": to_timestamp,
                    "token": self.api_key
                }
                
                response = requests.get(f"{self.base_url}/stock/candle", params=params)
                data = response.json()
                
                if data.get("s") != "ok":
                    logger.error(f"API error: {data}")
                    return None
                
                # Create DataFrame from the data
                df = pd.DataFrame({
                    "timestamp": pd.to_datetime([datetime.fromtimestamp(t) for t in data["t"]]),
                    "open": data["o"],
                    "high": data["h"],
                    "low": data["l"],
                    "close": data["c"],
                    "volume": data["v"]
                })
                
                # Sort by timestamp
                df = df.sort_values("timestamp")
                
                # Cache the results
                self.cache[cache_key] = df
                self.cache_expiry[cache_key] = datetime.now() + timedelta(seconds=self.cache_duration)
                
                return df
                
            else:
                logger.warning(f"Historical data not implemented for provider {self.provider}")
                return None
                
        except Exception as e:
            logger.exception(f"Error getting historical data: {e}")
            return None
    
    def find_correlated_pairs(self, symbols, lookback_days=60, min_correlation=0.7):
        """Find correlated pairs among a list of symbols
        
        This is useful for statistical arbitrage strategies.
        """
        if not self.api_key:
            logger.error("API key is required for correlation analysis")
            return []
            
        symbols_key = ",".join(sorted(symbols))
        cache_key = f"corr_{symbols_key}_{lookback_days}_{min_correlation}"
        
        if cache_key in self.cache and datetime.now() < self.cache_expiry.get(cache_key, datetime.min):
            logger.info(f"Using cached correlation results")
            return self.cache[cache_key]
            
        try:
            # Get historical data for each symbol
            price_data = {}
            for symbol in symbols:
                df = self.get_historical_data(symbol, interval="daily", period=f"{lookback_days}days")
                if df is not None:
                    price_data[symbol] = df["close"]
                    
            if len(price_data) < 2:
                logger.error("Insufficient data for correlation analysis")
                return []
                
            # Combine price data into a DataFrame
            price_df = pd.DataFrame(price_data)
            
            # Calculate correlation matrix
            corr_matrix = price_df.corr()
            
            # Find pairs with correlation above threshold
            pairs = []
            for i in range(len(symbols)):
                for j in range(i + 1, len(symbols)):
                    symbol1 = symbols[i]
                    symbol2 = symbols[j]
                    
                    if symbol1 in corr_matrix.index and symbol2 in corr_matrix.columns:
                        correlation = corr_matrix.loc[symbol1, symbol2]
                        
                        if abs(correlation) >= min_correlation:
                            pairs.append({
                                "symbol1": symbol1,
                                "symbol2": symbol2,
                                "correlation": correlation
                            })
            
            # Cache the results
            self.cache[cache_key] = pairs
            self.cache_expiry[cache_key] = datetime.now() + timedelta(seconds=self.cache_duration)
            
            return pairs
            
        except Exception as e:
            logger.exception(f"Error finding correlated pairs: {e}")
            return []
    
    def get_sector_performance(self, period="1day"):
        """Get performance data for market sectors
        
        Args:
            period: Time period ('1day', '5day', '1month', '3month', 'ytd', '1year')
            
        Returns:
            Dictionary mapping sector names to performance percentages
        """
        if not self.api_key:
            logger.error("API key is required for sector performance")
            return {}
            
        cache_key = f"sectors_{period}"
        if cache_key in self.cache and datetime.now() < self.cache_expiry.get(cache_key, datetime.min):
            logger.info(f"Using cached sector performance for {period}")
            return self.cache[cache_key]
            
        try:
            if self.provider == "alphavantage":
                params = {
                    "function": "SECTOR",
                    "apikey": self.api_key
                }
                response = requests.get(self.base_url, params=params)
                data = response.json()
                
                # Extract the relevant performance data based on period
                period_key = None
                if period == "1day":
                    period_key = "Rank A: Real-Time Performance"
                elif period == "5day":
                    period_key = "Rank B: 5 Day Performance"
                elif period == "1month":
                    period_key = "Rank C: 1 Month Performance"
                elif period == "3month":
                    period_key = "Rank D: 3 Month Performance"
                elif period == "ytd":
                    period_key = "Rank E: Year-to-Date (YTD) Performance"
                elif period == "1year":
                    period_key = "Rank F: 1 Year Performance"
                    
                if period_key and period_key in data:
                    # Parse percentage strings to floats
                    result = {}
                    for sector, perf_str in data[period_key].items():
                        try:
                            # Remove percentage sign and convert to float
                            perf = float(perf_str.strip('%'))
                            result[sector] = perf
                        except ValueError:
                            continue
                    
                    # Cache the results
                    self.cache[cache_key] = result
                    self.cache_expiry[cache_key] = datetime.now() + timedelta(seconds=self.cache_duration)
                    
                    return result
                    
            elif self.provider == "iex":
                # Implementation for IEX if available
                pass
                
            logger.warning(f"Sector performance not fully implemented for provider {self.provider}")
            return {}
                
        except Exception as e:
            logger.exception(f"Error getting sector performance: {e}")
            return {}

    def get_company_fundamentals(self, symbol, data_type="overview"):
        """Get fundamental data for a company
        
        Args:
            symbol: The stock symbol
            data_type: Type of fundamental data ('overview', 'income', 'balance', 'cash', 'earnings')
            
        Returns:
            Dictionary of fundamental data
        """
        if not self.api_key:
            logger.error("API key is required for company fundamentals")
            return {}
            
        cache_key = f"fundamentals_{symbol}_{data_type}"
        if cache_key in self.cache and datetime.now() < self.cache_expiry.get(cache_key, datetime.min):
            logger.info(f"Using cached fundamental data for {symbol}")
            return self.cache[cache_key]
            
        try:
            if self.provider == "alphavantage":
                function = ""
                if data_type == "overview":
                    function = "OVERVIEW"
                elif data_type == "income":
                    function = "INCOME_STATEMENT"
                elif data_type == "balance":
                    function = "BALANCE_SHEET"
                elif data_type == "cash":
                    function = "CASH_FLOW"
                elif data_type == "earnings":
                    function = "EARNINGS"
                    
                if not function:
                    logger.error(f"Invalid fundamental data type: {data_type}")
                    return {}
                    
                params = {
                    "function": function,
                    "symbol": symbol,
                    "apikey": self.api_key
                }
                
                response = requests.get(self.base_url, params=params)
                data = response.json()
                
                # Check for error messages
                if "Error Message" in data:
                    logger.error(f"API error: {data['Error Message']}")
                    return {}
                    
                # Process data based on type
                result = data
                
                # Cache the results
                self.cache[cache_key] = result
                self.cache_expiry[cache_key] = datetime.now() + timedelta(seconds=self.cache_duration)
                
                return result
                
            elif self.provider == "finnhub":
                endpoint = ""
                if data_type == "overview":
                    endpoint = "stock/profile2"
                elif data_type == "income":
                    endpoint = "stock/financials"
                elif data_type == "balance":
                    endpoint = "stock/financials"
                elif data_type == "cash":
                    endpoint = "stock/financials"
                elif data_type == "earnings":
                    endpoint = "stock/earnings"
                    
                if not endpoint:
                    logger.error(f"Invalid fundamental data type: {data_type}")
                    return {}
                
                params = {
                    "symbol": symbol,
                    "token": self.api_key
                }
                
                if data_type in ["income", "balance", "cash"]:
                    params["statement"] = data_type
                    
                response = requests.get(f"{self.base_url}/{endpoint}", params=params)
                data = response.json()
                
                # Process data based on type
                result = data
                
                # Cache the results
                self.cache[cache_key] = result
                self.cache_expiry[cache_key] = datetime.now() + timedelta(seconds=self.cache_duration)
                
                return result
                
            elif self.provider == "iex":
                # Implementation for IEX if available
                pass
                
            logger.warning(f"Company fundamentals not fully implemented for provider {self.provider}")
            return {}
                
        except Exception as e:
            logger.exception(f"Error getting company fundamentals: {e}")
            return {}
    
    def clear_cache(self):
        """Clear the cache"""
        self.cache = {}
        self.cache_expiry = {}
        logger.info("Cache cleared")
        
    def get_quote(self, symbol):
        """Get current quote for a symbol"""
        if not self.api_key:
            logger.error("API key is required for quotes")
            return None
            
        cache_key = f"quote_{symbol}"
        cache_duration = 60  # Shorter duration for real-time quotes
        
        if cache_key in self.cache and datetime.now() < self.cache_expiry.get(cache_key, datetime.min):
            logger.info(f"Using cached quote for {symbol}")
            return self.cache[cache_key]
            
        try:
            if self.provider == "alphavantage":
                params = {
                    "function": "GLOBAL_QUOTE",
                    "symbol": symbol,
                    "apikey": self.api_key
                }
                
                response = requests.get(self.base_url, params=params)
                data = response.json()
                
                if "Global Quote" not in data:
                    logger.error(f"API error: {data}")
                    return None
                
                quote = data["Global Quote"]
                result = {
                    "symbol": quote.get("01. symbol", symbol),
                    "price": float(quote.get("05. price", 0)),
                    "change": float(quote.get("09. change", 0)),
                    "change_percent": float(quote.get("10. change percent", "0%").strip("%")),
                    "volume": int(quote.get("06. volume", 0)),
                    "latest_trading_day": quote.get("07. latest trading day", "")
                }
                
                # Cache the results
                self.cache[cache_key] = result
                self.cache_expiry[cache_key] = datetime.now() + timedelta(seconds=cache_duration)
                
                return result
                
            elif self.provider == "finnhub":
                params = {
                    "symbol": symbol,
                    "token": self.api_key
                }
                
                response = requests.get(f"{self.base_url}/quote", params=params)
                data = response.json()
                
                result = {
                    "symbol": symbol,
                    "price": data.get("c", 0),
                    "change": data.get("d", 0),
                    "change_percent": data.get("dp", 0),
                    "high": data.get("h", 0),
                    "low": data.get("l", 0),
                    "open": data.get("o", 0),
                    "previous_close": data.get("pc", 0)
                }
                
                # Cache the results
                self.cache[cache_key] = result
                self.cache_expiry[cache_key] = datetime.now() + timedelta(seconds=cache_duration)
                
                return result
                
            elif self.provider == "iex":
                params = {
                    "token": self.api_key
                }
                
                response = requests.get(f"{self.base_url}/stock/{symbol}/quote", params=params)
                data = response.json()
                
                result = {
                    "symbol": data.get("symbol", symbol),
                    "price": data.get("latestPrice", 0),
                    "change": data.get("change", 0),
                    "change_percent": data.get("changePercent", 0) * 100,
                    "volume": data.get("volume", 0),
                    "market_cap": data.get("marketCap", 0),
                    "pe_ratio": data.get("peRatio", None)
                }
                
                # Cache the results
                self.cache[cache_key] = result
                self.cache_expiry[cache_key] = datetime.now() + timedelta(seconds=cache_duration)
                
                return result
                
            else:
                logger.warning(f"Quotes not implemented for provider {self.provider}")
                return None
                
        except Exception as e:
            logger.exception(f"Error getting quote for {symbol}: {e}")
            return None
            
    def get_market_news(self, limit=10):
        """Get latest market news"""
        if not self.api_key:
            logger.error("API key is required for market news")
            return []
            
        cache_key = f"news_{limit}"
        if cache_key in self.cache and datetime.now() < self.cache_expiry.get(cache_key, datetime.min):
            logger.info(f"Using cached market news")
            return self.cache[cache_key]
            
        try:
            if self.provider == "finnhub":
                params = {
                    "category": "general",
                    "token": self.api_key
                }
                
                response = requests.get(f"{self.base_url}/news", params=params)
                data = response.json()
                
                if not isinstance(data, list):
                    logger.error(f"API error: {data}")
                    return []
                
                # Process the news items
                results = []
                for i, news in enumerate(data):
                    if i >= limit:
                        break
                        
                    results.append({
                        "headline": news.get("headline", ""),
                        "summary": news.get("summary", ""),
                        "source": news.get("source", ""),
                        "url": news.get("url", ""),
                        "datetime": datetime.fromtimestamp(news.get("datetime", 0))
                    })
                
                # Cache the results
                self.cache[cache_key] = results
                self.cache_expiry[cache_key] = datetime.now() + timedelta(seconds=self.cache_duration)
                
                return results
                
            elif self.provider == "iex":
                params = {
                    "token": self.api_key
                }
                
                response = requests.get(f"{self.base_url}/stock/market/news/last/{limit}", params=params)
                data = response.json()
                
                if not isinstance(data, list):
                    logger.error(f"API error: {data}")
                    return []
                
                # Process the news items
                results = []
                for news in data:
                    results.append({
                        "headline": news.get("headline", ""),
                        "summary": news.get("summary", ""),
                        "source": news.get("source", ""),
                        "url": news.get("url", ""),
                        "datetime": datetime.fromisoformat(news.get("datetime", "").replace("Z", "+00:00"))
                    })
                
                # Cache the results
                self.cache[cache_key] = results
                self.cache_expiry[cache_key] = datetime.now() + timedelta(seconds=self.cache_duration)
                
                return results
                
            else:
                logger.warning(f"Market news not implemented for provider {self.provider}")
                return []
                
        except Exception as e:
            logger.exception(f"Error getting market news: {e}")
            return []
    
    def get_economic_calendar(self, start_date=None, end_date=None):
        """Get economic event calendar"""
        if not self.api_key:
            logger.error("API key is required for economic calendar")
            return []
            
        # Set default dates if not provided
        if start_date is None:
            start_date = datetime.now().strftime("%Y-%m-%d")
        if end_date is None:
            end_date = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
            
        cache_key = f"econ_calendar_{start_date}_{end_date}"
        if cache_key in self.cache and datetime.now() < self.cache_expiry.get(cache_key, datetime.min):
            logger.info(f"Using cached economic calendar")
            return self.cache[cache_key]
            
        try:
            if self.provider == "finnhub":
                params = {
                    "from": start_date,
                    "to": end_date,
                    "token": self.api_key
                }
                
                response = requests.get(f"{self.base_url}/calendar/economic", params=params)
                data = response.json()
                
                if "economicCalendar" not in data:
                    logger.error(f"API error: {data}")
                    return []
                
                # Process the calendar events
                results = []
                for event in data["economicCalendar"]:
                    results.append({
                        "event": event.get("event", ""),
                        "time": event.get("time", ""),
                        "country": event.get("country", ""),
                        "actual": event.get("actual", None),
                        "estimate": event.get("estimate", None),
                        "impact": event.get("impact", "")
                    })
                
                # Cache the results
                self.cache[cache_key] = results
                self.cache_expiry[cache_key] = datetime.now() + timedelta(seconds=self.cache_duration)
                
                return results
                
            else:
                logger.warning(f"Economic calendar not implemented for provider {self.provider}")
                return []
                
        except Exception as e:
            logger.exception(f"Error getting economic calendar: {e}")
            return []
    
    def get_insider_trading(self, symbol, limit=10):
        """Get insider trading activity for a symbol"""
        if not self.api_key:
            logger.error("API key is required for insider trading data")
            return []
            
        cache_key = f"insider_{symbol}_{limit}"
        if cache_key in self.cache and datetime.now() < self.cache_expiry.get(cache_key, datetime.min):
            logger.info(f"Using cached insider trading data for {symbol}")
            return self.cache[cache_key]
            
        try:
            if self.provider == "finnhub":
                params = {
                    "symbol": symbol,
                    "token": self.api_key
                }
                
                response = requests.get(f"{self.base_url}/stock/insider-transactions", params=params)
                data = response.json()
                
                if "data" not in data:
                    logger.error(f"API error: {data}")
                    return []
                
                # Process the insider transactions
                results = []
                for i, transaction in enumerate(data["data"]):
                    if i >= limit:
                        break
                        
                    results.append({
                        "name": transaction.get("name", ""),
                        "share": transaction.get("share", 0),
                        "change": transaction.get("change", 0),
                        "filing_date": transaction.get("filingDate", ""),
                        "transaction_date": transaction.get("transactionDate", ""),
                        "transaction_code": transaction.get("transactionCode", ""),
                        "transaction_price": transaction.get("transactionPrice", 0)
                    })
                
                # Cache the results
                self.cache[cache_key] = results
                self.cache_expiry[cache_key] = datetime.now() + timedelta(seconds=self.cache_duration)
                
                return results
                
            else:
                logger.warning(f"Insider trading data not implemented for provider {self.provider}")
                return []
                
        except Exception as e:
            logger.exception(f"Error getting insider trading data: {e}")
            return []
            
    def get_earnings_calendar(self, from_date=None, to_date=None, symbols=None):
        """Get earnings announcements calendar"""
        if not self.api_key:
            logger.error("API key is required for earnings calendar")
            return []
            
        # Set default dates if not provided
        if from_date is None:
            from_date = datetime.now().strftime("%Y-%m-%d")
        if to_date is None:
            to_date = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
            
        symbols_str = ",".join(symbols) if symbols else "all"
        cache_key = f"earnings_calendar_{from_date}_{to_date}_{symbols_str}"
        
        if cache_key in self.cache and datetime.now() < self.cache_expiry.get(cache_key, datetime.min):
            logger.info(f"Using cached earnings calendar")
            return self.cache[cache_key]
            
        try:
            if self.provider == "finnhub":
                params = {
                    "from": from_date,
                    "to": to_date,
                    "token": self.api_key
                }
                
                if symbols:
                    params["symbol"] = symbols[0]  # Finnhub only supports one symbol at a time
                
                response = requests.get(f"{self.base_url}/calendar/earnings", params=params)
                data = response.json()
                
                if "earningsCalendar" not in data:
                    logger.error(f"API error: {data}")
                    return []
                
                # Process the earnings announcements
                results = []
                for announcement in data["earningsCalendar"]:
                    # If symbols is provided, filter results
                    if symbols and announcement.get("symbol") not in symbols:
                        continue
                        
                    results.append({
                        "symbol": announcement.get("symbol", ""),
                        "company": announcement.get("name", ""),
                        "date": announcement.get("date", ""),
                        "eps_estimate": announcement.get("epsEstimate", None),
                        "eps_actual": announcement.get("epsActual", None),
                        "revenue_estimate": announcement.get("revenueEstimate", None),
                        "revenue_actual": announcement.get("revenueActual", None),
                        "quarter": announcement.get("quarter", 0),
                        "year": announcement.get("year", 0)
                    })
                
                # Cache the results
                self.cache[cache_key] = results
                self.cache_expiry[cache_key] = datetime.now() + timedelta(seconds=self.cache_duration)
                
                return results
                
            else:
                logger.warning(f"Earnings calendar not implemented for provider {self.provider}")
                return []
                
        except Exception as e:
            logger.exception(f"Error getting earnings calendar: {e}")
            return []


# Example usage
if __name__ == "__main__":
    # Replace with your API key
    api_key = "HRNVIIR8YAH6982C"
    
    # Create a stock screener instance
    screener = StockScreener(api_key=api_key, provider="alphavantage")
    
    # Search for symbols
    results = screener.search_symbols("AAPL")
    print(f"Search results for 'AAPL': {results}")
    
    # Get historical data
    hist_data = screener.get_historical_data("AAPL", interval="daily", period="1month")
    if hist_data is not None:
        print(f"Historical data for AAPL: {hist_data.head()}")
    
    # Find correlated pairs
    pairs = screener.find_correlated_pairs(["AAPL", "MSFT", "GOOGL", "FB"], min_correlation=0.7)
    print(f"Correlated pairs: {pairs}")
    
    # Get sector performance
    sectors = screener.get_sector_performance(period="1month")
    print(f"Sector performance: {sectors}")
    
    # Get company fundamentals
    fundamentals = screener.get_company_fundamentals("AAPL", data_type="overview")
    print(f"AAPL fundamentals: {fundamentals}")