#!/usr/bin/env python3
"""
Command-line interface for stock screening in the Mainstreet HFT project.
"""

import argparse
import logging
import sys
import json
from stock_screener import StockScreener
from tabulate import tabulate
import pandas as pd
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Mainstreet HFT Stock Screener")
    
    parser.add_argument("--api-key", type=str, required=True,
                        help="API key for stock data provider")
    parser.add_argument("--provider", type=str, default="alphavantage",
                        choices=["alphavantage", "finnhub", "iex"],
                        help="Stock data provider (default: alphavantage)")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search for symbols")
    search_parser.add_argument("--keywords", type=str, required=True,
                             help="Keywords to search for")
    
    # Gainers command
    gainers_parser = subparsers.add_parser("gainers", help="Get top gainers")
    gainers_parser.add_argument("--limit", type=int, default=10,
                              help="Number of results to return (default: 10)")
    
    # Losers command
    losers_parser = subparsers.add_parser("losers", help="Get top losers")
    losers_parser.add_argument("--limit", type=int, default=10,
                              help="Number of results to return (default: 10)")
    
    # Historical data command
    history_parser = subparsers.add_parser("history", help="Get historical data")
    history_parser.add_argument("--symbol", type=str, required=True,
                              help="Symbol to get historical data for")
    history_parser.add_argument("--interval", type=str, default="daily",
                              choices=["daily", "hourly", "1min", "5min", "15min", "30min", "60min"],
                              help="Data interval (default: daily)")
    history_parser.add_argument("--period", type=str, default="1month",
                              choices=["1month", "3months", "6months", "1year"],
                              help="How far back to go (default: 1month)")
    history_parser.add_argument("--output", type=str, default="table",
                              choices=["table", "csv", "json"],
                              help="Output format (default: table)")
    
    # Correlation command
    corr_parser = subparsers.add_parser("correlate", help="Find correlated pairs")
    corr_parser.add_argument("--symbols", type=str, required=True,
                           help="Comma-separated list of symbols")
    corr_parser.add_argument("--min-correlation", type=float, default=0.7,
                           help="Minimum correlation threshold (default: 0.7)")
    corr_parser.add_argument("--lookback", type=int, default=60,
                           help="Lookback period in days (default: 60)")
    
    # Sector performance command
    sector_parser = subparsers.add_parser("sectors", help="Get sector performance")
    sector_parser.add_argument("--period", type=str, default="1day",
                             choices=["1day", "5day", "1month", "3month", "ytd", "1year"],
                             help="Performance period (default: 1day)")
    
    # Fundamentals command
    fundamentals_parser = subparsers.add_parser("fundamentals", help="Get company fundamentals")
    fundamentals_parser.add_argument("--symbol", type=str, required=True,
                                   help="Symbol to get fundamentals for")
    fundamentals_parser.add_argument("--type", type=str, default="overview",
                                   choices=["overview", "income", "balance", "cash", "earnings"],
                                   help="Type of fundamental data (default: overview)")
    
    # Screener command
    screener_parser = subparsers.add_parser("screen", help="Screen stocks based on criteria")
    screener_parser.add_argument("--criteria", type=str, required=True,
                               help="JSON string or path to JSON file with screening criteria")
    screener_parser.add_argument("--limit", type=int, default=20,
                               help="Maximum number of results (default: 20)")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export data for multiple symbols")
    export_parser.add_argument("--symbols", type=str, required=True,
                             help="Comma-separated list of symbols")
    export_parser.add_argument("--type", type=str, default="price",
                             choices=["price", "volume", "fundamentals", "all"],
                             help="Type of data to export (default: price)")
    export_parser.add_argument("--days", type=int, default=30,
                             help="Number of days of historical data (default: 30)")
    export_parser.add_argument("--output", type=str, required=True,
                             help="Output file path (.csv or .xlsx)")
    
    return parser.parse_args()

def format_number(value):
    """Format large numbers for display"""
    if value is None:
        return "N/A"
    
    try:
        value = float(value)
    except (ValueError, TypeError):
        return str(value)
    
    if abs(value) >= 1_000_000_000:
        return f"${value/1_000_000_000:.2f}B"
    elif abs(value) >= 1_000_000:
        return f"${value/1_000_000:.2f}M"
    elif abs(value) >= 1_000:
        return f"${value/1_000:.2f}K"
    else:
        return f"${value:.2f}"

def load_criteria_json(criteria_arg):
    """Load screening criteria from JSON string or file"""
    if criteria_arg.startswith('{'):
        # Treat as JSON string
        try:
            return json.loads(criteria_arg)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON string: {e}")
            return {}
    else:
        # Treat as file path
        try:
            with open(criteria_arg, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.error(f"Error loading JSON file: {e}")
            return {}

def main():
    """Main entry point for the stock screener CLI"""
    args = parse_arguments()
    
    if not args.command:
        logger.error("No command specified. Use --help for usage information.")
        return 1
    
    # Initialize stock screener
    screener = StockScreener(api_key=args.api_key, provider=args.provider)
    
    # Execute the appropriate command
    if args.command == "search":
        results = screener.search_symbols(args.keywords)
        if results:
            headers = ["Symbol", "Name", "Type", "Region"]
            table_data = []
            for result in results:
                row = [
                    result.get("symbol", ""),
                    result.get("name", ""),
                    result.get("type", ""),
                    result.get("region", "")
                ]
                table_data.append(row)
            
            print(tabulate(table_data, headers=headers, tablefmt="grid"))
            print(f"Found {len(results)} results for '{args.keywords}'")
        else:
            print(f"No results found for '{args.keywords}'")
    
    elif args.command == "gainers":
        results = screener.get_top_gainers(args.limit)
        if results:
            headers = ["Symbol", "Name", "Price", "Change %"]
            table_data = []
            for result in results:
                row = [
                    result.get("symbol", ""),
                    result.get("name", ""),
                    result.get("price", 0),
                    f"{result.get('change_pct', 0):.2f}%"
                ]
                table_data.append(row)
            
            print(tabulate(table_data, headers=headers, tablefmt="grid"))
            print(f"Top {len(results)} gainers")
        else:
            print("No gainers found")
    
    elif args.command == "losers":
        results = screener.get_top_losers(args.limit)
        if results:
            headers = ["Symbol", "Name", "Price", "Change %"]
            table_data = []
            for result in results:
                row = [
                    result.get("symbol", ""),
                    result.get("name", ""),
                    result.get("price", 0),
                    f"{result.get('change_pct', 0):.2f}%"
                ]
                table_data.append(row)
            
            print(tabulate(table_data, headers=headers, tablefmt="grid"))
            print(f"Top {len(results)} losers")
        else:
            print("No losers found")
    
    elif args.command == "history":
        df = screener.get_historical_data(args.symbol, args.interval, args.period)
        if df is not None:
            if args.output == "table":
                # Print as a table
                print(tabulate(df.tail(20), headers="keys", tablefmt="grid"))
                print(f"Showing last 20 rows of {len(df)} total rows")
            elif args.output == "csv":
                # Print as CSV
                print(df.to_csv(index=False))
            elif args.output == "json":
                # Print as JSON
                print(df.to_json(orient="records"))
        else:
            print(f"No historical data found for '{args.symbol}'")
    
    elif args.command == "correlate":
        symbols = [s.strip() for s in args.symbols.split(",")]
        results = screener.find_correlated_pairs(symbols, args.lookback, args.min_correlation)
        if results:
            headers = ["Symbol 1", "Symbol 2", "Correlation"]
            table_data = []
            for result in results:
                row = [
                    result.get("symbol1", ""),
                    result.get("symbol2", ""),
                    f"{result.get('correlation', 0):.4f}"
                ]
                table_data.append(row)
            
            print(tabulate(table_data, headers=headers, tablefmt="grid"))
            print(f"Found {len(results)} correlated pairs")
        else:
            print("No correlated pairs found")
    
    elif args.command == "sectors":
        sectors = screener.get_sector_performance(args.period)
        if sectors:
            headers = ["Sector", "Performance"]
            table_data = []
            for sector, performance in sectors.items():
                row = [
                    sector,
                    f"{performance:.2f}%"
                ]
                table_data.append(row)
            
            # Sort by performance (descending)
            table_data.sort(key=lambda x: float(x[1].strip('%')), reverse=True)
            
            print(tabulate(table_data, headers=headers, tablefmt="grid"))
            print(f"Sector performance for {args.period}")
        else:
            print("No sector performance data found")
    
    elif args.command == "fundamentals":
        fundamentals = screener.get_company_fundamentals(args.symbol, args.type)
        if fundamentals:
            if args.type == "overview":
                # Print company overview as a table
                table_data = []
                for key, value in fundamentals.items():
                    if key in ['MarketCapitalization', 'EBITDA', 'RevenueTTM', 'GrossProfitTTM',
                               'QuarterlyEarningsGrowthYOY', 'QuarterlyRevenueGrowthYOY']:
                        value = format_number(value)
                    row = [key, value]
                    table_data.append(row)
                
                print(tabulate(table_data, headers=["Metric", "Value"], tablefmt="grid"))
                print(f"Company fundamentals for {args.symbol}")
            else:
                # For other types, print as JSON
                print(json.dumps(fundamentals, indent=2))
        else:
            print(f"No fundamental data found for '{args.symbol}'")
    
    elif args.command == "screen":
        criteria = load_criteria_json(args.criteria)
        if not criteria:
            print("No valid screening criteria provided")
            return 1
            
        results = screener.screen_stocks(criteria)
        if results:
            # Limit results if needed
            results = results[:args.limit]
            
            # Get common keys for headers
            all_keys = set()
            for result in results:
                all_keys.update(result.keys())
            
            # Prioritize important keys
            key_order = ['symbol', 'name', 'price', 'change_pct', 'volume', 'market_cap', 'pe_ratio']
            headers = [k for k in key_order if k in all_keys]
            headers.extend([k for k in all_keys if k not in key_order])
            
            # Prepare table data
            table_data = []
            for result in results:
                row = []
                for header in headers:
                    value = result.get(header, "")
                    if header in ['market_cap']:
                        value = format_number(value)
                    elif header in ['change_pct']:
                        value = f"{value:.2f}%" if value is not None else ""
                    row.append(value)
                table_data.append(row)
            
            print(tabulate(table_data, headers=headers, tablefmt="grid"))
            print(f"Found {len(results)} stocks matching criteria")
        else:
            print("No stocks found matching criteria")
    
    elif args.command == "export":
        symbols = [s.strip() for s in args.symbols.split(",")]
        
        # Initialize output data
        output_data = {}
        
        for symbol in symbols:
            print(f"Processing {symbol}...")
            
            # Get historical data
            hist_data = None
            if args.type in ["price", "volume", "all"]:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=args.days)
                hist_data = screener.get_historical_data(symbol, "daily", f"{args.days}days")
            
            # Get fundamentals data
            fundamentals = None
            if args.type in ["fundamentals", "all"]:
                fundamentals = screener.get_company_fundamentals(symbol, "overview")
            
            # Store data for this symbol
            output_data[symbol] = {
                "historical": hist_data,
                "fundamentals": fundamentals
            }
        
        # Determine output format based on file extension
        output_format = args.output.split(".")[-1].lower()
        
        if output_format == "csv":
            # For CSV, we'll output historical data for all symbols
            all_hist_data = []
            for symbol, data in output_data.items():
                if data["historical"] is not None:
                    # Add symbol column
                    hist_df = data["historical"].copy()
                    hist_df["symbol"] = symbol
                    all_hist_data.append(hist_df)
            
            if all_hist_data:
                combined_df = pd.concat(all_hist_data)
                combined_df.to_csv(args.output, index=False)
                print(f"Exported data to {args.output}")
            else:
                print("No data to export")
                
        elif output_format in ["xlsx", "xls"]:
            # For Excel, we can output multiple sheets
            with pd.ExcelWriter(args.output) as writer:
                # Historical data sheet for each symbol
                for symbol, data in output_data.items():
                    if data["historical"] is not None:
                        data["historical"].to_excel(writer, sheet_name=f"{symbol}_historical", index=False)
                
                # Fundamentals sheet
                if args.type in ["fundamentals", "all"]:
                    fund_data = []
                    for symbol, data in output_data.items():
                        if data["fundamentals"] is not None:
                            # Convert dict to series
                            fund_series = pd.Series(data["fundamentals"])
                            fund_series.name = symbol
                            fund_data.append(fund_series)
                    
                    if fund_data:
                        fund_df = pd.concat(fund_data, axis=1)
                        fund_df.to_excel(writer, sheet_name="Fundamentals")
                
                print(f"Exported data to {args.output}")
        else:
            print(f"Unsupported output format: {output_format}")
            return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())