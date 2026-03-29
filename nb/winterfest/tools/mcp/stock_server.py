import json
import logging
from enum import Enum

import yfinance as yf
from fastmcp import FastMCP

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FinancialPeriod(str, Enum):
    annual = "annual"
    quarterly = "quarterly"


class FinancialStatementType(str, Enum):
    income = "income"
    balance_sheet = "balance_sheet"
    cashflow = "cashflow"


# Initialize FastMCP server
mcp = FastMCP(
    "stock_market",
    instructions="""
    A stock market data server providing 4 essential tools for financial analysis:
    
    1. get_stock_news - Get the 5 most recent news articles for any stock ticker
    2. get_financials - Retrieve financial statements (income, balance sheet, cashflow) 
    3. get_price_history - Get historical price data with summary statistics
    4. get_recommendations - Get analyst recommendations and upgrades/downgrades
    
    All tools accept standard stock ticker symbols (e.g., "AAPL", "GOOGL", "TSLA").
    Use these tools to provide comprehensive financial analysis and market insights.
    """
)


@mcp.tool()
async def get_stock_news(ticker: str) -> str:
    """
    Get the 5 most recent news articles for a stock ticker.
    
    This tool retrieves the latest news articles from Yahoo Finance for a given stock symbol,
    including article titles, summaries, publisher information, publication dates, and links.
    
    Args:
        ticker: Stock ticker symbol (e.g., "AAPL" for Apple, "GOOGL" for Google, "TSLA" for Tesla)
    
    Returns:
        JSON string containing:
        - ticker: The stock symbol queried
        - news_count: Number of articles returned
        - news: Array of news articles with:
            - title: Article headline
            - summary: Article summary/description
            - url: Link to full article
            - publisher: News source name
            - published: Publication date/time
    
    Example:
        >>> await get_stock_news("AAPL")
        {"ticker": "AAPL", "news_count": 5, "news": [...]}
    """
    try:
        company = yf.Ticker(ticker.upper())
        
        # Verify ticker exists
        try:
            if company.isin is None:
                return json.dumps({
                    "error": f"Ticker {ticker} not found",
                    "ticker": ticker
                })
        except Exception:
            return json.dumps({
                "error": f"Ticker {ticker} not found",
                "ticker": ticker
            })
        
        # Get news
        news = company.news[:5]  # Get first 5 articles
        
        if not news:
            return json.dumps({
                "ticker": ticker.upper(),
                "news": [],
                "message": "No news found"
            })
        
        # Format news articles
        news_list = []
        for article in news:
            # Handle both old and new yfinance news structure
            content = article.get("content", article)
            
            news_item = {
                "title": content.get("title", article.get("title", "")),
                "summary": content.get("summary", content.get("description", "")),
                "url": (
                    content.get("canonicalUrl", {}).get("url", "")
                    or content.get("clickThroughUrl", {}).get("url", "")
                    or article.get("link", "")
                ),
                "publisher": (
                    content.get("provider", {}).get("displayName", "")
                    or article.get("publisher", "")
                ),
                "published": content.get("pubDate", article.get("providerPublishTime", ""))
            }
            news_list.append(news_item)
        
        return json.dumps({
            "ticker": ticker.upper(),
            "news_count": len(news_list),
            "news": news_list
        }, indent=2)
        
    except Exception as e:
        logger.error(f"Error getting news for {ticker}: {e}")
        return json.dumps({
            "error": f"Failed to get news: {str(e)}",
            "ticker": ticker
        })


@mcp.tool()
async def get_financials(
    ticker: str,
    statement_type: FinancialStatementType,
    period: FinancialPeriod = FinancialPeriod.annual
) -> str:
    """
    Get financial statements for a stock (income statement, balance sheet, or cashflow).
    
    This tool retrieves comprehensive financial data from Yahoo Finance, allowing analysis
    of a company's financial health through its official financial statements.
    
    Args:
        ticker: Stock ticker symbol (e.g., "AAPL", "GOOGL", "TSLA")
        statement_type: Type of financial statement to retrieve:
            - "income" - Income statement (revenue, expenses, net income)
            - "balance_sheet" - Balance sheet (assets, liabilities, equity)
            - "cashflow" - Cash flow statement (operating, investing, financing activities)
        period: Reporting period (default: "annual"):
            - "annual" - Annual financial statements (yearly)
            - "quarterly" - Quarterly financial statements (every 3 months)
    
    Returns:
        JSON string containing:
        - ticker: The stock symbol queried
        - statement_type: Type of financial statement
        - period: Reporting period (annual or quarterly)
        - data: Array of financial data by date, with each period containing:
            - date: Financial reporting date
            - Various financial metrics depending on statement_type:
                * Income: Total Revenue, Gross Profit, Operating Income, Net Income, EBITDA
                * Balance Sheet: Total Assets, Total Liabilities, Stockholders Equity
                * Cashflow: Operating Cash Flow, Investing Cash Flow, Financing Cash Flow
    
    Example:
        >>> await get_financials("AAPL", "income", "annual")
        {"ticker": "AAPL", "statement_type": "income", "period": "annual", "data": [...]}
    """
    try:
        company = yf.Ticker(ticker.upper())
        
        # Verify ticker exists
        try:
            if company.isin is None:
                return json.dumps({
                    "error": f"Ticker {ticker} not found",
                    "ticker": ticker
                })
        except Exception:
            return json.dumps({
                "error": f"Ticker {ticker} not found",
                "ticker": ticker
            })
        
        # Get appropriate financial statement
        if statement_type == FinancialStatementType.income:
            df = company.quarterly_income_stmt if period == "quarterly" else company.income_stmt
        elif statement_type == FinancialStatementType.balance_sheet:
            df = company.quarterly_balance_sheet if period == "quarterly" else company.balance_sheet
        elif statement_type == FinancialStatementType.cashflow:
            df = company.quarterly_cashflow if period == "quarterly" else company.cashflow
        else:
            return json.dumps({
                "error": f"Invalid statement type: {statement_type}",
                "valid_types": ["income", "balance_sheet", "cashflow"]
            })
        
        # Check if data exists
        if df is None or df.empty:
            return json.dumps({
                "ticker": ticker.upper(),
                "statement_type": statement_type,
                "period": period,
                "data": [],
                "message": "No financial data available"
            })
        
        # Convert to JSON format (transpose so dates are rows)
        result = []
        for column in df.columns:
            date_str = column.strftime("%Y-%m-%d") if hasattr(column, "strftime") else str(column)
            date_obj = {"date": date_str}
            
            for index, value in df[column].items():
                # Convert numpy types to native Python types
                if hasattr(value, 'item'):
                    date_obj[str(index)] = None if str(value) == 'nan' else value.item()
                else:
                    date_obj[str(index)] = None if str(value) == 'nan' else value
            
            result.append(date_obj)
        
        return json.dumps({
            "ticker": ticker.upper(),
            "statement_type": statement_type,
            "period": period,
            "data": result
        }, indent=2, default=str)
        
    except Exception as e:
        logger.error(f"Error getting financials for {ticker}: {e}")
        return json.dumps({
            "error": f"Failed to get financials: {str(e)}",
            "ticker": ticker
        })


@mcp.tool()
async def get_price_history(
    ticker: str,
    period: str = "6mo",
    interval: str = "1wk"
) -> str:
    """
    Get historical stock price data with summary statistics.
    
    This tool retrieves historical OHLCV (Open, High, Low, Close, Volume) data for a stock,
    along with calculated summary statistics like price ranges and total change percentage.
    Useful for analyzing price trends, volatility, and trading patterns.
    
    Args:
        ticker: Stock ticker symbol (e.g., "AAPL", "GOOGL", "TSLA")
        period: Time period to retrieve (default: "6mo"):
            - "6mo" - 6 months
            - "1y" - 1 year
            - "2y" - 2 years
            - "5y" - 5 years
            - "10y" - 10 years
            - "max" - All available data
        interval: Data interval/granularity (default: "1wk"):
            - "1d" - Daily prices
            - "1wk" - Weekly prices
            - "1mo" - Monthly prices
    
    Returns:
        JSON string containing:
        - ticker: The stock symbol queried
        - period: Time period retrieved
        - interval: Data interval used
        - data_points: Number of price data points returned
        - summary: Summary statistics including:
            - min_price: Lowest price in period
            - max_price: Highest price in period
            - avg_price: Average price in period
            - total_change_pct: Percentage change from start to end
        - data: Array of price data with each entry containing:
            - date: Trading date
            - open: Opening price
            - high: Highest price
            - low: Lowest price
            - close: Closing price
            - volume: Trading volume
    
    Example:
        >>> await get_price_history("AAPL", period="1y", interval="1wk")
        {"ticker": "AAPL", "period": "1y", "data_points": 52, "summary": {...}, "data": [...]}
    """
    try:
        company = yf.Ticker(ticker.upper())
        
        # Verify ticker exists
        try:
            if company.isin is None:
                return json.dumps({
                    "error": f"Ticker {ticker} not found",
                    "ticker": ticker
                })
        except Exception:
            return json.dumps({
                "error": f"Ticker {ticker} not found",
                "ticker": ticker
            })
        
        # Get historical data
        hist = company.history(period=period, interval=interval)
        
        if hist.empty:
            return json.dumps({
                "ticker": ticker.upper(),
                "period": period,
                "interval": interval,
                "data": [],
                "message": "No historical data available"
            })
        
        # Convert to list of dictionaries
        hist_reset = hist.reset_index()
        hist_data = []
        
        for _, row in hist_reset.iterrows():
            data_point = {
                "date": row["Date"].isoformat() if hasattr(row["Date"], "isoformat") else str(row["Date"]),
                "open": float(row["Open"]) if str(row["Open"]) != 'nan' else None,
                "high": float(row["High"]) if str(row["High"]) != 'nan' else None,
                "low": float(row["Low"]) if str(row["Low"]) != 'nan' else None,
                "close": float(row["Close"]) if str(row["Close"]) != 'nan' else None,
                "volume": int(row["Volume"]) if str(row["Volume"]) != 'nan' else None,
            }
            hist_data.append(data_point)
        
        # Calculate summary statistics
        closes = [d["close"] for d in hist_data if d["close"] is not None]
        summary = {}
        if closes:
            summary = {
                "min_price": round(min(closes), 2),
                "max_price": round(max(closes), 2),
                "avg_price": round(sum(closes) / len(closes), 2),
                "total_change_pct": round(((closes[-1] - closes[0]) / closes[0] * 100), 2) if len(closes) > 1 else 0
            }
        
        return json.dumps({
            "ticker": ticker.upper(),
            "period": period,
            "interval": interval,
            "data_points": len(hist_data),
            "summary": summary,
            "data": hist_data
        }, indent=2)
        
    except Exception as e:
        logger.error(f"Error getting price history for {ticker}: {e}")
        return json.dumps({
            "error": f"Failed to get price history: {str(e)}",
            "ticker": ticker
        })


@mcp.tool()
async def get_recommendations(ticker: str, months_back: int = 12) -> str:
    """
    Get analyst recommendations and recent upgrades/downgrades for a stock.
    
    This tool retrieves analyst opinions, ratings, and recent changes in recommendations
    from major financial institutions and research firms. Useful for understanding market
    sentiment and professional analyst views on a stock's prospects.
    
    Args:
        ticker: Stock ticker symbol (e.g., "AAPL", "GOOGL", "TSLA")
        months_back: Number of months of historical upgrades/downgrades to retrieve (default: 12)
            - Retrieves the most recent upgrade/downgrade from each firm within this timeframe
            - Minimum: 1 month, Maximum: typically 24 months
    
    Returns:
        JSON string containing:
        - ticker: The stock symbol queried
        - recommendations: Array of recent analyst recommendations with:
            - Date: Recommendation date
            - Firm: Name of the analyst/research firm
            - To Grade: Current recommendation (e.g., "Buy", "Hold", "Sell")
            - Action: Type of action (e.g., "init", "main", "up", "down")
        - upgrades_downgrades: Array of recent rating changes with:
            - Firm: Name of the analyst/research firm
            - GradeDate: Date of the rating change
            - Action: Type of action (Upgrade, Downgrade, Initiate, Maintain, etc.)
            - FromGrade: Previous rating (if applicable)
            - ToGrade: New rating
            
            Common rating grades: Strong Buy, Buy, Outperform, Hold, Underperform, Sell, Strong Sell
    
    Example:
        >>> await get_recommendations("AAPL", months_back=6)
        {"ticker": "AAPL", "recommendations": [...], "upgrades_downgrades": [...]}
    """
    try:
        import pandas as pd
        
        company = yf.Ticker(ticker.upper())
        
        # Verify ticker exists
        try:
            if company.isin is None:
                return json.dumps({
                    "error": f"Ticker {ticker} not found",
                    "ticker": ticker
                })
        except Exception:
            return json.dumps({
                "error": f"Ticker {ticker} not found",
                "ticker": ticker
            })
        
        result = {
            "ticker": ticker.upper(),
            "recommendations": [],
            "upgrades_downgrades": []
        }
        
        # Get recommendations summary
        try:
            recs = company.recommendations
            if recs is not None and not recs.empty:
                # Get most recent recommendations
                recs_reset = recs.reset_index()
                recs_recent = recs_reset.tail(10).to_dict(orient="records")
                result["recommendations"] = recs_recent
        except Exception as e:
            logger.warning(f"Could not get recommendations: {e}")
        
        # Get upgrades/downgrades
        try:
            upgrades = company.upgrades_downgrades
            if upgrades is not None and not upgrades.empty:
                # Filter by date
                upgrades_reset = upgrades.reset_index()
                cutoff_date = pd.Timestamp.now() - pd.DateOffset(months=months_back)
                upgrades_filtered = upgrades_reset[upgrades_reset["GradeDate"] >= cutoff_date]
                
                # Sort by date descending
                upgrades_sorted = upgrades_filtered.sort_values("GradeDate", ascending=False)
                
                # Get latest by firm
                latest_by_firm = upgrades_sorted.drop_duplicates(subset=["Firm"])
                
                # Convert to dict
                upgrades_list = latest_by_firm.to_dict(orient="records")
                result["upgrades_downgrades"] = upgrades_list
        except Exception as e:
            logger.warning(f"Could not get upgrades/downgrades: {e}")
        
        return json.dumps(result, indent=2, default=str)
        
    except Exception as e:
        logger.error(f"Error getting recommendations for {ticker}: {e}")
        return json.dumps({
            "error": f"Failed to get recommendations: {str(e)}",
            "ticker": ticker
        })


if __name__ == "__main__":
    mcp.run(
        transport="http",
        port=8001
    )
