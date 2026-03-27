"""
Stock research tools — synchronous Python functions + OpenAI tool schemas.

Extracted from tools/mcp/stock_server.py for use in:
- nb/bbb/tool_calling_agent.ipynb (teacher agent)
- nb/bbb/tool_calling_data_generator.ipynb (bulk data collection)
- nb/bbb/tool_calling_baseline.ipynb (raw model evaluation)
- nb/bbb/tool_calling_rl.ipynb (RL environment)
"""

import json
import logging

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool implementations (synchronous, return JSON strings)
# ---------------------------------------------------------------------------

def get_stock_news(ticker: str) -> str:
    """Get the 5 most recent news articles for a stock ticker."""
    try:
        company = yf.Ticker(ticker.upper())
        news = company.news[:5]

        if not news:
            return json.dumps({"ticker": ticker.upper(), "news": [], "message": "No news found"})

        news_list = []
        for article in news:
            content = article.get("content", article)
            news_list.append({
                "title": content.get("title", article.get("title", "")),
                "summary": content.get("summary", content.get("description", "")),
                "publisher": (
                    content.get("provider", {}).get("displayName", "")
                    or article.get("publisher", "")
                ),
                "published": content.get("pubDate", article.get("providerPublishTime", "")),
            })

        return json.dumps({"ticker": ticker.upper(), "news_count": len(news_list), "news": news_list}, indent=2)
    except Exception as e:
        logger.error(f"Error getting news for {ticker}: {e}")
        return json.dumps({"error": f"Failed to get news: {e}", "ticker": ticker})


def get_financials(ticker: str, statement_type: str = "income", period: str = "annual") -> str:
    """Get financial statements (income, balance_sheet, or cashflow) for a stock."""
    try:
        company = yf.Ticker(ticker.upper())

        stmt_map = {
            ("income", "annual"): company.income_stmt,
            ("income", "quarterly"): company.quarterly_income_stmt,
            ("balance_sheet", "annual"): company.balance_sheet,
            ("balance_sheet", "quarterly"): company.quarterly_balance_sheet,
            ("cashflow", "annual"): company.cashflow,
            ("cashflow", "quarterly"): company.quarterly_cashflow,
        }

        df = stmt_map.get((statement_type, period))
        if df is None or df.empty:
            return json.dumps({
                "ticker": ticker.upper(), "statement_type": statement_type,
                "period": period, "data": [], "message": "No financial data available",
            })

        result = []
        for column in df.columns:
            date_str = column.strftime("%Y-%m-%d") if hasattr(column, "strftime") else str(column)
            date_obj = {"date": date_str}
            for index, value in df[column].items():
                if hasattr(value, "item"):
                    date_obj[str(index)] = None if str(value) == "nan" else value.item()
                else:
                    date_obj[str(index)] = None if str(value) == "nan" else value
            result.append(date_obj)

        return json.dumps({
            "ticker": ticker.upper(), "statement_type": statement_type,
            "period": period, "data": result,
        }, indent=2, default=str)
    except Exception as e:
        logger.error(f"Error getting financials for {ticker}: {e}")
        return json.dumps({"error": f"Failed to get financials: {e}", "ticker": ticker})


def get_price_history(ticker: str, period: str = "6mo", interval: str = "1wk") -> str:
    """Get historical stock price data with summary statistics."""
    try:
        company = yf.Ticker(ticker.upper())
        hist = company.history(period=period, interval=interval)

        if hist.empty:
            return json.dumps({
                "ticker": ticker.upper(), "period": period, "interval": interval,
                "data": [], "message": "No historical data available",
            })

        hist_reset = hist.reset_index()
        hist_data = []
        for _, row in hist_reset.iterrows():
            hist_data.append({
                "date": row["Date"].isoformat() if hasattr(row["Date"], "isoformat") else str(row["Date"]),
                "open": round(float(row["Open"]), 2) if str(row["Open"]) != "nan" else None,
                "high": round(float(row["High"]), 2) if str(row["High"]) != "nan" else None,
                "low": round(float(row["Low"]), 2) if str(row["Low"]) != "nan" else None,
                "close": round(float(row["Close"]), 2) if str(row["Close"]) != "nan" else None,
                "volume": int(row["Volume"]) if str(row["Volume"]) != "nan" else None,
            })

        closes = [d["close"] for d in hist_data if d["close"] is not None]
        summary = {}
        if closes:
            summary = {
                "min_price": round(min(closes), 2),
                "max_price": round(max(closes), 2),
                "avg_price": round(sum(closes) / len(closes), 2),
                "total_change_pct": round(((closes[-1] - closes[0]) / closes[0] * 100), 2) if len(closes) > 1 else 0,
            }

        return json.dumps({
            "ticker": ticker.upper(), "period": period, "interval": interval,
            "data_points": len(hist_data), "summary": summary, "data": hist_data,
        }, indent=2)
    except Exception as e:
        logger.error(f"Error getting price history for {ticker}: {e}")
        return json.dumps({"error": f"Failed to get price history: {e}", "ticker": ticker})


def get_recommendations(ticker: str, months_back: int = 12) -> str:
    """Get analyst recommendations and recent upgrades/downgrades for a stock."""
    try:
        company = yf.Ticker(ticker.upper())
        result = {"ticker": ticker.upper(), "recommendations": [], "upgrades_downgrades": []}

        try:
            recs = company.recommendations
            if recs is not None and not recs.empty:
                recs_reset = recs.reset_index()
                result["recommendations"] = recs_reset.tail(10).to_dict(orient="records")
        except Exception as e:
            logger.warning(f"Could not get recommendations: {e}")

        try:
            upgrades = company.upgrades_downgrades
            if upgrades is not None and not upgrades.empty:
                upgrades_reset = upgrades.reset_index()
                cutoff_date = pd.Timestamp.now() - pd.DateOffset(months=months_back)
                upgrades_filtered = upgrades_reset[upgrades_reset["GradeDate"] >= cutoff_date]
                upgrades_sorted = upgrades_filtered.sort_values("GradeDate", ascending=False)
                latest_by_firm = upgrades_sorted.drop_duplicates(subset=["Firm"])
                result["upgrades_downgrades"] = latest_by_firm.to_dict(orient="records")
        except Exception as e:
            logger.warning(f"Could not get upgrades/downgrades: {e}")

        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        logger.error(f"Error getting recommendations for {ticker}: {e}")
        return json.dumps({"error": f"Failed to get recommendations: {e}", "ticker": ticker})


# ---------------------------------------------------------------------------
# Function registry — maps function names to callables
# ---------------------------------------------------------------------------

TOOL_FUNCTIONS = {
    "get_stock_news": get_stock_news,
    "get_financials": get_financials,
    "get_price_history": get_price_history,
    "get_recommendations": get_recommendations,
}


# ---------------------------------------------------------------------------
# OpenAI-compatible tool schemas
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_news",
            "description": "Get the 5 most recent news articles for a stock ticker from Yahoo Finance.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol, e.g. 'AAPL', 'NVDA'"},
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_financials",
            "description": "Get financial statements (income statement, balance sheet, or cash flow) for a stock.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol, e.g. 'AAPL', 'NVDA'"},
                    "statement_type": {
                        "type": "string",
                        "enum": ["income", "balance_sheet", "cashflow"],
                        "description": "Type of financial statement to retrieve.",
                    },
                    "period": {
                        "type": "string",
                        "enum": ["annual", "quarterly"],
                        "description": "Reporting period. Defaults to 'annual'.",
                    },
                },
                "required": ["ticker", "statement_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_price_history",
            "description": "Get historical stock price data (OHLCV) with summary statistics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol, e.g. 'AAPL', 'NVDA'"},
                    "period": {
                        "type": "string",
                        "enum": ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
                        "description": "Time period to retrieve. Defaults to '6mo'.",
                    },
                    "interval": {
                        "type": "string",
                        "enum": ["1d", "1wk", "1mo"],
                        "description": "Data granularity. Defaults to '1wk'.",
                    },
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recommendations",
            "description": "Get analyst recommendations and recent upgrades/downgrades for a stock.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol, e.g. 'AAPL', 'NVDA'"},
                    "months_back": {
                        "type": "integer",
                        "description": "Number of months of history to retrieve. Defaults to 12.",
                    },
                },
                "required": ["ticker"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Agent loop — works with any OpenAI-compatible endpoint
# ---------------------------------------------------------------------------

def run_tool_calling_agent(
    client,
    model: str,
    user_prompt: str,
    system_prompt: str,
    tools: list[dict] | None = None,
    tool_functions: dict | None = None,
    max_iterations: int = 15,
) -> list[dict]:
    """
    Run a tool-calling agent loop. Works with OpenAI API or any
    OpenAI-compatible endpoint (vLLM, llama-server, etc.).

    Returns the full message history (the training trajectory).
    """
    if tools is None:
        tools = TOOL_SCHEMAS
    if tool_functions is None:
        tool_functions = TOOL_FUNCTIONS

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    for i in range(max_iterations):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

        choice = response.choices[0]
        assistant_msg = {"role": "assistant", "content": choice.message.content or ""}

        # Check for tool calls
        if choice.message.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in choice.message.tool_calls
            ]
            messages.append(assistant_msg)

            # Execute each tool call
            for tc in choice.message.tool_calls:
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments)

                if fn_name in tool_functions:
                    result = tool_functions[fn_name](**fn_args)
                else:
                    result = json.dumps({"error": f"Unknown tool: {fn_name}"})

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

                print(f"  [{i+1}] Called {fn_name}({', '.join(f'{k}={v!r}' for k, v in fn_args.items())})")
        else:
            # No tool calls — model is done
            messages.append(assistant_msg)
            print(f"  [{i+1}] Agent finished — produced final response")
            break

    return messages
