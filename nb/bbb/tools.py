"""
Stock research tools — synchronous Python functions + auto-generated OpenAI tool schemas.

Schemas are derived from function signatures, type hints, and docstrings so
you only maintain one source of truth. Changing a function automatically
updates its schema.

Extracted from tools/mcp/stock_server.py for use in all nb/bbb/ notebooks.
Agent loop is in agent.py.
"""

import inspect
import json
import logging
import typing

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema auto-generation from function signatures
# ---------------------------------------------------------------------------

# Type hint → JSON Schema type mapping
_TYPE_MAP = {str: "string", int: "integer", float: "number", bool: "boolean"}


def _build_tool_schema(fn, *, enums: dict[str, list[str]] | None = None) -> dict:
    """
    Build an OpenAI Responses API tool schema from a Python function.

    Uses inspect.signature for param names/defaults/types and the docstring
    for the function description. Pass `enums` to constrain specific params
    to a fixed set of values, e.g. {"period": ["annual", "quarterly"]}.
    """
    enums = enums or {}
    sig = inspect.signature(fn)
    props = {}
    required = []

    for name, param in sig.parameters.items():
        annotation = param.annotation
        # Resolve Optional[X], Union[X, None], etc.
        origin = typing.get_origin(annotation)
        if origin is typing.Union:
            args = [a for a in typing.get_args(annotation) if a is not type(None)]
            annotation = args[0] if args else str

        json_type = _TYPE_MAP.get(annotation, "string")
        prop: dict = {"type": json_type}

        if name in enums:
            prop["enum"] = enums[name]

        # Use default value as description hint
        if param.default is not inspect.Parameter.empty:
            prop["description"] = f"Defaults to {param.default!r}."

        props[name] = prop
        # strict mode requires all params listed in required
        required.append(name)

    return {
        "type": "function",
        "name": fn.__name__,
        "description": (fn.__doc__ or "").strip(),
        "parameters": {
            "type": "object",
            "properties": props,
            "required": required,
            "additionalProperties": False,
        },
        "strict": True,
    }


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
# OpenAI Responses API tool schemas — auto-generated from functions above
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    _build_tool_schema(get_stock_news),
    _build_tool_schema(get_financials, enums={
        "statement_type": ["income", "balance_sheet", "cashflow"],
        "period": ["annual", "quarterly"],
    }),
    _build_tool_schema(get_price_history, enums={
        "period": ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
        "interval": ["1d", "1wk", "1mo"],
    }),
    _build_tool_schema(get_recommendations),
]


