"""
Data generation helpers for Phase 2 of the BBB pipeline.

- Ticker list and research focus prompts for diverse trajectory generation
- Serialization of Responses API SDK objects to dicts
- Truncation of tool outputs (for SFT — masked tokens = zero gradient, pure overhead)
- Conversion from Responses API format → Hermes/chat format (what Qwen3 tokenizer expects)
- Quality filtering for training data
"""

import json
import random
from datetime import datetime

import tiktoken

# ---------------------------------------------------------------------------
# Token counting — used for truncation
# ---------------------------------------------------------------------------

_ENCODING = tiktoken.get_encoding("o200k_base")


# ---------------------------------------------------------------------------
# System prompt — shared between Phase 1 (teacher) and Phase 2 (data gen)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = f"""\
You are a sell-side equity research analyst producing a brief research snapshot.

Today's date is {datetime.now().strftime("%Y-%m-%d")}.

## Instructions
- Use the available tools to gather the data you need. Be efficient — call only what's necessary.
- Think briefly about what data points matter most, then call tools.
- After gathering data, produce a concise equity research snapshot.

## Output Format
Produce a brief **Equity Research Snapshot** (~half page). Structure:

**{{COMPANY}} ({{TICKER}})** | {{Sector}} | {{Market Cap}}

**Key Metrics:** Revenue (TTM), EPS, P/E, margins, debt/equity — whatever is most relevant.
**Recent Developments:** 1-3 bullet points on material news, earnings, or events.
**Financial Highlights:** Key takeaways from the most recent financials.
**Price Action:** Current price context, 52-week range, recent trend summary.
**Analyst Consensus:** Target price, buy/hold/sell split, notable recent upgrades or downgrades.
**Bull Case:** 2-3 bullets.
**Bear Case:** 2-3 bullets.
**Bottom Line:** One-sentence takeaway.

## Important
- Use ONLY the tools provided. Do not fabricate data.
- Be concise — this is a flash note, not a full report.
- Include specific numbers: prices, percentages, ratios. No vague statements.
- If a tool returns an error, note it and move on.
- Keep your internal reasoning brief — a few sentences of planning, not lengthy analysis.
"""


# ---------------------------------------------------------------------------
# Tickers — ~200 diverse publicly traded companies across sectors
# ---------------------------------------------------------------------------

TICKERS = [
    # Mega-cap tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AVGO",
    "ORCL", "CRM", "AMD", "INTC", "CSCO", "ADBE", "NFLX",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "USB",
    "PNC", "TFC", "COF", "BK", "STT", "ICE", "CME", "SPGI", "MCO", "AIG",
    # Healthcare
    "JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY", "TMO", "ABT", "DHR", "BMY",
    "AMGN", "GILD", "ISRG", "MDT", "SYK", "CVS", "CI", "HUM", "ELV", "ZTS",
    # Consumer staples & discretionary
    "WMT", "PG", "KO", "PEP", "COST", "MCD", "SBUX", "NKE", "TGT", "HD",
    "LOW", "TJX", "DG", "EL", "CL", "KMB", "GIS", "SJM", "MNST", "HSY",
    # Energy
    "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "VLO", "PSX", "OXY", "HES",
    "DVN", "FANG", "HAL", "BKR", "KMI",
    # Industrials
    "CAT", "DE", "HON", "UNP", "UPS", "FDX", "GE", "RTX", "LMT", "NOC",
    "BA", "MMM", "EMR", "ITW", "PH", "ETN", "WM", "RSG", "FAST", "GD",
    # Real estate
    "AMT", "PLD", "CCI", "EQIX", "SPG", "O", "VICI", "PSA", "DLR", "WELL",
    # Communication & media
    "DIS", "CMCSA", "T", "VZ", "TMUS", "CHTR", "WBD", "FOX", "OMC", "IPG",
    # Materials
    "LIN", "APD", "SHW", "ECL", "DD", "NEM", "FCX", "NUE", "STLD", "VMC",
    # Utilities
    "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL", "WEC", "ED",
    # High-growth / mid-cap tech
    "CRWD", "SNOW", "DDOG", "NET", "ZS", "MDB", "PANW", "FTNT", "HUBS", "TTD",
    "WDAY", "VEEV", "ABNB", "DASH", "RBLX", "DKNG", "BILL", "OKTA",
    # Fintech & payments
    "SQ", "PYPL", "COIN", "SOFI", "HOOD", "MELI", "NU",
    # Auto & transport
    "F", "GM", "RIVN", "LUV", "DAL", "UAL", "UBER", "LYFT",
    # International ADRs
    "TSM", "BABA", "NVO", "ASML", "SAP", "TM", "SONY", "SHOP",
]

FOCUS_AREAS = [
    "financial health and balance sheet strength",
    "growth potential and market expansion opportunities",
    "competitive position and market share dynamics",
    "recent news and analyst sentiment",
    "revenue trends and profitability trajectory",
    "cash flow generation and capital allocation strategy",
    "valuation relative to peers and historical trends",
    "risk factors and key challenges ahead",
]


def make_user_prompt(ticker: str, focus: str | None = None) -> str:
    """Build a user prompt for the teacher agent."""
    if focus is None:
        focus = random.choice(FOCUS_AREAS)
    return f"Research {ticker} focusing on {focus}."


# ---------------------------------------------------------------------------
# Serialization — convert SDK objects to JSON-serializable dicts
# ---------------------------------------------------------------------------

def serialize_input_list(input_list: list) -> list[dict]:
    """Convert a mix of dicts and OpenAI SDK response objects to JSON-serializable dicts."""
    serialized = []
    for item in input_list:
        if isinstance(item, dict):
            serialized.append(item)
        elif hasattr(item, "model_dump"):
            serialized.append(item.model_dump())
        else:
            serialized.append({"type": "unknown", "data": str(item)})
    return serialized


# ---------------------------------------------------------------------------
# Truncation — reduce tool output tokens for SFT
# ---------------------------------------------------------------------------

def truncate_tool_output(output: str, max_tokens: int = 600) -> str:
    """
    Truncate a tool output string to max_tokens.

    Tool results are masked (labels=-100) during SFT, so they contribute zero
    gradient — they're pure context overhead. Keeping them short frees up
    sequence length for the assistant turns that the model actually learns from.
    """
    tokens = _ENCODING.encode(output)
    if len(tokens) <= max_tokens:
        return output
    truncated = _ENCODING.decode(tokens[:max_tokens])
    return truncated + "\n... [truncated]"


def count_tokens(text: str) -> int:
    """Count tokens in a string using o200k_base encoding."""
    return len(_ENCODING.encode(text))


# ---------------------------------------------------------------------------
# Format conversion: Responses API → Hermes / Chat Completions
# ---------------------------------------------------------------------------

def _responses_tool_to_chat(tool: dict) -> dict:
    """
    Convert a Responses API flat tool schema to Chat Completions nested format.

    Responses API:  {"type": "function", "name": "...", "parameters": {...}, ...}
    Chat Completions: {"type": "function", "function": {"name": "...", "parameters": {...}}}
    """
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool["parameters"],
            **({"strict": tool["strict"]} if "strict" in tool else {}),
        },
    }


def responses_to_hermes(
    raw_record: dict,
    max_tool_tokens: int = 600,
) -> dict | None:
    """
    Convert a saved Responses API trajectory to Hermes/chat format for SFT.

    Expects a dict with keys: input, output, tools.
    Returns: {"messages": [...], "tools": [...]} or None if conversion fails.

    The conversion:
      - developer → system
      - reasoning items → <think> tags in assistant content
      - message items → assistant text content
      - function_call items → assistant tool_calls array
      - function_call_output → role: tool messages (with truncation)
      - Tool schemas: flat → nested format
    """
    all_items = list(raw_record["input"]) + list(raw_record["output"])
    tool_schemas = [_responses_tool_to_chat(t) for t in raw_record["tools"]]

    messages = []
    i = 0

    while i < len(all_items):
        item = all_items[i]

        # --- System / developer message ---
        if isinstance(item, dict) and item.get("role") in ("developer", "system"):
            messages.append({"role": "system", "content": item["content"]})
            i += 1
            continue

        # --- User message ---
        if isinstance(item, dict) and item.get("role") == "user":
            messages.append({"role": "user", "content": item["content"]})
            i += 1
            continue

        # --- Tool result ---
        if isinstance(item, dict) and item.get("type") == "function_call_output":
            messages.append({
                "role": "tool",
                "tool_call_id": item["call_id"],
                "content": truncate_tool_output(item["output"], max_tool_tokens),
            })
            i += 1
            continue

        # --- Assistant turn: accumulate reasoning + message + function_calls ---
        think_parts: list[str] = []
        text_parts: list[str] = []
        tool_calls: list[dict] = []

        while i < len(all_items):
            it = all_items[i]
            it_type = it.get("type") if isinstance(it, dict) else None

            # Stop at boundaries: tool results, user, system
            if isinstance(it, dict) and (
                it_type == "function_call_output"
                or it.get("role") in ("developer", "system", "user")
            ):
                break

            if it_type == "reasoning":
                for s in it.get("summary") or []:
                    text = s.get("text", "") if isinstance(s, dict) else str(s)
                    if text:
                        think_parts.append(text)

            elif it_type == "function_call":
                tool_calls.append({
                    "id": it.get("call_id", f"call_{len(tool_calls)}"),
                    "type": "function",
                    "function": {
                        "name": it["name"],
                        "arguments": it.get("arguments", "{}"),
                    },
                })

            elif it_type == "message":
                for c in it.get("content") or []:
                    if isinstance(c, dict) and c.get("text"):
                        text_parts.append(c["text"])

            i += 1

        # Build assistant content
        content_parts = []
        if think_parts:
            joined_thinking = "\n\n".join(think_parts)
            content_parts.append(f"<think>\n{joined_thinking}\n</think>")
        content_parts.extend(text_parts)

        content = "\n\n".join(content_parts) if content_parts else None

        msg: dict = {"role": "assistant"}
        if content:
            msg["content"] = content
        if tool_calls:
            msg["tool_calls"] = tool_calls
        # Skip empty assistant messages (no content, no tool_calls)
        if content or tool_calls:
            messages.append(msg)

    return {"messages": messages, "tools": tool_schemas}


# ---------------------------------------------------------------------------
# Quality filtering
# ---------------------------------------------------------------------------

def filter_trajectory(raw_record: dict, min_memo_chars: int = 200) -> tuple[bool, str]:
    """
    Check if a trajectory meets quality standards for SFT training.

    Returns (passes: bool, reason: str).
    """
    all_items = list(raw_record["input"]) + list(raw_record["output"])

    # 1. Must have at least one tool call
    tool_calls = [
        it for it in all_items
        if isinstance(it, dict) and it.get("type") == "function_call"
    ]
    if not tool_calls:
        return False, "no_tool_calls"

    # 2. All tool calls must have valid JSON arguments
    for tc in tool_calls:
        try:
            json.loads(tc.get("arguments", "{}"))
        except (json.JSONDecodeError, TypeError):
            return False, "malformed_tool_args"

    # 3. Must have a final text output (the memo)
    final_text = ""
    for it in reversed(all_items):
        if isinstance(it, dict) and it.get("type") == "message":
            for c in it.get("content") or []:
                if isinstance(c, dict) and c.get("text"):
                    final_text = c["text"]
                    break
            if final_text:
                break

    if len(final_text) < min_memo_chars:
        return False, f"memo_too_short ({len(final_text)} chars)"

    # 4. Should use at least 2 different tools
    tool_names = {tc["name"] for tc in tool_calls}
    if len(tool_names) < 2:
        return False, f"too_few_tools ({tool_names})"

    return True, "ok"


# ---------------------------------------------------------------------------
# Token stats for a converted (Hermes) trajectory
# ---------------------------------------------------------------------------

def trajectory_token_stats(hermes_record: dict) -> dict:
    """Compute token counts per role for a Hermes-format trajectory."""
    stats = {"system": 0, "user": 0, "assistant": 0, "tool": 0, "total": 0}
    for msg in hermes_record["messages"]:
        role = msg["role"]
        tokens = 0
        if msg.get("content"):
            tokens += count_tokens(msg["content"])
        if msg.get("tool_calls"):
            tokens += count_tokens(json.dumps(msg["tool_calls"]))
        stats[role] = stats.get(role, 0) + tokens
        stats["total"] += tokens
    return stats
