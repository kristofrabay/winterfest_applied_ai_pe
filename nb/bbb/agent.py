"""
Async tool-calling agent loop using the OpenAI Responses API with reasoning support.

Expects an AsyncOpenAI client. Tool functions (synchronous yfinance calls) are
dispatched to a thread pool via asyncio.to_thread so they don't block the event loop.
"""

import asyncio
import json

from .tools import TOOL_SCHEMAS, TOOL_FUNCTIONS


async def run_tool_calling_agent(
    client,
    model: str,
    user_prompt: str,
    system_prompt: str,
    tools: list[dict] | None = None,
    tool_functions: dict | None = None,
    max_iterations: int = 15,
    reasoning_effort: str = "medium",
    verbose: bool = True,
) -> dict:
    """
    Run a tool-calling agent loop using the OpenAI Responses API (async).

    Args:
        client: AsyncOpenAI client instance.
        verbose: If True, print reasoning summaries and tool calls to stdout.

    Returns a dict with:
      - "input": the full input list (for training data)
      - "output": the final response output items
      - "output_text": the final assistant response text
      - "reasoning_summaries": list of reasoning summary strings
      - "usage": token usage breakdown
    """
    if tools is None:
        tools = TOOL_SCHEMAS
    if tool_functions is None:
        tool_functions = TOOL_FUNCTIONS

    input_list = [
        {"role": "developer", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    reasoning_summaries = []

    for i in range(max_iterations):
        response = await client.responses.create(
            model=model,
            input=input_list,
            tools=tools,
            reasoning={"effort": reasoning_effort, "summary": "auto"},
        )

        # Collect reasoning summaries from this turn
        for item in response.output:
            if item.type == "reasoning" and getattr(item, "summary", None):
                for s in item.summary:
                    summary_text = getattr(s, "text", str(s))
                    reasoning_summaries.append(summary_text)
                    if verbose:
                        print(f"  [{i+1}] Reasoning: {summary_text[:120]}...")

        # Check if there are any function calls in this turn
        has_tool_calls = any(
            item.type == "function_call" for item in response.output
        )

        if not has_tool_calls:
            if verbose:
                print(f"  [{i+1}] Agent finished — produced final response")
            break

        # Append ALL output items (reasoning + function_calls) to preserve context
        input_list += response.output

        # Execute each function call — run sync yfinance in thread pool
        for item in response.output:
            if item.type == "function_call":
                fn_name = item.name
                fn_args = json.loads(item.arguments)

                if fn_name in tool_functions:
                    result = await asyncio.to_thread(
                        tool_functions[fn_name], **fn_args
                    )
                else:
                    result = json.dumps({"error": f"Unknown tool: {fn_name}"})

                input_list.append({
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": result,
                })

                if verbose:
                    args_str = ", ".join(
                        f"{k}={v!r}" for k, v in fn_args.items()
                    )
                    print(f"  [{i+1}] Called {fn_name}({args_str})")

    return {
        "input": input_list,
        "output": response.output,
        "output_text": response.output_text,
        "reasoning_summaries": reasoning_summaries,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "reasoning_tokens": getattr(
                getattr(response.usage, "output_tokens_details", None),
                "reasoning_tokens", 0,
            ),
        },
    }
