"""
Async tool-calling agent loops.

Two variants:
  - run_tool_calling_agent()      — OpenAI Responses API (reasoning models, GPT-5.x)
  - run_tool_calling_agent_chat() — Chat Completions API (local servers, Ollama, mlx-lm)

Both are async. Tool functions (synchronous yfinance calls) are dispatched to a
thread pool via asyncio.to_thread so they don't block the event loop.
"""

import asyncio
import json

from .tools import TOOL_SCHEMAS, TOOL_FUNCTIONS
from .helpers__data_gen import _responses_tool_to_chat


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


# ---------------------------------------------------------------------------
# Chat Completions API variant (Ollama, llama-server, mlx-lm, etc.)
# ---------------------------------------------------------------------------

# Auto-generate Chat Completions tool schemas from the Responses API ones
TOOL_SCHEMAS_CHAT = [_responses_tool_to_chat(t) for t in TOOL_SCHEMAS]


async def run_tool_calling_agent_chat(
    client,
    model: str,
    user_prompt: str,
    system_prompt: str,
    tools: list[dict] | None = None,
    tool_functions: dict | None = None,
    max_iterations: int = 15,
    verbose: bool = True,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    max_tool_output_chars: int = 2000,
) -> dict:
    """
    Run a tool-calling agent loop using the Chat Completions API (async).

    Works with any OpenAI-compatible server: Ollama, llama-server, mlx-lm, vLLM, etc.
    No reasoning support — use run_tool_calling_agent() for Responses API models.

    Returns the SAME shape as run_tool_calling_agent():
      - "input": the full conversation history (list of message dicts)
      - "output": the final assistant message(s) as a list
      - "output_text": the final assistant response text
      - "reasoning_summaries": always [] (Chat Completions has no reasoning)
      - "usage": token usage breakdown
    """
    #if tools is None:
    #    tools = TOOL_SCHEMAS_CHAT
    #if tool_functions is None:
    #    tool_functions = TOOL_FUNCTIONS

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    total_usage = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}

    for i in range(max_iterations):
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Track usage if provided
        if response.usage:
            total_usage["input_tokens"] += response.usage.prompt_tokens or 0
            total_usage["output_tokens"] += response.usage.completion_tokens or 0

        msg = response.choices[0].message

        # No tool calls — final response
        if not msg.tool_calls:
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
            })
            if verbose:
                print(f"  [{i+1}] Agent finished — produced final response")
            break

        # Append assistant message with tool calls
        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ],
        })

        # Execute each tool call
        for tc in msg.tool_calls:
            fn_name = tc.function.name
            fn_args = json.loads(tc.function.arguments)

            if fn_name in tool_functions:
                result = await asyncio.to_thread(
                    tool_functions[fn_name], **fn_args
                )
            else:
                result = json.dumps({"error": f"Unknown tool: {fn_name}"})

            # Truncate tool output to avoid flooding the context window
            if len(result) > max_tool_output_chars:
                result = result[:max_tool_output_chars] + "\n... [truncated]"

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

            if verbose:
                args_str = ", ".join(
                    f"{k}={v!r}" for k, v in fn_args.items()
                )
                print(f"  [{i+1}] Called {fn_name}({args_str})")

    # Extract final output text
    output_text = ""
    for m in reversed(messages):
        if m["role"] == "assistant" and not m.get("tool_calls"):
            output_text = m.get("content", "")
            break

    # Build output list (final assistant message(s) — matches Responses API shape)
    final_output = [m for m in messages if m["role"] == "assistant" and not m.get("tool_calls")]

    return {
        "input": messages,
        "output": final_output,
        "output_text": output_text,
        "reasoning_summaries": [],
        "usage": total_usage,
    }
