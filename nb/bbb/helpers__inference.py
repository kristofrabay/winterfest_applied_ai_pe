"""
Local inference helpers for Qwen3-4B tool-calling evaluation.

- Parse <tool_call> blocks from model output
- Multi-turn local agent loop (Unsloth/transformers)
- Composite reward function for scoring trajectories
"""

import json
import re


# ---------------------------------------------------------------------------
# Parse tool calls from raw model output
# ---------------------------------------------------------------------------

_TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL
)


def parse_tool_calls(text: str) -> tuple[str, list[dict]]:
    """
    Extract <tool_call> blocks from model output text.

    Returns:
        (content, tool_calls) where content is everything before the first
        <tool_call> tag (may include <think> blocks), and tool_calls is a
        list of parsed tool call dicts in OpenAI format.
    """
    matches = _TOOL_CALL_PATTERN.findall(text)

    tool_calls = []
    for raw in matches:
        try:
            parsed = json.loads(raw)
            tool_calls.append({
                "type": "function",
                "function": {
                    "name": parsed["name"],
                    "arguments": json.dumps(parsed.get("arguments", {})),
                },
            })
        except (json.JSONDecodeError, KeyError):
            # Malformed tool call — record it for the reward function
            tool_calls.append({
                "type": "function",
                "function": {"name": "__malformed__", "arguments": raw},
            })

    # Content before the first tool call
    first_pos = text.find("<tool_call>")
    content = text[:first_pos].strip() if first_pos > 0 else ("" if first_pos == 0 else text.strip())

    return content, tool_calls


def clean_response(text: str) -> str:
    """Strip chat template special tokens, keep <think> and <tool_call> tags."""
    for token in ["<|im_end|>", "<|endoftext|>", "<|im_start|>", "<|end|>"]:
        text = text.replace(token, "")
    return text.strip()


# ---------------------------------------------------------------------------
# Local agent loop — multi-turn inference with tool execution
# ---------------------------------------------------------------------------

def run_local_agent_loop(
    model,
    tokenizer,
    user_prompt: str,
    system_prompt: str,
    tools_chat: list[dict],
    tool_functions: dict,
    max_iterations: int = 8,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    enable_thinking: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Run a multi-turn tool-calling agent loop with a local model.

    Args:
        model: Unsloth/transformers model.
        tokenizer: Corresponding tokenizer.
        tools_chat: Tool schemas in Chat Completions format (nested).
        tool_functions: Dict mapping function names to callables.
        enable_thinking: Whether to enable <think> blocks in generation.
        verbose: Print each step.

    Returns dict with:
        messages: full conversation history
        output_text: final assistant response text (no tool calls)
        n_tool_calls: total tool calls made
        n_iterations: number of generate() calls
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    total_tool_calls = 0

    for i in range(max_iterations):
        # Format with chat template
        text = tokenizer.apply_chat_template(
            messages,
            tools=tools_chat,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        # Generate
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=0.95,
        )

        # Decode only the new tokens
        new_tokens = outputs[0][inputs.input_ids.shape[-1]:]
        response_text = tokenizer.decode(new_tokens, skip_special_tokens=False)
        response_text = clean_response(response_text)

        # Parse tool calls
        content, tool_calls = parse_tool_calls(response_text)

        if verbose:
            has_think = "<think>" in content
            if tool_calls:
                names = [tc["function"]["name"] for tc in tool_calls]
                print(f"  [{i+1}] think={has_think}, tool_calls={names}")
            else:
                print(f"  [{i+1}] think={has_think}, final response ({len(content)} chars)")

        if not tool_calls:
            # Final response — no tool calls
            messages.append({"role": "assistant", "content": response_text})
            break

        # Add assistant message with tool calls
        assistant_msg = {"role": "assistant", "tool_calls": tool_calls}
        if content:
            assistant_msg["content"] = content
        messages.append(assistant_msg)

        # Execute tools and add results
        for tc in tool_calls:
            fn_name = tc["function"]["name"]
            total_tool_calls += 1

            if fn_name == "__malformed__":
                result = json.dumps({"error": "Malformed tool call"})
            elif fn_name in tool_functions:
                try:
                    fn_args = json.loads(tc["function"]["arguments"])
                    result = tool_functions[fn_name](**fn_args)
                except Exception as e:
                    result = json.dumps({"error": str(e)})
            else:
                result = json.dumps({"error": f"Unknown tool: {fn_name}"})

            messages.append({"role": "tool", "content": result})

            if verbose:
                print(f"        → {fn_name}: {len(result)} chars")

    # Extract final output text
    output_text = ""
    for msg in reversed(messages):
        if msg["role"] == "assistant" and not msg.get("tool_calls"):
            output_text = msg.get("content", "")
            break

    return {
        "messages": messages,
        "output_text": output_text,
        "n_tool_calls": total_tool_calls,
        "n_iterations": i + 1,
    }


# ---------------------------------------------------------------------------
# Composite reward function
# ---------------------------------------------------------------------------


def compute_reward(
    messages: list[dict],
    valid_tool_names: set[str],
    reasoning_summaries: list[str] | None = None,
) -> dict:
    """
    Score a tool-calling trajectory with a composite reward function.

    Components:
        valid_json (+1): All tool calls have parseable JSON arguments
        thinking (+1): Model uses <think> before tool calls
        tool_selection (0 to +2): Jaccard overlap with valid tool set
        efficiency (-1 per excess): Penalize > 5 tool calls
        completion (+1): Produces a final text response (not just tool calls)
        no_hallucination (-2): Only calls tools that exist

    Returns dict with each component and total (range ~[-3, +6]).
    """
    tool_calls = []
    has_thinking = False
    has_final_response = False

    # Check reasoning_summaries (from Chat Completions — server separates thinking)
    if reasoning_summaries:
        has_thinking = True

    for msg in messages:
        if msg["role"] == "assistant":
            tcs = msg.get("tool_calls", [])
            tool_calls.extend(tcs)
            content = msg.get("content", "") or ""
            # Check for <think> tags in content (Unsloth / raw inference path)
            if "<think>" in content:
                has_thinking = True
            if not tcs and len(content) > 50:
                has_final_response = True

    components = {}

    # 1. Valid JSON arguments
    all_valid = True
    for tc in tool_calls:
        args = tc.get("function", {}).get("arguments", "")
        if tc["function"]["name"] == "__malformed__":
            all_valid = False
            break
        try:
            json.loads(args)
        except (json.JSONDecodeError, TypeError):
            all_valid = False
            break
    components["valid_json"] = 1.0 if (all_valid and tool_calls) else 0.0

    # 2. Thinking before tool calls
    components["thinking"] = 1.0 if has_thinking else 0.0

    # 3. Tool selection overlap
    used_tools = {
        tc["function"]["name"]
        for tc in tool_calls
        if tc["function"]["name"] != "__malformed__"
    }
    if valid_tool_names:
        overlap = len(used_tools & valid_tool_names) / len(valid_tool_names)
    else:
        overlap = 0.0
    components["tool_selection"] = round(overlap * 2.0, 2)

    # 4. Efficiency
    n_calls = len(tool_calls)
    components["efficiency"] = -1.0 * max(0, n_calls - 5)

    # 5. Completion
    components["completion"] = 1.0 if has_final_response else 0.0

    # 6. No hallucinated tools
    hallucinated = used_tools - valid_tool_names
    components["no_hallucination"] = -2.0 if hallucinated else 0.0

    components["total"] = round(sum(components.values()), 2)
    return components
