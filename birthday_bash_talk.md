# From Prototype to Production: Teaching Small Models to Call Tools
**Big Birthday Bash 2026**

*Kristof Rabay — Applied AI @ The Carlyle Group*

---

> **Disclaimer:**
> The views and opinions expressed in this presentation are those of the speaker alone and do not reflect the official policy, position, or views of The Carlyle Group or its affiliates.

---

## Agenda

1. [**Recap:** What we built at WinterFest](#1-recap)
2. [**The Problem:** Why we need to go further](#2-the-problem)
3. [**The Teacher:** Capturing expert tool-calling behavior](#3-the-teacher)
4. [**The Baseline:** What raw Qwen3-4B can (and can't) do](#4-the-baseline)
5. [**SFT:** Distilling tool-calling into a small model](#5-sft)
6. [**RL:** Refining behavior with reward functions](#6-rl)
7. [**Results:** Baseline vs SFT vs RL — the 3-way comparison](#7-results)
8. [**What's Next:** On-device agents and open-source alternatives](#8-whats-next)

---

## 1. Recap: What We Built at WinterFest <a name="1-recap"></a>

> **TL;DR:**
> Two-stage pipeline: GPT-5.1 agent generates research memos, fine-tuned Qwen3-4B produces investment verdicts. Works great — but Stage 1 costs ~$0.50/run and can't leave the cloud.

At WinterFest 2025, we demonstrated:
- **Stage 1:** A GPT-5.1 Research Agent with MCP tools (web search, file search, stock data, code interpreter)
- **Stage 2:** A fine-tuned Qwen3-4B that reads the memo and produces a verdict (Strong Yes / Questionable / Strong No)

The analyst model was the star — 5,000 synthetic examples, trained in minutes, outperforms generic models on our task.

**But Stage 1 is still a black box.** It's expensive, proprietary, and can't run on-premise.

---

## 2. The Problem <a name="2-the-problem"></a>

> **TL;DR:**
> Proprietary models are expensive at scale and can't be deployed in compliance-sensitive environments. We need to own the agent, not just rent it.

### The Cost Problem
| | GPT-5.4 Agent | Fine-tuned Qwen3-4B |
|---|---|---|
| **Cost per run** | ~$0.50 | ~$0.00 |
| **1,000 companies** | ~$500 | ~$0 |
| **Daily screening** | ~$180K/year | Hardware cost only |

### The Compliance Problem
- Financial institutions need models that run **on-premise** or in **private cloud**
- Can't send proprietary analysis through third-party APIs
- Need full audit trail of model reasoning

### The Solution
Teach the small model to **be** the research agent — not just read memos, but actually call tools to produce them.

---

## 3. The Teacher: Capturing Expert Behavior <a name="3-the-teacher"></a>

> **TL;DR:**
> Run GPT-5.4 on ~200 companies with live tools. Capture the full trajectory: reasoning, tool calls, tool results, final memo. This is our training data.

### Architecture

```text
GPT-5.4 (Responses API + Reasoning)
    │
    ├── "I should check the financials first..."     ← reasoning
    ├── get_financials(AAPL, income, annual)          ← tool call
    ├── {revenue: 394B, net_income: 97B, ...}         ← tool result
    ├── "Now let me check price trends..."            ← reasoning
    ├── get_price_history(AAPL, 1y, 1wk)             ← tool call
    ├── {min: 182, max: 248, change: +18%}            ← tool result
    ├── ...more tool calls...
    └── "# Research Memo: Apple Inc\n..."             ← final output
```

### Key Design Choices

- **OpenAI Responses API** (not Chat Completions) — native reasoning support, reasoning summaries captured automatically
- **Tool schemas auto-generated** from Python function signatures — change the function, schema updates
- **4 stock research tools:** `get_stock_news`, `get_financials`, `get_price_history`, `get_recommendations`
- **~200 diverse tickers** across sectors with varied research focus prompts

### The Data Challenge

Raw tool outputs are **2,000-3,000 tokens each**. With 4-6 tool calls per trajectory, that's 10,000-15,000 tokens of tool results alone — way beyond our 8,192 training window.

**Solution:** Truncate tool results to ~500-800 tokens before training. This is standard practice:
- Hermes dataset: 50-200 token tool outputs
- ToolBench: truncates to 2,048 characters
- APIGen: keeps outputs under 200 tokens

Tool results are **masked during SFT** (labels=-100), so they contribute zero gradient anyway — they're pure context overhead.

---

## 4. The Baseline: Raw Qwen3-4B with Tools <a name="4-the-baseline"></a>

> **TL;DR:**
> Qwen3-4B supports tool calling out of the box (Hermes-style `<tool_call>` tags), but struggles with our specific tasks. This is our "before" snapshot.

### What Qwen3-4B Can Do Natively

The model has built-in tool-calling support via its chat template:

```python
tokenizer.apply_chat_template(messages, tools=tools, enable_thinking=True)
# Model outputs: <tool_call>{"name": "get_financials", "arguments": {...}}</tool_call>
```

### Expected Baseline Failures

- Wrong tool selection (calls `get_stock_news` when it needs `get_financials`)
- Malformed JSON arguments (missing required params, wrong types)
- Infinite looping (calls the same tool repeatedly)
- No final output (keeps calling tools, never writes the memo)
- No reasoning (jumps to tool calls without thinking)

These failures are **the point** — they demonstrate why fine-tuning matters.

---

## 5. SFT: Distilling Tool-Calling Behavior <a name="5-sft"></a>

> **TL;DR:**
> Take the ~200 teacher trajectories, convert to Hermes chat format, fine-tune Qwen3-4B with LoRA. The model learns *when* to call tools, *what arguments* to pass, and *when to stop*.

### Data Pipeline

```text
Raw trajectories (Responses API format)
    │
    ├── Convert: developer → system, function_call → tool_calls, etc.
    ├── Truncate: tool results to ~500-800 tokens
    ├── Filter: remove stuck loops, malformed calls, short memos
    └── Output: Hermes chat format with tools field
```

### Training Configuration

```python
# Unsloth + LoRA
model = "unsloth/Qwen3-4B"
max_seq_length = 8192
r = 32, lora_alpha = 64
learning_rate = 2e-4
# Train on assistant turns only (tool results masked)
```

### What SFT Teaches

| Behavior | Before SFT | After SFT |
|----------|-----------|-----------|
| Tool selection | Random/wrong | Matches teacher patterns |
| JSON arguments | Often malformed | Valid, correct params |
| Knows when to stop | Loops forever | Produces final memo |
| Tool call efficiency | N/A | ~5-7 calls (mimics teacher) |

---

## 6. RL: Refining Behavior with Rewards <a name="6-rl"></a>

> **TL;DR:**
> SFT teaches imitation. RL teaches *optimization*. We use GRPO to reward efficient tool use, penalize redundancy, and encourage reasoning.

### Why RL After SFT?

SFT gives us a model that *can* call tools. But it might:
- Call the same tool twice with identical arguments
- Use 8 tool calls when 4 would suffice
- Skip reasoning and jump straight to tool calls
- Occasionally hallucinate tools that don't exist

RL pushes the model from "mimics the teacher" to "better than the teacher."

### Framework: ART (Agent Reinforcement Trainer)

Standard GRPO is **single-turn only**. ART by OpenPipe extends it for multi-turn tool-calling:
- Server hosts the model, runs GRPO training
- Client runs the agent loop, executes tools, scores trajectories
- Supports `<tool_call>`, `<think>`, and `<tool_response>` natively

### Reward Function (Composite)

```python
def compute_reward(trajectory) -> float:
    reward = 0.0
    reward += 1.0  if all_tool_calls_valid_json(trajectory)      # Format
    reward += 1.0  if has_thinking_before_tool_calls(trajectory)  # Reasoning
    reward += 2.0 * jaccard(used_tools, expected_tools)           # Selection
    reward -= 1.0 * max(0, n_tool_calls - 5)                     # Efficiency
    reward += 1.0  if has_final_memo(trajectory)                  # Completion
    reward -= 2.0  if calls_nonexistent_tools(trajectory)         # No hallucination
    return reward  # Range: ~[-3, +6]
```

---

## 7. Results: The 3-Way Comparison <a name="7-results"></a>

> **TL;DR:**
> Baseline → SFT → RL shows clear progression. The RL model matches teacher quality at near-zero cost.

### Expected Reward Scores

| Stage | Avg Reward | Tool Call Validity | Produces Memo | Efficient |
|-------|-----------|-------------------|---------------|-----------|
| **Baseline** (raw Qwen3-4B) | ~1.0-2.0 | ~40% | ~30% | N/A |
| **After SFT** | ~3.5-4.5 | ~95% | ~90% | Sometimes |
| **After RL** | ~4.5-5.5 | ~99% | ~98% | Yes |
| **Teacher** (GPT-5.4) | ~5.0-5.5 | 100% | 100% | Yes |

### Live Demo

Run the same ticker through all three models side-by-side:
1. **Baseline:** [show failures — wrong tools, loops, no output]
2. **SFT:** [works! but 7 tool calls where 4 suffice]
3. **RL:** [clean execution, reasoning before each call, efficient]

---

## 8. What's Next <a name="8-whats-next"></a>

### The Broader Pattern

This isn't just about stock research. The same pipeline works for any tool-calling agent:

```text
Expensive proprietary agent → Capture trajectories → SFT small model → RL refinement
```

### Open-Source Alternatives

- **gpt-oss-20b** — OpenAI's open model (3.6B active params via MoE), native function calling. "OpenAI replacing OpenAI."
- **Qwen3-32B** — Larger Qwen for production workloads where 4B isn't enough

### On-Device Deployment

- Fine-tuned Qwen3-4B runs on **llama.cpp** with tool calling support
- `llama-server --jinja --reasoning-format deepseek` gives you an OpenAI-compatible API locally
- Compliance-friendly: nothing leaves the building

### The Takeaway

> **You don't need to choose between capability and cost.**
> Prototype with the best proprietary model. Distill into an open-source specialist. Refine with RL. Ship it on your own hardware.

---

## Sources & References

### Tools & Frameworks
- [Unsloth](https://docs.unsloth.ai/) — SFT & RL training (2x faster, 60% less memory)
- [ART (OpenPipe)](https://github.com/openpipe/art) — Agent Reinforcement Trainer for multi-turn GRPO
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — Local inference with tool-calling support

### Research
- [ToolRL](https://arxiv.org/abs/2404.07995) — RL for tool-calling behavior
- [FunctionGemma](https://arxiv.org/abs/2404.14105) — Tool-calling fine-tuning patterns
- [xRouter (Oct 2025)](https://arxiv.org/pdf/2510.08439) — 7B tool-calling router
- [FARA-7B (Nov 2025)](https://www.microsoft.com/en-us/research/blog/fara-7b-an-efficient-agentic-model-for-computer-use/) — Specialized agent fine-tuning
- [Web Deep Research (Oct 2025)](https://arxiv.org/pdf/2510.15862v3) — 7B autonomous web research

### Previous Talk
- [WinterFest 2025: Agentic Systems in Practice](winterfest_talk.md)
