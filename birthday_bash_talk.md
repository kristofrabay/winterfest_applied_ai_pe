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

### Inference Parameters Matter (Learned the Hard Way)

Getting a thinking model to behave requires the right sampling parameters. Wrong settings cause infinite reasoning loops where the model spirals ("Wait, I should check... Yes. Okay. Wait, I should check...").

| Parameter | Value | Why |
|-----------|-------|-----|
| `temperature` | **0.6** | Official Qwen3 recommendation for thinking mode. Too low → repetition loops. |
| `top_p` | **0.95** | Keeps diversity while staying coherent. |
| `presence_penalty` | **1.5** | Breaks repetition loops by penalizing already-generated tokens. Official recommendation. |
| `enable_thinking` | **True** | But do NOT mention thinking in the prompt. Qwen3.5 doesn't support `/nothink` soft switches — meta-instructions about thinking confuse the model. |

**What NOT to do:**
- Never use greedy decoding (`temperature=0`) with thinking models — causes infinite loops
- Never put "use low thinking" or "/nothink" in the system prompt for Qwen3.5 — the model meta-reasons about whether to think and loops
- Qwen3.5 `/think` and `/nothink` soft switches are Qwen3-only — they silently fail on Qwen3.5
- Always send a warm-up request after starting a local server with thinking models — cold KV cache causes 0% reuse and the first request spirals into extremely long reasoning ([mlx-lm#1042](https://github.com/ml-explore/mlx-lm/pull/1042))

### Expected Baseline Failures

- Wrong tool selection (calls `get_stock_news` when it needs `get_financials`)
- Malformed JSON arguments (missing required params, wrong types)
- Infinite looping (calls the same tool repeatedly)
- No final output (keeps calling tools, never writes the memo)
- **Thinking loops** — model spirals in `<think>` block without producing output

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
    ├── Inject reasoning as <think> tags in assistant content
    ├── Truncate: tool results to ~500-800 tokens
    ├── Filter: remove stuck loops, malformed calls, short memos
    └── Output: Hermes chat format with tools field
```

### What Gets Masked (and Why)

SFT with `mask_prompt` / `train_on_responses_only` means we only compute loss on **assistant turns**. Everything else is context — the model sees it but doesn't learn to predict it.

```text
<|im_start|>system                    ← MASKED (labels=-100)
You are a sell-side analyst...
<|im_end|>
<|im_start|>user                      ← MASKED
Research AAPL focusing on...
<|im_end|>
<|im_start|>assistant                 ← TRAINED ✓
<think>I should check financials</think>
<tool_call>{"name": "get_financials", ...}</tool_call>
<|im_end|>
<|im_start|>tool                      ← MASKED
{"ticker": "AAPL", "revenue": 416B...}
<|im_end|>
<|im_start|>assistant                 ← TRAINED ✓
<think>Now compile the snapshot</think>
**AAPL** | Technology | $3.68T...
<|im_end|>
```

**The model learns the decision pattern, not the data.** It learns "after seeing financials data, call price_history next" — not "Apple's revenue is $416B." This is why tool output truncation works: shorter masked results = more room for the assistant turns that actually get gradient.

### SFT Hyperparameter Guide

#### Understanding `max_seq_length`

`max_seq_length` is the **total token window** — the entire sequence from `<|im_start|>system` to the last `<|im_end|>`. It includes everything: system prompt, user message, assistant turns, tool calls, tool results, and the final response. It is NOT just the output or just the prompt — it's the whole thing.

If a training sample exceeds `max_seq_length`, it gets **truncated from the end**. For tool-calling trajectories, the end is typically the final assistant response — the most important part. This is why tool output truncation matters: you need the assistant content to fit within the window.

```text
Example token budget (max_seq_length = 8192):
  System prompt + tool schemas:  ~800 tokens
  User message:                  ~20 tokens
  Assistant <think> + tool_call: ~200 tokens
  Tool results (4-5 calls):     ~3000 tokens (truncated from ~12K)
  Assistant <think> + tool_call: ~200 tokens
  More tool results:            ~1500 tokens
  Final assistant response:      ~800 tokens
  ─────────────────────────────────────
  Total:                        ~6520 tokens ✓ fits in 8192
```

#### Effective Batch Size

```text
effective_batch_size = batch_size × grad_accumulation_steps

Example: batch_size=1, grad_accumulation=8 → effective batch = 8
```

The optimizer updates weights once per effective batch. Gradient accumulation simulates a larger batch by accumulating gradients across multiple forward passes before stepping. This is how you train with large effective batches on limited memory.

#### Steps vs Epochs

```text
steps_per_epoch = num_training_samples / effective_batch_size
total_steps = steps_per_epoch × num_epochs

Example: 200 samples, effective batch=8 → 25 steps/epoch
         If iters=200 → 200/25 = 8 epochs over the data
```

MLX uses `iters` (steps), not epochs. To convert: `iters = desired_epochs × (num_samples / effective_batch_size)`.

#### LoRA Parameters

| Parameter | Unsloth | MLX | What it controls |
|-----------|---------|-----|-----------------|
| Rank (`r`) | `r=32` | `rank: 32` | Dimensionality of LoRA matrices. Higher = more capacity, more memory. 16-64 is typical. |
| Alpha | `lora_alpha=64` | `scale: 2.0` | Scaling factor. **MLX uses `scale = alpha / rank`**, not alpha directly. So `alpha=64, r=32` → `scale=2.0`. |
| Dropout | `lora_dropout=0` | `dropout: 0.0` | Regularization. Usually 0 for SFT, sometimes 0.05-0.1 for small datasets. |
| Target modules | All attention + MLP | `num_layers: -1` | Which layers get LoRA. `-1` = all. Reducing this is the primary memory knob. |

#### Learning Rate

| Framework | Default | Notes |
|-----------|---------|-------|
| Unsloth/TRL | `2e-4` | Uses 8-bit Adam (`adamw_8bit`) — higher LR compensates for quantization noise |
| MLX | `1e-5` | Uses full-precision Adam — lower LR because updates are more precise |

#### Training on 16GB Apple Silicon (The Hard Way)

Apple Silicon uses **unified memory** — GPU and CPU share the same 16GB pool. Unlike CUDA which throws a clean `OutOfMemoryError`, exceeding memory on Metal causes macOS to swap to SSD. Since GPU memory access patterns are terrible for disk I/O, this freezes the entire machine — no crash, no error, just an unresponsive laptop for 30+ minutes.

**The memory hierarchy** (most to least impact on memory):

| Knob | Aggressive | Conservative | Impact |
|------|-----------|-------------|--------|
| `max_seq_length` | 16384 | **8192** | Biggest single factor. Sequence length × layers × hidden dim. 16K at 4B params will freeze your Mac. |
| `num_layers` | -1 (all 36) | **8-16** | How many transformer layers get LoRA adapters. Top layers matter most for behavioral SFT. |
| `rank` | 32 | **8-16** | LoRA matrix dimensionality. Lower rank = less memory, less capacity. 8 is the floor for tool-calling. |
| `grad_checkpoint` | false | **true** | Recomputes activations during backward pass. ~30% slower, ~40% less memory. Non-negotiable on 16GB. |
| `batch_size` | 2+ | **1** | Already at minimum. |

**What actually fits on 16GB M1:**

| Model | max_seq_length | num_layers | rank | Fits? |
|-------|---------------|------------|------|-------|
| Qwen3.5-4B (4-bit) | 16384 | -1 (all) | 32 | OOM on backward pass |
| Qwen3.5-4B (4-bit) | 16384 | 16 | 16 | OOM on backward pass |
| Qwen3.5-4B (4-bit) | 8192 | 16 | 16 | Fits with tight tool truncation |
| Qwen3.5-2B (4-bit) | 16384 | -1 | 32 | **Comfortable — our pick** |
| Qwen3.5-0.8B (4-bit) | 16384 | -1 | 32 | Very comfortable, limited capacity |

**The practical answer for 16GB Mac:** Use **Qwen3.5-2B** for local training — 16K context with full LoRA, no OOM risk, fast iterations. For a conference demo, training a 2B model on a MacBook Pro is a clean narrative. Fighting OOM on 4B is not a slide.

**Close your browser.** Safari/Chrome easily consume 2-4GB of unified memory. Close them before training — that's the difference between OOM and success.

#### Smaller Qwen3.5 Models for Local Training

| Model | MLX 4-bit size | Tool calling | Sweet spot |
|-------|---------------|-------------|------------|
| Qwen3.5-0.8B | ~500MB | Basic | Quick experiments, proof of concept |
| **Qwen3.5-2B** | **~1.2GB** | **Good** | **Best for 16GB Mac — 16K context, full LoRA, fast training** |
| Qwen3.5-4B | ~2.6GB | Strong | OOMs during training on 16GB Mac |
| Qwen3.5-9B | ~5GB | Excellent | Needs 32GB+ Mac or GPU |

### Training Configuration

**MLX (Apple Silicon — 16GB safe):**
```yaml
model: mlx-community/Qwen3.5-2B-4bit
max_seq_length: 16384
batch_size: 1
grad_accumulation_steps: 8   # effective batch = 8
mask_prompt: true
grad_checkpoint: true
learning_rate: 1e-5
num_layers: -1             # all layers — 2B fits comfortably
lora_parameters:
  rank: 32
  scale: 2.0               # = alpha(64) / rank(32)
```

**Unsloth (Databricks GPU):**
```python
model = "unsloth/Qwen3-4B"
max_seq_length = 8192
r = 32, lora_alpha = 64
per_device_train_batch_size = 4
gradient_accumulation_steps = 8  # effective batch = 32
learning_rate = 2e-4
```

### Gotchas We Hit

1. **Qwen3.5 Jinja template expects `arguments` as dict, not string.** OpenAI API returns tool call arguments as JSON strings. Qwen3.5's template does `arguments | items` which crashes on strings. Fix: `json.loads()` in data prep.
2. **`max_seq_length=4096` → NaN loss.** Tool-calling trajectories are 7K-17K tokens. At 4096, the final assistant response (the only part with gradient) gets truncated entirely → zero training tokens → NaN.
3. **`max_seq_length=16384` → frozen Mac.** Metal/MLX doesn't OOM — it swaps to SSD and freezes the machine. No error, no crash, just 30 minutes of an unresponsive laptop. Always test memory before long runs.
4. **MLX `mask_prompt` is Unsloth's `train_on_responses_only`.** Same concept, different name, different implementation. Both mask everything except assistant turns.
5. **Validation passes but training OOMs.** Validation is forward-only (no gradients stored). The backward pass during training needs ~2x more memory. If val loss prints but training crashes, you're right at the memory edge.

### What SFT Teaches

| Behavior | Before SFT | After SFT |
|----------|-----------|-----------|
| Tool selection | Random/wrong | Matches teacher patterns |
| JSON arguments | Often malformed | Valid, correct params |
| Knows when to stop | Loops forever | Produces final memo |
| Tool call efficiency | N/A | ~5-7 calls (mimics teacher) |
| Reasoning | None or loops | Brief `<think>` before each action |

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
