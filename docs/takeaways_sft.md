# SFT Takeaways — Everything We Learned

*Consolidated from: notebooks, debugging sessions, old conversations, and the WinterFest experience.*
*Last updated: 2026-04-20*

---

## The Big Picture

SFT (Supervised Fine-Tuning) = teaching a model to predict the next token on labeled examples. It's imitation learning. The model sees what the teacher did and learns to replicate the pattern. No judgment of quality — just "do it like this."

**What SFT teaches:** format, structure, tool selection patterns, when to stop, reasoning depth.
**What SFT doesn't teach:** efficiency, quality judgment, edge case handling, "better than the teacher."

---

## 1. Data Is 80% of the Work

### 1a. Tool Output Truncation Is Mandatory

Raw yfinance responses are 2,000–3,000 tokens each. With 4–6 tool calls per trajectory, tool results alone consume 10,000–15,000 tokens — far beyond our 8,192 `max_seq_length`.

**Why it doesn't matter:** Tool results are masked (`labels=-100`) during SFT. They contribute **zero gradient**. They're pure context overhead — the model sees them to understand what happened, but doesn't learn to predict them.

**The principle: "Compress data to fit, don't expand window."** It's better to truncate tool outputs to fit within `max_seq_length=8192` than to increase to 16K. The attention cost is O(s²) — doubling sequence length quadruples attention compute. And the extra masked tokens add zero training signal.

**Industry standard:**
- Hermes dataset: 50–200 token tool outputs
- ToolBench: truncates to 2,048 characters
- APIGen: keeps outputs under 200 tokens

We truncate to ~250 tokens via `truncate_tool_output()` in `helpers__data_gen.py`.

### 1b. Format Conversion Is Non-Trivial

The OpenAI Responses API format is fundamentally different from Chat Completions / Hermes format:

| Responses API | Chat Completions / Hermes |
|--------------|--------------------------|
| `developer` role | `system` role |
| Flat tool schemas: `{"type": "function", "name": "...", "parameters": {...}}` | Nested: `{"type": "function", "function": {"name": "...", "parameters": {...}}}` |
| `function_call` items | `tool_calls` array on assistant message |
| `function_call_output` items | `role: tool` messages with `tool_call_id` |
| Reasoning as separate items | `<think>` tags in assistant content |

Our `responses_to_hermes()` function in `helpers__data_gen.py` handles all of this. Reasoning summaries from GPT-5.4 become `<think>` blocks — this teaches the student model the think-call-observe loop.

### 1c. Qwen3.5 Jinja Template Expects Arguments as Dict, Not String

OpenAI API returns tool call arguments as JSON strings (`"arguments": "{\"ticker\": \"AAPL\"}"`). Qwen3.5's Jinja template does `arguments | items` which crashes on strings.

**Fix:** `json.loads()` in data prep. The `to_mlx_format()` function in Phase 4 handles this.

### 1d. Quality Filtering Catches Real Issues

Our `filter_trajectory()` checks:
- At least 1 tool call with valid JSON arguments
- Final memo exists with minimum length (200 chars)
- At least 2 different tools used (single-tool trajectories are too narrow)
- Tool error rate below 50% (yfinance failures = bad training signal)
- Error keywords: "no data found", "delisted", "http error 404", etc.

**Result:** 955 clean trajectories from ~200 tickers (multiple focus areas per ticker).

---

## 2. It's All Just Text

### 2a. The Chat Template Is the Contract

Everything — system prompt, user message, tool schemas, assistant reasoning, tool calls, tool results, final output — gets flattened into a single text string by `tokenizer.apply_chat_template()`. The model sees:

```
<|im_start|>system
# Tools
{"type": "function", "function": {"name": "get_financials", ...}}
...
You are a sell-side equity research analyst...
<|im_end|>
<|im_start|>user
Research AAPL focusing on financial health.
<|im_end|>
<|im_start|>assistant
<think>I should check financials first...</think>
<tool_call>{"name": "get_financials", "arguments": {"ticker": "AAPL", ...}}</tool_call>
<|im_end|>
<|im_start|>tool
{"ticker": "AAPL", "revenue": "416B"...}
<|im_end|>
<|im_start|>assistant
**AAPL** | Technology | $3.68T...
<|im_end|>
```

**This is exactly what Claude Code, Codex, and every tool-calling agent does.** It's text in, text out, with parsing of special tags. The `<tool_call>` tags, `<think>` blocks, `<|im_start|>`/`<|im_end|>` tokens — they're all just tokens the model learns to predict.

### 2b. Masking Controls What Gets Gradient

`train_on_responses_only` (Unsloth) / `mask_prompt` (MLX) sets `labels=-100` for everything except assistant turns:

| Role | Masked? | Why |
|------|---------|-----|
| `system` | Yes (`labels=-100`) | Context, not prediction target |
| `user` | Yes | Context |
| `assistant` | **No — gets gradient** | This is what the model learns to produce |
| `tool` | Yes | Context (and truncated, see above) |

**The model learns the decision pattern, not the data.** It learns "after seeing financials, call price_history next" — not "Apple's revenue is $416B."

Our training ratio: ~36.5% of tokens get gradient (verified in notebooks).

### 2c. `<tool_call>` Parsing Is Just Regex

```python
_TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
```

Malformed tool calls get tagged as `__malformed__` rather than dropped — this gives the reward function signal during evaluation.

---

## 3. Training Knobs and What They Cost

### 3a. What `max_seq_length` Actually Means (And Why It Matters)

`max_seq_length` is the **total token window for the entire training example** — from the very first `<|im_start|>system` token to the last `<|im_end|>`. It includes everything: system prompt, tool schemas, user message, every assistant turn (thinking + tool calls + final memo), and every tool result. It is NOT just the output, NOT just the prompt — it's the whole thing concatenated.

**What happens to examples that DON'T fit:**

```
max_seq_length = 8192

Example A: 5,400 tokens → fits entirely, no problem
Example B: 11,234 tokens → TRUNCATED FROM THE END to 8,192 tokens
```

Truncation from the end is catastrophic for tool-calling SFT because **the final assistant response lives at the end**. That's the memo — the most important part. With masking, it's the only part that gets gradient. Truncate it and you might have zero training tokens for that example → NaN loss.

**What happens to examples that are SHORTER:**

With `packing=False` (which we use to preserve message boundaries), shorter examples get **padded** to `max_seq_length`. A 2K-token example padded to 8K wastes 75% of the sequence. And attention cost is O(s²) — so you're paying 8192² = 67M attention operations instead of 2048² = 4.2M. That's **16× more compute on padding tokens that contribute nothing.**

`packing=True` solves this by concatenating multiple short examples into one sequence, but it breaks message boundary masking for multi-turn conversations.

**The practical principle: "Compress data to fit, don't expand window."**

It's better to truncate tool outputs (masked tokens that contribute zero gradient anyway) to fit within 8,192 than to increase to 16,384 and pay quadratic attention cost. Our tool output truncation to ~250 tokens brings the max sample from ~17K to ~6.4K tokens — everything fits comfortably in 8,192.

```
Token budget example (max_seq_length = 8192):
  System prompt + tool schemas:  ~800 tokens
  User message:                  ~20 tokens
  Assistant <think> + tool_call: ~200 tokens
  Tool results (4-5 calls):     ~1,250 tokens (truncated from ~12K)
  More assistant turns:          ~400 tokens
  Final assistant response:      ~800 tokens
  ─────────────────────────────────────────────
  Total:                        ~3,470 tokens  ✓ fits with room to spare
```

**What we learned the hard way:**
- `max_seq_length=4096` → NaN loss (final memo truncated entirely, zero training tokens)
- `max_seq_length=16384` → OOM on Colab T4 (tried to allocate 11 GB for lm_head logits with Qwen3.5's 248K vocab)
- `max_seq_length=8192` → sweet spot (max sample after truncation is 6.4K, 45 samples exceed 8K but only lose some tool output, not the memo)

### 3b. The Scaling Cheat Sheet

| Knob | Memory | Compute/step | Intuition |
|------|--------|-------------|-----------|
| **Sequence length** (s) | O(s) to O(s²) | O(s²) attention | **The big one.** 8K→16K = 4x more attention compute. |
| **Batch size** (b) | O(b) | O(b) per step | Linear. Doubling = 2x memory, half the steps. |
| **LoRA rank** (r) | ~Negligible | ~Negligible | **Cheapest knob.** rank 8→32 barely moves the needle. |
| **Model size** (P) | O(P) | O(P) | Linear. 2B→4B ≈ 2x everything. |
| **num_layers** (LoRA) | O(L) activations | Weak | More adapted layers = more stored activations. |

### 3b. Cross-Platform: Unsloth vs MLX (Same Knobs, Different Names)

| Concept | Unsloth (Colab/Kaggle) | MLX (Mac) |
|---------|----------------------|-----------|
| LoRA rank | `r=32` | `rank: 32` |
| LoRA alpha | `lora_alpha=64` | `scale: 2.0` (**`scale = alpha/rank`**) |
| Mask non-assistant | `train_on_responses_only()` | `mask_prompt: true` |
| Target modules | Explicit list of 7 projections | `keys: [self_attn.q_proj, ...]` |
| Learning rate | `2e-4` (8-bit Adam) | `1e-5` (full-precision Adam) |
| Steps vs epochs | `num_train_epochs` | `iters` (manual conversion) |
| Gradient checkpointing | `use_gradient_checkpointing="unsloth"` | `grad_checkpoint: true` |

**Critical gotcha:** MLX `scale` is NOT the same as Unsloth's `lora_alpha`. Getting this wrong silently produces a model that learned nothing.

### 3c. Our Config (Qwen3.5-2B on Colab T4)

```python
model = "unsloth/Qwen3.5-2B"
max_seq_length = 8192
r = 32, lora_alpha = 64
per_device_train_batch_size = 1  # T4 memory constraint
gradient_accumulation_steps = 8  # effective batch = 8
learning_rate = 2e-4
num_train_epochs = 1  # ~102 steps for 811 samples
```

Training time: ~30 minutes on free T4 (Unsloth claims 5–10 min — optimistic).

---

## 4. What Broke and Why

### 4a. MLX Training: Dead End on 16GB Mac

| Model | Config | Result |
|-------|--------|--------|
| Qwen3.5-4B | rank=32, all layers, 16K seq | OOM on backward pass |
| Qwen3.5-2B | rank=8, 4 layers, 4K seq | Metal OOM: `kIOGPUCommandBufferCallbackErrorOutOfMemory` |
| Qwen3.5-0.8B | rank=8, 4 layers, 8K seq | Same result |

**Root cause:** MLX backward pass peaks at ~48 GB regardless of model size or config. The forward pass uses ~4 GB (12x ratio, vs normal 2–3x). Without `mx.compile`, MLX builds the entire backward graph eagerly in Python. With `mx.compile`, the JIT compilation itself causes kernel panics.

**Key insight:** `mx.set_wired_limit` controls Metal GPU allocation but NOT the CPU-side memory used by `mx.compile`'s JIT tracer. On 16GB unified memory — GPU and CPU share the same pool.

**Our solution:** Train on Colab (free T4), infer locally on Mac. MLX is excellent for inference (~4 GB for Qwen3.5-2B).

### 4b. `max_seq_length=4096` → NaN Loss

Tool-calling trajectories are 7K–17K tokens. At 4096, the final assistant response gets truncated entirely. With masking, zero training tokens remain → NaN loss.

### 4c. Qwen3.5 Vocab (248K tokens) → Eval OOM

The logits tensor: `batch(2) x seq(8192) x vocab(248,320) x 2 bytes ≈ 8 GB`. Eval at batch_size=2 blows T4 VRAM.

**Fix:** `per_device_eval_batch_size=1` — separate from train batch size.

### 4d. Validation Passes but Training OOMs

Validation is forward-only (no gradients stored). Training backward pass needs ~2x more memory. If val loss prints but training crashes, you're at the memory edge. Val success tells you nothing about training feasibility.

### 4e. `resource_tracker: leaked semaphore` Is Always a Red Herring

Surfaces whenever a process using multiprocessing is killed ungracefully (SIGKILL). NOT the cause of any crash. Known CPython issue #90549.

### 4f. `warmup_steps` in MLX YAML Config Is Silently Ignored

The `TrainingArgs` dataclass in mlx-lm's trainer doesn't have this field. YAML accepts it, never uses it.

---

## 5. Architecture Insight

**Same agent loop for teacher and student — just change `base_url`:**

```python
# Teacher (GPT-5.4)
client = AsyncOpenAI()  # hits api.openai.com

# Student (local Qwen3.5-2B)
client = AsyncOpenAI(base_url="http://localhost:8080/v1", api_key="none")
```

All tool execution, message formatting, trajectory saving, and reward computation is identical. This is the key reusability insight — the agent loop is model-agnostic.
