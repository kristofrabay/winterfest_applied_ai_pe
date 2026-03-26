# Plan: Tool-Calling Fine-Tuning & RL Conference Demo

## Context

Kristof is preparing a follow-up conference talk to his WinterFest 2025 presentation. The original demo showed a two-stage pipeline: (1) a GPT-5.1 research agent with MCP tools, and (2) a fine-tuned Qwen3-4B analyst model for investment verdicts.

**The new talk's thesis:** Take a small open-source model and teach it to *be* the research agent itself — not just analyze reports, but actually call tools (stock data, web search, file search) to produce them. This demonstrates the full journey: prototype with expensive proprietary models → distill into specialized open-source models via SFT → refine behavior with RL.

**Problem this solves:** The current agent costs ~$0.50/run on GPT-5.1. A fine-tuned 4B model running locally would cost near zero and could be deployed on-premise for compliance-sensitive financial use cases.

### Decisions Made

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Base model** | **Qwen3-4B** (primary) | More documentation, larger community, existing project familiarity. gpt-oss mentioned in talk as alternative. |
| **Teacher model** | **GPT-5.4** (latest at talk time) | Keep OpenAI API format, upgrade from 4.1-mini for higher quality traces |
| **GPU environment** | **Databricks cluster** | Develop locally, run notebooks on Databricks compute with GPUs. No Colab constraints. |
| **Framework** | **Unsloth** | Consistent with existing project, 2x faster training, first-class Qwen3 support |

---

## Phase 1: Build the Teacher Agent & Tool-Calling Loop

**Goal:** Create a simple, self-contained tool-calling agent using OpenAI's API (GPT-5.4) that we can run 1000+ times to generate training trajectories.

### Notebook: `nb/tool_calling_agent.ipynb`

**1a. Define the tools as plain Python functions + OpenAI tool schemas**

Reuse the 4 existing MCP tools from `tools/mcp/stock_server.py`, but expose them as **direct Python functions** (no MCP server needed). This simplifies the data collection pipeline:

```python
# Tools to implement (wrapping existing yfinance logic):
get_stock_news(ticker: str) -> str
get_financials(ticker: str, statement_type: str, period: str) -> str
get_price_history(ticker: str, period: str, interval: str) -> str
get_recommendations(ticker: str, months_back: int) -> str
```

Also add a simple `web_search(query: str) -> str` tool using OpenAI's web search or a free alternative (e.g., DuckDuckGo via `duckduckgo-search` package).

**1b. Build the agent loop**

A simple while-loop that:
1. Sends messages + tool definitions to GPT-5.4 (or GPT-4.1-mini for cheaper bulk runs)
2. If the model returns `tool_calls`, execute them, append results, loop
3. If no tool calls, the model is done — return the full conversation trajectory
4. Cap at 15 iterations max (safety limit)

This is essentially the pattern from `docs/tutorial_content/tool-calling-guide-for-local-llms.md` (the `unsloth_inference` function), but using OpenAI's API directly.

**1c. System prompt for research**

Adapt the existing agent system prompt from `nb/agent.ipynb` but simplified:
- Role: equity research analyst
- Available tools: the 5 tool schemas
- Task: research a given company ticker, produce a structured memo
- Key instruction: "Think step-by-step about which tools to call and why. After gathering data, synthesize a research memo."

**Critical files:**
- `tools/mcp/stock_server.py` — reuse the yfinance logic
- `nb/agent.ipynb` — reuse the system prompt structure
- `nb/helpers/llm_helpers.py` — reference for streaming patterns

---

## Phase 2: Generate the Tool-Calling Training Dataset

**Goal:** Run the teacher agent ~1000 times across diverse companies, capturing full multi-turn tool-calling trajectories.

### Notebook: `nb/tool_calling_data_generator.ipynb`

**2a. Company list generation**

Generate a diverse list of ~1000 real publicly traded tickers:
- Use a mix of S&P 500, NASDAQ 100, Russell 2000 tickers
- Ensure diversity: sectors, market caps, geographies
- Can use `yfinance` to get ticker lists, or hardcode a curated set
- Add diversity in the *task prompt* too: "Research {ticker} focusing on {random_focus}" where focus varies (growth potential, competitive position, financial health, recent news, etc.)

**2b. Trajectory collection loop**

For each company:
1. Run the teacher agent with GPT-5.4
2. Capture the **full conversation** as a list of messages:
   - `system` → system prompt with tool definitions
   - `user` → "Research {ticker}..."
   - `assistant` (with `tool_calls`) → model decides to call tools
   - `tool` → tool execution results
   - `assistant` (with `tool_calls`) → more tool calls...
   - `assistant` → final research memo (no tool calls)
3. Save each trajectory as JSONL

**2c. Dataset format**

Use the **OpenAI/Hermes-compatible format** since it's what Unsloth expects:

```json
{
  "messages": [
    {"role": "system", "content": "You are a research analyst..."},
    {"role": "user", "content": "Research AAPL focusing on financial health"},
    {"role": "assistant", "content": "<think>I should start by getting the financials...</think>", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_financials", "arguments": "{\"ticker\": \"AAPL\", \"statement_type\": \"income\", \"period\": \"annual\"}"}}]},
    {"role": "tool", "tool_call_id": "call_1", "content": "{...financial data...}"},
    {"role": "assistant", "content": "<think>Good, now let me check price history...</think>", "tool_calls": [...]},
    {"role": "tool", "tool_call_id": "call_2", "content": "{...}"},
    {"role": "assistant", "content": "# Research Memo: Apple Inc (AAPL)\n\n## Competition\n..."}
  ],
  "tools": [
    {"type": "function", "function": {"name": "get_stock_news", ...}},
    ...
  ]
}
```

**Key design choice — `<think>` tags in assistant messages:**
The teacher model should be prompted to include reasoning *before* each tool call. This is the same pattern as FunctionGemma's "reason before tool calling" notebook. We inject this into the system prompt: "Before each tool call, briefly explain your reasoning in <think> tags."

**2d. Cost & time estimate**
- GPT-5.4: pricing TBD at talk time (likely comparable to GPT-4.1 range)
- ~1000 trajectories × ~5 tool calls each × ~2K tokens/trajectory ≈ 10M tokens total
- Estimated cost: ~$20-50 depending on GPT-5.4 pricing
- Estimated time: ~2-3 hours (with rate limiting from `limiter`/`tenacity` as in existing `training_data_generator.ipynb`)

**2e. Data quality filtering**
After collection, filter out:
- Trajectories that hit the 15-iteration cap (likely stuck in loops)
- Trajectories with malformed tool calls
- Trajectories where the final memo is too short (<500 chars)
- Duplicates or near-duplicates

**Output:** `data/tool_calling_trajectories.jsonl` (~800-1000 clean trajectories)

---

## Phase 3: SFT — Teach the Small Model to Call Tools

**Goal:** Fine-tune a small open-source model to replicate the teacher's tool-calling behavior.

### Notebook: `nb/tool_calling_sft.ipynb`

**3a. Base model selection**

**Chosen: `Qwen3-4B`** (with `Qwen3-4B-Thinking` variant for native `<think>` support)

Why Qwen3-4B:
- Dominates tool-calling benchmarks after fine-tuning (F1 0.933)
- Already used in the existing project for the analyst model — audience sees consistency
- Native `<think>` tag support in the chat template
- Massive community documentation → faster debugging
- Unsloth has first-class support with optimized chat templates

**Talk mention: `gpt-oss-20b`** as the "next step" / production alternative:
- Only 3.6B active params (MoE), native function calling
- Narrative: "OpenAI's own open model replacing their proprietary one"
- Could show a comparison slide without needing to fully train it

**3b. Data formatting for SFT**

Follow the pattern from FunctionGemma notebook (`docs/tutorial_content/FunctionGemma_(270M).ipynb`):

1. Load trajectories from JSONL
2. Apply the model's chat template with `tokenizer.apply_chat_template(messages, tools=tools)`
3. Use `train_on_responses_only` to mask system/user/tool turns — only train on assistant outputs
4. This teaches the model: given context + tool results, what to say/call next

**3c. Training configuration**

Based on existing `nb/training_recipe.ipynb` and FunctionGemma patterns. With Databricks GPU cluster, we can use larger batch sizes:

```python
# Model loading
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-4B-Thinking",
    max_seq_length=8192,  # longer for multi-turn tool trajectories
    load_in_4bit=True,
)

# LoRA config (matching existing recipe)
model = FastLanguageModel.get_peft_model(
    model, r=32, lora_alpha=64,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_dropout=0, use_gradient_checkpointing="unsloth",
)

# SFT config — adjust batch size based on Databricks GPU memory
SFTConfig(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    max_steps=500,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    optim="adamw_8bit",
    dataset_text_field="text",
    max_seq_length=8192,
    eval_strategy="steps",
    eval_steps=25,
    save_steps=150,
    report_to="mlflow",  # Databricks-native tracking
)
```

**3d. Evaluation**

- Hold out 15% of trajectories for eval
- Track eval loss (should decrease, gap with train <0.05)
- Manual inspection: run inference on 10 unseen tickers, check:
  - Does it call the right tools?
  - Does it produce valid JSON arguments?
  - Does it know when to stop calling tools?
  - Is the final memo reasonable?

---

## Phase 4: RL — Refine Tool-Calling Behavior

**Goal:** Use GRPO to reward desirable behaviors and penalize bad habits.

### Notebook: `nb/tool_calling_rl.ipynb`

**4a. Framework: Unsloth GRPO with DAPO loss**

GRPO (Group Relative Policy Optimization) generates multiple completions per prompt, scores them with reward functions, and updates the policy. DAPO loss is recommended by Unsloth for stability.

**4b. RL environment setup**

For each training step:
1. Sample a company ticker + research task
2. Model generates a response (potentially with tool calls)
3. If tool calls are present, execute them and feed results back
4. Continue until model stops calling tools or hits max iterations
5. Score the full trajectory with reward functions

**Use Unsloth's ART (Agent Reinforcement Trainer)** or implement a custom environment that wraps the tool execution loop.

**4c. Reward function design (ToolRL-inspired composite)**

```python
def compute_reward(trajectory: list[dict]) -> float:
    reward = 0.0

    # 1. FORMAT REWARD (+1): All tool calls have valid JSON arguments
    if all_tool_calls_valid_json(trajectory):
        reward += 1.0

    # 2. THINKING REWARD (+1): Model uses <think> before tool calls
    if has_thinking_before_tool_calls(trajectory):
        reward += 1.0

    # 3. TOOL SELECTION REWARD (+1 to +2): Uses relevant tools
    relevant_tools = get_expected_tools_for_task(task_type)
    tool_overlap = jaccard(used_tools, relevant_tools)
    reward += tool_overlap * 2.0

    # 4. EFFICIENCY PENALTY (-1 per excess): Penalize > 5 tool calls
    n_calls = count_tool_calls(trajectory)
    if n_calls > 5:
        reward -= (n_calls - 5) * 1.0

    # 5. COMPLETION REWARD (+1): Produces a final memo (not just tool calls)
    if has_final_memo(trajectory):
        reward += 1.0

    # 6. NO HALLUCINATED TOOLS (-2): Only calls tools that exist
    if calls_nonexistent_tools(trajectory):
        reward -= 2.0

    return reward
```

**Reward range:** approximately [-3, +6], with a well-behaved trajectory scoring +4 to +5.

**4d. Training configuration**

```python
from trl import GRPOConfig, GRPOTrainer

config = GRPOConfig(
    num_generations=8,          # 8 completions per prompt
    max_completion_length=4096,
    max_prompt_length=2048,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=5e-6,  # lower than SFT
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    optim="adamw_8bit",
    loss_type="dapo",  # Recommended by Unsloth
)
```

**4e. RL reward scenarios to demo in the talk**

Show before/after comparisons:
1. **Before RL:** Model calls 8+ tools redundantly (e.g., gets news twice)
2. **After RL:** Model calls 3-5 tools efficiently, thinks between calls
3. **Before RL:** Model sometimes forgets to write the final memo
4. **After RL:** Always produces structured output after research

---

## Deliverables Summary

| # | Notebook | Purpose | Key Output |
|---|----------|---------|------------|
| 1 | `nb/tool_calling_agent.ipynb` | Teacher agent + tool loop | Working agent that researches any ticker |
| 2 | `nb/tool_calling_data_generator.ipynb` | Dataset generation | `data/tool_calling_trajectories.jsonl` |
| 3 | `nb/tool_calling_sft.ipynb` | SFT fine-tuning | LoRA adapters for tool-calling Qwen3-4B |
| 4 | `nb/tool_calling_rl.ipynb` | GRPO reinforcement learning | Refined model with better tool-use behavior |

**Additional files:**
- `tools/stock_tools.py` — Extracted tool functions (shared by notebooks 1, 2, and the RL environment)
- `data/tickers.json` — Curated list of ~1000 diverse tickers
- `data/tool_calling_trajectories.jsonl` — Training dataset

---

## Talk Narrative Arc

1. **"Last time"** — Recap the WinterFest demo (GPT-5.1 agent → Qwen3-4B analyst)
2. **"The problem"** — Proprietary models are expensive, can't deploy on-prem
3. **"Step 1: Prototype"** — Show the teacher agent working (live demo with a ticker)
4. **"Step 2: Distill"** — Show the dataset, explain the SFT process
5. **"Step 3: Specialize"** — Show SFT model calling tools (it works! but inefficiently)
6. **"Step 4: Refine"** — RL reward design, before/after comparison
7. **"The punchline"** — A 4B model running locally does 80%+ of what GPT-5.4 does, at ~0 marginal cost
8. **"What's next"** — Mention gpt-oss-20b as production alternative, on-device deployment possibilities

---

## Verification Plan

1. **Phase 1:** Run the teacher agent on 5 tickers manually, verify tool calls execute and memos are generated
2. **Phase 2:** Generate 50 test trajectories, inspect format, verify JSONL is valid, check tool call diversity
3. **Phase 3:** After SFT, run inference on 10 unseen tickers — check tool calls are valid JSON, correct tools selected, memo produced
4. **Phase 4:** After RL, compare reward scores before/after on a held-out set of 50 prompts — expect +1.5 average improvement
5. **End-to-end:** Run the full pipeline on a ticker never seen in training, present the result in talk format

---

## Implementation Order

We will build these **sequentially**, one notebook at a time:

1. **Start with Phase 1** (`nb/tool_calling_agent.ipynb`) — get the teacher agent working with a few tickers
2. **Then Phase 2** (`nb/tool_calling_data_generator.ipynb`) — bulk data generation
3. **Then Phase 3** (`nb/tool_calling_sft.ipynb`) — SFT training on Databricks
4. **Finally Phase 4** (`nb/tool_calling_rl.ipynb`) — GRPO refinement

Each phase validates the previous one's output before moving on. We'll also extract shared tool definitions into `tools/stock_tools.py` during Phase 1 so all notebooks import from the same place.
