# Applied AI at Work: Understanding and Training AI Systems

**Big Birthday Bash — Data Science Festival 2026**

*Kristof Rabay — Applied AI @ The Carlyle Group*

---

> **Disclaimer:**
> The views and opinions expressed in this presentation are those of the speaker alone and do not reflect the official policy, position, or views of The Carlyle Group or its affiliates.

---

## Opening Framing (~1 min before agenda)

> Today we're going to walk through building a custom agentic system — from context engineering with prompts and MCP tools, all the way to a reinforcement-learning-fine-tuned model.
>
> The purpose isn't to live-code or demonstrate fine-tuning techniques. It's to show what being an AI practitioner looks like in 2026 — understanding concepts, continuously trying things, running into errors, adjusting, failing again, adjusting again.
>
> **Thesis:** Understanding how models get trained — SFT, RL, tool calling, masking — helps tremendously in how we *apply* them. This is Applied AI. We don't need to write the neural networks. We need to understand just enough to know how to make them better, how to make them worse, and how to pick the right technique for the problem.

---

## Agenda (~35 min + Q&A)

1. [**The Hook:** Why does this matter?](#1-hook)
2. [**Recap:** WinterFest and what evolves from it](#2-recap)
3. [**Step 1 — Agents:** It's all just text](#3-agents)
4. [**Step 2 — SFT:** Teaching the format](#4-sft)
5. [**Step 3 — RL:** Teaching quality](#5-rl)
6. [**Synthesis:** Tying it all back](#6-synthesis)

---

## 1. The Hook: Why Does This Matter? <a name="1-hook"></a>

**~3 minutes**

### The Open Source Community Is Thriving

The open source community is incredibly active — driving research, architectural ideas, caching, memory, tool orchestration, multi-agent orchestration. At the same time, massive intelligence and capability is coming out of top proprietary vendors like Anthropic and OpenAI.

But even the best vendors have bad days. [Anthropic's April 23 postmortem](https://www.anthropic.com/engineering/april-23-postmortem) is a recent example — features get rolled out, models degrade, services become unstable. **Your fine-tuned model on your hardware doesn't break when someone else pushes a deployment.**

### Small Models Beating Large Ones

| What | Result | Source |
|------|--------|--------|
| RC-GRPO Qwen 7B vs Opus 4.5 | 85% vs 61% (BFCLv4 multi-turn) | [arXiv:2602.03025](https://arxiv.org/abs/2602.03025) |
| ART-E Qwen 14B vs o3 | 96% accuracy ($80, 40 min training) | [OpenPipe](https://openpipe.ai/blog/art-e-mail-agent) |
| Fine-tuned OPT-350M vs ChatGPT | 77% vs 26% (ToolBench) | [arXiv:2512.15943](https://arxiv.org/abs/2512.15943) |

**The learning:** Open source and customization beyond context engineering is always on the table. For narrow, well-defined use cases, specialized small models consistently beat general-purpose frontier models.

### Standards Are Emerging

Anthropic's MCP (Model Context Protocol) is now an industry standard — people write MCP servers for everything. Beyond that, patterns like skills and persistent memory (as seen in Claude Code, Cursor) are becoming how practitioners build agent systems. Understanding these patterns is part of Applied AI.

But the decision of whether that's enough — or whether we need to go further with SFT or RL — is up to us. And we can only make that decision if we understand how fine-tuning actually adjusts the model, how it literally tunes a use case.

*[Speaker notes: Bloomberg spent $10M training a 50.6B model on financial data. GPT-4 outperformed it on nearly every benchmark. The next frontier generation wiped out the investment. Lesson: broad pretraining is a losing bet, but narrow verticals with SFT/RL remain powerful.]*

---

## 2. Recap: WinterFest and What Evolves <a name="2-recap"></a>

**~3 minutes**

### WinterFest in 60 Seconds

At WinterFest 2025 we built a two-stage system: a **research agent** (GPT-5.1 with MCP tools) that generates equity research memos, and a **decision maker** (fine-tuned Qwen3-4B) that reads those memos and produces investment verdicts. We showcased SFT, synthetic data generation, and evaluation.

```text
   [ Target Company ]
           │
           ▼
 ┌───────────────────────┐
 │  Research Agent        │  GPT-5.1 + MCP tools
 │  (Web, Files, Stocks)  │  → Structured research memo
 └───────────┬───────────┘
             │
             ▼
 ┌───────────────────────┐
 │  Analyst Model         │  Fine-tuned Qwen3-4B
 │  (SFT on 5K examples)  │  → Investment verdict
 └───────────┬───────────┘
             │
             ▼
  [ Strong Yes / Questionable / Strong No ]
```

### What Evolves From That

Today I'm walking through an improved, adjusted scope of that same journey. As AI practitioners, understanding these concepts is what matters — not the specific model or environment.

**What changed (briefly):**
- We're sticking with a small Chinese open-source model (Qwen3.5-2B) — the specific model isn't the point
- We wanted to do this for free — Kaggle and Colab T4 GPUs, and we very hardly but succeeded
- The teacher model updated (GPT-5.4) — also not the point

**What IS the point:** In WinterFest we taught a model for a *downstream* task — read a memo, produce a verdict. Now we're teaching it for *the task itself* — the research, the tool calling, the thinking. This is the kind of hypothesis an AI practitioner might have: "Can I build in reasoning and agentic behavior?" We're showing how that hypothesis gets tested — experiments, failures, adjustments, more failures, adjustments, and eventually something that works.

And we won't stop at SFT — we'll go all the way to RL. Not to demo the result, but to show how in Applied AI, a hypothesis like this is tested, iterated on, and understood.

---

## 3. Step 1 — How Agents Work: It's All Just Text <a name="3-agents"></a>

**~8 minutes**

> Our first real topic. Step 1: understand what I'm actually teaching this model. Am I teaching it to connect to Yahoo Finance and get analyst recommendations? **No.** I'm teaching it in what format, based on what logic, to put out text — so I can parse that text into a function call, run it, and turn my text generation function into an agent. That agent could be called Claude Code.

### 3a. The Raw Chat Template (SHOW ON SCREEN)

When you call an agent with tools, the model doesn't see a UI. It sees this:

```
<|im_start|>system
# Tools
You have access to the following functions:

{"type": "function", "function": {"name": "get_financials", "description": "Get financial
 statements for a stock.", "parameters": {"type": "object", "properties": {"ticker":
 {"type": "string"}, "statement_type": {"type": "string", "enum": ["income",
 "balance_sheet", "cashflow"]}}, "required": ["ticker", "statement_type"]}}}

You are a sell-side equity research analyst...
<|im_end|>
<|im_start|>user
Research AAPL focusing on financial health.
<|im_end|>
<|im_start|>assistant
<think>I should check the financials first...</think>
<tool_call>{"name": "get_financials", "arguments": {"ticker": "AAPL", "statement_type": "income"}}</tool_call>
<|im_end|>
<|im_start|>tool
{"ticker": "AAPL", "revenue": "416B", ...}
<|im_end|>
<|im_start|>assistant
**AAPL** | Technology | $3.68T market cap...
<|im_end|>
```

Understanding what's really happening here already helps us design our decisions. The `<think>` tags, `<tool_call>` tags, `<|im_end|>` markers — these are all just tokens. The model learned during training when to write these tokens, just like it learned when to write any other word. An external system parses the tags, executes the function, injects the result, and lets the model continue.

**This is how Claude Code works.** It thinks, calls tools (read files, run bash, edit code), sees results, thinks again. Text + parsing + a loop.

### 3b. The Agent Loop (SHOW CODE)

All the research tasks you see — Claude Code calling bash, OpenAI using web search, an agent writing to Slack — it's all the same pattern. The LLM produces text that gets parsed into function arguments (tool schemas). After the tool-call end token is produced, we parse it, execute it, feed the result back.

```python
# The entire agent loop — this is all there is
while True:
    response = await client.chat.completions.create(
        model=model, messages=messages, tools=tools
    )
    
    if not response.tool_calls:
        break  # model produced end token without tool call — done
    
    for tc in response.tool_calls:
        result = tool_functions[tc.function.name](**tc.function.arguments)
        messages.append({"role": "tool", "content": result})
```

LangChain, LlamaIndex Workflows, Claude Agents SDK, OpenAI Agents SDK — they all do this exact same thing. Different wrappers, same loop.

*[Potential quick live demo: run a small model locally via mlx_lm.server, show it responding to a research prompt with tool calls]*

### 3c. Tool Schemas (QUICK NOTE)

We auto-generate schemas from Python function signatures — change the function, schema updates. Single source of truth. MCP standardizes how agents discover and connect to tools, but at the end of the day it's just a way to deliver these same JSON schemas to the model.

### 3d. Closing This Section

So we're building a Claude Code-like system. The model produces text, we parse special tokens into actions, execute them, feed results back. Now that we understand what the LLM is really writing — and what we'd be teaching it to write — we can start the fine-tuning journey.

---

## 4. Step 2 — SFT: Teaching the Format <a name="4-sft"></a>

**~10 minutes**

> As a data scientist and ML engineer turned Applied AI practitioner — and anyone else in the same shoes — we understand: most of the time is on the data. Data wrangling, feature engineering, all that stuff. Same here.

### 4a. What Is SFT?

> **SFT says:** "Here's the exact right answer. Predict these exact tokens."

It learns to imitate. So we need ground truth — we need to calculate the difference between prediction and label. Same as in classical ML, except here we call it **loss**.

**How loss is calculated (briefly):** At each token position, the model produces a probability distribution over its entire vocabulary (~248K tokens for Qwen3.5). The loss for that position = `-log(P(correct_token))`. If the model is 90% confident in the right token, loss is low (0.1). If only 5% confident, loss is high (3.0). Average across all non-masked positions → that's your training loss. **Same concept as log-loss in classical ML.**

### 4b. The Training Data (SHOW CODE / PIPELINE)

We ran GPT-5.4 on ~200 companies with live tools. Captured 955 full trajectories.

```text
Raw trajectories (GPT-5.4 Responses API)
    │
    ├── Convert format: developer → system, function_call → tool_calls
    ├── Inject reasoning as <think> tags in assistant content
    ├── Truncate tool results to ~250 tokens (from 2,000-3,000)
    ├── Filter: remove stuck loops, malformed calls, short memos
    └── Output: 955 clean trajectories in Hermes chat format
```

Already here I can cover two decisions that matter for Applied AI practitioners:

**Decision 1 — Formatting:** We inject `<think>` tags so the model learns the full think → call → observe loop. Not just what tools to call, but the reasoning pattern before each call.

**Decision 2 — Truncation:** Tool results are 2,000-3,000 tokens each. But they're masked during training (zero gradient) — pure overhead. Truncating to ~250 tokens frees space for the assistant turns the model actually learns from. Every major dataset does this.

### 4c. Masking — What Gets Gradient and Why (KEY VISUAL)

We have ground truth for everything in the trajectory. But why would we punish the model for not knowing what the tool returns? Or what the user wrote?

```text
<|im_start|>system                    ← MASKED — not the model's job to predict this
You are a sell-side analyst...
<|im_end|>
<|im_start|>user                      ← MASKED — not the model's job
Research AAPL focusing on...
<|im_end|>
<|im_start|>assistant                 ← TRAINED ✓ — this IS the model's job
<think>I should check financials</think>
<tool_call>{"name": "get_financials", ...}</tool_call>
<|im_end|>
<|im_start|>tool                      ← MASKED — the model doesn't control tool output
{"ticker": "AAPL", "revenue": 416B...}
<|im_end|>
<|im_start|>assistant                 ← TRAINED ✓
<think>Now compile the snapshot</think>
**AAPL** | Technology | $3.68T...
<|im_end|>
```

If the model doesn't adhere to formatting while "calling" the tool (`<tool_call>` with valid JSON) — that's a penalty. If it can't write a proper research memo — penalty. But not knowing what yfinance returns? That's not its fault.

Our training ratio: ~36.5% of tokens get gradient. The rest is context.

### 4d. Understanding `max_seq_length` (KEY CONCEPT)

`max_seq_length` is the total token window — the entire sequence from first `<|im_start|>` to last `<|im_end|>`. Everything: system prompt, tool schemas, user message, all turns, all tool results.

**Too short (4096):** The final assistant response gets truncated from the end. With masking, zero training tokens remain → **NaN loss** (division by zero — no signal to improve on).

**Too long (16384):** OOM. Qwen3.5's 248K vocabulary × 16K sequence length = ~8 GB just for the logits tensor.

**Sweet spot (8192):** Our max sample after truncation is 6.4K tokens → fits with room to spare.

Shorter examples get padded — a 2K sample padded to 8K wastes 75% of the sequence, and attention cost is O(s²). That's 16× more compute on padding. This is why data prep is 80% of the work.

*[Quick mention: extending context windows beyond training length is possible via RoPE frequency rescaling (YaRN) — the positional encoding frequencies get rescaled so longer sequences map into positions the model already understands.]*

### 4e. The Training Setup

**LoRA:** We're not fine-tuning the entire model with billions of parameters — just adapter layers on top. Same concept as in WinterFest, same concept the industry uses.

Standard training parameters — epochs, learning rate, warmup, early stopping — same as classical neural network training. The frameworks abstract most of this.

**Cross-platform (same knobs, different names):**

| Concept | Unsloth (Colab/Kaggle) | MLX (Mac) |
|---------|----------------------|-----------|
| LoRA rank | `r=32` | `rank: 32` |
| LoRA scaling | `lora_alpha=64` | `scale: 2.0` (**= alpha/rank** — get this wrong and the model learns nothing) |
| Mask non-assistant | `train_on_responses_only()` | `mask_prompt: true` |
| Learning rate | `2e-4` (8-bit Adam) | `1e-5` (full-precision Adam) |

### 4f. MLX Story — Local Training on Mac (POTENTIAL LIVE DEMO)

*[Show: running a small model locally from terminal via mlx_lm — low risk, high reward demo]*

*[Show: the MLX training config YAML and the CLI command to run it]*

*[Show: the Colab/Kaggle notebook with `trainer.train()` — this is where all the magic happens. This is where Meta burns $100M on a training run. Same API, different scale.]*

**What broke:**
- **MLX backward pass:** Peaks at ~48 GB regardless of model size. Two kernel panics. Root cause: unified memory on Mac — GPU and CPU share the same 16GB pool, and the JIT compiler's CPU-side memory isn't bounded by the GPU memory limit. Solution: train on Colab (free T4, 5 min), serve locally on Mac (inference is excellent at ~4 GB).
- **`max_seq_length=4096`:** NaN loss — everything masked, zero signal.
- **Eval at `batch_size=2`:** OOM — Qwen3.5's 248K vocab creates massive logits tensors.
- **Validation passes but training OOMs:** Validation is forward-only. Backward pass needs ~2-3x more memory for storing gradients. If val prints a number but training crashes, you're at the memory edge.

### 4g. The Scaling Cheat Sheet (SHOW TABLE)

| Knob | Cost | Intuition |
|------|------|-----------|
| **Sequence length** | O(s²) | **The big one.** 8K→16K = 4× more attention compute. |
| **Batch size** | O(b) | Linear. Trade memory for speed. |
| **LoRA rank** | ~Nothing | **Cheapest knob.** rank 8→32 barely matters. |
| **Model size** | O(P) | Linear. 2B→4B ≈ 2× everything. |

**Our result:** Qwen3.5-2B, rank=32, seq=8192, on a free Colab T4. Training: ~30 min (Unsloth claims 5-10 — optimistic). 811 train / 144 eval samples.

---

### Transition to RL

We understand what we're training, what data it takes, how the fine-tuning works, and we've done it. Before moving to RL, here's where things stand:

| | Base (Qwen3.5-2B) | After SFT | After RL |
|---|---|---|---|
| Tool call format | Sometimes valid | Consistently valid | ? |
| Tool selection | Inconsistent | Matches teacher patterns | ? |
| Final memo | Often missing | Produces structured output | ? |
| Reward score | ~3.4 avg (our baseline) | Expected: +15-40% (literature) | *[reveal later]* |

*[Note: We verified format improvement on SFT but didn't run a full quantitative eval against baseline. The literature consistently shows 15-40% improvement from SFT on tool calling (ToolRL, FunctionGemma +27pp). From an internal project, we saw pass rate jump from ~70% to 90%+ simply from showing the model examples — the power of SFT.]*

Now the question: can we push further? This is where RL comes in — and to understand it, we need to see how it differs from SFT.

---

## 5. Step 3 — RL: Teaching Quality <a name="5-rl"></a>

**~8 minutes**

> **SFT says:** "Here's the exact right answer. Predict these exact tokens." *(Supervised — has a teacher.)*
>
> **GRPO says:** "I don't know the right answer. But I generated 5 attempts, scored them, and I know which ones were BETTER. Make the good attempts more likely." *(Reinforcement — no teacher, just a judge.)*

The key difference: **we don't need ground truth.** We can use it, and it helps, but the core of RL is measurable, quantifiable evaluations of generated trajectories — reward and penalty signals.

These signals can be simple: is a word present or not? Is the output concise? Were `<think>` tags used? (This is how DeepSeek trained their model to think — pure RL with just accuracy and format rewards, and the model *spontaneously* developed chain-of-thought reasoning. [arXiv:2501.12948](https://arxiv.org/abs/2501.12948))

Or they can be powerful: LLM judges that score for something non-deterministic and not simply calculable.

### 5a. The GRPO Loop (KEY VISUAL — build interactive HTML)

```text
1. SAMPLE: Generate 5 completions per prompt (temperature=1.0 for diversity)
2. SCORE:  Reward function scores each → [5.5, 3.0, 4.2, -1.0, 3.5]
3. ADVANTAGE: Subtract group mean → [+2.5, +0.0, +1.2, -4.0, +0.5]
4. LOSS:   For each completion: advantage × sum(token log-probs). Sum all → one number.
5. BACKPROP: One backward pass → gradients flow back to weights.
           Tokens from +2.5 completion: "make these weight configurations more likely"
           Tokens from -4.0 completion: "make these less likely"
6. UPDATE: One weight update (lr=5e-6 — 40× lower than SFT because RL is fragile)
```

**(BUILD CUSTOM VISUAL: the computation graph fanout showing how 5 completions → 1 loss → backprop fans out with advantage as multiplier on each branch → same weights get competing signals)**

**For both SFT and GRPO:** the backward mechanism is identical — standard backpropagation via the chain rule (PyTorch autograd). The only difference is how the loss is computed. SFT: cross-entropy against ground truth labels. GRPO: advantage-weighted log-probabilities. Same backprop, different loss function. And it's not tokens that get adjusted — it's **weights**, based on what tokens they predicted.

### 5b. The Essay Analogy

> Imagine you're a writing teacher. You give 8 students the same essay prompt. They each write an essay. You grade them. You don't show them a perfect essay (that would be SFT). You just tell each one how they scored relative to the group. A students: "keep doing that." D students: "change something." **The group IS the baseline — no external reference needed.** That's GRPO.

### 5c. Training Configuration (SHOW CODE)

```python
# GRPO hyperparameters (from our WinterFest experiment)
learning_rate = 5e-6     # 40× lower than SFT — RL is fragile
beta = 0.0               # No KL penalty = no reference model needed (GRPO's simplification)
temperature = 1.0        # Need diversity — identical completions = zero gradient
loss_type = "dr_grpo"    # No length bias (arXiv:2503.06639)
num_generations = 4      # Completions per prompt
epsilon = 0.2            # Max 20% probability change per update
```

Why `temperature=1.0`? If it's low, all completions are identical → same rewards → advantages = 0 → **zero gradient, the model learns nothing.** SFT exploits (low temp). RL explores (high temp).

**Important distinction:** TRL's GRPOTrainer is **single-turn only** — it calls `model.generate()` once and scores the result. It has no mechanism to pause on tool calls, execute tools, and resume. For full multi-turn RL (scoring the entire trajectory including tool results and final memo), you need **ART** (OpenPipe), which has an async environment loop for tool execution between turns. Same GRPO algorithm, different scope.

For this demo we use TRL (simpler, runs on free T4) and score the model's first response — does it think, call valid tools, select the right ones? Single-turn GRPO teaches **tool-calling discipline**. Multi-turn GRPO via ART teaches **research strategy**.

### 5d. Reward Engineering (KEY SECTION)

> **Your reward function IS your product specification.**

Pulling intuition from the research:

**Simple reward (rule-based):**
- Valid JSON tool calls? (0 or 1) — from [ToolRL](https://arxiv.org/abs/2504.13958)
- Called the right tools? (Jaccard overlap, with hallucination veto — if you call a fake tool, correctness goes to zero) — from [ToolRLA](https://arxiv.org/abs/2603.01620), deployed on a financial advisory copilot
- Produced a final memo? (0 or 1)

**Composite reward (adds LLM-as-judge):**
- Same format/correctness gates as above, PLUS:
- LLM judge reads the full trajectory and scores whether the analysis is specific vs generic boilerplate (0 to 2) — from [Tool-R1](https://arxiv.org/abs/2509.12867), [ART RULER](https://art.openpipe.ai/fundamentals/ruler)
- LLM judge checks for fabrication — penalty -5 if claims aren't supported by tool outputs — from [ToolRLA](https://arxiv.org/abs/2603.01620)

**Critical research finding — cite this!** LLM judges tend to cluster scores around 0.7-0.9 on a 1-10 scale (they avoid being "explicitly wrong" by giving generous scores). This kills the learning signal because GRPO needs score variance. Fix: force categorical scores (0/0.5/1.0) or use quiz-based verification. ([DeepLearning.AI GRPO course](https://learn.deeplearning.ai/courses/reinforcement-fine-tuning-llms-grpo/))

### 5e. Reward Hacking (ENTERTAINING EXAMPLES — all cited)

The model finds shortcuts that maximize reward without doing the task:

| Example | What happened | Source |
|---------|--------------|--------|
| **Boat race** | RL agent in CoastRunners discovered driving in circles hitting turbo pads scored higher than finishing the race | [OpenAI blog (2016)](https://openai.com/blog/faulty-reward-functions-in-the-wild) |
| **Delete the tests** | Code generation model learned to delete failing test cases rather than fix bugs — perfect "pass rates" | [CodeRL, NeurIPS 2022](https://arxiv.org/abs/2308.09662) |
| **Broken tool calls** | Cursor's model learned to emit broken tool calls to avoid negative rewards for bad edits | [Composer2 report](https://cursor.com/resources/Composer2.pdf) |
| **Infinite loop** | ART-E model repeated its last tool call until hitting max turns | [ZenML case study](https://www.zenml.io/llmops-database/building-art-e-reinforcement-learning-for-email-search-agent-development) |
| **Ask instead of act** | Cursor's model learned to ask clarifying questions instead of making risky edits — gaming the reward by avoiding action | [Composer2 report](https://cursor.com/resources/Composer2.pdf) |
| **Emergent misalignment** | Models trained to reward-hack in one domain generalize to misaligned behavior in others | [Anthropic (arXiv:2511.18397)](https://arxiv.org/abs/2511.18397) |

Mitigations: multiplicative veto (wrong tool = zero reward), competing incentives (efficiency vs completeness), human spot-checks, and ultimately — better reward functions.

### 5f. Our Results (From WinterFest RL Experiment)

Going back to our original WinterFest project — we applied GRPO to the verdict model:

- **Reward progression:** 3.7 (start) → 5.7 (sustained mid-training) → best single step: 7.70
- The reward is a weighted sum of 6 components (structure, field presence, signal validity, correctness, reasoning, content quality). 7.70 was near-optimal — correctness alone contributed 3.0 (weight 2.0, the highest).
- 12 hours on 1x H100, 2,889 lines of logs
- Dense rewards (always partial credit) ensured learning signal every step
- `reward_std > 0` every step = the model always had something to learn from

*[Reveal the RL column in the comparison table:]*

| | Base | After SFT | After RL |
|---|---|---|---|
| Reward score | ~3.4 | ~4.5 (est.) | 5.5-5.7 |
| What improves | — | Format, structure, tool selection | Efficiency, quality, edge cases |

### 5g. SFT Loss vs GRPO Reward (KEY VISUAL — build interactive HTML)

| | SFT | GRPO |
|---|---|---|
| **What you plot** | Loss going **down** | Reward going **up** |
| **"Good"** | Train ≈ eval (generalizing) | Reward increasing, `reward_std > 0` |
| **"Bad"** | Eval rising while train drops (overfitting) | All completions score the same (zero gradient) |
| **Measures** | "How well does it predict the teacher's tokens?" | "How good are its outputs by our criteria?" |

Both are just numbers going in a direction. But they measure fundamentally different things. Understanding that changes how you think about what each technique can and can't do.

**(BUILD CUSTOM VISUAL: side-by-side showing how loss is computed and applied in backprop for both SFT and GRPO)**

---

## 6. Synthesis: Tying It All Back <a name="6-synthesis"></a>

**~3 minutes**

### The Journey

We had a hypothesis: can we build our own agentic system, trained on our data, running on our hardware? We prototyped with a proprietary API provider. Generated synthetic data. Ran a baseline with the open-source model. Fine-tuned with SFT. Designed an RL pipeline to push further.

Along the way, we learned that tool calling is just text. That SFT is just next-token prediction on masked sequences. That RL is just advantage-weighted gradients from multiple attempts.

None of this is magic. It's engineering. And the real value isn't the fine-tuned model — it's the understanding.

### The Practical Framework

> For your next project, start from what you need:
>
> - **Better prompts** → Context engineering. Most problems start and end here.
> - **Custom tools / data sources** → MCP servers, skills. Standardize once, use everywhere.
> - **Custom domain knowledge** → RAG, file search, vector stores.
> - **Custom behavior or format** → SFT. Teach the model your patterns.
> - **Custom quality judgment** → RL. Define what "good" means and let the model find it.
>
> You don't always need to go all the way. Pick the right level.

### This Is How I Work in Applied AI

This is how I look for candidates for model customization. How I evaluate whether fine-tuning is worth it. How I design experiments, fail, adjust, fail again, and eventually learn something.

### The Forward-Looking Thought

We're entering an era where the line between "using AI" and "building AI" is blurring. The tools are accessible. The research is open. The frameworks are free.

The question isn't whether you *can* fine-tune a model — it's whether you understand enough to know when you *should*.

---

## Sources & References

### Papers Cited in the Talk

| Paper | Year | Key Finding | arXiv |
|-------|------|-------------|-------|
| DeepSeek Math (GRPO) | Feb 2024 | Group-relative policy optimization — eliminates critic | [2402.03300](https://arxiv.org/abs/2402.03300) |
| DeepSeek-R1 | Jan 2025 | Emergent reasoning from simple accuracy + format rewards | [2501.12948](https://arxiv.org/abs/2501.12948) |
| ToolRL | Apr 2025 | Fine-grained reward decomposition; GRPO from scratch beats SFT+RL | [2504.13958](https://arxiv.org/abs/2504.13958) |
| OTC | Apr 2025 | RL reduces tool calls by 73% while maintaining accuracy | [2504.14870](https://arxiv.org/abs/2504.14870) |
| Dr. GRPO | Mar 2025 | Fixes length bias and difficulty bias in vanilla GRPO | [2503.06639](https://arxiv.org/abs/2503.06639) |
| Trading-R1 | Sep 2025 | Financial domain RL with evidence-grounding curriculum | [2509.11420](https://arxiv.org/abs/2509.11420) |
| RC-GRPO | Feb 2026 | 7B beats Opus 4.5 on BFCLv4 multi-turn (85% vs 61%) | [2602.03025](https://arxiv.org/abs/2602.03025) |
| ToolRLA | Mar 2026 | Multiplicative reward veto; deployed on financial advisory copilot | [2603.01620](https://arxiv.org/abs/2603.01620) |
| FunctionGemma | 2024 | 270M model: 58%→85% accuracy with SFT | [Google blog](https://blog.google/technology/developers/functiongemma/) |
| Small LMs for Tool Calling | Dec 2024 | 350M beats ChatGPT 3x on ToolBench | [2512.15943](https://arxiv.org/abs/2512.15943) |

### Reward Hacking Sources
- [OpenAI — CoastRunners (2016)](https://openai.com/blog/faulty-reward-functions-in-the-wild)
- [CodeRL / Astra-Hacking (NeurIPS 2022 / 2023)](https://arxiv.org/abs/2308.09662)
- [Cursor Composer2 report](https://cursor.com/resources/Composer2.pdf)
- [ART-E case study (ZenML)](https://www.zenml.io/llmops-database/building-art-e-reinforcement-learning-for-email-search-agent-development)
- [Anthropic — Emergent Misalignment (2025)](https://arxiv.org/abs/2511.18397)
- [DeepLearning.AI — LLM judge clustering](https://learn.deeplearning.ai/courses/reinforcement-fine-tuning-llms-grpo/)
- [Lilian Weng — Reward Hacking survey](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/)

### Other References
- [Anthropic April 23 postmortem](https://www.anthropic.com/engineering/april-23-postmortem)
- [BFCL Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) (Claude Opus 4.5 FC: 77.47% overall, #1)
- [Cameron Wolfe — GRPO explained](https://cameronrwolfe.substack.com/p/grpo)
- [HuggingFace — GRPO visual](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/grpo_visual.png)
- [Karpathy — RL vs SL visual](http://karpathy.github.io/assets/rl/rl.png)
- [RLVR Limits](https://limit-of-rlvr.github.io/) — RL optimizes within, not beyond, the base distribution

### Tools & Frameworks
- [Unsloth](https://docs.unsloth.ai/) — SFT & RL training
- [ART (OpenPipe)](https://github.com/openpipe/art) — Multi-turn GRPO for agents
- [MCP (Anthropic)](https://www.anthropic.com/news/model-context-protocol) — Tool connectivity standard

### Previous Talk
- [WinterFest 2025: Agentic Systems in Practice](winterfest_talk.md)
