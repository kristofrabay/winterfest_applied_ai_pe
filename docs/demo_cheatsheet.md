# Demo Cheatsheet — BBB Talk

*Exact commands, notebooks, cells, and fallbacks for each demo moment in the talk.*

---

## Pre-Talk Setup (Do 10 min before)

```bash
# 1. Start local model server (needed for D1 and optionally D6)
uv run mlx_lm.server \
  --model mlx-community/Qwen3.5-2B-4bit \
  --port 8080 \
  --chat-template-args '{"enable_thinking":true}' \
  --prompt-cache-size 4

# 2. Warm up the cache (first request to thinking models spirals without this)
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Say hello."}], "max_tokens": 10}'

# 3. Verify .env has OPENAI_API_KEY (needed for D2 teacher demo)
cat .env | grep OPENAI_API_KEY

# 4. Open these in browser tabs (ready to switch to):
#    - Colab notebook: nb/bbb/_phase_4_sft_colab.ipynb (with frozen outputs)
#    - Kaggle notebook: nb/bbb/_phase_4_sft_kaggle.ipynb (with frozen outputs)
```

---

## D1: Local OSS Model Inference (Section 3b — Agents)

**Risk:** Very low | **Reward:** High — "an open source model running on my laptop"

**What:** Show Qwen3.5-2B responding to a research prompt locally.

**Option A — curl from terminal (simplest):**
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful equity research analyst. Be concise."},
      {"role": "user", "content": "What are the main revenue segments for Apple (AAPL)?"}
    ],
    "max_tokens": 500,
    "temperature": 0.6
  }' | python -m json.tool
```

**Option B — AsyncOpenAI client (matches talk's code, more impressive):**
Use `nb/bbb/_phase_3_baseline.ipynb`, cells 2-7:
- Cell 2: imports + client setup (`base_url="http://localhost:8080/v1"`)
- Cell 3: warm-up request
- Cell 6: simple question test (no tools)
- Cell 7: print result

**Option C — with tool calling (most impressive, slightly higher risk):**
Use `nb/bbb/_phase_3_baseline.ipynb`, cell 10:
- Runs `run_tool_calling_agent_chat()` with full tool schemas
- Model calls tools, gets real yfinance data, writes memo
- Takes ~45 seconds per ticker on Mac

**Fallback:** Screenshot of cell 10's output (AAPL research with 4 tool calls, reward 4.5)

---

## D2: Teacher Agent — GPT-5.4 Calling Tools Live (Section 3b — Agents)

**Risk:** Medium (needs network + OpenAI API) | **Reward:** High — shows the full agent loop

**What:** Run the teacher on one ticker, watch it reason and call tools in real time.

**Notebook:** `nb/bbb/_phase_1_teacher.ipynb`
- **Cell 2:** Setup (imports, client, model = "gpt-5.4")
- **Cell 8:** Single ticker run — `run_tool_calling_agent()` with NVDA
  - Shows reasoning summaries, tool calls with args, final "Agent finished"
  - Takes ~30-60 seconds
- **Cell 12:** Display the final memo as formatted markdown

**Pre-test ticker:** NVDA (tested, works, ~5 tool calls, rich output)

**Alternative ticker if NVDA fails:** AAPL or MSFT (both tested in cell 16)

**Fallback:** The frozen output in cell 8 already shows the full trajectory:
```
[1] Reasoning: Researching financials for NVDA...
[1] Called get_financials(ticker='NVDA', statement_type='income', period='quarterly')
[1] Called get_financials(ticker='NVDA', statement_type='balance_sheet', period='quarterly')
[1] Called get_price_history(ticker='NVDA', period='1y', interval='1wk')
[1] Called get_recommendations(ticker='NVDA', months_back=12)
[1] Called get_stock_news(ticker='NVDA')
[2] Agent finished — produced final response
```

---

## D4: Show Training Config + CLI Command (Section 4f — SFT)

**Risk:** Very low (just showing files) | **Reward:** Medium — "this is all it takes"

**What:** Show the MLX training config YAML and the command to run it.

**File to show:** `nb/bbb/sft_config.yaml`
```bash
# Open in editor or cat it
cat nb/bbb/sft_config.yaml
```

**CLI command to show (don't run — just display):**
```bash
MLX_DISABLE_COMPILE=1 uv run mlx_lm.lora --config nb/bbb/sft_config.yaml
```

**Talking point:** "This YAML and this one command — that's all it takes to fine-tune locally. Same concept whether you're on a Mac or a $500K GPU cluster. The frameworks abstract the complexity."

**Fallback:** None needed — this is just showing static files.

---

## D5: Colab/Kaggle Notebook with `trainer.train()` (Section 4f — SFT)

**Risk:** Very low (frozen notebook in browser) | **Reward:** High — "this is where Meta burns $100M"

**What:** Open the SFT notebook, scroll to the training cell, show the setup and frozen output.

**Notebook:** `nb/bbb/_phase_4_sft_colab.ipynb` (or `_phase_4_sft_kaggle.ipynb`)

**Key cells to show:**
- **Cell 10:** Model loading — `FastVisionModel.from_pretrained("unsloth/Qwen3.5-2B")` + LoRA config
  - Shows: VRAM used: 2.1 GB, model patching output
- **Cell 12:** Data formatting — `tokenizer.apply_chat_template()` with tools
  - Shows: "Train: 811 | Eval: 144", token length stats, first 500 chars of formatted text
- **Cell 14:** Trainer setup — `SFTTrainer()` + `train_on_responses_only()`
  - Shows: the masking setup
- **Cell 15:** Masking verification — "Training tokens: 1995, Masked tokens: 3476, Train ratio: 36.5%"
  - **This is the money cell** — directly validates the masking diagram
- **Cell 16:** `trainer.train()` — "This is where all the magic happens"
  - Shows: Unsloth banner, step count, timing

**Talking point at cell 16:** "trainer.train(). That's it. This is the same API call whether you're training a 2B model on a free T4 or Meta is training Llama on 16,000 H100s. Same `trainer.train()`. Different scale."

**Fallback:** None needed — frozen outputs are the demo.

---

## D6: Fine-Tuned Model vs Base (Section 4f — Optional)

**Risk:** Medium-High (need SFT model converted and served) | **Reward:** Very high if it works

**What:** Same prompt, base model vs SFT model, side by side.

**Prerequisites (test before talk):**
```bash
# 1. Convert SFT model to MLX format (if not already done)
uv run mlx_lm.convert --hf-path nb/bbb/sft_results/qwen35_2b_sft_merged \
    --mlx-path models/mlx_sft_fused -q

# 2. Serve the SFT model (different port from base)
uv run mlx_lm.server --model models/mlx_sft_fused --port 8081 \
    --chat-template-args '{"enable_thinking":true}' \
    --prompt-cache-size 4
```

**Demo:** Run same prompt against both:
```python
# Base model (port 8080)
base_client = AsyncOpenAI(base_url="http://localhost:8080/v1", api_key="none")

# SFT model (port 8081)
sft_client = AsyncOpenAI(base_url="http://localhost:8081/v1", api_key="none")
```

Use `nb/bbb/_phase_4_sft_mlx.ipynb` cells 13-15 for the SFT eval loop.

**Status:** NOT YET TESTED. Need to verify:
- [ ] SFT model converts to MLX successfully
- [ ] SFT model serves without errors
- [ ] SFT model produces tool calls (the whole point)
- [ ] Noticeable difference from base model

**Fallback:** Skip entirely. The frozen Colab notebook outputs + literature numbers tell the story.

---

## Slide-to-Code Mapping

| Talk Section | What to show | File | Cell/Line |
|-------------|-------------|------|-----------|
| 3a. Raw chat template | The full `<\|im_start\|>` sequence | Talk markdown itself (pre-formatted) | — |
| 3b. Agent loop | While loop code | `nb/bbb/agent.py` | Lines 56-110 (Responses API) or 177-263 (Chat Completions) |
| 3b. Same loop, different client | `base_url` swap | `nb/bbb/agent.py` | Compare `_phase_1_teacher.ipynb` cell 2 vs `_phase_3_baseline.ipynb` cell 2 |
| 3c. Auto-generated schemas | `_build_tool_schema()` | `nb/bbb/tools.py` | Lines 31-77 |
| 3c. Schema output | Printed schemas | `_phase_1_teacher.ipynb` | Cell 6 (output shows all 4 tools) |
| 4a. Loss explained | Talk markdown | — | — |
| 4b. Data pipeline | Pipeline diagram | Talk markdown | — |
| 4b. Data prep code | `responses_to_hermes()` | `nb/bbb/helpers__data_gen.py` | Lines 185-291 |
| 4b. Truncation | `truncate_tool_output()` | `nb/bbb/helpers__data_gen.py` | Lines 143-155 |
| 4c. Masking diagram | Color-coded tokens | Talk markdown (pre-formatted) or **V1 visual** | — |
| 4c. Masking verification | "36.5% training ratio" | `_phase_4_sft_colab.ipynb` | Cell 15 |
| 4d. max_seq_length | Token budget breakdown | Talk markdown | — |
| 4e. Cross-platform table | Unsloth vs MLX | Talk markdown | — |
| 4e. MLX config | YAML file | `nb/bbb/sft_config.yaml` | Full file |
| 4f. `trainer.train()` | SFT training cell | `_phase_4_sft_colab.ipynb` | Cell 16 |
| 4g. Scaling cheat sheet | Table | Talk markdown | — |
| 5a. GRPO loop | Diagram | **V3 visual** (adapt HuggingFace) | — |
| 5a. Computation graph | Fanout diagram | **V4 visual** (custom build) | — |
| 5c. GRPO config | Hyperparameters + footguns table | `nb/bbb/_phase_5_rl.ipynb` | Cell `2d199430` (GRPOConfig) + markdown cell `7b539e40` (footguns table) |
| 5c. Multi-turn limitation | TRL vs ART explanation | `nb/bbb/_phase_5_rl.ipynb` | Intro markdown cell `573816aa` + final cell `fdcc0ffc` |
| 5d. Reward function (TRL signature) | `tool_calling_reward_func()` | `nb/bbb/_phase_5_rl.ipynb` | Cell `ca82054c` |
| 5d. Simple reward | Code | `nb/bbb/helpers__inference.py` | Lines 194-274 |
| 5d. Composite reward | Code | `docs/research_reward_design.md` | Reward B section |
| 5e. Reward hacking | Table | Talk markdown | — |
| 5f. WinterFest results | Numbers | Talk markdown + `docs/grpo_learnings_winterfest.md` | — |
| 5g. SFT vs GRPO visual | Side-by-side | **V5 visual** (custom build) | — |
| 6. BYOAI spectrum | Framework diagram | **V6 visual** | — |

---

## Emergency Fallback Plan

If everything breaks (network down, laptop freezes, etc.):

1. The talk markdown has ALL the code blocks inline — you can present from the markdown alone
2. Every notebook has frozen outputs — even without running, the cells show results
3. The key diagrams (masking, GRPO loop, scaling table) are all text-based in the talk markdown
4. The only true "live" moments are D1 and D2 — everything else is frozen output or static files

**Minimum viable demo:** Open `_phase_4_sft_colab.ipynb` in browser and walk through cells 10-16 with frozen outputs. This covers model loading, data formatting, masking verification, and training — all the SFT content without running anything.
