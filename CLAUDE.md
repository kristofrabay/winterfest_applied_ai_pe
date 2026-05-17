# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Conference demo project for Applied AI in equity investment research — presented across two talks at Data Science Festival:

1. **WinterFest 2025** — Two-stage pipeline: GPT-5.1 research agent + fine-tuned Qwen3-4B analyst model
2. **Big Birthday Bash 2026** — Teaching a small model to *be* the research agent itself via tool-calling SFT + RL

*Kristof Rabay — Applied AI @ The Carlyle Group*

### Talk Thesis (BBB 2026)

> Understanding how models get trained — SFT, RL, tool calling, masking — helps tremendously in how we *apply* them. This is Applied AI.

The talk is educational code-showing (no live running), walking through building a custom agentic system from context engineering to RL. Format: ~35 min content + 5-10 min Q&A.

### BBB Pipeline Status (as of April 2026)

| Phase | Notebook | Status | Notes |
|-------|----------|--------|-------|
| 1. Teacher Agent | `_phase_1_teacher.ipynb` | **DONE** | GPT-5.4 via Responses API, tested on multiple tickers |
| 2. Data Generation | `_phase_2_data_gen.ipynb` | **DONE** | 955 trajectories in `trajectories_sft.jsonl` |
| 3. Baseline | `_phase_3_baseline.ipynb` | **DONE** | 5 MLX baseline runs, avg reward 3.40 |
| 4. SFT | `_phase_4_sft_colab.ipynb` / `_kaggle` / `_mlx` | **DONE** | Trained on Colab+Kaggle (Unsloth). MLX training failed (48GB backward pass). Model weights saved but no quantitative eval against baseline. |
| 5. RL | `_phase_5_rl.ipynb` | **SKELETON** | Demo notebook with GRPO config, reward function, TRL setup. Not run. TRL is single-turn only — full multi-turn needs ART (OpenPipe). |

### BBB Code Structure (`nb/bbb/`)

- `tools.py` — Stock tool functions + auto-generated OpenAI Responses API schemas (single source of truth)
- `agent.py` — Two async agent loops: `run_tool_calling_agent()` (Responses API for GPT-5.x) and `run_tool_calling_agent_chat()` (Chat Completions for local servers — Ollama, mlx-lm, llama-server)
- `helpers__data_gen.py` — System prompt, ticker list (~200), format conversion (Responses API → Hermes), truncation, quality filtering
- `helpers__inference.py` — `<tool_call>` parsing, local Unsloth agent loop, composite reward function
- `birthday_bash_talk.md` — The talk outline (source of truth for presentation content)
- Notebooks import from these shared modules

### BBB Documentation (`docs/`)

| Doc | Purpose |
|-----|---------|
| `takeaways_sft.md` | Consolidated SFT learnings — data, masking, max_seq_length, what broke |
| `takeaways_rl.md` | Consolidated RL learnings — GRPO mechanics, reward engineering, benchmarks |
| `research_sft_rl_benchmarks.md` | Citable SFT/RL deltas from papers (verified BFCL numbers) |
| `research_reward_design.md` | Reward function patterns — simple + composite, with LLM-as-judge |
| `research_grpo_explained.md` | Educational GRPO explainer — essay analogy, token→params, computation graph |
| `research_financial_rl_rewards.md` | Survey of 13 papers with financial domain reward functions |
| `research_reward_examples.md` | Production RL examples — Cursor, ART-E, RULER, reward hacking |
| `grpo_learnings_winterfest.md` | GRPO experiment retrospective from Baseten H100 run |
| `demo_cheatsheet.md` | Exact commands, notebooks, cells for each talk demo |

### Key Technical Decisions

- **Model:** Qwen3.5-2B — trains in ~30 min on free Colab T4, serves locally on Mac via mlx_lm.server
- **API:** OpenAI Responses API for teacher; Chat Completions for local inference
- **Tool schemas:** Auto-generated from function signatures via `_build_tool_schema()`
- **SFT data:** Tool outputs truncated to ~250 tokens (masked tokens = zero gradient = pure overhead)
- **max_seq_length:** 8192 (compress data to fit; 4096 → NaN loss, 16384 → OOM)
- **Training:** Unsloth on Colab/Kaggle. MLX for inference only (backward pass ~48GB on any Qwen3.5).
- **RL:** TRL GRPOTrainer for single-turn, ART for multi-turn. GRPO from scratch can outperform SFT+RL (ToolRL finding).
- **Qwen3.5 Jinja template:** expects tool call `arguments` as dict, not JSON string

### Qwen3.5 Inference Parameters (Critical)

- **`temperature=0.6, top_p=0.95, presence_penalty=1.5`** — official Qwen3 thinking mode settings
- **Never use greedy decoding** (`temperature=0`) with thinking models — infinite loops
- **Never mention thinking/nothink in prompts for Qwen3.5** — Qwen3.5 doesn't support `/think` `/nothink` soft switches (Qwen3-only). Meta-instructions cause spiraling.
- **`--prompt-cache-size 4`** on 16GB Mac — default 10 causes OOM
- **Warm-up request required** after server start — cold KV cache causes 0% reuse and long reasoning ([mlx-lm#1042](https://github.com/ml-explore/mlx-lm/pull/1042))

## Commands

```bash
# Environment setup
uv sync

# Local model serving (inference on Mac)
uv run mlx_lm.server --model mlx-community/Qwen3.5-2B-4bit --port 8080 \
    --chat-template-args '{"enable_thinking":true}' --prompt-cache-size 4

# WinterFest notebooks
python nb/winterfest/tools/mcp/stock_server.py  # MCP server (port 8001, must run first)
jupyter notebook nb/winterfest/agent.ipynb

# BBB notebooks
jupyter notebook nb/bbb/_phase_1_teacher.ipynb    # Teacher agent (GPT-5.4)
jupyter notebook nb/bbb/_phase_2_data_gen.ipynb    # Bulk data generation
jupyter notebook nb/bbb/_phase_3_baseline.ipynb    # Raw model baseline (MLX)
jupyter notebook nb/bbb/_phase_4_sft_colab.ipynb   # SFT training (Colab)
jupyter notebook nb/bbb/_phase_4_sft_kaggle.ipynb  # SFT training (Kaggle)
jupyter notebook nb/bbb/_phase_4_sft_mlx.ipynb     # MLX config + eval (Mac)
jupyter notebook nb/bbb/_phase_5_rl.ipynb          # RL via GRPO (demo skeleton)
```

## Architecture Details

### WinterFest Pipeline
- **Research Agent** (`nb/winterfest/agent.ipynb`) — `openai-agents` SDK with `gpt-5.1`, MCP stock tools, web search, file search, code interpreter
- **Training Data** (`nb/winterfest/training_data_generator.ipynb`) — ~5,000 synthetic verdicts via `gpt-4.1-mini`
- **Analyst Model** (`nb/winterfest/training_recipe.ipynb`) — Unsloth SFT on `Qwen3-4B-Thinking`

### BBB Pipeline
- **Teacher** — GPT-5.4 via Responses API with reasoning summaries, direct yfinance tool calls (no MCP)
- **Data** — 955 trajectories, Responses API → Hermes format conversion, tool output truncation to ~250 tokens
- **SFT** — Unsloth on Colab/Kaggle, `train_on_responses_only` masking, ~36.5% tokens get gradient
- **RL** — TRL GRPOTrainer (single-turn scoring), reward function scores tool-calling quality

### MCP Stock Server (`nb/winterfest/tools/mcp/stock_server.py`)
- Built with `fastmcp`, HTTP port 8001
- 4 yfinance tools: `get_stock_news`, `get_financials`, `get_price_history`, `get_recommendations`
- Used by WinterFest agent only — BBB uses direct tool calls via `tools.py`

## Key Dependencies

- `openai` + `openai-agents[viz]` — Agent framework and OpenAI API
- `fastmcp` — MCP server framework
- `yfinance` (pinned to 0.2.62) — Stock data source
- `unsloth` — SFT/RL training (Colab/GPU only, not in pyproject.toml)
- `tiktoken` — Token counting for truncation
- `tenacity` + `limiter` — Rate limiting and retry logic

## Environment

- Python 3.11 (managed via `uv`)
- Requires `.env` with `OPENAI_API_KEY` and `VECTOR_STORE_ID`

## Important Conventions

- Training data uses Hermes chat format with `<think>` tags for reasoning traces
- Tool schemas auto-generated — never hand-write JSON schemas
- The talk outline (`nb/bbb/birthday_bash_talk.md`) is the source of truth for all presentation content
- All research is saved in `docs/` with full citations — verify against original papers before presenting
