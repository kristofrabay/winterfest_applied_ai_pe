# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Conference demo project for **"Agentic Systems in Practice: Applying AI to Investments"** — originally presented at Data Science Festival WinterFest 2025, now being extended for a follow-up conference talk.

The system is a **two-stage pipeline** for automated equity investment research:
1. **Research Agent** (`nb/winterfest/agent.ipynb`) — GPT-5.1-powered agent with MCP tools that generates structured research memos on target companies
2. **Analyst Model** — A fine-tuned small language model (Qwen3-4B) that reads research memos and produces investment verdicts (`Strong Yes` / `Questionable` / `Strong No`) with chain-of-thought reasoning

### Current Conference Goal (Big Birthday Bash — BBB)

Teach **Qwen3.5** (2B locally on Mac, 4B on GPU) to be the research agent itself via tool-calling fine-tuning + RL:
1. **Teacher Agent** (`nb/bbb/_phase_1_teacher.ipynb`) — GPT-5.4 via Responses API with reasoning, generates training trajectories. DONE.
2. **Data Generation** (`nb/bbb/_phase_2_data_gen.ipynb`) — Run teacher on ~200 companies, save full tool-calling trajectories
3. **Baseline** (`nb/bbb/_phase_3_baseline.ipynb`) — Run raw Qwen3.5 with tools to establish "before" metrics
4. **SFT** (`nb/bbb/_phase_4_sft.ipynb`) — Fine-tune Qwen3.5 on trajectories via MLX (Mac) or Unsloth (GPU)
5. **RL** (`nb/bbb/_phase_5_rl.ipynb`) — GRPO via ART (OpenPipe) to refine tool-calling behavior

### BBB Code Structure (`nb/bbb/`)
- `tools.py` — Stock tool functions + auto-generated OpenAI Responses API schemas (single source of truth)
- `agent.py` — Two async agent loops: `run_tool_calling_agent()` (Responses API for GPT-5.x) and `run_tool_calling_agent_chat()` (Chat Completions for local servers — Ollama, mlx-lm, llama-server)
- `helpers__data_gen.py` — System prompt, ticker list, format conversion (Responses API → Hermes), truncation, quality filtering
- `helpers__inference.py` — `<tool_call>` parsing, local Unsloth agent loop, composite reward function
- Notebooks import from these shared modules

### Key Technical Decisions
- **API:** OpenAI Responses API (not Chat Completions) for teacher; Chat Completions for local inference (mlx-lm, Ollama, llama-server)
- **Tool schemas:** Auto-generated from function signatures via `_build_tool_schema()` — change the function, schema updates automatically
- **SFT data:** Tool outputs must be truncated to ~500-800 tokens before training (standard practice — all major datasets do this). Raw yfinance responses are 2000-3000 tokens each, which wastes compute on masked tokens.
- **max_seq_length:** 8192 for SFT (compress data to fit, don't expand window for zero-loss masked tokens)
- **16GB Mac training limits:** Qwen3.5-4B OOMs during backward pass at any useful sequence length. Use Qwen3.5-2B for local training — fits 16K context with full LoRA (`rank=32, num_layers=-1`). 4B freezes the Mac (Metal swaps to SSD instead of OOM).
- **Qwen3.5 Jinja template:** expects tool call `arguments` as dict, not JSON string. The `to_mlx_format()` function in the Phase 4 notebook handles this conversion.

### Qwen3.5 Inference Parameters (Critical)
Thinking models require specific sampling parameters to avoid infinite reasoning loops:
- **`temperature=0.6, top_p=0.95, presence_penalty=1.5`** — official Qwen3 recommendations for thinking mode
- **Never use greedy decoding** (`temperature=0`) with thinking models — causes infinite repetition
- **Never mention thinking/nothink in prompts for Qwen3.5** — the `/think` and `/nothink` soft switches are Qwen3-only. Qwen3.5 does not support them. Including meta-instructions about thinking in the system prompt causes the model to spiral into reasoning loops about whether it should be reasoning.
- **`--prompt-cache-size 4`** when running mlx_lm.server on 16GB Apple Silicon — default of 10 causes OOM with concurrent requests
- **Warm-up request required** after server start for thinking models — first request has 0% KV cache reuse and produces extremely long reasoning. Send a throwaway request to prime the cache before real work ([mlx-lm#1042](https://github.com/ml-explore/mlx-lm/pull/1042))

Full plan: `docs/tutorial_content/claude_plan.md`

## Commands

```bash
# Environment setup
uv sync                                    # Install all dependencies
python nb/winterfest/tools/mcp/stock_server.py  # Start MCP server on port 8001 (must be running before winterfest agent notebook)

# WinterFest notebooks (original talk)
jupyter notebook nb/winterfest/agent.ipynb
jupyter notebook nb/winterfest/training_data_generator.ipynb
jupyter notebook nb/winterfest/training_recipe.ipynb

# BBB notebooks (new talk — tool-calling fine-tuning + RL)
jupyter notebook nb/bbb/_phase_1_teacher.ipynb
jupyter notebook nb/bbb/_phase_2_data_gen.ipynb
jupyter notebook nb/bbb/_phase_3_baseline.ipynb
jupyter notebook nb/bbb/_phase_4_sft.ipynb
jupyter notebook nb/bbb/_phase_5_rl.ipynb
```

## Architecture Details

### MCP Stock Server (`nb/winterfest/tools/mcp/stock_server.py`)
- Built with `fastmcp`, runs on HTTP port 8001
- 4 tools via Yahoo Finance (`yfinance`): `get_stock_news`, `get_financials`, `get_price_history`, `get_recommendations`
- The agent notebook connects to this as an MCP tool source — the server **must be running** before executing the agent

### Research Agent (`nb/winterfest/agent.ipynb`)
- Uses `openai-agents` SDK (OpenAI's agentic framework) with `gpt-5.1`
- Tools: Web Search, File Search (RAG over `data/winterfest/docs/`), Code Interpreter, MCP stock tools
- Output: Structured markdown reports covering Competition, Customers, Financials, Growth Opportunities → saved to `data/winterfest/output/`
- Streaming helper in `nb/winterfest/helpers/llm_helpers.py` handles all event types from the agent SDK

### Training Data Pipeline (`nb/winterfest/training_data_generator.ipynb`)
- Generates ~5,000 synthetic examples using `gpt-4.1-mini`
- Each example: fictional company report → expert investment verdict with reasoning
- Output format: JSONL files in `data/winterfest/` (main dataset: `data/winterfest/training_data_examples_all.jsonl`)
- Includes a hallucination detection dataset variant

### Fine-Tuning (`nb/winterfest/training_recipe.ipynb`)
- Base model: `unsloth/Qwen3-4B-Thinking` (4-bit quantized)
- Framework: Unsloth with LoRA adapters
- Training format: model outputs `<think>` block (chain-of-thought) followed by structured verdict
- Training history tracked in `data/winterfest/training_history.csv`

## Key Dependencies

- `openai` + `openai-agents[viz]` — Agent framework and OpenAI API
- `fastmcp` — MCP server framework
- `yfinance` (pinned to 0.2.62) — Stock data source for MCP tools
- `unsloth` — SFT training (used in Colab/GPU notebooks, not in pyproject.toml)
- `tiktoken` — Token counting for context window management
- `tenacity` + `limiter` — Rate limiting and retry logic for API calls

## Environment

- Python 3.11 (managed via `uv`)
- Requires `.env` with `OPENAI_API_KEY` and `VECTOR_STORE_ID`
- The vector store ID points to an OpenAI-hosted file search index over documents in `data/docs/`

## Important Conventions

- Research memos must follow the structured markdown format (Competition / Customers / Financials / Growth Opportunities) — this is the contract between Stage 1 output and Stage 2 input
- Training data uses chat-completion format with `<think>` tags for reasoning traces
- The MCP server uses HTTP transport (not stdio) — relevant when integrating with different agent frameworks
- Individual company output files in `data/winterfest/output/individual/` are gitignored
