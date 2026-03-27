# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Conference demo project for **"Agentic Systems in Practice: Applying AI to Investments"** — originally presented at Data Science Festival WinterFest 2025, now being extended for a follow-up conference talk.

The system is a **two-stage pipeline** for automated equity investment research:
1. **Research Agent** (`nb/winterfest/agent.ipynb`) — GPT-5.1-powered agent with MCP tools that generates structured research memos on target companies
2. **Analyst Model** — A fine-tuned small language model (Qwen3-4B) that reads research memos and produces investment verdicts (`Strong Yes` / `Questionable` / `Strong No`) with chain-of-thought reasoning

### Current Conference Goal (Big Birthday Bash — BBB)

Teach **Qwen3-4B** to be the research agent itself via tool-calling fine-tuning + RL:
1. **Teacher Agent** (`nb/bbb/tool_calling_agent.ipynb`) — GPT-5.4 agent with stock tools, generates training trajectories
2. **Data Generation** (`nb/bbb/tool_calling_data_generator.ipynb`) — Run teacher on ~200 companies, save full tool-calling trajectories
3. **Baseline** (`nb/bbb/tool_calling_baseline.ipynb`) — Run raw Qwen3-4B with tools to establish "before" metrics
4. **SFT** (`nb/bbb/tool_calling_sft.ipynb`) — Fine-tune Qwen3-4B on trajectories via Unsloth
5. **RL** (`nb/bbb/tool_calling_rl.ipynb`) — GRPO via ART (OpenPipe) to refine tool-calling behavior

Full plan: `docs/tutorial_content/claude_plan.md`

## Commands

```bash
# Environment setup
uv sync                                    # Install all dependencies
python tools/mcp/stock_server.py           # Start MCP server on port 8001 (must be running before winterfest agent notebook)

# WinterFest notebooks (original talk)
jupyter notebook nb/winterfest/agent.ipynb
jupyter notebook nb/winterfest/training_data_generator.ipynb
jupyter notebook nb/winterfest/training_recipe.ipynb

# BBB notebooks (new talk — tool-calling fine-tuning + RL)
jupyter notebook nb/bbb/tool_calling_agent.ipynb
jupyter notebook nb/bbb/tool_calling_data_generator.ipynb
jupyter notebook nb/bbb/tool_calling_baseline.ipynb
jupyter notebook nb/bbb/tool_calling_sft.ipynb
jupyter notebook nb/bbb/tool_calling_rl.ipynb
```

## Architecture Details

### MCP Stock Server (`tools/mcp/stock_server.py`)
- Built with `fastmcp`, runs on HTTP port 8001
- 4 tools via Yahoo Finance (`yfinance`): `get_stock_news`, `get_financials`, `get_price_history`, `get_recommendations`
- The agent notebook connects to this as an MCP tool source — the server **must be running** before executing the agent

### Research Agent (`nb/winterfest/agent.ipynb`)
- Uses `openai-agents` SDK (OpenAI's agentic framework) with `gpt-5.1`
- Tools: Web Search, File Search (RAG over `data/docs/`), Code Interpreter, MCP stock tools
- Output: Structured markdown reports covering Competition, Customers, Financials, Growth Opportunities → saved to `data/output/`
- Streaming helper in `nb/helpers/llm_helpers.py` handles all event types from the agent SDK

### Training Data Pipeline (`nb/winterfest/training_data_generator.ipynb`)
- Generates ~5,000 synthetic examples using `gpt-4.1-mini`
- Each example: fictional company report → expert investment verdict with reasoning
- Output format: JSONL files in `data/` (main dataset: `data/training_data_examples_all.jsonl`)
- Includes a hallucination detection dataset variant

### Fine-Tuning (`nb/winterfest/training_recipe.ipynb`)
- Base model: `unsloth/Qwen3-4B-Thinking` (4-bit quantized)
- Framework: Unsloth with LoRA adapters
- Training format: model outputs `<think>` block (chain-of-thought) followed by structured verdict
- Training history tracked in `data/training_history.csv`

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
- Individual company output files in `data/output/individual/` are gitignored
