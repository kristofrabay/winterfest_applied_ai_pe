# Applied AI: Equity Investment Opportunity Sourcing

This project implements an end-to-end AI system for automated public equity investment research and analysis, developed across two conference talks:

1. **WinterFest 2025** — Built a two-stage pipeline: proprietary Research Agent + fine-tuned Analyst Model
2. **Big Birthday Bash (BBB)** — Teaching the small model to *be* the agent itself via tool-calling SFT + RL

*Kristof Rabay — Applied AI @ The Carlyle Group*

> **Disclaimer:** The views expressed are those of the speaker alone and do not reflect the official position of The Carlyle Group or its affiliates.

---

## Architecture Overview

### WinterFest: Two-Stage Pipeline

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

### BBB: Teaching the Small Model to Call Tools

```text
   Phase 1: Teacher Agent (GPT-5.4)
       │  generates ~200 tool-calling trajectories
       ▼
   Phase 2: Data Generation
       │  truncated trajectories → SFT training data
       ▼
   Phase 3: Baseline (raw Qwen3-4B with tools)
       │  establish "before" metrics
       ▼
   Phase 4: SFT (Unsloth + LoRA)
       │  teach tool-calling behavior
       ▼
   Phase 5: RL (GRPO via ART)
       │  refine: fewer redundant calls, better reasoning
       ▼
   [ Qwen3-4B that calls tools autonomously at ~$0/run ]
```

---

## Project Structure

```text
├── nb/
│   ├── winterfest/                        # WinterFest 2025 notebooks
│   │   ├── agent.ipynb                    # Research Agent (GPT-5.1 + MCP)
│   │   ├── training_data_generator.ipynb  # 5K synthetic verdicts via GPT-4.1-mini
│   │   ├── training_recipe.ipynb          # SFT: Qwen3-4B-Thinking analyst
│   │   ├── helpers/                       # Streaming, LLM utilities
│   │   └── tools/mcp/stock_server.py     # FastMCP server (Yahoo Finance, port 8001)
│   │
│   └── bbb/                               # Big Birthday Bash notebooks
│       ├── tools.py                       # Stock tool functions + auto-generated schemas
│       ├── agent.py                       # Tool-calling agent loops (Responses API + Chat Completions)
│       ├── helpers__data_gen.py            # Data gen: serialization, truncation, format conversion
│       ├── helpers__inference.py           # Local inference: tool call parsing, reward function
│       ├── _phase_1_teacher.ipynb         # Teacher agent demo (GPT-5.4)
│       ├── _phase_2_data_gen.ipynb        # Bulk trajectory generation
│       ├── _phase_3_baseline.ipynb        # Raw Qwen3-4B with tools (MLX + Databricks)
│       ├── _phase_4_sft.ipynb             # SFT fine-tuning
│       └── _phase_5_rl.ipynb              # GRPO reinforcement learning
│
├── data/
│   ├── winterfest/                        # WinterFest training data + outputs
│   │   ├── training_data_examples_all.jsonl
│   │   ├── docs/                          # Internal docs for RAG
│   │   └── output/                        # Generated research reports
│   └── bbb/                               # BBB trajectory data
│       ├── trajectories_raw.jsonl         # Raw Responses API trajectories
│       ├── trajectories_sft.jsonl         # Converted to Hermes format for SFT
│       ├── baseline_results_mlx.jsonl     # MLX local baseline metrics
│       └── baseline_examples_mlx.json     # Best/worst examples for talk
│
├── docs/                                  # Images, tutorial content, plan
├── winterfest_talk.md                     # WinterFest talk outline
├── birthday_bash_talk.md                  # BBB talk outline
└── CLAUDE.md                              # Claude Code instructions
```

---

## Getting Started

```bash
# Install dependencies
uv sync

# WinterFest pipeline
python nb/winterfest/tools/mcp/stock_server.py  # Start MCP server (required for agent)
jupyter notebook nb/winterfest/agent.ipynb  # Run research agent

# BBB pipeline
jupyter notebook nb/bbb/_phase_1_teacher.ipynb  # Teacher agent demo
```

Requires `.env` with `OPENAI_API_KEY` and `VECTOR_STORE_ID`.

---

## Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Teacher API** | OpenAI Responses API | Native reasoning support, flat tool schemas |
| **Base model** | Qwen3-4B | Best tool-calling benchmarks at size, native `<think>` support |
| **Tool schemas** | Auto-generated from function signatures | Single source of truth, change function → schema updates |
| **SFT data** | Tool outputs truncated to ~500-800 tokens | Standard practice (Hermes, ToolBench, APIGen all do this) |
| **RL framework** | ART by OpenPipe | Only option for multi-turn GRPO with tool calling |
| **Training** | Unsloth + LoRA on Databricks | 2x faster, 60% less memory, first-class Qwen3 support |

---

## Talk Materials

- [WinterFest Talk](winterfest_talk.md) — *Agentic Systems in Practice* (Nov 2025)
- [Birthday Bash Talk](birthday_bash_talk.md) — *From Prototype to Production: Teaching Small Models to Call Tools*

## Sources

- [Unsloth](https://docs.unsloth.ai/) — SFT & RL training framework
- [ART (OpenPipe)](https://github.com/openpipe/art) — Agent Reinforcement Trainer
- [FunctionGemma](https://arxiv.org/abs/2404.14105) — Tool-calling fine-tuning patterns
- [ToolRL](https://arxiv.org/abs/2404.07995) — RL for tool-calling behavior
