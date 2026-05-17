# Applied AI: Equity Investment Opportunity Sourcing

This project implements an end-to-end AI system for automated public equity investment research and analysis, developed across two conference talks:

1. **WinterFest 2025** — Built a two-stage pipeline: proprietary Research Agent + fine-tuned Analyst Model
2. **Big Birthday Bash 2026** — Teaching the small model to *be* the agent itself via tool-calling SFT + RL

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
       │  generates 955 tool-calling trajectories
       ▼
   Phase 2: Data Generation
       │  Responses API → Hermes format, truncation, filtering
       ▼
   Phase 3: Baseline (raw Qwen3.5-2B with tools)
       │  establish "before" metrics (avg reward 3.4)
       ▼
   Phase 4: SFT (Unsloth on Colab/Kaggle, free T4)
       │  teach tool-calling behavior via LoRA
       ▼
   Phase 5: RL (GRPO — single-turn via TRL, multi-turn via ART)
       │  refine: reward engineering, quality beyond format
       ▼
   [ Qwen3.5-2B that calls tools autonomously at ~$0/run ]
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
│   └── bbb/                               # Big Birthday Bash 2026 notebooks
│       ├── tools.py                       # Stock tool functions + auto-generated schemas
│       ├── agent.py                       # Tool-calling agent loops (Responses API + Chat Completions)
│       ├── helpers__data_gen.py           # Data gen: serialization, truncation, format conversion
│       ├── helpers__inference.py          # Local inference: tool call parsing, reward function
│       ├── birthday_bash_talk.md          # Talk outline (source of truth for presentation)
│       ├── _phase_1_teacher.ipynb         # Teacher agent demo (GPT-5.4)
│       ├── _phase_2_data_gen.ipynb        # Bulk trajectory generation (955 trajectories)
│       ├── _phase_3_baseline.ipynb        # Raw Qwen3.5-2B with tools (MLX)
│       ├── _phase_4_sft_colab.ipynb       # SFT fine-tuning (Google Colab, Unsloth)
│       ├── _phase_4_sft_kaggle.ipynb      # SFT fine-tuning (Kaggle, Unsloth)
│       ├── _phase_4_sft_mlx.ipynb         # MLX config + eval (Mac — training failed)
│       └── _phase_5_rl.ipynb              # GRPO reinforcement learning (demo skeleton)
│
├── data/
│   ├── winterfest/                        # WinterFest training data + outputs
│   │   ├── training_data_examples_all.jsonl
│   │   ├── docs/                          # Internal docs for RAG
│   │   └── output/                        # Generated research reports
│   └── bbb/                               # BBB trajectory data
│       ├── trajectories_raw.jsonl         # Raw Responses API trajectories (955)
│       ├── trajectories_sft.jsonl         # Converted to Hermes format for SFT (955)
│       ├── baseline_results_mlx.jsonl     # MLX baseline metrics
│       └── mlx_sft/                       # MLX train/valid splits
│
├── docs/                                  # Research, learnings, talk prep
│   ├── takeaways_sft.md                   # Consolidated SFT learnings
│   ├── takeaways_rl.md                    # Consolidated RL learnings
│   ├── research_sft_rl_benchmarks.md      # Citable SFT/RL deltas (verified BFCL numbers)
│   ├── research_reward_design.md          # Reward function patterns (simple + composite)
│   ├── research_grpo_explained.md         # Educational GRPO explainer
│   ├── research_financial_rl_rewards.md   # Financial domain RL paper survey
│   ├── research_reward_examples.md        # Production RL examples (Cursor, ART-E, etc.)
│   ├── grpo_learnings_winterfest.md       # GRPO experiment retrospective (Baseten H100)
│   ├── demo_cheatsheet.md                 # Exact commands and cells for talk demos
│   └── tutorial_content/                  # Reference materials
│       ├── claude_plan.md                 # Original project plan (outdated)
│       └── rl/                            # RL tutorials, ART docs, GRPO notebooks
│
├── winterfest_talk.md                     # WinterFest talk outline
├── CLAUDE.md                              # Claude Code instructions
└── README.md                              # This file
```

---

## Getting Started

```bash
# Install dependencies
uv sync

# Local model inference (Mac)
uv run mlx_lm.server --model mlx-community/Qwen3.5-2B-4bit --port 8080

# WinterFest pipeline
python nb/winterfest/tools/mcp/stock_server.py  # Start MCP server first
jupyter notebook nb/winterfest/agent.ipynb

# BBB pipeline
jupyter notebook nb/bbb/_phase_1_teacher.ipynb   # Teacher agent demo
```

Requires `.env` with `OPENAI_API_KEY` and `VECTOR_STORE_ID`.

---

## Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Base model** | Qwen3.5-2B | Fast local inference on Mac, trains in ~30 min on free Colab T4 |
| **Teacher API** | OpenAI Responses API | Native reasoning support, flat tool schemas |
| **Tool schemas** | Auto-generated from function signatures | Single source of truth |
| **SFT data** | Tool outputs truncated to ~250 tokens | Masked tokens = zero gradient = pure overhead |
| **Training** | Unsloth on Colab/Kaggle (free T4) | MLX backward pass ~48 GB on any Qwen3.5 — infeasible on 16GB Mac |
| **RL framework** | TRL GRPOTrainer (single-turn) + ART by OpenPipe (multi-turn) | TRL for demo simplicity, ART for full agent RL |
| **Inference params** | `temp=0.6, top_p=0.95, presence_penalty=1.5` | Official Qwen3 thinking-mode settings |
| **Thinking control** | Never mention thinking in prompts for Qwen3.5 | Qwen3.5 doesn't support `/nothink` — causes spiraling |

---

## Talk Materials

- [WinterFest Talk](winterfest_talk.md) — *Agentic Systems in Practice* (Nov 2025)
- [Birthday Bash Talk](nb/bbb/birthday_bash_talk.md) — *Applied AI at Work: Understanding and Training AI Systems* (2026)

## Sources

- [Unsloth](https://docs.unsloth.ai/) — SFT & RL training framework
- [ART (OpenPipe)](https://github.com/openpipe/art) — Agent Reinforcement Trainer
- [ToolRL](https://arxiv.org/abs/2504.13958) — Reward design for tool-calling RL
- [ToolRLA](https://arxiv.org/abs/2603.01620) — Multiplicative reward veto (financial domain)
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) — Emergent reasoning from RL
- [RC-GRPO](https://arxiv.org/abs/2602.03025) — 7B beats Opus 4.5 on multi-turn
- [BFCL Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) — Berkeley Function Calling benchmark
