# Plan: Applied AI Conference Talk (BBB 2026)

*Last updated: 2026-04-26. Replaces the original plan which referenced Qwen3-4B, Databricks, and live demos.*

## Context

Follow-up to WinterFest 2025. The original demo showed a two-stage pipeline: GPT-5.1 research agent + fine-tuned Qwen3-4B analyst. The BBB talk teaches the audience how these systems work under the hood — from tool calling mechanics to SFT to RL.

**Thesis:** Understanding how models get trained helps tremendously in how we apply them. This is Applied AI.

**Format:** ~35 min educational code-showing (no live running) + 5-10 min Q&A.

## Decisions Made

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Base model** | Qwen3.5-2B | Trains in ~30 min on free Colab T4, serves locally on Mac |
| **Teacher model** | GPT-5.4 (Responses API) | Native reasoning, captures reasoning summaries |
| **Training env** | Colab + Kaggle (free T4) | MLX training OOMs at 48GB. Free GPUs work. |
| **SFT framework** | Unsloth | 2x faster, first-class Qwen3.5 support |
| **RL framework** | TRL GRPOTrainer (single-turn demo) + ART (multi-turn, mentioned) | TRL runs on free T4. ART for production multi-turn. |
| **Talk format** | Code-showing, not live demo | Low risk, higher educational density |
| **Presentation tool** | Slideshow (PPTX) + HTML interactives | Slides for structure, HTML for GRPO/SFT visualizations |

## Pipeline Status

| Phase | Status | Key Artifacts |
|-------|--------|--------------|
| 1. Teacher Agent | **DONE** | `_phase_1_teacher.ipynb`, tested on NVDA, AAPL, TSLA, JPM |
| 2. Data Generation | **DONE** | 955 trajectories in `data/bbb/trajectories_sft.jsonl` |
| 3. Baseline | **DONE** | 5 MLX runs, avg reward 3.4, in `data/bbb/baseline_results_mlx_2b.jsonl` |
| 4. SFT | **DONE** | Trained on Colab + Kaggle. Weights in `nb/bbb/sft_results/`. No quantitative eval. |
| 5. RL | **SKELETON** | `_phase_5_rl.ipynb` — demo notebook, not run. TRL single-turn limitation documented. |

## Talk Structure

| # | Section | Time | Content |
|---|---------|------|---------|
| 0 | Opening | 1 min | "Building a custom agentic system, from context engineering to RL" |
| 1 | Hook | 3 min | Small models beating frontier, Anthropic postmortem, MCP/Skills standards |
| 2 | Recap | 3 min | WinterFest → BBB evolution, hypothesis shift (downstream → the task itself) |
| 3 | Agents | 8 min | Raw chat template, agent loop, tool schemas, "it's all just text" |
| 4 | SFT | 10 min | Data pipeline, masking, max_seq_length, cross-platform, MLX failures, scaling |
| 5 | RL | 8 min | GRPO loop, gradient mechanics, reward engineering, hacking examples, results |
| 6 | Synthesis | 3 min | Journey recap, BYOAI framework, "this is how I work in Applied AI" |

## Remaining Work

| Priority | Task | Status |
|----------|------|--------|
| ~~Talk outline~~ | ~~`birthday_bash_talk.md`~~ | **DONE** |
| ~~Demo cheatsheet~~ | ~~`docs/demo_cheatsheet.md`~~ | **DONE** |
| ~~Phase 5 notebook~~ | ~~`_phase_5_rl.ipynb`~~ | **DONE** |
| ~~Research docs~~ | ~~Benchmarks, reward design, GRPO explained~~ | **DONE** |
| ~~Doc sync~~ | ~~CLAUDE.md, README.md, this plan~~ | **DONE** |
| Build visualizations | V1-V7 (masking, GRPO loop, computation graph, etc.) | PENDING |
| Build slideshow | PPTX from talk.md with speaker notes | PENDING |
| Test demos | D1 (local model), D2 (teacher), D5 (frozen notebook) | PENDING |

## Key Research Findings (For Reference)

All research is in `docs/research_*.md`. Headline numbers:

- **RC-GRPO:** 7B beats Opus 4.5 on BFCLv4 multi-turn (85% vs 61%) — [arXiv:2602.03025](https://arxiv.org/abs/2602.03025)
- **ToolRLA:** Multiplicative reward veto deployed on financial advisory copilot, +47% task completion — [arXiv:2603.01620](https://arxiv.org/abs/2603.01620)
- **ToolRL:** GRPO from scratch outperforms SFT+RL, +5 to +42pp over SFT — [arXiv:2504.13958](https://arxiv.org/abs/2504.13958)
- **LLM judges cluster scores 0.7-0.9** — kills GRPO signal. Fix: categorical scores or quiz-based. — [DeepLearning.AI](https://learn.deeplearning.ai/courses/reinforcement-fine-tuning-llms-grpo/)
- **TRL GRPOTrainer is single-turn only** — no mechanism for multi-turn tool calling. ART handles this.
- **Our WinterFest GRPO:** Reward 3.7 → 5.7 sustained, best step 7.7, 12h on 1x H100
