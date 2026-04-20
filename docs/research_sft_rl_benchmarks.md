# SFT/RL Delta Benchmarks — Citable Research for the Talk

*Compiled: 2026-04-20. Numbers verified against official BFCL leaderboard (github.com/HuanzhiMao/BFCL-Result) and original papers. BFCL data snapshot: 2025-12-16.*

---

## Key Slide Numbers

These are the strongest contrasts for maximum visual impact:

| # | Claim | Numbers | Source |
|---|-------|---------|--------|
| 1 | RL adds significant delta over SFT | ToolRL GRPO adds +5 to +42pp over SFT depending on model size and benchmark | ToolRL, arXiv:2504.13958 (Apr 2025) |
| 2 | Small model + RL beats frontier on multi-turn | RC-GRPO Qwen2.5-7B at **85%** vs Claude Opus 4.5 at **61.25%** on BFCLv4 **multi-turn subset** (80 examples) | RC-GRPO, arXiv:2602.03025 (Feb 2026) |
| 3 | RL training is cheap | OpenPipe ART-E: $80 on one H100, 40 minutes, beats o3 | OpenPipe blog (2025-2026) |
| 4 | The Bloomberg lesson | $10M specialized training wiped out by next-gen frontier model | Bloomberg/GPT-4 comparison (2023-2024) |
| 5 | Generation leapfrog | Qwen3.5-9B (2026) BFCL 66.1 beats Qwen3-30B (2025) at 42.4 | Qwen3.5 Technical Report (2026) |
| 6 | Frontier models lead overall | Claude Opus 4.5 FC: **77.47%** (#1 on full BFCLv4). But specialized 7B beats it on multi-turn. | BFCL official leaderboard |

**Important context:** Claude Opus 4.5 is #1 overall on BFCLv4 (77.47%), but the RC-GRPO result shows a fine-tuned 7B can dominate frontier models on the *multi-turn* subset specifically. This is the "narrow task specialization" story — not "small model beats big model everywhere."

---

## 1. SFT Deltas for Tool Calling

### FunctionGemma (Google, 2024)
- **Source:** [Google Blog](https://blog.google/technology/developers/functiongemma/)
- **Model:** Gemma 3 270M, specialized for function calling
- **Results:** Mobile Actions accuracy: **58% → 85%** (+27pp). Tool-Call Character Error Rate: ~33% at baseline.
- **Takeaway:** Even a 270M model becomes a viable function-calling agent after targeted SFT.

### APIGen / xLAM (Salesforce, NeurIPS 2024)
- **Paper:** "APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets" — [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/61cce86d180b1184949e58939c4f983d-Paper-Datasets_and_Benchmarks_Track.pdf)
- **Results:**
  - **xLAM-7b-fc-r:** 88.24% on BFCL (3rd place as of July 2024), surpassing GPT-4o, GPT-4-Turbo, Llama3-70B, Claude-3 models
  - **xLAM-1b-fc-r:** 78.94% on BFCL — the only sub-2B model on the leaderboard, outperforming GPT-3.5-Turbo and Claude-3-Opus
- **Takeaway:** High-quality synthetic training data (60K verified examples) lets a 1B model beat GPT-3.5 and a 7B model beat GPT-4.

### Gorilla LLM (UC Berkeley, NeurIPS 2023)
- **Paper:** "Gorilla: Large Language Model Connected with Massive APIs" — [arXiv:2305.15334](https://arxiv.org/abs/2305.15334)
- **Results:** Fine-tuned LLaMA-7B achieved **20.43% higher accuracy than GPT-4** and **10.75% higher than ChatGPT**. 83% improvement over base LLaMA on APIBench.
- **Takeaway:** The original proof that SFT on tool-calling data can make a 7B model beat frontier models.

### Small Language Models for Tool Calling (Amazon, Dec 2024)
- **Paper:** "Small Language Models for Efficient Agentic Tool Calling" — [arXiv:2512.15943](https://arxiv.org/abs/2512.15943)
- **Results:** Fine-tuned OPT-350M: **77.55% on ToolBench** vs ChatGPT-CoT: 26.00%, ToolLLaMA-DFS: 30.18%, Claude-CoT: 2.73%
- **Takeaway:** 350M model beats ChatGPT by 3x on ToolBench — most extreme SFT delta demonstration.

### ToolLLM / ToolBench (OpenBMB, ICLR 2024 Spotlight)
- **Paper:** "ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs" — [ICLR 2024](https://openreview.net/forum?id=dHng2O0Jjr)
- **Results:** ToolLLaMA achieved comparable performance to ChatGPT across 16,000+ APIs. Open 7B-32B models routinely exceed 70% pass rate.
- **Takeaway:** SFT with well-structured trajectories closes the gap between open-source and proprietary models.

### Qwen3 / Qwen3.5 (Alibaba, 2025-2026)
- **Source:** [arXiv:2505.09388](https://arxiv.org/pdf/2505.09388), Qwen3.5 Technical Report
- **Results:**
  - Qwen3 4B outperforms Qwen2.5-72B-Instruct on multiple benchmarks
  - Qwen3.5-9B: **66.1 on BFCL-V4**, beats Qwen3-30B (42.4) — 3x smaller, next generation
  - Qwen3.5-122B-A10B: **72.2 on BFCL-V4**, outperforming GPT-5 mini (55.5) by 30%
- **Takeaway:** Base model tool-calling improves so fast that a 2026 9B model beats a 2025 30B model out of the box.

---

## 2. RL Deltas Over SFT

### ToolRL (Apr 2025)
- **Paper:** "ToolRL: Reward is All Tool Learning Needs" — [arXiv:2504.13958](https://arxiv.org/abs/2504.13958)
- **Results (BFCL V3):**

| Model | Raw Base | Best SFT | GRPO Cold Start | RL Gain over SFT |
|-------|----------|----------|-----------------|-------------------|
| Qwen2.5-1.5B | 19.41% | 40.67% | **46.20%** | +5.5pp |
| Qwen2.5-3B | 33.04% | 41.97% | **52.98%** | +11.0pp |
| Qwen2.5-7B | 41.97% | 36.53% | **58.38%** | +21.9pp |

- Bamboogle QA: SFT 30.4% → GRPO **72.0%** (+41.6pp)
- API-Bank: SFT 47.07% → GRPO **64.66%** (+17.6pp)
- **Key finding:** GRPO from scratch outperforms SFT and SFT+RL. SFT initialization actually hurts RL by causing overfitting.
- **Takeaway:** RL adds +5 to +42pp over SFT, gains scale with model size.

### RC-GRPO (Feb 2026)
- **Paper:** "RC-GRPO: Reward-Conditioned GRPO for Multi-Turn Tool Calling" — [arXiv:2602.03025](https://arxiv.org/abs/2602.03025)
- **Eval:** BFCLv4 **multi-turn subset only** (80 test examples from 200 curated trajectories — NOT the full BFCLv4 benchmark)
- **Results (Qwen2.5-7B-Instruct):**
  - Base: 11.25%
  - SFT + GRPO: 48.75%
  - RCTP-FT + RC-GRPO: **85.00%** (+36.25pp over SFT+GRPO)
- **Closed-source baselines on same multi-turn subset:** Claude Opus 4.5: 61.25%, GLM-4.7: 58.75%, Claude Sonnet 4.5: 57.50%, Gemini 3 Pro: 53.75%, GPT-5.2: 50.00%
- **Context:** On the FULL BFCLv4, Claude Opus 4.5 FC leads at 77.47% overall. The RC-GRPO result demonstrates that a fine-tuned 7B can dominate frontier models on **multi-turn tool calling specifically**, not across the board.
- **Takeaway:** Narrow task specialization via RL lets a 7B model beat frontier models where it matters — multi-turn agentic tool use.

### DeepSeek-R1 (Jan 2025)
- **Paper:** "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL" — [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)
- **Results:**
  - R1-Zero (pure RL, no SFT): AIME 2024 pass@1 15.6% → **71.0%**
  - R1 (SFT + RL pipeline): AIME 2024 **79.8%**
  - Distillation: Qwen-7B → 55.5% AIME (surpasses QwQ-32B-Preview's 50.0%)
- **Critical comparison:** RL-only on Qwen-32B: 47.0% AIME. Distillation (SFT from R1): 72.6%. Distillation from stronger model beats RL on the smaller model directly.
- **Takeaway:** SFT+RL pipeline is most reliable. Distillation of teacher trajectories is extremely effective.

### OpenPipe ART Results (2025-2026)
- **Source:** [OpenPipe ART GitHub](https://github.com/OpenPipe/ART), [ART-E Blog](https://openpipe.ai/blog/art-e-mail-agent)
- **ART-E (email agent):** Qwen2.5-14B with GRPO: **96% accuracy**, beating o3, o4-mini, Gemini 2.5 Pro, GPT-4.1. +56% over base. Cost: single H100, under $80, under 40 min.
- **Summarizer:** Qwen2.5-14B: **37% → 70%** with GRPO, outperforming Sonnet 4 and GPT-4.1. Cost: $22, 4 hours.
- **Takeaway:** GRPO on 14B can beat frontier models for under $100.

### OTC: Optimal Tool Calls (Apr 2025)
- **Paper:** "OTC: Optimal Tool Calls via RL" — [arXiv:2504.14870](https://arxiv.org/abs/2504.14870)
- **Results:** Reduces tool calls by up to **73.1%**, improves tool productivity by up to **229.4%**, maintains comparable accuracy. Identifies "cognitive offloading" where larger models over-rely on tools.
- **Takeaway:** RL can optimize tool efficiency — models learn when NOT to call tools.

### ARTIST (Microsoft, May 2025)
- **Paper:** "Agentic Reasoning and Tool Integration for LLMs via RL" — [arXiv:2505.01441](https://arxiv.org/abs/2505.01441)
- **Results:** Up to **22% absolute improvement** over base models. More than **doubles accuracy** of prompt-based baselines on t-bench (multi-turn).
- **Takeaway:** RL with interleaved reasoning and tool use is a general technique.

### Tool-Zero (EMNLP 2025)
- **Paper:** "Tool Zero: Training Tool-Augmented LLMs via Pure RL from Scratch" — [ACL Anthology](https://aclanthology.org/2025.findings-emnlp.485/)
- **Results:** Over **7% improvement** over both SFT and RL-with-SFT baselines on BFCL.
- **Takeaway:** Pure RL from scratch can outperform SFT+RL, paralleling DeepSeek-R1-Zero.

---

## 3. Delta Trends Over Time (2023 → 2026)

### The Bloomberg Cautionary Tale (2023-2024)
- Bloomberg spent **$10M** training 50.6B params on financial data (1.3M GPU hours, 512 A100s, 53 days)
- GPT-4 outperformed it: FinQA **68.79% vs 43%**, ConvFinQA **76% vs 43%**, NER **83% vs 61%**
- **Lesson:** In 2023, the next frontier generation wiped out a $10M domain-specific investment.

### BFCL Leaderboard Evolution (Verified Against Official Data)
- **2024 (v1):** Early leaderboard dominated by large models (Llama 3.1 405B, GPT-4)
- **2026 (v3):** GLM 4.5 Thinking leads at 76.7%. Surprising: Claude Opus 4 regressed to 25.3% (known multi-turn regression for newer thinking models)
- **2026 (v4, official — data snapshot 2025-12-16):**
  - Claude Opus 4.5 FC: **77.47%** (#1)
  - Claude Sonnet 4.5 FC: **73.24%** (#2)
  - Gemini 3 Pro Preview: 72.51%
  - GLM-4.6 FC (thinking): 72.38%
  - GPT-5.2 FC: 55.87% (#16)
  - *(Note: Opus 4.6, Sonnet 4.6 not yet on leaderboard)*
- **Trend:** Top models cluster tightly (72-77%), but GPT-5.2 lags significantly. Multi-turn remains the hardest category.

### The Counterargument: Fine-Tuning Is MORE Valuable in 2026
- Fine-tuned xLAM-7B beat GPT-4 on BFCL in 2024
- RC-GRPO Qwen2.5-7B (85%) beat Claude Opus 4.5 (61.25%) on BFCLv4 **multi-turn subset** in 2026
- The "sweet spot" has shifted: **3-8B models with RL are the new production agent tier** for high-volume specialized tasks
- Fine-tune 7B with LoRA: under $5. GRPO on 14B: $22-80. Break-even vs frontier API: 1-3 months.

### The Nuanced Answer
- Frontier models are better → the *floor* rises → prompting alone works for more tasks
- But specialized fine-tuning still creates enormous deltas for narrow tasks
- The value of fine-tuning has shifted from "making it work at all" (2023) to "making it work better, cheaper, and privately" (2026)
- Sources: [seldo.com](https://seldo.com/posts/2026-is-the-year-of-fine-tuned-small-models/), [kadoa.com](https://www.kadoa.com/blog/is-fine-tuning-still-worth-it), [spheron.network](https://www.spheron.network/blog/how-to-fine-tune-llm-2026/)

---

## 4. The "When to Stop" Framework

| Approach | When It Wins | Evidence |
|----------|-------------|----------|
| **Prompting alone** | Task well-covered by base training; low volume; changing requirements | Kadoa (2025): "Many problems that previously required FT can now be solved with better prompting" |
| **SFT** | Specialized domain with 200+ examples; consistent output format; high volume | xLAM-1b 78.9% BFCL; FARA-7B beating GPT-4o; FunctionGemma 58%→85% |
| **SFT + RL** | Need to exceed frontier on specific task; multi-turn tool calling; optimize efficiency | RC-GRPO 85% beating Opus 4.5 at 61.25%; ART-E 96% beating o3 |
| **Pure RL** | Sufficient compute; want max generalization; clear verifiable rewards | ToolRL GRPO outperforms SFT+RL; Tool-Zero +7% over both |

### Cost Thresholds (2026)
- Fine-tune 7B with LoRA: **under $5**, hours not weeks
- GRPO training on 14B: single H100, **$22-80**, 40 min to 4 hours
- Break-even vs frontier API: **1-3 months** for high-volume apps

### FARA-7B (Microsoft, Nov 2025)
- **Paper:** [arXiv:2511.19663](https://arxiv.org/abs/2511.19663)
- Qwen2.5-VL-7B, SFT with 145K trajectories → **73.5%** vs GPT-4o 65.1% on WebVoyager
- Completes tasks in ~16 steps vs 41 steps for alternatives
- **Takeaway:** Specialization wins when the task is well-defined and high-volume.

### xRouter (Salesforce, Oct 2025)
- **Paper:** [arXiv:2510.08439](https://arxiv.org/abs/2510.08439)
- 7B router achieves near GPT-5 accuracy at **~1/8 the cost** on Olympiad Bench
- Cost reductions of **60-80%** across evaluations
- **Takeaway:** Train a small router that dispatches to the right model — the meta-strategy.
