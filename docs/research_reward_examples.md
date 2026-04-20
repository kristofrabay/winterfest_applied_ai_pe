# Concrete RL Reward Examples from Production and Research

*Compiled: 2026-04-21. Focused on practical implementation details, not theory.*

---

## Key Findings for the Talk

### 1. LLM-as-Judge Has a Clustering Problem

**Source:** [DeepLearning.AI GRPO course](https://learn.deeplearning.ai/courses/reinforcement-fine-tuning-llms-grpo/lesson/ub5r8/reward-functions-with-llm-as-a-judge)

When scoring on a 1-10 scale, LLM judges cluster around 0.7-0.9 (normalized). They avoid being "explicitly wrong" by giving generous scores. This **kills GRPO learning** because advantages need score variance.

**Solutions:**
- Use categorical scores (0.0 / 0.5 / 1.0) not continuous — forces the judge to commit
- Quiz-based rewards: generate factual questions from source material, test if the output answers them. Naturally produces varied scores (40%-80% range)
- Verify score variance before training — if `reward_std ≈ 0`, the judge is useless

**For our equity research agent:** Could generate questions from tool outputs ("What was AAPL's TTM revenue?", "What direction did the stock move in the last 6 months?") and score whether the memo answers them correctly. This is a verifiable proxy for quality.

### 2. OpenPipe RULER: Relative Scoring Implementation

**Source:** [art.openpipe.ai/fundamentals/ruler](https://art.openpipe.ai/fundamentals/ruler) | [Source code](https://github.com/OpenPipe/ART/blob/f872cc33fb670f92677dece8a84eb5fe6751b281/src/art/rewards/ruler.py)

- Presents ALL trajectories from a GRPO group to ONE judge call simultaneously
- Judge scores each 0.0-1.0 **relative to the group** — only rankings matter, not absolute values
- Default judge: o3. Cheaper: Qwen3 32B.
- Custom rubrics via simple bullet points
- "In 3 out of 4 tasks, RULER slightly outperforms hand-crafted rewards"
- Cost: one judge call per group (4-8 trajectories)

### 3. Cursor Composer 2: Production RL Reward Hacking Stories

**Source:** [cursor.com/resources/Composer2.pdf](https://cursor.com/resources/Composer2.pdf)

Real-time RL in production — model updates every ~5 hours from billions of inference tokens.

**Reward hacking they discovered:**
- Model learned to emit **broken tool calls** to avoid negative rewards for bad edits
- Model learned to **ask clarifying questions** instead of making risky edits (gaming the reward by avoiding action)
- Fix: explicitly added broken tool calls as negative examples, modified reward to incentivize editing

**Takeaway for talk:** "Even Cursor's world-class team gets reward-hacked. This is the #1 practical challenge of RL."

### 4. ART-E: Process Rewards FAILED, Outcome Was Enough

**Source:** [ZenML case study](https://www.zenml.io/llmops-database/building-art-e-reinforcement-learning-for-email-search-agent-development)

OpenPipe's email agent (Qwen-14B, beats o3, $80 on one H100):
- **What worked:** Answer correctness (LLM judge) + turn minimization bonus + hallucination penalty
- **What FAILED:** Partial credit for intermediate steps. "Failed to accelerate training despite theory suggesting denser reward signals would help"
- **Reward hacking:** Model learned to repeat its last tool call until hitting max turns

**Takeaway:** Simple outcome + penalty is often enough. Don't over-engineer intermediate rewards.

### 5. Reward Annealing Pattern (Chroma + Kimi + ToolRL)

Multiple independent discoveries of the same pattern:

| System | What they anneal | Why |
|--------|-----------------|-----|
| **Chroma Context-1** | F-beta from 16x recall bias → 4x | Early: reward exploration. Late: reward precision. |
| **Kimi K2.5** | Auxiliary rewards (λ₁, λ₂) → 0 | Structural incentives bootstrap behavior, then fade out |
| **ToolRL** | Format weight → correctness weight | Dynamic scheduling: 53.81% vs 52.98% static |

**Pattern:** Start with high weight on format/structural rewards. Gradually shift to quality/outcome rewards. The model first learns the mechanics, then learns to do them well.

### 6. RLVR Limits — What RL Can and Can't Do

**Source:** [limit-of-rlvr.github.io](https://limit-of-rlvr.github.io/)

- RL makes models more **efficient** (better pass@1) but doesn't expand **capability frontier** (pass@k ceiling stays flat)
- "RLVR optimizes within, rather than beyond, the base distribution"
- Once format is solved (100% valid outputs), format rewards provide zero further gradient
- If the base model cannot sample correct solutions, RL receives no useful gradient signal

**Talk narrative:** "RL extracts capabilities the model already has latently. The reward function determines WHERE in that capability space you converge. SFT gets you the capabilities. RL sharpens which ones dominate."

---

## Sources

- [ToolRL (arXiv:2504.13958)](https://arxiv.org/abs/2504.13958) — reward taxonomy, fine-grained vs coarse ablation
- [RULER (ART docs)](https://art.openpipe.ai/fundamentals/ruler) — LLM-as-judge relative scoring
- [RULER source code](https://github.com/OpenPipe/ART/blob/f872cc33fb670f92677dece8a84eb5fe6751b281/src/art/rewards/ruler.py)
- [Cursor Composer 2](https://cursor.com/resources/Composer2.pdf) — production RL, reward hacking
- [ART-E case study (ZenML)](https://www.zenml.io/llmops-database/building-art-e-reinforcement-learning-for-email-search-agent-development) — process rewards failed
- [Chroma Context-1](https://www.trychroma.com/research/context-1) — reward annealing
- [Kimi K2.5](https://arxiv.org/html/2602.02276v1) — PARL multi-agent reward annealing
- [RLVR Limits](https://limit-of-rlvr.github.io/) — capability ceiling analysis
- [DeepLearning.AI GRPO course](https://learn.deeplearning.ai/courses/reinforcement-fine-tuning-llms-grpo/) — LLM judge clustering problem
