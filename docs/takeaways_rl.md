# RL Takeaways — Everything We Learned

*Consolidated from: WinterFest GRPO experiment (Baseten/H100), tutorial materials, ART docs, research papers, and deep-dive into GRPO mechanics.*
*Last updated: 2026-04-20*

---

## The Big Picture

RL (Reinforcement Learning) = the model generates completions, they get scored by a reward function, and weights update to make high-reward completions more likely. No labeled "correct answer" needed — just a way to say "this is better than that."

**What RL teaches:** efficiency, quality judgment, edge case handling, behaviors beyond what the training data shows.
**What RL needs:** a model that already roughly knows the task (from SFT), and a well-designed reward function. (Though ToolRL and DeepSeek-R1-Zero showed you can skip SFT entirely with enough compute.)

### The Two Sentences (USE IN PRESENTATION)

> **SFT says:** "Here's the exact right answer. Predict these exact tokens." *(Supervised — has a teacher.)*
>
> **GRPO says:** "I don't know the right answer. But I generated 5 attempts, scored them, and I know which ones were BETTER. Make the good attempts more likely." *(Reinforcement — no teacher, just a judge.)*

**SFT vs RL — the core difference:**

| | SFT | RL (GRPO) |
|---|---|---|
| Learning signal | Next-token prediction on labeled data | Reward score on generated completions |
| Requires | Ground truth (training data) | Reward function (no ground truth needed, but helps) |
| Teaches | Format, structure, imitation | Quality, efficiency, optimization |
| Risk | Learns to copy, including mistakes | Reward hacking — optimizes reward, not intent |
| Loss | Cross-entropy on masked tokens | Policy gradient (GRPO: group-relative advantages) |

---

## 2. GRPO — How It Actually Works

GRPO (Group Relative Policy Optimization) is the RL algorithm. Introduced by DeepSeek Math ([arXiv:2402.03300](https://arxiv.org/abs/2402.03300)). A simplification of PPO that eliminates the critic network.

### The Analogy (USE IN PRESENTATION)

> **Imagine you're a writing teacher.** You give 8 students the same essay prompt. They each write an essay. You grade them: two get A's, four get B's, two get D's.
>
> You don't tell them what a good essay looks like (that would be SFT). Instead, you just tell each student: "Here's how you scored *relative to the group average*." The A students hear "keep doing what you did." The D students hear "change something."
>
> **That's GRPO.** No answer key needed — just a way to compare attempts.

### The Full Loop (One Training Step)

```
1. SAMPLE: Pick a batch of prompts
   For each prompt, generate G=5 completions (temperature=1.0 for diversity)

2. SCORE: Run reward function on each completion
   Prompt "Research AAPL" → rewards: [5.5, 3.0, 4.2, -1.0, 3.5]

3. ADVANTAGE: Compute group-relative advantages
   mean = 3.04
   advantage_A = 5.5 - 3.04 = +2.46   ← "much better than average"
   advantage_E = -1.0 - 3.04 = -4.04   ← "much worse than average"

4. LOSS: Combine all completions into ONE loss number
   For each token in each completion:
     loss contribution = -(advantage) × log_prob(token)
   Sum across all tokens in all completions → one scalar

5. BACKPROP: Trace backward through computation graph → one gradient per weight
   (The advantage survives through the chain rule — each token's gradient 
    is proportional to its completion's advantage)

6. UPDATE: weight -= learning_rate × gradient
   (ONE update, not five separate updates)
```

### How the Gradient Knows Which Completion It Came From

The final loss is one number, but **backprop doesn't just propagate one number — it follows the computation graph**. PyTorch remembers every operation:

```
final_loss (one scalar)
  │
  ├── loss_from_A = -advantage_A × (logprob_token1 + logprob_token2 + ...)
  │     The chain rule carries advantage_A (+2.46) to every weight
  │     that produced these tokens → "make get_financials more likely"
  │
  └── loss_from_E = -advantage_E × (logprob_token1 + logprob_token2 + ...)
        The chain rule carries advantage_E (-4.04) to every weight
        that produced these tokens → "make get_stock_info less likely"
```

**The advantages don't get averaged away.** They survive through backprop because the computation graph remembers which path each token took. The weight that produced "get_financials" in completion A gets a gradient ∝ +2.46. The weight that produced "get_stock_info" in completion E gets a gradient ∝ -4.04. Both happen in ONE backward pass, producing ONE gradient vector that combines all signals.

After the update: `get_financials` slightly more likely in similar contexts, `get_stock_info` slightly less likely. Repeat for thousands of prompts × hundreds of steps → model converges.

### Why GRPO, Not PPO?

**PPO:** Policy model + Critic model + Reference model = 3 models, 2 trained
**GRPO:** Policy model only = 1 model, 1 trained

DeepSeek Math's insight: for language generation, you don't need a learned critic. **The group IS the critic.** ~60% less memory, no instability from training two networks.

Tradeoff: must generate G completions per prompt (4-8), slower at generation time but cheap with vLLM.

### Why Temperature, Beta, and Learning Rate Matter

**Temperature = 1.0 (diversity for learning signal):**
Low temp → all completions identical → same rewards → advantages ≈ 0 → **zero gradient**. SFT exploits (low temp). RL explores (high temp).

**Beta = 0.0 (no KL penalty):**
No reference model needed → saves ~50% memory. The clipping mechanism (ε=0.2) is the only constraint on drift. GRPO's group-relative advantage provides implicit regularization.

**Learning rate = 5e-6 (40× lower than SFT):**
RL signal is noisy (reward approximation, advantage estimation). Too high → model collapses. SFT uses 2e-4 because the signal is clean (ground truth labels).

### Key Hyperparameters (From Our WinterFest Experiment)

```python
learning_rate = 5e-6          # Much lower than SFT (2e-4)
max_grad_norm = 0.1           # Tight clipping — RL-specific
beta = 0.0                    # No KL penalty = no reference model
loss_type = "dr_grpo"         # Dr. GRPO: no length bias (arXiv:2503.06639)
scale_rewards = False         # Dr. GRPO recommends this
num_generations = 4           # Completions per prompt
temperature = 1.0             # High diversity for exploration
epsilon = 0.2                 # PPO-style clipping (max 20% change per update)
```

**Best educational sources for GRPO:**
- Cameron Wolfe: [cameronrwolfe.substack.com/p/grpo](https://cameronrwolfe.substack.com/p/grpo) (the best walkthrough)
- Dr. GRPO: [arXiv:2503.06639](https://arxiv.org/abs/2503.06639) (fixes length bias and difficulty bias)
- DeepSeek-R1: [arXiv:2501.12948](https://arxiv.org/abs/2501.12948) (emergent reasoning from simple rewards)

---

## 3. Reward Engineering — The Hard Part

**Your reward function IS your product specification.** It defines exactly what "good" means.

### 3a. Two Reward Functions: Simple and Composite

**Reward A (Simple — explain in 60 seconds):** 4 components, all rule-based.
- Format: valid JSON? (0 or 1) — from [ToolRL](https://arxiv.org/abs/2504.13958)
- Tool coverage: fraction of expected tools called, with hallucination veto (0 to 2) — from [ToolRLA](https://arxiv.org/abs/2603.01620)
- Completion: produced final memo? (0 or 1) — from ToolRL
- Efficiency: penalty for excess calls (-1 per extra) — from [OTC](https://arxiv.org/abs/2504.14870)
- **Range: [-2, 4].** Perfect trajectory = 4.0.

**Reward B (Composite — production-grade):** 5 components, adds LLM-as-judge quality signal.
- Everything in Simple (minus arbitrary efficiency penalty), PLUS:
- Parameter accuracy: correct ticker in all tool calls? (0 to 1) — from ToolRL fine-grained
- **LLM-as-judge for analysis quality:** A judge model reads the full trajectory and scores whether the memo is specific/data-grounded vs generic boilerplate (0 to 2) — from [Tool-R1](https://arxiv.org/abs/2509.12867) + [ART RULER](https://art.openpipe.ai/fundamentals/ruler)
- **LLM-as-judge for fabrication:** Judge checks if claims are supported by tool outputs (penalty -5) — from ToolRLA compliance principle
- **Range: [-5, 6].** The LLM judge is where RL earns its keep beyond SFT — rules can't tell if analysis is insightful, but a judge model can.

Full code for both: see `docs/research_reward_design.md`.

### 3b. Key Design Principles

1. **Multiplicative > Additive for correctness.** ToolRLA proved in production (financial advisory, 80+ advisors): additive rewards let the model compensate wrong tool selection with good parameters. Multiplicative veto prevents this. (+7pp improvement in ablation.)

2. **Dense > Sparse.** Reward of {0, 1} gives almost no signal. Reward of {-3, -1, 0, +0.3, +1, +2, +3} gives rich gradient. Every paper agrees on this.

3. **RL needs variance.** If all completions score identically → advantage = 0 → gradient = 0. Reward spread across multiple scales is intentional. `reward_std > 0` every step = learning signal present.

4. **Weight the most important reward highest.** WinterFest verdict model: correctness weight 2.0 (highest). ToolRLA: compliance penalty -10 (dominates everything).

5. **Format is a gate, not a score.** PORTool rescales format to [-0.25, 0.25] so it never overrides correctness. Format rewards are necessary but shouldn't dominate.

6. **Metric for best model = "reward", not "eval_loss".** RL optimizes reward. Loss can increase while model improves. `greater_is_better=True`.

### 3c. Reward Hacking — The Real Risk

The model finds shortcuts that maximize the reward without doing the task well.

**Show this to the audience:** "This scored 4.0/4.0. Called all 4 tools, valid JSON, produced a memo. Read the memo: *'AAPL is a company. The financials show numbers. Analysts have opinions. In conclusion, AAPL is a stock.'* That's reward hacking."

| Pattern | Exploits | Mitigation |
|---------|----------|------------|
| Calls every tool regardless of need | Coverage reward | Multiplicative veto (ToolRLA) |
| `<think>ok</think>` (empty reasoning) | Thinking reward | Require substantive content |
| Copies tool output verbatim as "analysis" | Completion reward | Evidence grounding (Trading-R1) |
| Always exactly N tool calls | Efficiency reward | Competing incentives |
| Invents plausible statistics | Quality reward | Compliance penalty -10 (ToolRLA) |

Sources: [Lilian Weng (2024)](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/), [Anthropic emergent misalignment (2025)](https://arxiv.org/abs/2511.18397)

---

## 4. Practical Learnings from WinterFest GRPO (Baseten, 1x H100)

*Full retrospective: `docs/grpo_learnings_winterfest.md`*

### 4a. What Worked

- **Reward progression:** 3.7–4.1 (early) → 5.5–5.7 (sustained mid-training) → 7.70 (best step)
- **12-hour training run** on 1x H100, 2,889 lines of logs
- **Dense rewards** ensured learning signal present every step
- **`beta=0.0`** (no KL penalty) simplified everything — no reference model

### 4b. GRPOTrainer Footguns (Silent Failures)

1. **`remove_unused_columns=False` is MANDATORY** — otherwise TRL strips `ground_truth` before reward functions run. Silently returns 0.
2. **`FastVisionModel.for_training(model)` before trainer construction** — otherwise inference mode, gradients don't flow.
3. **`GRPOTrainer(processing_class=tokenizer, ...)`** not `tokenizer=...` — old kwarg deprecated but doesn't error.
4. **Reward signature:** `fn(completions: List[str], **kwargs) -> List[float]` — dataset columns arrive via `**kwargs`.
5. **`fast_inference` and `use_vllm` must agree** — enabling one without the other silently breaks generation.

### 4c. Meta-Lessons

- **"It ran once" vs "it ran well"** = two different commits, 3 distinct runs.
- **Logging to print() + training.csv** = only reliable post-mortem. Don't trust platform logging.
- **Remove redundant thinking instructions** — Qwen3's template already wraps `<think>`. Duplicating in prompt pollutes reward signals.
- **8B on a single H100 is the realistic envelope.** 2B-4B is practical for Colab T4.
- **Dependency hell was the biggest time-sink** — PEFT 0.10.0 broke Unsloth, pin `peft~=0.17.0`.

---

## 5. Frameworks: Unsloth/TRL vs ART (Mention, Don't Deep-Dive)

Both implement GRPO. Different architectures:

| | Unsloth/TRL GRPOTrainer | ART (OpenPipe) |
|---|---|---|
| Turn support | Single-turn | Multi-turn (tool calling native) |
| Reward | `fn(completions, **kwargs) -> List[float]` | RULER auto-evaluation or custom |
| Best for | Learning GRPO, full control | Production multi-turn agents |

**For the talk:** "Unsloth/TRL for understanding and control. ART for multi-turn tool calling. Same underlying algorithm (GRPO), different wrappers."

---

## 6. Citable Benchmarks (Key Numbers for Slides)

*Full details: `docs/research_sft_rl_benchmarks.md`*

| Claim | Numbers | Source |
|-------|---------|--------|
| RL adds significant delta over SFT | +5 to +42pp depending on model/benchmark | [ToolRL](https://arxiv.org/abs/2504.13958) (Apr 2025) |
| Fine-tuned 7B beats frontier on multi-turn | RC-GRPO 85% vs Opus 4.5 61.25% (BFCLv4 multi-turn subset) | [RC-GRPO](https://arxiv.org/abs/2602.03025) (Feb 2026) |
| Frontier leads overall | Claude Opus 4.5 FC: 77.47% (#1 on full BFCLv4) | [BFCL official](https://gorilla.cs.berkeley.edu/leaderboard.html) |
| RL training is cheap | ART-E: $80, 40 min, single H100, beats o3 | [OpenPipe blog](https://openpipe.ai/blog/art-e-mail-agent) |
| GRPO from scratch can beat SFT+RL | ToolRL: GRPO cold-start outperforms SFT initialization | [ToolRL](https://arxiv.org/abs/2504.13958) |
| RL optimizes efficiency | OTC: 73% fewer tool calls, same accuracy | [OTC](https://arxiv.org/abs/2504.14870) |

---

## 7. Research Gaps — RESOLVED

All previously open research questions have been answered:

- [x] Citable SFT/RL benchmarks → `docs/research_sft_rl_benchmarks.md`
- [x] Delta trends 2023→2026 → same doc, BFCL evolution section
- [x] Better reward candidates → `docs/research_reward_design.md` (Simple + Composite)
- [x] DeepSeek-R1 patterns → radical simplicity (accuracy + format only)
- [x] Financial domain RL → ToolRLA (financial advisory), Trading-R1 (equity research), Fin-PRM (financial reasoning)
- [x] GRPO mechanics explained → `docs/research_grpo_explained.md`
- [x] How gradient traces back → token-level advantage broadcasting through computation graph
