# GRPO Explained — Educational Content for the Talk

*For a 10-minute conference section. Audience knows ML basics but not RL internals.*

**Key sources:**
- DeepSeek Math paper (original GRPO): [arXiv:2402.03300](https://arxiv.org/abs/2402.03300) (Feb 2024)
- Cameron Wolfe's deep dive: [cameronrwolfe.substack.com/p/grpo](https://cameronrwolfe.substack.com/p/grpo) + [/p/grpo-tricks](https://cameronrwolfe.substack.com/p/grpo-tricks)
- Dr. GRPO: [arXiv:2503.06639](https://arxiv.org/abs/2503.06639) (Mar 2025)
- DeepSeek-R1: [arXiv:2501.12948](https://arxiv.org/abs/2501.12948) (Jan 2025)
- HuggingFace TRL GRPOTrainer docs
- Unsloth GRPO tutorial: [unsloth.ai/blog/grpo](https://unsloth.ai/blog/grpo)

---

## The Analogy (Use This on the Slide)

> **Imagine you're a writing teacher.** You give 8 students the same essay prompt. They each write an essay. You grade them: two get A's, four get B's, two get D's.
>
> You don't tell them what a good essay looks like (that would be SFT). Instead, you just tell each student: "Here's how you scored *relative to the group average*." The A students hear "keep doing what you did." The D students hear "change something."
>
> **That's GRPO.** The model generates multiple completions, scores them, and adjusts weights so behaviors shared by high-scoring completions become more likely. **No teacher's answer key needed** — just a way to compare attempts.

For tool-calling specifically:

> Give the model "Research AAPL." It generates 4 trajectories. One calls the right tools efficiently and produces a clean memo (score: 5.5). Another hallucinates a tool and loops (score: -1.0). GRPO says: "be more like trajectory 1, less like trajectory 4." The *group* is the baseline.

---

## The Full Loop (One Training Step)

```
GRPO Training Step
==================

1. SAMPLE: Pick a batch of prompts
   For each prompt, generate G=4 completions (temperature=1.0 for diversity)

2. SCORE: Run reward function on each completion
   Prompt "Research AAPL" → rewards: [5.5, 3.0, 4.2, -1.0]

3. ADVANTAGE: Compute group-relative advantages
   mean = 2.925
   advantages: [+2.575, +0.075, +1.275, -3.925]
   (Dr. GRPO: just r_i - mean, no std normalization)

4. LOSS: For each token in each completion:
   loss += -clip(ratio, 0.8, 1.2) × advantage_of_this_completion
   High-reward completions → make these tokens MORE likely
   Low-reward completions → make these tokens LESS likely

5. UPDATE: Backprop through the transformer, update LoRA weights
   (lr=5e-6, much smaller than SFT's 2e-4)

6. REPEAT for next batch
```

---

## The Math, Made Accessible

### Step 1: The Core Idea (Policy Gradient)

```
L = -E[A(x) × log π(y|x)]
```

In words: "Increase the log-probability of completions with positive advantage. Decrease it for negative advantage."

- `π(y|x)` = probability the model assigns to completion y given prompt x
- `A(x)` = advantage — "how much better was this than average?"

### Step 2: How GRPO Computes Advantages

For each prompt, generate G completions: {y₁, y₂, ..., y_G}

Score each: {r₁, r₂, ..., r_G}

Compute advantages **relative to the group**:

```
Vanilla GRPO:  A_i = (r_i - mean(r)) / std(r)
Dr. GRPO:      A_i = r_i - mean(r)        ← what we use (no std normalization)
```

**This is the "Group Relative" part.** PPO uses a trained critic network to estimate advantages. GRPO throws away the critic and just uses the other completions in the group as the baseline.

Dr. GRPO drops the std normalization to avoid difficulty bias — uniformly hard or easy prompts get naturally smaller advantages.

### Step 3: Clipping (Stability)

Raw policy gradient is unstable — one big update can collapse the model. Clip the ratio between new and old policy:

```
ratio = π_new(y|x) / π_old(y|x)
L = -min(ratio × A, clip(ratio, 1-ε, 1+ε) × A)
```

With ε=0.2, no single update can change any token's probability by more than 20%. This prevents catastrophic forgetting.

### Step 4: How the Gradient Traces Back to Specific Tokens

**This is the mystery: "How does the gradient know which completion it came from, and which tokens to reinforce?"**

Three parts:

1. **Each completion is a sequence of token-level decisions.** When generating completion y_i, the model made choices: token₁, token₂, ..., token_T. Each has a log-probability: `log π(token_t | prompt, tokens₁..t₋₁)`.

2. **Log-probability of the full completion decomposes:**
```
log π(y_i | x) = Σ_t log π(token_t | x, y_<t)
```

3. **The advantage for completion i applies to EVERY token in that completion:**
```
gradient for token_t in completion_i = A_i × ∂/∂θ log π(token_t | x, y_<t)
```

The gradient "knows which completion" because we compute a **separate loss term for each completion**, each with its own advantage. The gradient "knows which tokens" because the loss decomposes into a sum over tokens, and **backpropagation through the transformer naturally attributes the loss to the parameters that produced each token prediction.**

### Concrete Example — Traced with Real Numbers

5 completions for "Research AAPL", temperature=1.0:

```
Completion A: calls get_financials → get_price_history → writes memo     reward = 5.5
Completion B: calls get_news → get_financials → writes memo              reward = 4.0
Completion C: calls get_financials → get_financials again → loops         reward = 1.0
Completion D: calls get_recommendations → writes memo                    reward = 3.5
Completion E: hallucinates get_stock_info → crashes                      reward = -1.0

Group mean = (5.5 + 4.0 + 1.0 + 3.5 + -1.0) / 5 = 2.6

Advantages:
  A: 5.5 - 2.6 = +2.9    "much better than average"
  B: 4.0 - 2.6 = +1.4    "somewhat better"
  C: 1.0 - 2.6 = -1.6    "worse than average"
  D: 3.5 - 2.6 = +0.9    "slightly better"
  E: -1.0 - 2.6 = -3.6   "much worse"
```

**Now compute ONE combined loss:** Each completion contributes a loss term = its advantage × sum of its token log-probs. We add them all up:

```
loss = loss_A + loss_B + loss_C + loss_D + loss_E    ← one scalar number

Where:
  loss_A = -(+2.9) × (logprob("get_financials") + logprob("AAPL") + ...)
  loss_E = -(-3.6) × (logprob("get_stock_info") + logprob("APPLE") + ...)
```

**Why `+=` not `=`?** We're summing contributions from all 5 completions into one number. Each `loss_X` is itself one number (advantage × sum of that completion's token log-probs).

### How Backprop Fans Out (The Key Diagram)

The final loss is one scalar, but backprop follows the **computation graph** — it remembers every operation that produced it. The advantage survives through the chain rule at each branch:

```
                    ┌── Completion A (advantage +2.9) ─────────────┐
                    │   tokens: [get_financials, AAPL, ...]        │
                    │   each token's logprob weighted by +2.9      │
                    │                                              │
                    ├── Completion B (advantage +1.4) ─────────────┤
                    │   each token's logprob weighted by +1.4      │
                    │                                              │
Prompt ──► Model ──┼── Completion C (advantage -1.6) ─────────────┼──► SUM ──► ONE loss
                    │   each token's logprob weighted by -1.6      │           (scalar)
                    │                                              │              │
                    ├── Completion D (advantage +0.9) ─────────────┤         BACKPROP
                    │   each token's logprob weighted by +0.9      │         (chain rule)
                    │                                              │              │
                    └── Completion E (advantage -3.6) ─────────────┘              ▼
                        each token's logprob weighted by -3.6
                                                                          ONE gradient
                                                                          per weight
```

Backprop traces backward from the single loss through each branch:

```
final_loss (one scalar)
  │
  ├── loss_A: chain rule carries +2.9 to every weight that produced A's tokens
  │     ├── weight for "get_financials" → gradient ∝ +2.9 → make MORE likely
  │     ├── weight for "AAPL"           → gradient ∝ +2.9 → make MORE likely
  │     └── weight for "annual"         → gradient ∝ +2.9 → make MORE likely
  │
  ├── loss_C: chain rule carries -1.6 to every weight that produced C's tokens
  │     ├── weight for "get_financials" → gradient ∝ -1.6 → make LESS likely
  │     │   (but A's +2.9 outweighs C's -1.6 → net: slightly more likely)
  │     └── weight for second "get_financials" → gradient ∝ -1.6 → LESS likely
  │
  └── loss_E: chain rule carries -3.6 to every weight that produced E's tokens
        ├── weight for "get_stock_info" → gradient ∝ -3.6 → make MUCH less likely
        └── weight for "APPLE"          → gradient ∝ -3.6 → make MUCH less likely
```

**The advantages don't get averaged away into one number.** They survive through backprop because the computation graph remembers which path each token took. All 5 contributions are combined in ONE backward pass, producing ONE gradient vector that contains all signals. Then ONE weight update: `weight -= lr × gradient`.

**After this single update:**
- `get_financials` is slightly more likely (A pulled it up at +2.9, C pulled down at -1.6, net positive)
- `get_stock_info` is less likely (E pulled it down at -3.6, nothing pulled it up)
- `get_price_history` is slightly more likely (only A used it, at +2.9)

Repeat for thousands of prompts × hundreds of steps → the model converges.

### Why Not 5 Separate Updates?

You might ask: "Why not just do 5 separate backward passes, one per completion?"

You could — mathematically it would give the same gradients (addition is commutative). But combining them:
- Is faster (one backward pass, not five)
- Is how PyTorch/TRL actually implements it
- Lets the optimizer see all signals at once for a more stable update

### From Tokens to Parameters — How the Gradient Reaches the Weights

**Tokens are outputs** — what the model predicted ("get_financials", "AAPL", ...).
**Parameters are weights** — the millions of numbers inside the network that produce those outputs.

The same weights produce *every* token. What changes between tokens is the **input flowing through those weights** (the hidden state from all previous tokens). Think of it like a factory machine that stamps different shapes depending on what raw material you feed in — the machine (weights) is the same, the material (hidden state) changes.

**A tiny concrete example (2 weights, 2 completions):**

The model has `weight_1 = 0.5` and `weight_2 = -0.3`. These weights (plus millions of others) determine the probability distribution over the vocabulary at each step.

```
Token "get_financials" in Completion A (advantage +2.9):
───────────────────────────────────────────────────────
  Forward: context flows through weight_1, weight_2 → P("get_financials") = 0.30
           log_prob = log(0.30) = -1.2

  Loss contribution: -(+2.9) × (-1.2) = +3.48

  Gradient (chain rule — "how much did each weight contribute?"):
    ∂loss/∂weight_1 = -(+2.9) × (∂log_prob/∂weight_1)
                    = -(+2.9) × (+0.4)     ← this 0.4 is the "local gradient"
                    = -1.16                    (depends on the input/hidden state)
    
    ∂loss/∂weight_2 = -(+2.9) × (-0.2) = +0.58


Token "get_stock_info" in Completion E (advantage -3.6):
───────────────────────────────────────────────────────
  Forward: DIFFERENT context flows through SAME weights
           → P("get_stock_info") = 0.15, log_prob = -1.9

  Loss contribution: -(-3.6) × (-1.9) = -6.84

  Gradient (same weights, different local gradients because different input):
    ∂loss/∂weight_1 = -(-3.6) × (+0.1) = +0.36
    ∂loss/∂weight_2 = -(-3.6) × (-0.5) = -1.80
```

**Now combine — same weight, two contributions summed:**

```
weight_1 total gradient = -1.16 (from A's token) + 0.36 (from E's token) = -0.80
weight_2 total gradient = +0.58 (from A's token) + (-1.80) (from E's token) = -1.22

Update (learning_rate = 0.000005):
  weight_1: 0.5   - 0.000005 × (-0.80) = 0.500004   (nudged slightly)
  weight_2: -0.3  - 0.000005 × (-1.22) = -0.299994  (nudged slightly)
```

After this tiny update, `get_financials` is very slightly more likely and `get_stock_info` very slightly less likely when the model sees a similar context. Repeat thousands of times → convergence.

**The full picture:**

```
                         SAME weights (millions of params)
                         ┌─────────────────┐
Context from comp A ───► │ weight_1 = 0.5  │──► P("get_financials")=0.30
(hidden state after      │ weight_2 = -0.3 │    advantage +2.9
 "Research AAPL")        │ weight_3 = ...  │    ∂loss/∂w1 = -1.16 (pulls w1 down)
                         │ ...             │
                         │                 │
Context from comp E ───► │ (same weights!) │──► P("get_stock_info")=0.15
(different hidden state  │                 │    advantage -3.6
 because E chose         │                 │    ∂loss/∂w1 = +0.36 (pulls w1 up)
 different earlier       │                 │
 tokens)                 └─────────────────┘
                         
weight_1 final gradient = -1.16 + 0.36 = -0.80  (net: pull down → more likely to
                                                   produce A-like tokens next time)
```

**Three components per gradient:**
1. **The advantage** (+2.9 or -3.6) — "how much do we care?" (from GRPO)
2. **The local gradient** (0.4 or 0.1) — "how sensitive is this token's probability to this weight?" (from backprop, depends on the hidden state / input context)
3. **The sign** — determines direction: make more likely or less likely

**The final gradient for each weight = sum of (advantage × local gradient) across ALL tokens in ALL completions.** High-advantage completions pull harder. Negative-advantage completions pull in the opposite direction. The weights move in the direction that makes good completions collectively more likely.

---

## Why Temperature, Beta, and Learning Rate Matter

### Temperature = 1.0 (Need Diversity)

If temp is low (0.3), all G completions are nearly identical → same rewards → advantages ≈ 0 → **zero gradient, zero learning.**

At temp=1.0, completions diverge — different tools, different orderings, different reasoning. Reward spread creates real advantages.

**SFT exploits (low temp, deterministic). RL explores (high temp, diverse).**

### Beta = 0.0 (No KL Penalty)

Standard PPO: `L = policy_gradient - β × KL(π_new || π_ref)`
This requires a **frozen copy of the original model** in memory.

With β=0.0:
- No reference model needed (saves ~50% memory)
- No KL computation
- Only constraint on drift = the **clipping mechanism** (ε=0.2)

Works because GRPO's group-relative advantage already provides implicit regularization.

### Learning Rate = 5e-6 (40x Lower Than SFT)

RL is fragile. SFT uses lr=2e-4 because the training signal is clean (ground truth labels). RL's signal is noisy (reward function approximation, advantage estimation). Too high → model collapses. Too low → too slow.

5e-6 with tight gradient clipping (max_grad_norm=0.1) worked in our WinterFest experiment.

---

## GRPO vs PPO: The Key Insight

**PPO architecture:**
```
Policy model + Critic model + Reference model = 3 models, 2 trained
```

**GRPO architecture:**
```
Policy model only = 1 model, 1 trained
```

DeepSeek Math's insight: for language generation, you don't need a learned critic. You can just **generate multiple completions and compare them directly.** The group IS the critic.

- ~60% less memory
- No instability from training two networks simultaneously
- Tradeoff: must generate G completions per prompt (4-8), which is slower at generation time but cheap with vLLM

---

## Existing Visualizations to Reference or Adapt

### Tier 1 — Use Directly in Slides

| What | Source | URL |
|------|--------|-----|
| **GRPO training loop (full pipeline)** | HuggingFace TRL | [grpo_visual.png](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/grpo_visual.png) |
| **PPO vs GRPO architecture (3 models vs 1)** | DeepSeek Math paper, Figure 4 | [arXiv HTML](https://arxiv.org/html/2402.03300v3) |
| **Advantage as gradient multiplier (the classic)** | Karpathy's "Pong from Pixels" (2016) | [rl.png](http://karpathy.github.io/assets/rl/rl.png) vs [sl.png](http://karpathy.github.io/assets/rl/sl.png) |
| **Animated RL training loop** | Cameron Wolfe GRPO post | [cameronrwolfe.substack.com/p/grpo](https://cameronrwolfe.substack.com/p/grpo) (animated GIF in article) |

### Tier 2 — Deep Reference for Study

| What | Source | URL |
|------|--------|-----|
| 22-diagram step-by-step GRPO walkthrough | Vizuara Substack | [vizuara.substack.com/p/how-does-group-relative-policy-optimization](https://vizuara.substack.com/p/how-does-group-relative-policy-optimization) |
| GRPO flow for DeepSeek R1 (detailed) | FareedKhan GitHub | [github.com/FareedKhan-dev/train-deepseek-r1](https://github.com/FareedKhan-dev/train-deepseek-r1) |
| PPO vs GRPO architecture + advantage formula | Oxen.ai blog | [ghost.oxen.ai/why-grpo-is-important](https://ghost.oxen.ai/why-grpo-is-important-and-how-it-works/) |
| Interactive backprop explainer | xnought | [xnought.github.io/backprop-explainer](https://xnought.github.io/backprop-explainer/) |
| Backprop animations (chain rule) | 3Blue1Brown | [youtube.com/watch?v=tIeHLnjs5U8](https://www.youtube.com/watch?v=tIeHLnjs5U8) |
| Policy gradient math derivation | OpenAI Spinning Up | [spinningup.openai.com RL Intro 3](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html) |

### What Doesn't Exist (We Should Build)

The agent searched extensively and concluded: **no existing visualization combines all three of our concepts** (multiple completions → one loss → backprop fans out with advantage as multiplier on each branch → same weights, different hidden states → per-weight gradient).

Our ASCII diagrams in this doc (the "How Backprop Fans Out" section and "From Tokens to Parameters" section) are more detailed and specific than anything publicly available. **We should convert these into an HTML/JS interactive visualization for the talk.**

---

## Suggested Slide Sequence (10 Minutes)

| Time | Content | Visual |
|------|---------|--------|
| 0:00-1:00 | "SFT taught imitation. Now we teach optimization." | SFT vs RL comparison table |
| 1:00-3:00 | The essay grading analogy. 4 completions, score them, reinforce the good ones. | 4 text boxes with score badges, green/red arrows |
| 3:00-5:00 | The GRPO loop: Sample → Score → Advantage → Update. Walk through one step with AAPL. | HuggingFace `grpo_visual.png` or adapted version |
| 5:00-6:30 | "How does the gradient know?" Tokens → params, advantage as multiplier, backprop fans out. | **Custom diagram** (build from our ASCII art) |
| 6:30-7:30 | Why temp=1.0, beta=0.0, lr=5e-6. | Config block with annotations |
| 7:30-8:30 | GRPO vs PPO: 1 model vs 3. The group IS the critic. | DeepSeek Math Figure 4 |
| 8:30-10:00 | Our reward function. Dense rewards. "Your reward function IS your product spec." | Code + reward hacking example |
