# GRPO Learnings — WinterFest Verdict Model Retrospective

*Captured from Kristof's earlier GRPO experiment training the investment verdict model on Baseten (1x H100, Jan 2026).*

---

## Starting Point

- Cloned SFT scripts into grpo folder — SFTTrainer/SFTConfig/UnslothVisionDataCollator/`train_on_responses_only` all had to go
- Swapped in GRPOTrainer / GRPOConfig from trl
- Kept same LoRA config (r=32, alpha=64, language + attention + MLP, no vision layers, no dropout)

---

## Dependency Hell (Biggest Time-Sink)

- **PEFT 0.10.0** broke Unsloth's compiled cache with a `VARIANT_NAMING_KEYS` error — THE blocker
  - Fix: `pip install peft~=0.17.0` after pip install unsloth
- Required `trl>=0.22.6` for new GRPO features
- `apt-get install gcc g++` before pip install unsloth — Triton kernels compile at import time
- Tried patched Unsloth multi-GPU PRs (pull/3889 / pull/423) — commented out, kept single-GPU
- Sanity check in run.sh: `from trl import GRPOConfig, GRPOTrainer` + print PEFT/PyTorch/CUDA versions before launching

---

## Model & Memory

- Qwen3-VL-2B-Thinking was too heavy for 1x H100 = OOM
- Swapped to `unsloth/Qwen3-VL-4B-Thinking-bnb-4bit` (7250MB)
- 4-bit quantization (`load_in_4bit=True`), gradient checkpointing set to `"unsloth"`
- `MAX_SEQ_LENGTH=32_768, MAX_PROMPT_LENGTH=20_000, MAX_COMPLETION_LENGTH=min(192, MAX_SEQ — MAX_PROMPT)` — uncapped math blew memory with vLLM

### The `fast_inference` Flag Flip-Flop (Non-Obvious)

- Initial working setup: `fast_inference=False` — required to get GRPO generating working at all
- Later vLLM upgrade flipped to `fast_inference=True` and added `use_vllm=True, vllm_device="cuda:0"`
- **Rule:** don't enable `use_vllm=True` without matching `fast_inference=True` — they must agree
- Colocate mode (`vllm_mode="colocate"`, `vllm_gpu_memory_utilization=0.6`) stayed commented out

---

## Data / Prompt Layer

- GRPO needs `prompt` + `ground_truth` columns (NOT prompt + completion like SFT)
- VL prompts must be list of message dicts with content blocks, not strings:
  `[{"role": "...", "content": [{"type": "text", "text": "..."}]}]`
- `ground_truth` column parsed from the verdict field (handled both JSON string and Python-dict-literal)
- **Critical:** Removed the training instruction "Provide your thinking, then output a valid JSON verdict" — the Qwen3-thinking chat template already wraps `<think>` automatically; the redundant instruction polluted reward signals
- Train/eval split: 70% / 15%

---

## GRPOTrainer API Footguns (Silent Failures)

1. **`remove_unused_columns=False` is MANDATORY** — otherwise TRL strips `ground_truth` before reward funcs run and `correctness_reward_func` receives nothing
2. **`FastVisionModel.for_training(model)`** must be called before trainer construction — otherwise model stays in inference mode, gradients misbehave
3. **New TRL API:** `GRPOTrainer(processing_class=tokenizer, ...)` not `tokenizer=...` — old kwarg is deprecated
4. **Reward function signature:** `fn(completions: List[str], **kwargs) -> List[float]` — dataset columns arrive via `**kwargs`

---

## Reward Function Design (6 Weighted Funcs)

| Function | Reward | Weight | Description |
|----------|--------|--------|-------------|
| `structure_reward_func` | +2.0 for exactly 5 JSON fields, linear penalty for distance, -1.0 for no valid JSON | 1.0 | Structure compliance |
| `field_presence_reward_func` | +0.3 per required field | 1.0 | Field completeness |
| `proceed_signal_validity_reward_func` | +1.0 if enum valid, -0.5 if exists but invalid | 1.5 | Signal correctness |
| `correctness_reward_func` | +3.0 exact match, -1.5 same category, +0.5 one neutral, 0.0 opposite | **2.0** (highest) | Verdict accuracy |
| `reasoning_quality_reward_func` | +0.5 for substantive reasoning before `</think>`, -0.5 if no `</think>` | 0.5 | Reasoning presence |
| `content_quality_reward_func` | +0.2 per evidence-cited field filled | 1.0 | Evidence quality |

**Key principles:**
- **Dense rewards > sparse** — model always gets partial credit
- **JSON extraction:** split on `</think>`, then regex `r'(\{[\s\S]*\}')`, then `json.loads`

---

## GRPO Hyperparameters That Worked

```python
# train.py
learning_rate = 5e-6          # Lower than SFT — RL is sensitive
max_grad_norm = 0.1           # Tight clipping — RL-specific
beta = 0.0                    # No KL penalty, no reference model needed — GRPO's big simplification
loss_type = "dr_grpo"         # No length bias
scale_rewards = False         # Dr. GRPO recommends this
importance_sampling_level = "sequence"  # Better for structured output
epsilon = 0.2                 # PPO-style clipping
num_iterations = 1
num_generations = 4           # Completions per prompt
# Batch math: per_device=1 × num_generations=4 × grad_accum=4 → 16 effective completions/step
temperature = 1.0             # Dropped from 0.95 for diversity
top_p = 1.0
top_k = 0
warmup_ratio = 0.1
weight_decay = 0.1
lr_scheduler = "cosine"
optim = "adamw_8bit"
max_steps = 10
save_steps = 5
seed = 3407
bf16 = True
mask_truncated_completions = False
```

### Eval & Early Stopping

```python
eval_strategy = "steps"
eval_steps = 10
load_best_model_at_end = True
metric_for_best_model = "reward"    # NOT eval_loss!
greater_is_better = True
EarlyStoppingCallback(patience=10, threshold=0.01)
logging_steps = 1
logging_first_step = True
log_completions = True
num_completions_to_print = 1
```

---

## Baseten Platform Gotchas

- Custom Docker images blocked — couldn't use `unsloth/unsloth:2025.10.9-pt2.8.0-cu12.8`
  - Fallback: `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime` + install deps in run.sh
- Python logging output not captured — logs went missing
  - Fix: swap `logger.info(...)` → `print(...)` everywhere
- `set -euo` echoed every command; noisy → changed to `set -eu`
- Shared team Databricks token didn't work → personal DATABRICKS_TOKEN_KRABAY secret
- Output dir: `os.environ.get("BT_CHECKPOINT_DIR", os.environ.get("OUTPUT_DIR"))` — Baseten injects `BT_CHECKPOINT_DIR`
- MLflow: `MLFLOW_TRACKING_URI="databricks"`, experiment path per user

---

## What Finally Worked (The Win)

- **Successful run:** Jan 3 14:44 → Jan 4 02:33 (~12h on 1x H100), 2,889 lines of logs
- Training metrics saved: `training_history.csv` + `training_metrics.png` (reward + loss plots)
- Final LoRA adapters saved

### Observed Reward Behavior

- **Early steps:** reward = 3.7–4.1
- **Mid training:** reward = 5.5–5.7 sustained
- **Best single step:** reward=7.70 (structure=2.0, field_presence=1.5, proceed_signal=1.0, correctness=3.0, content=0.2) — near-optimal
- `reward_std` consistently > 0 = learning signal present every step
- `frac_reward_zero_std = 0.0` — no batches where all completions tied (which would kill training)
- `kl=0.0` always (expected, since `beta=0.0` disables KL term)
- Completion lengths averaged 3,000–4,000 tokens (within 8,192 cap)

---

## Meta-Lessons

1. **RL needs variance** — if all completions in a batch score identically, gradient is zero. The reward-function spread across multiple scales was intentional.
2. **32B model dream stayed a dream** — 8B on a single H100 is the realistic envelope for this stack.
3. **"It ran once" vs "it ran well"** are two different commits — the 455e8b8 → 42f7473 arc shows 3 distinct runs.
4. **Logging to print + saving training.csv** turned out to be the only reliable post-mortem record on Baseten.
5. **Theoretical companion doc** (1,459 lines) was written alongside the experiment — pairs nicely with the practical retrospective.
