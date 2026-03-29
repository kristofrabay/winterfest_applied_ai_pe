# Running Open-Source LLMs Locally on Mac (Apple Silicon)

A practical guide to inference and training on your MacBook Pro M1/M2/M3/M4.

---

## The Ecosystem at a Glance

There are four tools you'll hear about. Here's what each one actually does:

| Tool | Inference | Training | Platform | What it is |
|------|-----------|----------|----------|------------|
| **llama.cpp** | Yes | No | Any (CPU, Metal, CUDA) | C++ inference engine for GGUF model files. The foundation that everything else builds on. |
| **Ollama** | Yes | No | Any | Go wrapper around llama.cpp. Adds model management, a registry, and an OpenAI-compatible API. |
| **MLX / mlx-lm** | Yes | **Yes** | Mac only | Apple's ML eXploration framework + its LLM library. The only real option for LoRA/QLoRA training on Apple Silicon. |
| **Unsloth** | No | Yes | **NVIDIA only** | Fast LoRA (Low-Rank Adaptation) training on CUDA GPUs. Does not work on Mac. Period. |

**The key insight:** For inference on Mac, you choose between Ollama (easy) and llama-server (flexible). For training on Mac, MLX is your only option. Unsloth is for cloud GPUs / Databricks.

---

## Glossary

Terms you'll encounter throughout this guide:

| Term | Full Name | What it means |
|------|-----------|---------------|
| **GGUF** | GPT-Generated Unified Format | The file format llama.cpp uses for model weights. One `.gguf` file = one model. Replaced the older GGML format. |
| **LoRA** | Low-Rank Adaptation | Fine-tuning technique that adds small trainable matrices to a frozen base model. Much cheaper than training all weights. |
| **QLoRA** | Quantized LoRA | LoRA applied on top of a 4-bit quantized base model. Saves memory — the base stays tiny, only the adapters train in full precision. |
| **SFT** | Supervised Fine-Tuning | Training a model on input-output examples (like our tool-calling trajectories). |
| **RL / GRPO** | Reinforcement Learning / Group Relative Policy Optimization | Training a model with a reward function instead of examples. GRPO is the specific RL algorithm used by Unsloth/ART. |
| **MLX** | ML eXploration | Apple's machine learning framework, built specifically for Apple Silicon's unified memory architecture. |
| **CUDA** | Compute Unified Device Architecture | NVIDIA's GPU computing platform. Required by Unsloth, PyTorch (default), and most ML frameworks. |
| **Metal** | — | Apple's GPU programming framework. What llama.cpp and MLX use for GPU acceleration on Mac. |
| **MPS** | Metal Performance Shaders | PyTorch's backend for running on Apple GPUs. Works but has many limitations. |
| **KV cache** | Key-Value cache | Memory that stores past attention states so the model doesn't recompute them for every new token. Grows with context length. |
| **Flash Attention** | — | A memory-efficient algorithm for computing attention. Reduces VRAM usage for long contexts. |
| **VRAM** | Video RAM | GPU memory. On Apple Silicon, this is shared with system RAM (unified memory). |
| **HF** | HuggingFace | The main platform for hosting and sharing ML models and datasets. |
| **Jinja** | — | A Python templating engine. Used by llama.cpp to format chat messages and tool definitions. The `--jinja` flag enables it. |
| **EOS** | End of Sequence | A special token that tells the model to stop generating. Different models use different EOS tokens. |
| **DoRA** | Weight-Decomposed Low-Rank Adaptation | A variant of LoRA that also adapts the magnitude of weights. Slightly better quality, slightly more memory. |

---

## Part 1: Inference

### Option A: Ollama — The Easy Path

Ollama is llama.cpp wrapped in a nice CLI with automatic model downloads, a model registry, and an OpenAI-compatible API.

#### Install and run

```bash
# Install (macOS)
brew install ollama

# Pull and run a model — that's it
ollama run qwen3:4b
```

You're now in an interactive chat. Type `/bye` to exit.

#### What's happening under the hood

When you `ollama run`, it:
1. Downloads the GGUF (GPT-Generated Unified Format) weights from Ollama's registry (cached at `~/.ollama/models/`)
2. Starts the Ollama server (`ollama serve`) if not already running
3. Spawns a llama.cpp subprocess that loads the model into unified memory
4. Opens an interactive chat session

The server stays running at `localhost:11434` after you exit the chat.

#### Key commands

```bash
ollama pull qwen3:4b          # Download without starting chat
ollama list                    # Show downloaded models
ollama ps                      # Show loaded (in-memory) models
ollama stop qwen3:4b           # Unload from memory
ollama rm qwen3:4b             # Delete from disk
ollama show qwen3:4b           # Show model metadata, template, params
```

#### Using it from Python

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # required by SDK, not validated
)

response = client.chat.completions.create(
    model="qwen3:4b",
    messages=[{"role": "user", "content": "What is Apple's P/E ratio?"}],
)
print(response.choices[0].message.content)
```

That's the same `OpenAI` SDK you use for GPT — just a different `base_url`.

#### Tool calling

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_financials",
        "description": "Get financial statements for a stock",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "statement_type": {"type": "string", "enum": ["income", "balance_sheet", "cashflow"]},
            },
            "required": ["ticker", "statement_type"],
        },
    },
}]

response = client.chat.completions.create(
    model="qwen3:4b",
    messages=[{"role": "user", "content": "Get AAPL's income statement"}],
    tools=tools,
)

msg = response.choices[0].message
if msg.tool_calls:
    for tc in msg.tool_calls:
        print(f"{tc.function.name}({tc.function.arguments})")
```

**Caveat:** Qwen3 tool calling works in Ollama 0.9.4+. Qwen 3.5 has known bugs — avoid it for tool calling in Ollama for now.

#### Loading custom GGUF files

Create a `Modelfile`:
```dockerfile
FROM /path/to/your-model.gguf
PARAMETER num_ctx 8192
PARAMETER temperature 0.7
```

```bash
ollama create my-model -f Modelfile
ollama run my-model
```

Or load directly from HuggingFace:
```bash
ollama run hf.co/bartowski/Qwen3-4B-GGUF:Q8_0
```

#### Loading LoRA adapters (after fine-tuning)

```dockerfile
FROM qwen3:4b
ADAPTER /path/to/your-lora.gguf
```

```bash
ollama create qwen3-finetuned -f Modelfile
```

#### Memory and performance knobs

| Env Variable | Default | What it does |
|---|---|---|
| `OLLAMA_CONTEXT_LENGTH` | Auto (4096 if <24GB) | Default context window |
| `OLLAMA_NUM_PARALLEL` | 1-4 (auto) | Concurrent request slots |
| `OLLAMA_MAX_LOADED_MODELS` | 3 | Models in memory at once |
| `OLLAMA_KEEP_ALIVE` | 5m | Idle time before unloading |
| `OLLAMA_FLASH_ATTENTION` | 0 | Set to 1 for long contexts |

---

### Option B: llama-server — Full Control

llama.cpp's built-in server. More flags, more control, slightly faster. No model management — you point it at a file.

#### Install

```bash
brew install llama.cpp
```

#### Start a server

```bash
llama-server \
    -hf unsloth/Qwen3-4B-GGUF:Q8_0 \
    -c 8192 \
    -np 2 \
    -fa \
    --jinja \
    --port 8080
```

What each flag does:

| Flag | Purpose |
|------|---------|
| `-hf user/repo:quant` | Download from HuggingFace (cached locally) |
| `-m /path/to/file.gguf` | Or load a local GGUF file |
| `-c 8192` | Context window (tokens) |
| `-np 2` | Parallel request slots (continuous batching) |
| `-fa` | Flash Attention (memory-efficient attention algorithm) — always use it |
| `--jinja` | Enable Jinja templating engine for chat formatting — **required for tool calling** |
| `--port 8080` | Bind port |
| `-ngl auto` | GPU layers (auto = all that fit, default) |

#### Stop the server

```bash
# If foreground: Ctrl+C
# If background:
pkill llama-server
# Or by port:
lsof -ti:8080 | xargs kill
```

#### Connect from Python

Identical to Ollama, different port:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="sk-no-key-required",
)

# Same chat completions, tool calling, etc.
```

#### API endpoints

| Endpoint | What it does |
|----------|-------------|
| `POST /v1/chat/completions` | Chat (supports tools, streaming) |
| `POST /v1/completions` | Text completion |
| `POST /v1/embeddings` | Embeddings (needs `--embedding` flag) |
| `GET /v1/models` | List loaded model |
| `GET /health` | Health check |
| `GET /metrics` | Prometheus metrics (needs `--metrics`) |

#### Continuous batching explained

`-np N` creates N independent processing slots. Continuous batching means new requests are inserted into the GPU's compute batch as soon as a slot frees up — the server doesn't wait for one request to finish before starting another. Each slot gets `context_size / N` tokens of KV cache (Key-Value cache — the memory that stores past attention states).

- `-np 1`: Single user, max context per request
- `-np 2-4`: Good for concurrent notebook cells or a small pipeline
- `-np 8+`: High throughput, less context per request

---

### Ollama vs llama-server — When to Use Which

| | Ollama | llama-server |
|---|---|---|
| **Setup** | `ollama run qwen3:4b` | Multiple CLI flags |
| **Model management** | Built-in (pull/push/list) | You manage GGUF files yourself |
| **Multi-model** | Auto-loads/unloads | One model per server |
| **Context window** | Defaults to 4096, auto-scales | You set it, can go higher |
| **Performance** | Slightly slower (Go overhead) | Slightly faster |
| **Tool calling** | Built-in but model-specific bugs | Reliable via `--jinja` |
| **Best for** | Quick prototyping, multi-model | Production serving, max control |

**Rule of thumb:** Start with Ollama. Switch to llama-server if you need larger context, more parallel slots, or hit tool-calling bugs.

---

### GGUF Quantization — What to Download

Quantization compresses model weights from 16-bit floats to fewer bits (4-bit, 8-bit, etc.), trading a small amount of quality for much smaller file sizes and faster inference. When you see `Q4_K_M`, `Q8_0`, etc., here's what matters:

| Quant | Bits/weight | Quality | Speed | When to use |
|-------|-------------|---------|-------|-------------|
| `Q4_K_M` | ~4.9 | Good | Fast | **Default choice.** Best quality/size tradeoff. |
| `Q5_K_M` | ~5.7 | Very good | Fast | If Q4_K_M feels lossy and you have RAM. |
| `Q8_0` | ~8.5 | Near-lossless | Moderate | Evaluation, quality-sensitive tasks. |
| `F16` | 16 | Lossless | Slow | Baseline reference only. |
| `Q2_K` / `Q3_K` | ~3-4 | Degraded | Fastest | Emergency (model doesn't fit otherwise). |

**The naming:** Q = quantized, number = bits, K = k-quant method, S/M/L = which layers get higher precision (M mixes Q4 and Q6 for important layers).

**Memory rule of thumb:**
```
Total RAM needed ≈ GGUF file size + 500MB overhead + ~200-400MB per parallel slot
```

For Qwen3-4B: Q4_K_M is ~2.5GB on disk → ~3.5GB in memory. Fits easily on 8GB M1.

---

## Part 2: Training

### MLX-LM — The Only Real Option on Mac

Unsloth requires NVIDIA CUDA (Compute Unified Device Architecture) — it will not run on Apple Silicon. PyTorch's MPS (Metal Performance Shaders) backend technically works but can't quantize (no bitsandbytes library), so a 4B model needs ~12-16GB just for weights + optimizer states. MLX is purpose-built for Apple Silicon's unified memory architecture, where CPU and GPU share the same RAM.

#### Install

```bash
uv add "mlx-lm[train]"    # updates pyproject.toml + uv.lock
# or standalone: pip install "mlx-lm[train]"
```

#### The training command

```bash
mlx_lm.lora \
  --model mlx-community/Qwen3-4B-4bit \
  --train \
  --data ./sft_data \
  --iters 600 \
  --batch-size 2 \
  --num-layers -1 \
  --max-seq-length 4096 \
  --mask-prompt \
  --grad-checkpoint \
  --learning-rate 1e-5 \
  --adapter-path ./adapters \
  --save-every 200 \
  --steps-per-report 10 \
  --steps-per-eval 100
```

#### What the flags mean

| Flag | Default | What it does |
|------|---------|-------------|
| `--model` | — | HuggingFace (HF) repo or local path. Use a pre-quantized 4-bit model for QLoRA (Quantized LoRA). |
| `--train` | false | Enable training mode (without this, it just evaluates). |
| `--data` | — | Directory with `train.jsonl` and optionally `valid.jsonl`. |
| `--iters` | 1000 | Training iterations (not epochs). |
| `--batch-size` | 4 | Minibatch size. Reduce to 1-2 if memory is tight. |
| `--num-layers` | 16 | How many transformer layers get LoRA (Low-Rank Adaptation) adapters. `-1` = all layers. **Primary memory knob.** |
| `--max-seq-length` | 2048 | Max sequence length. Longer sequences are truncated. |
| `--mask-prompt` | false | **Train only on assistant completions.** Equivalent to Unsloth's `train_on_responses_only`. |
| `--grad-checkpoint` | false | Trade ~30% speed for lower memory. Use on 8GB machines. |
| `--learning-rate` | 1e-5 | Adam learning rate. |
| `--adapter-path` | `adapters` | Where to save LoRA weights. |
| `--save-every` | 100 | Save checkpoint every N iters. |
| `--steps-per-report` | 10 | Print loss every N iters. |
| `--steps-per-eval` | 200 | Run validation every N iters. |
| `--grad-accumulation-steps` | 1 | Accumulate gradients over N steps (effective larger batch). |

#### LoRA parameters (via YAML config)

CLI flags don't cover LoRA rank/alpha. For full control, use a YAML config:

```yaml
# config.yaml
model: mlx-community/Qwen3-4B-4bit
train: true
data: ./sft_data
iters: 600
batch_size: 2
num_layers: -1
max_seq_length: 4096
mask_prompt: true
grad_checkpoint: true
learning_rate: 1e-5
adapter_path: ./adapters
save_every: 200
steps_per_report: 10
steps_per_eval: 100

lora_parameters:
  rank: 16
  scale: 2.0        # NOTE: this is alpha/rank, NOT alpha itself
  dropout: 0.0
```

**Important:** MLX uses `scale = alpha / rank`. So to match Unsloth's `r=32, lora_alpha=64`, set `rank: 32, scale: 2.0`.

Run with:
```bash
mlx_lm.lora --config config.yaml
```

#### Data format

Your `train.jsonl` supports four formats. For tool-calling SFT, use the **tools format** — one JSON object per line:

```json
{
  "messages": [
    {"role": "system", "content": "You are a research analyst..."},
    {"role": "user", "content": "Research AAPL"},
    {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "get_financials", "arguments": "{\"ticker\": \"AAPL\"}"}}]},
    {"role": "tool", "content": "{\"revenue\": 394000000000}"},
    {"role": "assistant", "content": "**AAPL** | Technology | $3.5T\n..."}
  ],
  "tools": [{"type": "function", "function": {"name": "get_financials", "description": "...", "parameters": {...}}}]
}
```

With `--mask-prompt`, loss is computed only on assistant turns. This maps directly to the `trajectories_sft.jsonl` format from Phase 2.

#### After training

**Generate with adapter (without fusing):**
```bash
mlx_lm.generate \
  --model mlx-community/Qwen3-4B-4bit \
  --adapter-path ./adapters \
  --prompt "Research TSLA" \
  --max-tokens 1024
```

**Fuse adapters into model:**
```bash
mlx_lm.fuse \
  --model mlx-community/Qwen3-4B-4bit \
  --adapter-path ./adapters \
  --save-path ./fused_model
```

**Evaluate on test set:**
```bash
mlx_lm.lora \
  --model mlx-community/Qwen3-4B-4bit \
  --adapter-path ./adapters \
  --data ./sft_data \
  --test
```

**Serve the fused model:**
```bash
mlx_lm.server --model ./fused_model --prompt-cache-warmup
# → OpenAI-compatible API at localhost:8080
# --prompt-cache-warmup is critical for thinking models (Qwen3.5) —
# without it, first request produces extremely long reasoning due to cold KV cache
```

#### Export to GGUF (for Ollama / llama-server)

GGUF export from mlx-lm is limited to Llama/Mistral architectures. For Qwen, the workaround:

```bash
# 1. Fuse and dequantize to HF format
mlx_lm.fuse \
  --model mlx-community/Qwen3-4B-4bit \
  --adapter-path ./adapters \
  --save-path ./fused_model \
  --dequantize

# 2. Convert to GGUF with llama.cpp's converter
python llama.cpp/convert_hf_to_gguf.py ./fused_model --outfile qwen3-4b-finetuned.gguf

# 3. Quantize
llama-quantize qwen3-4b-finetuned.gguf qwen3-4b-finetuned-Q8_0.gguf Q8_0

# 4. Load in Ollama or llama-server
ollama create qwen3-ft -f Modelfile  # Modelfile: FROM ./qwen3-4b-finetuned-Q8_0.gguf
```

#### Memory requirements

| Config | ~Memory | 8GB M1? | 16GB M1? |
|--------|---------|---------|----------|
| Qwen3-4B 4-bit, batch 1, 4 layers, grad ckpt | ~4-5 GB | Yes (tight) | Comfortable |
| Qwen3-4B 4-bit, batch 2, 16 layers | ~6-8 GB | Marginal | Yes |
| Qwen3-4B 4-bit, batch 4, all layers | ~10-12 GB | No | Yes |

**Performance:** ~100-250 tok/s training throughput depending on M1 variant. 200 examples at 600 iterations ≈ 15-45 minutes.

---

### MLX-LM vs Unsloth — Side by Side

| | MLX-LM | Unsloth |
|---|---|---|
| **Runs on** | Mac (Apple Silicon) | NVIDIA GPU (Colab, Databricks, etc.) |
| **Install** | `uv add "mlx-lm[train]"    # updates pyproject.toml + uv.lock
# or standalone: pip install "mlx-lm[train]"` | `pip install unsloth` |
| **Interface** | CLI + YAML config | Python API in notebooks |
| **LoRA config** | `rank`, `scale` (= alpha/rank) | `r`, `lora_alpha` |
| **Prompt masking** | `--mask-prompt` | `train_on_responses_only()` |
| **Quantization** | Pre-quantized model (separate step) | Built-in 4-bit loading |
| **Data format** | `train.jsonl` (chat/tools/text) | HuggingFace Dataset |
| **RL (GRPO)** | Not supported | Supported |
| **Export** | MLX native, GGUF (limited) | HF, GGUF, vLLM |

**The mental model:** Unsloth is a Python library you use inside notebooks. MLX-LM is a CLI tool you run from the terminal. Both do LoRA fine-tuning; they just target different hardware.

---

## Part 3: The Full Local Workflow

Putting it all together for the BBB project:

```
1. Generate data (Phase 2)
   → Run _phase_2_data_gen.ipynb against OpenAI API (GPT-5.4)
   → Output: trajectories_sft.jsonl

2. Train locally (Phase 4 — MLX alternative)
   → Split trajectories_sft.jsonl into train.jsonl / valid.jsonl
   → mlx_lm.lora --model mlx-community/Qwen3-4B-4bit --train --data ./sft_data --mask-prompt ...
   → Output: ./adapters/

3. Test locally (Phase 3)
   → Fuse: mlx_lm.fuse --model ... --adapter-path ./adapters --save-path ./fused
   → Serve: mlx_lm.server --model ./fused
   → Run baseline + SFT eval notebooks pointing at localhost:8080

4. RL (Phase 5) — still needs cloud GPUs
   → Push to Databricks, use Unsloth + ART
```

Or, to demo the SFT model live on stage:

```bash
# Terminal 1: serve the fine-tuned model
mlx_lm.server --model ./fused_model --port 8080

# Terminal 2: Jupyter notebook
# client = OpenAI(base_url="http://localhost:8080/v1", api_key="none")
# → Same agent code, runs locally, $0/call
```

---

## Quick Reference

```bash
# === OLLAMA ===
ollama run qwen3:4b                    # Interactive chat
ollama pull qwen3:4b                   # Download only
ollama list                            # Show models
ollama ps                              # Show loaded models
ollama stop qwen3:4b                   # Unload from memory
ollama rm qwen3:4b                     # Delete
# API at localhost:11434/v1

# === LLAMA-SERVER ===
brew install llama.cpp
llama-server -hf user/repo:Q8_0 -c 8192 -np 2 -fa --jinja --port 8080
pkill llama-server                     # Stop
# API at localhost:8080/v1

# === MLX-LM ===
uv add "mlx-lm[train]"    # updates pyproject.toml + uv.lock
# or standalone: pip install "mlx-lm[train]"
mlx_lm.generate --model mlx-community/Qwen3-4B-4bit -p "Hello"
mlx_lm.server --model mlx-community/Qwen3-4B-4bit
mlx_lm.lora --config config.yaml      # Train
mlx_lm.lora --model ... --adapter-path ./adapters --test  # Evaluate
mlx_lm.fuse --model ... --adapter-path ./adapters --save-path ./fused
```
