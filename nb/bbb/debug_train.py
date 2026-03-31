"""Test Qwen3.5-0.8B with mx.compile — the only path that works."""
import faulthandler, sys, os
from functools import partial
faulthandler.enable()

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import types
from pathlib import Path
from mlx.utils import tree_flatten

MODEL_ID = "mlx-community/Qwen3.5-0.8B-OptiQ-4bit"

print(f"[1] MLX device info:")
info = mx.device_info()
max_working = info["max_recommended_working_set_size"]
print(f"    memory: {info['memory_size'] / 1e9:.2f} GB, wired limit: {max_working / 1e9:.2f} GB")

# SAFETY: limit to 80% — process dies instead of kernel panic
safe_limit = int(max_working * 0.8)
mx.set_wired_limit(safe_limit)
print(f"    Safe wired limit: {safe_limit / 1e9:.2f} GB")

print(f"\n[2] Loading {MODEL_ID}...")
from mlx_lm.utils import load
model, tokenizer = load(MODEL_ID, tokenizer_config={"trust_remote_code": True})
print(f"    Peak mem: {mx.get_peak_memory() / 1e9:.2f} GB, layers: {len(model.layers)}")

print(f"\n[3] Applying LoRA (rank=8, 4 layers, auto keys)...")
from mlx_lm.tuner.utils import linear_to_lora_layers
model.freeze()
linear_to_lora_layers(model, 4, {"rank": 8, "scale": 2.0, "dropout": 0.0})
trainable = sum(v.size for _, v in tree_flatten(model.trainable_parameters())) / 1e6
print(f"    Trainable: {trainable:.2f}M")

print(f"\n[4] Gradient checkpointing...")
from mlx_lm.tuner.trainer import grad_checkpoint as enable_grad_ckpt, default_loss
enable_grad_ckpt(model.layers[0])

print(f"\n[5] Loading shortest training sample...")
from mlx_lm.tuner.datasets import load_local_dataset, CacheDataset
cfg = types.SimpleNamespace(mask_prompt=True, data=str(Path(__file__).resolve().parent.parent.parent / "data" / "bbb" / "mlx_sft"))
train_set, valid_set, _ = load_local_dataset(Path(cfg.data), tokenizer, cfg)
cache = CacheDataset(train_set)

tokens, offset = cache[1]  # 4813 tokens, shortest with trainable toks
max_seq = 8192
trunc_len = min(len(tokens), max_seq)
batch_arr = np.zeros((1, trunc_len), np.int32)
batch_arr[0, :trunc_len] = tokens[:trunc_len]
batch = mx.array(batch_arr)
lengths = mx.array([[offset, trunc_len]])
print(f"    {trunc_len} tokens, {trunc_len - offset} trainable")

print(f"\n[6] Setting up COMPILED step (like real trainer)...")
opt = optim.Adam(learning_rate=1e-5)
loss_value_and_grad = nn.value_and_grad(model, default_loss)
state = [model.state, opt.state, mx.random.state]

@partial(mx.compile, inputs=state, outputs=state)
def step(batch, lengths):
    (lvalue, toks), grad = loss_value_and_grad(model, batch, lengths)
    opt.update(model, grad)
    return lvalue, toks

print(f"    Compiled step function ready")

print(f"\n[7] Running compiled step (JIT compile + execute)...")
print(f"    This may take 1-3 minutes for JIT compilation...")
sys.stdout.flush()
mx.reset_peak_memory()
model.train()
try:
    lvalue, toks = step(batch, lengths)
    mx.eval(state, lvalue, toks)
    peak = mx.get_peak_memory() / 1e9
    print(f"    Loss: {lvalue.item():.4f}, toks: {toks.item()}")
    print(f"    Peak mem (compiled step): {peak:.2f} GB")
except Exception as e:
    print(f"    FAILED: {type(e).__name__}: {e}")
    print(f"    Peak mem at failure: {mx.get_peak_memory() / 1e9:.2f} GB")
    sys.exit(1)

print(f"\n[8] Second step (cached kernel, should be fast)...")
mx.reset_peak_memory()
try:
    lvalue, toks = step(batch, lengths)
    mx.eval(state, lvalue, toks)
    peak2 = mx.get_peak_memory() / 1e9
    print(f"    Loss: {lvalue.item():.4f}")
    print(f"    Peak mem (2nd step): {peak2:.2f} GB")
except Exception as e:
    print(f"    FAILED: {type(e).__name__}: {e}")

print(f"\n=== RESULTS ===")
print(f"Compiled peak: {peak:.2f} GB")
if peak < 8:
    print("EXCELLENT: Fits comfortably. Training will work!")
elif peak < 11:
    print("TIGHT but should work. Close other apps during training.")
else:
    print("TOO LARGE: Won't fit with wired limit. Need smaller model or cloud.")
