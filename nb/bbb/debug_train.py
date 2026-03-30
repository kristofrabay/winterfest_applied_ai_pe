"""Diagnostic script — test backward pass WITH mx.compile (like the real trainer)."""
import faulthandler, sys, os
from functools import partial
faulthandler.enable()

# NOTE: NOT disabling compile this time — testing the real path
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import types
from pathlib import Path
from mlx.utils import tree_flatten, tree_map

print(f"[1] MLX device info:")
info = mx.device_info()
for k, v in info.items():
    if "mem" in k.lower() or "size" in k.lower() or "buffer" in k.lower():
        print(f"    {k}: {v / 1e9:.2f} GB" if isinstance(v, (int, float)) and v > 1e6 else f"    {k}: {v}")

# Do NOT set wired limit (let it swap) — just measure peak memory
print(f"\n    NOTE: No wired limit set — measuring true peak memory")

print(f"\n[2] Loading model...")
from mlx_lm.utils import load
model, tokenizer = load("mlx-community/Qwen3.5-2B-4bit", tokenizer_config={"trust_remote_code": True})
print(f"    Peak mem: {mx.get_peak_memory() / 1e9:.2f} GB")

print(f"\n[3] Applying LoRA (rank=8, 4 layers)...")
from mlx_lm.tuner.utils import linear_to_lora_layers
model.freeze()
linear_to_lora_layers(model, 4, {"rank": 8, "scale": 2.0, "dropout": 0.0})
trainable = sum(v.size for _, v in tree_flatten(model.trainable_parameters())) / 1e6
print(f"    Trainable: {trainable:.2f}M params")

print(f"\n[3b] Enabling gradient checkpointing...")
from mlx_lm.tuner.trainer import grad_checkpoint as enable_grad_ckpt, default_loss
enable_grad_ckpt(model.layers[0])
print(f"    Done")

print(f"\n[4] Loading training sample...")
from mlx_lm.tuner.datasets import load_local_dataset, CacheDataset
cfg = types.SimpleNamespace(mask_prompt=True, data=str(Path(__file__).resolve().parent.parent.parent / "data" / "bbb" / "mlx_sft"))
train_set, valid_set, _ = load_local_dataset(Path(cfg.data), tokenizer, cfg)
cache = CacheDataset(train_set)

# Use sample 1 (same as before — 4813 tokens, 1626 trainable)
tokens, offset = cache[1]
max_seq = 8192
trunc_len = min(len(tokens), max_seq)
batch_arr = np.zeros((1, trunc_len), np.int32)
batch_arr[0, :trunc_len] = tokens[:trunc_len]
batch = mx.array(batch_arr)
lengths = mx.array([[offset, trunc_len]])
print(f"    Sample: {trunc_len} tokens, {trunc_len - offset} trainable")

print(f"\n[5] Setting up compiled step function (like the real trainer)...")
opt = optim.Adam(learning_rate=1e-5)
loss_value_and_grad = nn.value_and_grad(model, default_loss)
state = [model.state, opt.state, mx.random.state]

@partial(mx.compile, inputs=state, outputs=state)
def step(batch, lengths):
    (lvalue, toks), grad = loss_value_and_grad(model, batch, lengths)
    opt.update(model, grad)
    return lvalue, toks

print(f"    Compiled step function defined")

print(f"\n[6] Running compiled step (this triggers JIT compilation + execution)...")
sys.stdout.flush()
mx.reset_peak_memory()
model.train()
try:
    lvalue, toks = step(batch, lengths)
    mx.eval(state, lvalue, toks)
    print(f"    Loss: {lvalue.item():.4f}, toks: {toks.item()}")
    print(f"    Peak mem (compiled step): {mx.get_peak_memory() / 1e9:.2f} GB")
except Exception as e:
    print(f"    FAILED: {type(e).__name__}: {e}")
    print(f"    Peak mem at failure: {mx.get_peak_memory() / 1e9:.2f} GB")
    sys.exit(1)

print(f"\n[7] Second step (should be faster, compiled kernel cached)...")
sys.stdout.flush()
mx.reset_peak_memory()
try:
    lvalue, toks = step(batch, lengths)
    mx.eval(state, lvalue, toks)
    print(f"    Loss: {lvalue.item():.4f}, toks: {toks.item()}")
    print(f"    Peak mem (2nd step): {mx.get_peak_memory() / 1e9:.2f} GB")
except Exception as e:
    print(f"    FAILED: {type(e).__name__}: {e}")

print(f"\n=== RESULTS ===")
print(f"If peak mem < 11.45 GB: training should work with wired limit")
print(f"If peak mem > 11.45 GB: need to reduce wired limit or model size")
