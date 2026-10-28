#!/usr/bin/env bash
set -euo pipefail

# Download LFM3-1.7B-Base from Hugging Face (PyTorch format).
# MLX converter doesn't support lfm3 model_type yet, so we use PyTorch + MPS.

MODEL_ID="LiquidAI/LFM3-1.7B-Base"

echo "=== Downloading $MODEL_ID from Hugging Face ==="
python3 << 'PYEOF'
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Device: {device}")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("LiquidAI/LFM3-1.7B-Base", trust_remote_code=True)
print("Tokenizer loaded.")

print("Loading model (this downloads ~3.5 GB BF16 weights)...")
model = AutoModelForCausalLM.from_pretrained(
    "LiquidAI/LFM3-1.7B-Base",
    trust_remote_code=True,
    dtype=torch.bfloat16,
    device_map={"": device},
)
print("Model loaded successfully!")
print("Model type:", type(model).__name__)
PYEOF

echo "=== Done. Model cached in Hugging Face hub ==="