# lfm3-neurons

Local mechanistic interpretability for **Liquid AI LFM3-1.7B-Base** on Apple Silicon (Mac mini, MLX).

No server, no API keys, no cloud. Run everything locally, manage memory explicitly, find individual neurons/features that steer model behavior.

## What this does

1. **Downloads** LFM3-1.7B-Base from Hugging Face and converts it to MLX format.
2. **Hooks** into every layer of LFM3's tri-hybrid architecture (13 gated short-conv + 5 sliding-window attention + 6 global attention), each with a SwiGLU MLP (hidden 2048 → intermediate 7680 → 2048).
3. **Extracts** MLP intermediate activations (the 7680-neuron layer) across a text corpus.
4. **Trains** a TopK Sparse Autoencoder (SAE) on those activations to decompose superposed features.
5. **Finds** neurons/features that steer behavior — e.g., pushing a feature makes the model hostile, poetic, sycophantic, etc.
6. **Steers** by adding activation vectors at specific layers and observing output changes.

## Why LFM3

- Open weights (`LiquidAI/LFM3-1.7B-Base`, LFM Open License v1.0)
- Small enough to run on a Mac mini (~1.73B params, ~1 GB at 4-bit)
- Every layer has standard neurons (SwiGLU MLP) — not just attention/conv
- **Zero prior art** on LFM interpretability/SAEs. This is greenfield.

## Architecture

```
~/git/local/lfm3-neurons/
├── src/lfm3_neurons/
│   ├── model.py      # Load LFM3 in MLX, capture activations per layer
│   ├── sae.py        # TopK Sparse Autoencoder in MLX
│   ├── steer.py      # Activation-addition steering experiments
│   ├── find_neurons.py # Rank neurons by activation contrast, find steering candidates
│   └── utils.py      # Helpers (loading, saving, formatting)
├── scripts/
│   ├── download_model.sh      # HF → MLX conversion
│   ├── extract_activations.py # Run corpus, save activations per layer
│   ├── train_sae.py           # Train SAE on extracted activations
│   └── find_neurons.py        # Find and test steering neurons
├── configs/sae.yaml           # SAE + training hyperparameters
└── data/                      # (gitignored) activations, SAE weights, cache
```

## Quick start

```bash
# 1. Activate venv
source .venv/bin/activate

# 2. Download + convert LFM3-1.7B-Base to MLX (one-time, ~3.5 GB BF16)
./scripts/download_model.sh

# 3. Extract activations from a text corpus
python scripts/extract_activations.py --config configs/sae.yaml

# 4. Train a TopK SAE on layer 6 residual-stream activations
python scripts/train_sae.py --config configs/sae.yaml --layer 6

# 5. Find steering neurons / features
python scripts/find_neurons.py --config configs/sae.yaml --layer 6 --top-k 20
```

## Memory management (Mac mini)

- Model loads in BF16 (~3.5 GB) or 4-bit quantized (~1 GB)
- Activations are streamed — never dumped to disk all at once
- SAE trains on batches, with gradient accumulation if needed
- All tensors stay on the MLX default device (GPU on Apple Silicon)

## License

Model weights: LFM Open License v1.0 (check commercial use).
Code: MIT.