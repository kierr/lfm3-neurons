# lfm3-neurons — Agent Instructions

## Purpose
Local mechanistic interpretability for Liquid AI LFM3-1.7B-Base on Apple Silicon (Mac mini, MLX). Find individual neurons/features that steer model behavior. No server, no API keys, local-only.

## Key constraints
- **Run locally only.** MLX on Mac mini GPU. No remote inference, no cloud API.
- **Manage memory.** Model is ~1.73B params (~3.5 GB BF16, ~1 GB 4-bit). Stream activations, batch SAE training, don't dump full activation tensors to disk.
- **No server.** Everything is a Python script or MLX model load. Ollama may have the model but we need direct weight access for activation hooks — use HF + MLX conversion, not ollama API.
- **LFM3 specifics:** 24 layers (13 gated short-conv k=7, 5 sliding-window attn 256, 6 global attn), hidden=2048, intermediate=7680 (SwiGLU MLP in every layer), vocab=128K, context=256K, BF16, trust_remote_code=True.

## Repo layout
- `src/lfm3_neurons/` — core library (model loading, SAE, steering, neuron finding)
- `scripts/` — CLI entry points (download, extract, train, find)
- `configs/sae.yaml` — SAE + training hyperparameters
- `data/` — gitignored (activations, SAE weights, cache)

## Conventions
- Conventional commits: `feat(`, `fix(`, `docs(`, `test(`, `chore(`
- Commit promptly; this is a single-developer local repo but keep history clean
- MLX tensors: `mlx.core` arrays, default device (GPU)

## First experiment target
Layer 6 residual-stream output (2nd global-attention layer) — train one TopK SAE, find features, test activation-addition steering.