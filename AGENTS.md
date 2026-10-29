# lfm3-neurons — Agent Instructions

## Purpose
Mechanistic interpretability for LiquidAI LFM3-1.7B-Base on Apple Silicon (PyTorch + MPS). Find individual MLP neurons that steer model behavior and demonstrate single-neuron personality steering. No server, no API keys, local-only.

## Key constraints
- **Run locally only.** PyTorch on the Mac mini GPU via MPS. No remote inference, no cloud API.
- **MLX does not work here.** The MLX converter does not support the `lfm3` model_type — do not attempt MLX conversion; use HF + PyTorch with `trust_remote_code=True`.
- **Manage memory.** Model is ~1.73B params (~3.5 GB BF16). Stream activations, batch SAE training, don't dump full activation tensors to disk.
- **Deterministic results.** All committed results in `results/` use greedy decoding (`do_sample=False`) so outputs reproduce token-for-token. When rerunning experiments, keep greedy unless explicitly studying sampling variance.
- **LFM3 specifics:** 24 layers (13 gated short-conv k=7, 5 sliding-window attn 256, 6 global attn), hidden=2048, intermediate=7680 (SwiGLU MLP in every layer), vocab=128K, context=256K, BF16, `trust_remote_code=True`.
- **Layer indexing:** decoder layer module is `model.model.layers[i]`, its MLP is `.feed_forward` with `.w1/.w2/.w3`.

## Repo layout
- `src/lfm3_neurons/` — core library
  - `model.py` — `LFM3Model` (load, generate, `MLPInterceptor` for capture/steer)
  - `steering.py` — `find_contrastive_neurons`, `neuron_steered`, `steer_generate`
  - `metrics.py` — lexicon scores, repetition-loop detector, token divergence
  - `sae.py` — TopK SAE + training
  - `utils.py` — JSON/dir helpers
- `scripts/` — CLIs: `find_neurons`, `steer`, `demo_personality` (flagship), `dose_response`, `extract_activations`, `train_sae`
- `scripts/experiments/` — one-off discovery/sweep scripts kept as the record of how the neurons were originally found (sampling-based; outputs in `results/`)
- `results/` — committed experiment outputs (JSON with full generations)
- `data/` — gitignored (activations, SAE weights, rerun outputs)
- `FINDINGS.md` — full writeup of all results

## Conventions
- Conventional commits: `feat(`, `fix(`, `docs(`, `chore(`
- Commit promptly; single-developer repo but keep history clean
- Steer via `MLPInterceptor` / `steer_generate` (patched `feed_forward.forward`); don't add new hook mechanisms
