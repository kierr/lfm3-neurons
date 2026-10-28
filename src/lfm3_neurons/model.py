"""Core library for LFM3 neuron/feature interpretability on Apple Silicon (PyTorch + MPS).

Pipeline:
  1. Load LFM3-1.7B-Base in PyTorch (HF -> MPS)
  2. Capture per-layer activations via forward hooks (MLP intermediate = 7680-neuron layer)
  3. Train TopK SAE on activations to decompose superposed features
  4. Find steering neurons/features via activation contrast
  5. Steer by activation addition at specific layers
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from .utils import ensure_dir, save_json, load_json


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    model_id: str = "LiquidAI/LFM3-1.7B-Base"
    dtype: str = "bfloat16"
    device: str = "mps"  # "mps" | "cpu" | "cuda"
    trust_remote_code: bool = True
    hidden_size: int = 2048
    intermediate_size: int = 7680
    num_layers: int = 24


@dataclass
class SAEConfig:
    layer: int = 6
    d_in: int = 2048
    d_sae: int = 16384  # 8x expansion
    k: int = 64  # TopK sparsity
    lr: float = 1e-3
    batch_size: int = 256
    num_epochs: int = 10
    l1_coeff: float = 3e-4
    seed: int = 42


@dataclass
class ExtractConfig:
    dataset: str = "roneneldan/TinyStories"
    split: str = "train"
    max_samples: int = 5000
    max_tokens_per_sample: int = 512
    batch_size: int = 8
    layers: list[int] = field(default_factory=lambda: [6])
    activation_type: str = "mlp_intermediate"
    output_dir: str = "./data/activations"


# ---------------------------------------------------------------------------
# Model loading with activation capture
# ---------------------------------------------------------------------------

class LFM3Model:
    """Wrapper around LFM3 in PyTorch with per-layer activation capture via hooks."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = torch.device(config.device)
        self._captured: dict[int, torch.Tensor] = {}
        self._handles: list = []

    def load(self):
        """Load model + tokenizer from Hugging Face."""
        print(f"Loading tokenizer from {self.config.model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id, trust_remote_code=self.config.trust_remote_code
        )

        dtype = torch.bfloat16 if self.config.dtype == "bfloat16" else torch.float32
        print(f"Loading model (dtype={self.config.dtype}, device={self.device})...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            trust_remote_code=self.config.trust_remote_code,
            dtype=dtype,
            device_map={"": self.device},
        )
        self.model.eval()
        print(f"Model loaded: {self.config.model_id}")
        return self.model, self.tokenizer

    def generate(self, prompt: str, max_tokens: int = 100, temperature: float = 0.7) -> str:
        """Generate text from a prompt string."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Activation capture via forward hooks
    # ------------------------------------------------------------------

    def _make_hook(self, layer_idx: int):
        """Create a forward hook that captures MLP intermediate activation."""
        def hook(module, input_args, output):
            # output is the MLP intermediate: silu(w1(x)) * w3(x) — shape (batch, seq, 7680)
            self._captured[layer_idx] = output.detach()
        return hook

    def install_hooks(self, layers: list[int]):
        """Install forward hooks on MLP intermediate of specified layers.

        In LFM3, each decoder layer's `feed_forward` is a SwiGLU MLP.
        We hook `feed_forward.w1` output? No — we want the gated intermediate:
        silu(w1(x)) * w3(x). That's computed inside feed_forward.forward.
        
        The cleanest hook point is `feed_forward` module itself — but its output
        is w2(intermediate), not intermediate. So we hook the internal w1/w3
        computation. 

        Actually, looking at Lfm3MLP.forward: return self.w2(F.silu(self.w1(x)) * self.w3(x))
        The intermediate `F.silu(self.w1(x)) * self.w3(x)` is a local variable.
        To capture it, we need to hook either:
        - w1 and w3 separately and recompute the gate, or
        - Replace the forward method.

        Simplest: hook on `feed_forward.w1` output and `feed_forward.w3` output,
        then compute silu(w1_out) * w3_out ourselves.
        
        But w1 and w3 have the same input (h_normed). Their outputs are both
        (batch, seq, 7680). We can capture w1 output, apply silu, multiply by
        w3 output captured from its own hook.
        """
        self._captured.clear()

        w1_hooks = []
        w3_hooks = []
        w1_outs: dict[int, torch.Tensor] = {}
        w3_outs: dict[int, torch.Tensor] = {}

        def make_w1_hook(layer_idx):
            def hook(module, input_args, output):
                w1_outs[layer_idx] = output.detach()
            return hook

        def make_w3_hook(layer_idx):
            def hook(module, input_args, output):
                w3_outs[layer_idx] = output.detach()
                # Now we have both w1 and w3 — compute the gated intermediate
                if layer_idx in w1_outs:
                    import torch.nn.functional as F
                    intermediate = F.silu(w1_outs[layer_idx]) * output.detach()
                    self._captured[layer_idx] = intermediate
            return hook

        for layer_idx in layers:
            layer = self.model.model.layers[layer_idx]
            h1 = layer.feed_forward.w1.register_forward_hook(make_w1_hook(layer_idx))
            h3 = layer.feed_forward.w3.register_forward_hook(make_w3_hook(layer_idx))
            w1_hooks.append(h1)
            w3_hooks.append(h3)

        self._handles = w1_hooks + w3_hooks
        return self._captured

    def remove_hooks(self):
        for handle in self._handles:
            handle.remove()
        self._handles = []
        self._captured.clear()

    def forward_with_capture(self, input_ids: torch.Tensor, layers: list[int]) -> dict[int, torch.Tensor]:
        """Run forward pass, capturing MLP intermediate activations at `layers`."""
        self.install_hooks(layers)
        self._captured.clear()
        with torch.no_grad():
            _ = self.model(input_ids)
        captured = dict(self._captured)
        self.remove_hooks()
        return captured

    def get_layer_module(self, layer_idx: int):
        """Get the decoder layer module at `layer_idx`."""
        return self.model.model.layers[layer_idx]


def extract_cli():
    """CLI entry point: extract activations from a corpus."""
    import argparse
    from datasets import load_dataset

    parser = argparse.ArgumentParser(description="Extract LFM3 activations")
    parser.add_argument("--config", type=str, default="configs/sae.yaml")
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--max-samples", type=int, default=5000)
    args = parser.parse_args()

    cfg = ModelConfig()
    model = LFM3Model(cfg)
    model.load()

    ds = load_dataset("roneneldan/TinyStories", split="train")
    texts = [example["text"] for example in ds.select(range(min(args.max_samples, len(ds))))]

    all_acts: list[torch.Tensor] = []
    batch_size = 8

    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting"):
        batch_texts = texts[i : i + batch_size]
        tokens = model.tokenizer(
            batch_texts, return_tensors="pt", padding=True, truncation=True,
            max_length=512,
        )
        input_ids = tokens["input_ids"].to(model.device)
        captured = model.forward_with_capture(input_ids, layers=[args.layer])
        if args.layer in captured:
            acts = captured[args.layer]  # (batch, seq, intermediate)
            acts = acts.reshape(-1, acts.shape[-1])
            all_acts.append(acts.cpu())

    if not all_acts:
        print("No activations captured!")
        return

    combined = torch.cat(all_acts, dim=0)
    flat = combined.float().numpy().astype(np.float32)
    print(f"Total activations: {flat.shape}")

    out_dir = Path(args.output_dir if hasattr(args, 'output_dir') else "./data/activations")
    ensure_dir(str(out_dir))
    out_path = out_dir / f"layer_{args.layer}_activations.npy"
    np.save(out_path, flat)
    print(f"Saved to {out_path}")