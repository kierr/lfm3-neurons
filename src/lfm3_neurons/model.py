"""Loading LFM3-1.7B-Base and capturing MLP intermediate activations.

The model runs in PyTorch on MPS (Apple Silicon GPU). MLX was ruled out: its
converter does not support the ``lfm3`` architecture yet.

LFM3-1.7B is a tri-hybrid architecture (gated short convolutions,
sliding-window attention, global attention) where every layer has a SwiGLU
MLP::

    feed_forward(x) = w2( silu(w1(x)) * w3(x) )

The product ``silu(w1(x)) * w3(x)`` is the 7680-dimensional *MLP intermediate*
-- the closest thing LFM3 has to classic per-neuron units. All neuron finding
and steering in this repo operates on that vector.

The intermediate is a local variable inside ``Lfm3MLP.forward``, so it cannot
be reached with a plain forward hook on the module. Instead, this module
patches ``feed_forward.forward`` with an equivalent implementation that
records the intermediate and/or injects a steering vector before ``w2``
(see :class:`MLPInterceptor`). The same mechanism powers activation capture
and steering, so both operate on exactly the same quantity.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Iterator

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ModelConfig:
    model_id: str = "LiquidAI/LFM3-1.7B-Base"
    dtype: str = "bfloat16"  # "bfloat16" | "float32"
    device: str = "mps"  # "mps" | "cpu" | "cuda"
    trust_remote_code: bool = True
    num_layers: int = 24
    hidden_size: int = 2048
    intermediate_size: int = 7680


class MLPInterceptor:
    """Patch one ``Lfm3MLP`` module to expose (and optionally steer) its intermediate.

    On enter, replaces ``mlp.forward`` with an equivalent SwiGLU implementation
    that (a) appends the intermediate ``silu(w1(x)) * w3(x)`` to ``capture``
    (if given) and (b) adds ``alpha`` to position ``neuron_idx`` of the
    intermediate (if ``neuron_idx`` is given) before ``w2``. On exit the
    original forward is restored. Works during both plain forwards and
    ``generate()`` calls.
    """

    def __init__(
        self,
        mlp: torch.nn.Module,
        capture: list | None = None,
        neuron_idx: int | None = None,
        alpha: float = 0.0,
    ):
        self.mlp = mlp
        self.capture = capture
        self.neuron_idx = neuron_idx
        self.alpha = alpha
        self._original = None
        self._was_instance_attr = False

    def _patched_forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.mlp.w1(x))
        up = self.mlp.w3(x)
        intermediate = gate * up
        if self.capture is not None:
            self.capture.append(intermediate.detach())
        if self.neuron_idx is not None:
            intermediate = intermediate.clone()
            intermediate[:, :, self.neuron_idx] += self.alpha
        return self.mlp.w2(intermediate)

    def __enter__(self) -> "MLPInterceptor":
        self._original = self.mlp.forward
        self._was_instance_attr = "forward" in self.mlp.__dict__
        self.mlp.forward = self._patched_forward
        return self

    def __exit__(self, *exc) -> None:
        if self._was_instance_attr:
            self.mlp.forward = self._original
        else:
            del self.mlp.forward  # restore lookup to the class method


class LFM3Model:
    """Wrapper around LFM3-1.7B-Base with MLP intermediate capture and steering."""

    def __init__(self, config: ModelConfig | None = None):
        self.config = config or ModelConfig()
        self.model = None
        self.tokenizer = None
        self.device = torch.device(self.config.device)

    def load(self) -> "LFM3Model":
        """Load tokenizer + model from Hugging Face onto the target device."""
        cfg = self.config
        print(f"Loading {cfg.model_id} (dtype={cfg.dtype}, device={cfg.device})...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_id, trust_remote_code=cfg.trust_remote_code
        )
        dtype = torch.bfloat16 if cfg.dtype == "bfloat16" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            trust_remote_code=cfg.trust_remote_code,
            dtype=dtype,
            device_map={"": self.device},
        )
        self.model.eval()
        print("Model loaded.")
        return self

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_tokens: int = 60,
        temperature: float = 0.7,
        do_sample: bool = True,
        seed: int | None = None,
    ) -> str:
        """Generate a completion from a prompt string.

        ``do_sample=False`` gives greedy (deterministic) decoding, which all
        committed results in ``results/`` use. With sampling, pass ``seed``
        for reproducibility.
        """
        if seed is not None:
            torch.manual_seed(seed)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        gen_kwargs = dict(
            max_new_tokens=max_tokens,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Activation capture
    # ------------------------------------------------------------------

    def mlp_module(self, layer: int) -> torch.nn.Module:
        """The SwiGLU MLP module of decoder layer ``layer``."""
        return self.model.model.layers[layer].feed_forward

    def forward_with_capture(
        self, input_ids: torch.Tensor, layers: list[int]
    ) -> dict[int, torch.Tensor]:
        """Run one forward pass and return per-layer MLP intermediates.

        Returns a dict mapping layer index -> tensor of shape
        ``(batch, seq, intermediate_size)``.
        """
        captures: dict[int, list] = {layer: [] for layer in layers}
        with contextlib.ExitStack() as stack:
            for layer in layers:
                stack.enter_context(
                    MLPInterceptor(self.mlp_module(layer), capture=captures[layer])
                )
            with torch.no_grad():
                _ = self.model(input_ids)
        return {layer: torch.cat(tensors, dim=0) for layer, tensors in captures.items()}
