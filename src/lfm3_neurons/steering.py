"""Contrastive neuron finding and single-neuron steering.

**Finding.** Run the model on two prompt sets that differ along one axis
(e.g. "dealing with enemies" vs "making friends"), capture the MLP
intermediate at a chosen layer for every token, and rank neurons by the
absolute difference of their condition means::

    contrast = mean(activations | positive prompts) - mean(activations | negative prompts)

High-|contrast| neurons are candidates whose activation tracks the axis.

**Steering.** Add ``alpha`` to one neuron's intermediate activation at one
layer, for every token of a generation, by patching the layer's MLP forward
(see :class:`~lfm3_neurons.model.MLPInterceptor`). Positive ``alpha`` pushes
the neuron up, negative pulls it down; the sign often flips the behavioral
direction. Use :func:`neuron_steered` as a context manager or
:func:`steer_generate` for one-shot calls.
"""

from __future__ import annotations

import contextlib
from typing import Callable

import torch

from .model import LFM3Model, MLPInterceptor


def _capture_intermediates(
    model: LFM3Model, prompts: list[str], layer: int
) -> torch.Tensor:
    """Capture MLP intermediates at ``layer`` for every token of every prompt."""
    chunks = []
    for prompt in prompts:
        tokens = model.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        captured = model.forward_with_capture(tokens["input_ids"].to(model.device), [layer])
        acts = captured[layer]  # (1, seq, intermediate)
        chunks.append(acts.reshape(-1, acts.shape[-1]))
    if not chunks:
        return torch.zeros((0, model.config.intermediate_size), device=model.device)
    return torch.cat(chunks, dim=0)


def find_contrastive_neurons(
    model: LFM3Model,
    positive_prompts: list[str],
    negative_prompts: list[str],
    layer: int,
    top_k: int = 20,
) -> list[dict]:
    """Rank neurons at ``layer`` by activation contrast between two prompt sets.

    Returns the top-``k`` as dicts with keys ``layer``, ``neuron_idx``,
    ``pos_mean``, ``neg_mean``, ``contrast`` (pos - neg) and ``score``
    (``abs(contrast)``), sorted by score descending.
    """
    pos_acts = _capture_intermediates(model, positive_prompts, layer)
    neg_acts = _capture_intermediates(model, negative_prompts, layer)

    pos_mean = pos_acts.mean(dim=0)
    neg_mean = neg_acts.mean(dim=0)
    contrast = pos_mean - neg_mean
    scores = contrast.abs()

    k = min(top_k, scores.shape[0])
    values, indices = torch.topk(scores, k)
    return [
        {
            "layer": layer,
            "neuron_idx": int(idx),
            "pos_mean": float(pos_mean[idx]),
            "neg_mean": float(neg_mean[idx]),
            "contrast": float(contrast[idx]),
            "score": float(val),
        }
        for val, idx in zip(values, indices)
    ]


@contextlib.contextmanager
def neuron_steered(model: LFM3Model, layer: int, neuron_idx: int, alpha: float):
    """Context manager that adds ``alpha`` to one neuron during all generations."""
    with MLPInterceptor(model.mlp_module(layer), neuron_idx=neuron_idx, alpha=alpha):
        yield


def steer_generate(
    model: LFM3Model,
    prompt: str,
    layer: int,
    neuron_idx: int,
    alpha: float,
    max_tokens: int = 60,
    temperature: float = 0.7,
    do_sample: bool = True,
    seed: int | None = None,
) -> str:
    """Generate from ``prompt`` with one neuron steered at one layer."""
    with neuron_steered(model, layer, neuron_idx, alpha):
        return model.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            do_sample=do_sample,
            seed=seed,
        )
