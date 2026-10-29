"""lfm3-neurons: single-neuron steering for LiquidAI LFM3-1.7B-Base."""

from .metrics import (
    aggression_hits,
    deescalation_hits,
    has_repetition_loop,
    token_divergence,
)
from .model import LFM3Model, MLPInterceptor, ModelConfig
from .sae import SAEConfig, TopKSAE, train_sae
from .steering import find_contrastive_neurons, neuron_steered, steer_generate

__version__ = "1.0.0"

__all__ = [
    "LFM3Model",
    "MLPInterceptor",
    "ModelConfig",
    "SAEConfig",
    "TopKSAE",
    "aggression_hits",
    "deescalation_hits",
    "find_contrastive_neurons",
    "has_repetition_loop",
    "neuron_steered",
    "steer_generate",
    "token_divergence",
    "train_sae",
]
