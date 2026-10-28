"""LFM3 neuron interpretability package."""

from .model import LFM3Model, ModelConfig, ExtractConfig
from .sae import TopKSAE, SAEConfig, train_sae
from .find_neurons import find_contrastive_neurons, steer_generate, SteerConfig

__version__ = "0.1.0"