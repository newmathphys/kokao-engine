"""Kokao Engine v1.3.0 — Production-Plus Ultra."""

__version__ = "1.0.0"
__author__ = "Kokao Team"

from .autologic import AutoLogic
from .cf import CounterfactualKokao
from .clip import KokaoMultimodal
from .core import KokaoCoreV9, KOKAOEngine
from .dask import DaskBackend
from .decoder import KokaoDecoder
from .evolve import EvolveKokao
from .kg import KokaoKGStream
from .learnable import KokaoCoreWithLearnableForget
from .mlflow import MLflowLogger, log_signal, log_training
from .models import (QuantumCircuitConfig, QuantumResponse, RayActorConfig,
                     RayTrainRequest, RayTrainResponse, RLAction,
                     RLPolicyConfig, RLReward, RLState)
from .quantum import KokaoQPU, KokaoQuantumBackend
from .ray import KokaoRayActor, KokaoRayCluster
from .rl import KokaoPolicy, KokaoRLAgent
from .robustness import log_execution, retry, rollback
from .snn import KokaoSNNCore, KokaoSpikingLayer

__all__ = [
    # Core
    "KokaoCoreV9",
    "KOKAOEngine",
    # Modules
    "KokaoCoreWithLearnableForget",
    "CounterfactualKokao",
    "KokaoDecoder",
    "EvolveKokao",
    "KokaoSpikingLayer",
    "KokaoSNNCore",
    "KokaoMultimodal",
    "KokaoKGStream",
    "AutoLogic",
    "KokaoRayActor",
    "KokaoRayCluster",
    "KokaoQuantumBackend",
    "KokaoQPU",
    "KokaoPolicy",
    "KokaoRLAgent",
    "DaskBackend",
    # Models
    "RayActorConfig",
    "RayTrainRequest",
    "RayTrainResponse",
    "QuantumCircuitConfig",
    "QuantumResponse",
    "RLState",
    "RLAction",
    "RLReward",
    "RLPolicyConfig",
    # Utils
    "retry",
    "rollback",
    "log_execution",
    "MLflowLogger",
    "log_training",
    "log_signal",
    # Version
    "__version__",
]
