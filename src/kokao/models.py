"""Pydantic models для Kokao Engine."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

# === Ray Models ===


class RayActorConfig(BaseModel):
    """Конфигурация Ray актора для KokaoCore."""

    n_features: int = Field(..., ge=1, description="Размерность входа")
    n_actors: int = Field(default=1, ge=1, le=1000, description="Количество акторов")
    max_retries: int = Field(default=3, ge=1, le=10, description="Попытки перезапуска")


class RayClusterState(BaseModel):
    """Состояние Ray кластера."""

    active_actors: int
    pending_requests: int
    total_vectors_processed: int
    last_heartbeat: float


class RayTrainRequest(BaseModel):
    """Запрос на обучение через Ray."""

    x_batch: List[List[float]]
    target_batch: List[float]
    learning_rate: float = Field(default=0.01, gt=0.0, lt=1.0)
    epochs: int = Field(default=1, ge=1, le=100)


class RayTrainResponse(BaseModel):
    """Ответ на обучение через Ray."""

    success: bool
    loss: float
    new_weights: List[float]
    metrics: Dict[str, float]


# === Quantum Models ===


class QuantumCircuitConfig(BaseModel):
    """Конфигурация квантовой схемы."""

    n_qubits: int = Field(..., ge=1, le=100, description="Количество кубитов")
    depth: int = Field(default=5, ge=1, le=1000, description="Глубина схемы")
    backend_name: str = Field(default="simulator", description="Бэкенд (simulator, ibmq, etc.)")


class QiskitBackend(BaseModel):
    """Qiskit бэкенд."""

    name: str = "qiskit_simulator"
    shots: int = Field(default=1024, ge=1)


class CirqBackend(BaseModel):
    """Cirq бэкенд."""

    name: str = "cirq_simulator"
    repetitions: int = Field(default=1000, ge=1)


class QuantumResponse(BaseModel):
    """Ответ квантового бэкенда."""

    success: bool
    counts: Dict[str, int]
    probability: float
    error_message: Optional[str] = None


# === RL Models ===


class RLState(BaseModel):
    """Состояние RL агента."""

    features: List[float]
    timestamp: Optional[float] = None


class RLAction(BaseModel):
    """Действие RL агента."""

    action_type: str = Field(..., description="Тип действия (EXPLORE, EXPLOIT)")
    value: float = Field(..., ge=-1.0, le=1.0)


class RLReward(BaseModel):
    """Награда RL агента."""

    value: float
    is_terminal: bool = False


class RLPolicyConfig(BaseModel):
    """Конфигурация RL политики."""

    n_features: int = Field(..., ge=1)
    n_actions: int = Field(..., ge=2)
    learning_rate: float = Field(default=0.001, gt=0.0, lt=1.0)
    gamma: float = Field(default=0.99, gt=0.0, le=1.0)
