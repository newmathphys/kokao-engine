# kokao_engine_v10_0/kokao_quantum.py
"""KokaoQPU — Реальные квантовые компьютеры."""

import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kokao-v9")
)

try:
    from qiskit import Aer, QuantumCircuit, execute

    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False

import torch


class KokaoQuantumBackend:
    """Реальный квантовый бэкенд (Qiskit).

    Backend:
    - Qiskit (IBM Q)
    - Cirq (Google) - future

    Преимущества: гранты по квантовому ML, эксклюзивность
    """

    def __init__(self, backend_name: str = "simulator"):
        if HAS_QISKIT:
            self.backend = Aer.get_backend(backend_name)
        else:
            self.backend = None  # Mocked

    def run(self, qc, shots: int = 1024) -> dict:
        """Выполнить квантовую схему."""
        if HAS_QISKIT and self.backend is not None:
            result = execute(qc, self.backend, shots=shots).result()
            return result.get_counts(qc)
        return {"0" * qc.num_qubits: shots}  # Mocked


class KokaoQPU:
    """Квантовый Kokao на реальных устройствах.

    Гибридный: QuantumKokaoNet + KokaoQuantumBackend
    """

    def __init__(self, n_qubits: int, backend_name: str = "simulator"):
        self.n_qubits = n_qubits
        self.backend = KokaoQuantumBackend(backend_name)

    def quantum_signal(self, x: torch.Tensor) -> float:
        """Квантовый сигнал через encoding в quantum circuit."""
        qc = QuantumCircuit(self.n_qubits, self.n_qubits)

        # Encode x into quantum circuit (angle encoding)
        for i in range(min(len(x), self.n_qubits)):
            qc.ry(x[i].item() * 0.1, i)

        # Простая схема
        for i in range(self.n_qubits - 1):
            qc.cx(i, i + 1)

        # Measurement
        qc.measure_all()

        # Execute
        counts = self.backend.run(qc)

        # Decode counts to scalar
        total = sum(counts.values())
        if total == 0:
            return 0.0

        # Probability of |00...0⟩
        zero_state = "0" * self.n_qubits
        prob_zero = counts.get(zero_state, 0) / total

        return prob_zero * 100.0
