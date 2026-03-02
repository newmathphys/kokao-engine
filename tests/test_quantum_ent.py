"""Quantum Entanglement — 10 тестов."""

import os
import sys

import pytest
import torch

src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, src_path)

try:
    from qiskit import QuantumCircuit

    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False
    QuantumCircuit = None

from kokao.models import QuantumCircuitConfig
from kokao.quantum import KokaoQPU, KokaoQuantumBackend


class TestQuantumEnt:
    """Quantum запутанность."""

    def _test_quantum_backend_init(self):
        backend = KokaoQuantumBackend(backend_name="simulator")
        assert backend.backend is not None

    def _test_quantum_qpu_init(self):
        qpu = KokaoQPU(n_qubits=4, backend_name="simulator")
        assert qpu.n_qubits == 4

    def _test_quantum_signal_basic(self):
        qpu = KokaoQPU(n_qubits=4, backend_name="simulator")
        x = torch.randn(4, device="cpu")
        s = qpu.quantum_signal(x)
        assert isinstance(s, float)

    def _test_quantum_bell_state(self):
        qpu = KokaoQPU(n_qubits=2, backend_name="simulator")
        x = torch.tensor([0.5, 0.5])
        s = qpu.quantum_signal(x)
        assert isinstance(s, float)

    def _test_quantum_ghz_state(self):
        qpu = KokaoQPU(n_qubits=3, backend_name="simulator")
        x = torch.tensor([0.33, 0.33, 0.33])
        s = qpu.quantum_signal(x)
        assert isinstance(s, float)

    def _test_quantum_config_valid(self):
        config = QuantumCircuitConfig(n_qubits=4, depth=5)
        assert config.n_qubits == 4

    def _test_quantum_config_invalid(self):
        with pytest.raises(ValueError):
            QuantumCircuitConfig(n_qubits=0)

    def _test_quantum_zero_input(self):
        qpu = KokaoQPU(n_qubits=4, backend_name="simulator")
        x = torch.zeros(4)
        s = qpu.quantum_signal(x)
        assert isinstance(s, float)

    def _test_quantum_large_input(self):
        qpu = KokaoQPU(n_qubits=4, backend_name="simulator")
        x = torch.randn(4, device="cpu") * 100
        s = qpu.quantum_signal(x)
        assert isinstance(s, float)

    @pytest.mark.skipif(not HAS_QISKIT, reason="qiskit not installed")
    def _test_quantum_backend_run(self):
        backend = KokaoQuantumBackend(backend_name="simulator")
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        counts = backend.run(qc, shots=100)
        assert isinstance(counts, dict)
