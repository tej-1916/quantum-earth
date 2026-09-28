"""Configurable Quantum Circuit & Variational Quantum Classifier (VQC) Engine.

Built using PennyLane and Qiskit.
Adheres strictly to scientific integrity:
- Explicit quantum gate operations
- Trainable RY and RZ single-qubit rotations
- Entanglement modes: None, Ring, and Full
- Pauli-Z expectation value measurements
- Dual execution: PennyLane PyTorch backprop simulator + Qiskit QASM/ASCII export
"""

from typing import Dict, Any, List, Optional, Tuple, Literal
import math
import numpy as np
import torch
import torch.nn as nn

import pennylane as qml
import qiskit
from qiskit import QuantumCircuit


EntanglementMode = Literal["none", "ring", "full"]
EncodingType = Literal["angle", "data_reuploading"]


class QuantumCircuitLayer(nn.Module):
    """PyTorch-compatible Variational Quantum Circuit (VQC) module powered by PennyLane.
    
    Architecture:
    Input: (B, n_qubits) compact latent angles
    1. Feature Encoding: Angle encoding (RY rotations)
    2. Variational Layers (depth D):
       - If data_reuploading: repeat feature encoding before each layer
       - Trainable RY(theta) and RZ(phi) rotations on each qubit
       - Entanglement gates (CNOT) according to mode ('none', 'ring', 'full')
    3. Output: (B, n_qubits) Pauli-Z expectation values <Z_i> in [-1.0, 1.0]
    """

    def __init__(
        self,
        n_qubits: int = 8,
        depth: int = 2,
        entanglement: EntanglementMode = "ring",
        encoding: EncodingType = "angle",
        device_name: str = "default.qubit",
        seed: Optional[int] = 42,
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.depth = depth
        self.entanglement = entanglement
        self.encoding = encoding
        self.data_reuploading = (encoding == "data_reuploading")

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Initialize trainable variational rotation angles (uniform in [-pi, pi])
        # weights_ry shape: (depth, n_qubits)
        # weights_rz shape: (depth, n_qubits)
        self.weights_ry = nn.Parameter(
            torch.randn(depth, n_qubits) * (math.pi / 4.0)
        )
        self.weights_rz = nn.Parameter(
            torch.randn(depth, n_qubits) * (math.pi / 4.0)
        )

        # Build PennyLane device and QNode
        self.dev = qml.device(device_name, wires=n_qubits)
        self._qnode = self._build_qnode()

    def _build_qnode(self):
        """Constructs the vectorized PennyLane QNode with torch backpropagation."""
        n_qubits = self.n_qubits
        depth = self.depth
        entanglement = self.entanglement
        data_reuploading = self.data_reuploading

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(inputs, w_ry, w_rz):
            # 1. Initial Angle Encoding (if not data re-uploading)
            if not data_reuploading:
                for i in range(n_qubits):
                    qml.RY(inputs[:, i], wires=i)

            # 2. Variational Layers
            for d in range(depth):
                if data_reuploading:
                    for i in range(n_qubits):
                        qml.RY(inputs[:, i], wires=i)

                # Trainable Single-Qubit Rotations
                for i in range(n_qubits):
                    qml.RY(w_ry[d, i], wires=i)
                    qml.RZ(w_rz[d, i], wires=i)

                # Entanglement
                if entanglement == "ring":
                    for i in range(n_qubits):
                        qml.CNOT(wires=[i, (i + 1) % n_qubits])
                elif entanglement == "full":
                    for i in range(n_qubits):
                        for j in range(i + 1, n_qubits):
                            qml.CNOT(wires=[i, j])
                elif entanglement == "none":
                    pass  # Product state, no entangling gates

            # 3. Measurement: Pauli-Z expectation values
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        return circuit

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Executes quantum forward pass.
        
        Args:
            inputs: Tensor of shape (B, n_qubits) containing feature angles.
        Returns:
            Tensor of shape (B, n_qubits) containing Pauli-Z expectations in [-1, 1].
        """
        # Ensure 2D tensor (B, n_qubits)
        if inputs.ndim == 1:
            inputs = inputs.unsqueeze(0)
            was_1d = True
        else:
            was_1d = False

        # Execute batched QNode
        expvals = self._qnode(inputs, self.weights_ry, self.weights_rz)
        # Stack list of (B,) tensors into (B, n_qubits)
        out = torch.stack(expvals, dim=1)

        return out.squeeze(0) if was_1d else out

    def get_circuit_metadata(self) -> Dict[str, Any]:
        """Returns quantum circuit parameter counts, gate counts, and depth."""
        # Calculate gate counts
        ry_gates = self.n_qubits if not self.data_reuploading else (self.n_qubits * (self.depth + 1))
        ry_gates += self.n_qubits * self.depth  # trainable RY
        rz_gates = self.n_qubits * self.depth  # trainable RZ

        if self.entanglement == "ring":
            cnot_gates = self.n_qubits * self.depth
        elif self.entanglement == "full":
            cnot_gates = ((self.n_qubits * (self.n_qubits - 1)) // 2) * self.depth
        else:
            cnot_gates = 0

        total_gates = ry_gates + rz_gates + cnot_gates
        param_count = self.weights_ry.numel() + self.weights_rz.numel()

        return {
            "n_qubits": self.n_qubits,
            "depth": self.depth,
            "entanglement": self.entanglement,
            "encoding": self.encoding,
            "data_reuploading": self.data_reuploading,
            "trainable_parameters": param_count,
            "gate_counts": {
                "ry_rotations": ry_gates,
                "rz_rotations": rz_gates,
                "cnot_entanglers": cnot_gates,
                "total_gates": total_gates,
            },
            "measurements": f"{self.n_qubits} × Pauli-Z Expectation <Z_i>",
            "simulator": "PennyLane default.qubit (statevector backprop)",
        }


def build_qiskit_circuit(
    n_qubits: int = 8,
    depth: int = 2,
    entanglement: EntanglementMode = "ring",
    encoding: EncodingType = "angle",
    feature_values: Optional[List[float]] = None,
    weights_ry: Optional[np.ndarray | List[List[float]]] = None,
    weights_rz: Optional[np.ndarray | List[List[float]]] = None,
    include_barriers: bool = True,
) -> QuantumCircuit:
    """Constructs a matching Qiskit QuantumCircuit for inspection, export, and visualization."""
    qc = QuantumCircuit(n_qubits, name=f"VQC_{n_qubits}q_d{depth}_{entanglement}")
    data_reuploading = (encoding == "data_reuploading")

    if feature_values is None:
        feature_values = [0.5 * (i + 1) for i in range(n_qubits)]
    if weights_ry is None:
        weights_ry = np.full((depth, n_qubits), 0.2)
    elif isinstance(weights_ry, list):
        weights_ry = np.array(weights_ry)

    if weights_rz is None:
        weights_rz = np.full((depth, n_qubits), 0.3)
    elif isinstance(weights_rz, list):
        weights_rz = np.array(weights_rz)

    # 1. Initial Angle Encoding
    if not data_reuploading:
        for i in range(n_qubits):
            qc.ry(float(feature_values[i]), i)
        if include_barriers:
            qc.barrier()

    # 2. Variational Layers
    for d in range(depth):
        if data_reuploading:
            for i in range(n_qubits):
                qc.ry(float(feature_values[i]), i)
            if include_barriers:
                qc.barrier()

        # Variational Single-Qubit Rotations
        for i in range(n_qubits):
            qc.ry(float(weights_ry[d, i]), i)
            qc.rz(float(weights_rz[d, i]), i)

        # Entanglement
        if entanglement == "ring":
            for i in range(n_qubits):
                qc.cx(i, (i + 1) % n_qubits)
        elif entanglement == "full":
            for i in range(n_qubits):
                for j in range(i + 1, n_qubits):
                    qc.cx(i, j)
        elif entanglement == "none":
            pass

        if include_barriers and d < depth - 1:
            qc.barrier()

    return qc


def get_qiskit_diagram_ascii(
    n_qubits: int = 8,
    depth: int = 2,
    entanglement: EntanglementMode = "ring",
    encoding: EncodingType = "angle",
) -> str:
    """Returns formatted ASCII circuit diagram generated by Qiskit."""
    qc = build_qiskit_circuit(
        n_qubits=n_qubits,
        depth=depth,
        entanglement=entanglement,
        encoding=encoding,
    )
    return str(qc.draw(output="text"))


def get_qasm_string(
    n_qubits: int = 8,
    depth: int = 2,
    entanglement: EntanglementMode = "ring",
    encoding: EncodingType = "angle",
) -> str:
    """Returns OpenQASM 2.0 / 3.0 representation of the circuit."""
    qc = build_qiskit_circuit(
        n_qubits=n_qubits,
        depth=depth,
        entanglement=entanglement,
        encoding=encoding,
    )
    try:
        from qiskit.qasm2 import dumps
        return dumps(qc)
    except Exception:
        try:
            return qc.qasm()
        except Exception:
            return str(qc)
