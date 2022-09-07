from qiskit import QuantumCircuit
import torch
import numpy as np
from qiskit.quantum_info.operators import Operator
from qiskit import IBMQ, Aer, assemble, transpile
class QModel:
    def __init__(self, qc : QuantumCircuit, ancilla_bits: list, dataset_qubits : list, output) -> None:
        self.qc = qc
        self.ancilla_bits = ancilla_bits
        self.features = dataset_qubits[:-2]
        self.output = output
    def forward(self):
        pass
    def de_forward(self):
        pass
