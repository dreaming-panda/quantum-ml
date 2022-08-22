from ast import Assert
from qiskit import QuantumCircuit
class QuantumConfig:
    def __init__(self, weights : list, features : list[list], labels : list, output : list, qc : QuantumCircuit, classical_bits : list) -> None:
        self.qc = qc
        self.weights = weights
        self.features = features
        self.labels = labels
        self.output = output
        self.cbits = classical_bits
        Assert(len(self.cbits) >= len(self.weights)) ,'too few classical bits'
