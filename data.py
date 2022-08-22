from qiskit import QuantumCircuit


class QuantumDataSet:
    def __init__(self, features : list[list], label : list, qc : QuantumCircuit) -> None:
        self.features = features
        self.qc = qc
        self.label = label
    def encode(self) -> None:
        pass
    def decode(self) -> None:
        pass
