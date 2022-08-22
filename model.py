from config import QuantumConfig

class QuantumModel:
    def __init__(self, config : QuantumConfig) -> None:
        self.qc = config.qc
        self.weights = config.weights
        self.features = config.features
        self.outputs = config.output
    def compute(self) -> None:
        pass
    def uncompute(self) -> None:
        pass
