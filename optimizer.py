from config import QuantumConfig
from data import QuantumDataSet
from model import QuantumModel
from qiskit import QuantumCircuit
from qiskit import IBMQ, Aer, assemble, transpile
class QuantumOptimizer:
    def __init__(self, config : QuantumConfig, data : QuantumDataSet, model : QuantumModel) -> None:
        self.config = config
        self.data = data
        self.model = model
        self.cbits = config.cbits
        self.weights = config.weights
        self.qc = config.qc
    def optimize(self) -> None:
        pass
    def execute(self, draw : bool = False) -> dict:
        self.qc.measure(self.weights, self.cbits[0 : len(self.weights)])
        aer_sim = Aer.get_backend('aer_simulator')
        aer_sim.set_options(device='GPU')
        transpiled_grover_circuit = transpile(self.qc, aer_sim)
        qobj = assemble(transpiled_grover_circuit)
        results = aer_sim.run(qobj).result()
        counts = results.get_counts()
        return counts
class GroverOptimizer(QuantumOptimizer):
    def __init__(self, config: QuantumConfig, data: QuantumDataSet, model: QuantumModel, iter : int) -> None:
        super().__init__(config, data, model)
        self.iter = iter
        self.search_space = []
        self.search_space.extend(self.weights)
        self.search_space.extend(self.config.labels)
        for f in self.config.features:
            self.search_space.extend(f)
    def oracle(self) -> None:
        self.model.compute()
        self.qc.barrier()
        for label, output in zip(self.config.labels, self.config.output):
            self.qc.cx(label, output)
        self.qc.barrier()
        for output in self.config.output:
            self.qc.x(output)
        self.qc.barrier()
        self.qc.h(self.config.labels[0])
        self.qc.mct(self.config.output, self.config.labels[0])
        self.qc.h(self.config.labels[0])
        self.qc.barrier()
        self.qc.x(self.config.labels[0])
        self.qc.h(self.config.labels[0])
        self.qc.mct(self.config.output, self.config.labels[0])
        self.qc.h(self.config.labels[0])
        self.qc.x(self.config.labels[0])
        self.qc.barrier()
        for output in reversed(self.config.output):
            self.qc.x(output)
        self.qc.barrier()
        for label, output in reversed(list(zip(self.config.labels, self.config.output))):
            self.qc.cx(label, output)
        self.qc.barrier()
        self.model.uncompute()
        self.qc.barrier()
    def diffusion(self) -> None:
        for q in self.weights:
            self.qc.h(q)
        self.qc.barrier()
        self.data.decode()
        self.qc.barrier()
        self.qc.z(self.search_space[0])
        self.qc.x(self.search_space[0])
        self.qc.z(self.search_space[0])
        self.qc.x(self.search_space[0])
        self.qc.barrier()
        for i in self.search_space:
            self.qc.x(i)
        self.qc.barrier()
        self.qc.h(self.search_space[0])
        self.qc.mct(self.search_space[1:], self.search_space[0])
        self.qc.h(self.search_space[0])
        self.qc.barrier()
        for i in self.search_space:
            self.qc.x(i)
        self.qc.barrier()
        for q in self.weights:
            self.qc.h(q)
        self.qc.barrier()
        self.data.encode()
        self.qc.barrier()
    def optimize(self) -> None:
        for q in self.weights:
            self.qc.h(q)
        self.data.encode()
        for _ in range(self.iter):
            self.oracle()
            self.qc.barrier()
            self.diffusion()
            self.qc.barrier()
