from qram import QRAMDataSet, RowPatternDataset
from qmodel import QModel, RowdetectModel
from qoptimizer import QOptimizer
from qiskit import QuantumCircuit
import torch
from qiskit import IBMQ, Aer, assemble, transpile, execute
weights_bits = [0]
address_bits = list(range(1, 10))
dataset_bits = list(range(1, 12))
ancillas = [12, 13, 14, 15]
output = 16
allqubits = list(range(17))
qc = QuantumCircuit(17, 17)
train = torch.Tensor([1 for _ in range(512)]).long()
row_pattern_dataset = RowPatternDataset(address_qubits=address_bits,qc=qc,dataset_qubits=dataset_bits,train=train)
row_pattern_model = RowdetectModel(qc=qc,ancilla_bits=ancillas,dataset_qubits=dataset_bits, weights=weights_bits,output=output)
optimizer = QOptimizer(qc=qc,dataset_qubits=dataset_bits,output=output,data=row_pattern_dataset,model=row_pattern_model,allqubits=allqubits)
row_pattern_dataset.encode()
row_pattern_model.forward()
optimizer.optimize(iter=3)

qc.measure(weights_bits, 0)
aer_sim = Aer.get_backend('aer_simulator')
job = execute(qc, aer_sim, shots=1000)
counts = job.result().get_counts()
print(counts)