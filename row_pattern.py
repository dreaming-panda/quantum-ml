from turtle import title
from qram import QRAMDataSet, RowPatternDataset
from qmodel import QModel, RowdetectModel, RowdetectModelA, RowdetectModelB
from qoptimizer import QOptimizer
from qiskit import QuantumCircuit
import matplotlib.pyplot as plt
import torch
from qiskit.visualization import plot_histogram,plot_state_city
from qiskit import IBMQ, Aer, assemble, transpile, execute
import time
def Optimize():
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
    qc.measure(weights_bits, weights_bits)
    aer_sim = Aer.get_backend('aer_simulator')
    job = execute(qc, aer_sim, shots=1000000)
    counts = job.result().get_counts()
    print(counts)
    fig = plot_histogram(counts)
    fig.savefig("row_pattern.jpg")
def Optimize_A():
    weights_bits = [0, 1, 2]
    address_bits = list(range(3, 12))
    dataset_bits = list(range(3, 14))
    ancillas = [14, 15, 16, 17]
    output = 18
    allqubits = list(range(19))
    qc = QuantumCircuit(19, 3)
    train = torch.Tensor([1 for _ in range(512)]).long()
    row_pattern_dataset = RowPatternDataset(address_qubits=address_bits,qc=qc,dataset_qubits=dataset_bits,train=train)
    row_pattern_model = RowdetectModelA(qc=qc,ancilla_bits=ancillas,dataset_qubits=dataset_bits, weights=weights_bits,output=output)
    optimizer = QOptimizer(qc=qc,dataset_qubits=dataset_bits,output=output,data=row_pattern_dataset,model=row_pattern_model,allqubits=allqubits)
    row_pattern_dataset.encode()
    row_pattern_model.forward()
    optimizer.optimize(iter=3)
    qc.measure(weights_bits, weights_bits)
    aer_sim = Aer.get_backend('aer_simulator')
    job = execute(qc, aer_sim, shots=1000000)
    counts = job.result().get_counts()
    print(counts)
    fig = plot_histogram(counts)
    fig.savefig("row_pattern_A.jpg")
def Optimize_B(iter):
    weights_bits = [0, 1, 2]
    address_bits = list(range(3, 12))
    dataset_bits = list(range(3, 14))
    ancillas = [14, 15, 16, 17, 18, 19, 20, 21]
    output = 22
    allqubits = list(range(23))
    qc = QuantumCircuit(23, 3)
    train = torch.Tensor([1 for _ in range(512)]).long()
    row_pattern_dataset = RowPatternDataset(address_qubits=address_bits,qc=qc,dataset_qubits=dataset_bits,train=train)
    row_pattern_model = RowdetectModelB(qc=qc,ancilla_bits=ancillas,dataset_qubits=dataset_bits, weights=weights_bits,output=output)
    optimizer = QOptimizer(qc=qc,dataset_qubits=dataset_bits,output=output,data=row_pattern_dataset,model=row_pattern_model,allqubits=allqubits)
    row_pattern_dataset.encode()
    row_pattern_model.forward()
    optimizer.optimize(iter=iter)
    qc.measure(weights_bits, weights_bits)
    aer_sim = Aer.get_backend('aer_simulator')
    aer_sim.set_options(precision='single')
    transpiled_qc = transpile(qc, aer_sim)
    job = execute(transpiled_qc, aer_sim, shots=1000000)
    counts = job.result().get_counts()
    print(counts)

start = time.time()  
Optimize_B(6)
print(time.time() - start)
