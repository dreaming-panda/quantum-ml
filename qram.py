from ast import Assert
from cProfile import label
from ntpath import join
from random import shuffle
from tkinter.ttk import LabeledScale
from config import QuantumConfig
from data import QuantumDataSet
from model import QuantumModel
from optimizer import GroverOptimizer, QuantumOptimizer
from qiskit import QuantumCircuit
import torch
import numpy as np
from qiskit.quantum_info.operators import Operator
from qiskit import IBMQ, Aer, assemble, transpile

class QRAMDataSet:
    def __init__(self, address_qubits : list, dataset_qubits: list, qc : QuantumCircuit) -> None:
        self.address_qubits = address_qubits
        self.dataset_qubits = dataset_qubits
        self.qc = qc
    def encode(self):
        pass
    def decode(self):
        pass
class QueenDataset(QRAMDataSet):
    def __init__(self, address_qubits : list, dataset_qubits: list, qc: QuantumCircuit, train : torch.Tensor) -> None:
        super().__init__(address_qubits=address_qubits,dataset_qubits=dataset_qubits,qc=qc)
        self.data = torch.load('./queen_dataset/queen_full.pt')
        self.label = torch.load('./queen_dataset/queen_labels_full.pt')
        self.bitstring = []
        self.matrix = torch.zeros((self.data.shape[0] * 4, self.data.shape[0]*4)).long()
        self.inverse_matrix = torch.zeros((self.data.shape[0] * 4, self.data.shape[0]*4)).long()
        self.train = train
        self.operator = None
        self.inverse_operator = None
        self.build_bitstring()
        self.build_operator()
        print(self.bitstring)
    def build_bitstring(self):
        for i in range(self.data.shape[0]):
            sample = self.data[i]
            x = sample[0][0].item() * 1 +  sample[0][1].item() * 2 + sample[0][2].item() * 4 + sample[1][0].item() * 8 + sample[1][1].item() * 16 + sample[1][2].item() * 32 + sample[2][0].item() * 64 \
                + sample[2][1].item() * 128 + sample[2][2].item() * 256 + self.label[i].item() * 512 + self.train[i].item() * 1024
            self.bitstring.append(x)
        Assert(len(self.bitstring) == self.data.shape[0]), 'invalid dataset'
    def build_operator(self):
        for i in range(self.data.shape[0] * 4):
            if i not in self.bitstring:
                self.bitstring.append(i)
        Assert(len(self.bitstring) == self.data.shape[0] * 4), 'invalid dataset'
        for i in range(len(self.bitstring)):
            self.matrix[self.bitstring[i]][i] = 1
        self.operator = Operator(self.matrix.numpy())
        for i in range(len(self.bitstring)):
            self.inverse_matrix[i][self.bitstring[i]] = 1
        self.inverse_operator = Operator(self.inverse_matrix.numpy())
    def encode(self):
        self.qc.barrier()
        for x in self.address_qubits:
            self.qc.h(x)
        self.qc.barrier()
        self.qc.unitary(self.operator, self.dataset_qubits, label='encoder')
        self.qc.barrier()
    def decode(self):
        self.qc.barrier()
        self.qc.unitary(self.inverse_operator, self.dataset_qubits, label='decoder')
        self.qc.barrier()
        for x in self.address_qubits:
            self.qc.h(x)
        self.qc.barrier()
class RowPatternDataset(QRAMDataSet):
    def __init__(self, address_qubits : list, dataset_qubits: list, qc: QuantumCircuit, train : torch.Tensor) -> None:
        super().__init__(address_qubits=address_qubits,dataset_qubits=dataset_qubits,qc=qc)
        self.data = torch.load('./row_pattern_dataset/row_pattern_full.pt')
        self.label = torch.load('./row_pattern_dataset/row_pattern_labels_full.pt')
        self.bitstring = []
        self.matrix = torch.zeros((self.data.shape[0] * 4, self.data.shape[0]*4)).long()
        self.inverse_matrix = torch.zeros((self.data.shape[0] * 4, self.data.shape[0]*4)).long()
        self.train = train
        self.operator = None
        self.inverse_operator = None
        self.build_bitstring()
        self.build_operator()
        print(self.bitstring)
    def build_bitstring(self):
        for i in range(self.data.shape[0]):
            sample = self.data[i]
            x = sample[0][0].item() * 1 +  sample[0][1].item() * 2 + sample[0][2].item() * 4 + sample[1][0].item() * 8 + sample[1][1].item() * 16 + sample[1][2].item() * 32 + sample[2][0].item() * 64 \
                + sample[2][1].item() * 128 + sample[2][2].item() * 256 + self.label[i].item() * 512 + self.train[i].item() * 1024
            self.bitstring.append(x)
        Assert(len(self.bitstring) == self.data.shape[0]), 'invalid dataset'
    def build_operator(self):
        for i in range(self.data.shape[0] * 4):
            if i not in self.bitstring:
                self.bitstring.append(i)
        Assert(len(self.bitstring) == self.data.shape[0] * 4), 'invalid dataset'
        for i in range(len(self.bitstring)):
            self.matrix[self.bitstring[i]][i] = 1
        self.operator = Operator(self.matrix.numpy())
        for i in range(len(self.bitstring)):
            self.inverse_matrix[i][self.bitstring[i]] = 1
        self.inverse_operator = Operator(self.inverse_matrix.numpy())
    def encode(self):
        self.qc.barrier()
        for x in self.address_qubits:
            self.qc.h(x)
        self.qc.barrier()
        self.qc.unitary(self.operator, self.dataset_qubits, label='encoder')
        self.qc.barrier()
    def decode(self):
        self.qc.barrier()
        self.qc.unitary(self.inverse_operator, self.dataset_qubits, label='decoder')
        self.qc.barrier()
        for x in self.address_qubits:
            self.qc.h(x)
        self.qc.barrier()
if __name__ == '__main__':
    train = torch.Tensor([1 for _ in range(512)]).long()
    qc = QuantumCircuit(12, 12)
    dataset_qubits = list(range(11))
    address_qubits = list(range(9))
    qram = QueenDataset(address_qubits=address_qubits,dataset_qubits=dataset_qubits, qc=qc, train=train)
    qram.encode()
    qc.measure_all()
    mpl = qc.draw('mpl')
    mpl.savefig('Queen.jpg')
    aer_sim = Aer.get_backend('aer_simulator')
    #aer_sim.set_options(device='GPU')
    transpiled_grover_circuit = transpile(qc, aer_sim)
    qobj = assemble(transpiled_grover_circuit)
    results = aer_sim.run(qobj).result()
    counts = results.get_counts()
    print(counts)

        
        
    



