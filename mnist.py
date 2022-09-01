from ast import Assert
from cProfile import label
from ntpath import join
from tkinter.ttk import LabeledScale
from config import QuantumConfig
from data import QuantumDataSet
from model import QuantumModel
from optimizer import GroverOptimizer, QuantumOptimizer
from qiskit import QuantumCircuit
import torch
class MNISTDataSet(QuantumDataSet):
    def __init__(self, features: list[list], label: list, mark: list, qc: QuantumCircuit, targets : list) -> None:
        super().__init__(features, label, qc)
        Assert(len(targets) == 2) , "invalid dataset"
        Assert(targets[0] < targets[1]), "invalid dataset"
        self.targets = targets
        self.train_data : torch.Tensor = torch.load('./mnist_dataset/train_data{}_{}.pt'.format(targets[0], targets[1]))
        self.train_label : torch.Tensor = torch.load('./mnist_dataset/train_label{}_{}.pt'.format(targets[0], targets[1]))
        self.mark = mark
    def encode(self) -> None:
        for x in self.features[0]:
            self.qc.h(x)
        self.qc.h(self.label[0])
        self.qc.barrier()
        control_bits = [i for i in self.features[0]]
        control_bits.append(self.label[0])
        for i in range(self.train_data.shape[0]):
            feature = self.train_data[i].item()
            label = str(self.train_label[i].item())
            feature = list(bin(feature)[2:].rjust(9, '0'))
            for j in range(len(feature)):
                if feature[j] == '0':
                    self.qc.x(self.features[0][j])
            if label == '0':
                self.qc.x(self.label[0])

            self.qc.barrier()

            self.qc.mcx(control_qubits=control_bits, target_qubit=self.mark[0])

            self.qc.barrier()

            if label == '0':
                self.qc.x(self.label[0])
            
            for j in range(len(feature)):
                if feature[j] == '0':
                    self.qc.x(self.features[0][j])
            self.qc.barrier()

    def decode(self) -> None:
        control_bits = [i for i in self.features[0]]
        control_bits.append(self.label[0])
        for k in range(self.train_data.shape[0]):
            i = self.train_data.shape[0] - 1 - k
            feature = self.train_data[i].item()
            label = str(self.train_label[i].item())
            feature = list(bin(feature)[2:].rjust(9, '0'))
            for j in range(len(feature)):
                if feature[j] == '0':
                    self.qc.x(self.features[0][j])
            if label == '0':
                self.qc.x(self.label[0])

            self.qc.barrier()

            self.qc.mcx(control_qubits=control_bits, target_qubit=self.mark[0])

            self.qc.barrier()

            if label == '0':
                self.qc.x(self.label[0])
            
            for j in range(len(feature)):
                if feature[j] == '0':
                    self.qc.x(self.features[0][j])
            self.qc.barrier()

        self.qc.barrier()
        for x in self.features[0]:
            self.qc.h(x)
        self.qc.h(self.label[0])
        self.qc.barrier()
        





if __name__ == '__main__':
    features = [[0, 1, 2, 3, 4, 5, 6, 7, 8]]
    labels = [9]
    mark = [10]

    qc = QuantumCircuit(11, 11)

    mnist = MNISTDataSet(features=features, label=labels, mark=mark,qc=qc,targets=[0, 1])

    mnist.encode()

    mpl = qc.draw('mpl')
    mpl.savefig('mnist01.jpg')
   