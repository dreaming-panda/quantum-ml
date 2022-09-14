from turtle import forward
from qiskit import QuantumCircuit
import torch
import numpy as np
from qiskit.quantum_info.operators import Operator
from qiskit import IBMQ, Aer, assemble, transpile
class QModel:
    def __init__(self, qc : QuantumCircuit, ancilla_bits: list, dataset_qubits : list, weights: list, output) -> None:
        self.qc = qc
        self.ancilla_bits = ancilla_bits
        self.features = dataset_qubits[:-2]
        self.output = output
        self.weights = weights
    def forward(self):
        pass
    def de_forward(self):
        pass
class RowdetectModel(QModel):
    def forward(self):
        for x in self.weights:
            self.qc.h(x)
        self.qc.barrier()
        self.qc.mcx(control_qubits=[self.features[0], self.features[1], self.features[2]], target_qubit=self.ancilla_bits[0])
        self.qc.mcx(control_qubits=[self.features[3], self.features[4], self.features[5]], target_qubit=self.ancilla_bits[1])
        self.qc.mcx(control_qubits=[self.features[6], self.features[7], self.features[8]], target_qubit=self.ancilla_bits[2])
        self.qc.barrier()
        self.qc.x(self.ancilla_bits[0])
        self.qc.x(self.ancilla_bits[1])
        self.qc.x(self.ancilla_bits[2])
        self.qc.x(self.ancilla_bits[3])
        self.qc.barrier()
        self.qc.mcx(control_qubits=[self.ancilla_bits[0], self.ancilla_bits[1], self.ancilla_bits[2]], target_qubit=self.ancilla_bits[3])
        self.qc.barrier()
        self.qc.x(self.ancilla_bits[0])
        self.qc.x(self.ancilla_bits[1])
        self.qc.x(self.ancilla_bits[2])
        self.qc.barrier()
        self.qc.mcx(control_qubits=[self.weights[0], self.ancilla_bits[3]], target_qubit=self.output)
        self.qc.barrier()

    def de_forward(self):
        self.qc.mcx(control_qubits=[self.weights[0], self.ancilla_bits[3]], target_qubit=self.output)
        self.qc.barrier()
        self.qc.x(self.ancilla_bits[0])
        self.qc.x(self.ancilla_bits[1])
        self.qc.x(self.ancilla_bits[2])
        self.qc.barrier()
        self.qc.mcx(control_qubits=[self.ancilla_bits[0], self.ancilla_bits[1], self.ancilla_bits[2]], target_qubit=self.ancilla_bits[3])
        self.qc.barrier()
        self.qc.x(self.ancilla_bits[0])
        self.qc.x(self.ancilla_bits[1])
        self.qc.x(self.ancilla_bits[2])
        self.qc.x(self.ancilla_bits[3])
        self.qc.barrier()
        self.qc.mcx(control_qubits=[self.features[6], self.features[7], self.features[8]], target_qubit=self.ancilla_bits[2])
        self.qc.mcx(control_qubits=[self.features[3], self.features[4], self.features[5]], target_qubit=self.ancilla_bits[1])
        self.qc.mcx(control_qubits=[self.features[0], self.features[1], self.features[2]], target_qubit=self.ancilla_bits[0])
        self.qc.barrier()
        for x in self.weights:
            self.qc.h(x)
        self.qc.barrier()

        

    
