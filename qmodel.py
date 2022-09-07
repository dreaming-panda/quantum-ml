from ast import Assert
from cProfile import label
from ntpath import join
from random import shuffle
from tkinter.ttk import LabeledScale
from turtle import forward
from config import QuantumConfig
from data import QuantumDataSet
from model import QuantumModel
from optimizer import GroverOptimizer, QuantumOptimizer
from qiskit import QuantumCircuit
import torch
import numpy as np
from qiskit.quantum_info.operators import Operator
from qiskit import IBMQ, Aer, assemble, transpile
class QModel:
    def __init__(self, qc : QuantumCircuit, ancilla_bits: list, features: list, output) -> None:
        self.qc = qc
        self.ancilla_bits = ancilla_bits
        self.features = features
        self.output = output
    def forward(self):
        pass
    def de_forward(self):
        pass
