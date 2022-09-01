from config import QuantumConfig
from data import QuantumDataSet
from model import QuantumModel
from optimizer import GroverOptimizer, QuantumOptimizer
from qiskit import QuantumCircuit

qc = QuantumCircuit(15, 15)
qc.mcx(control_qubits=[i for i in range(14)], target_qubit=14)
mpl = qc.draw('mpl')
mpl.savefig('mnist.jpg')