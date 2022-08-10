import matplotlib.pyplot as plt
import numpy as np
from qiskit import IBMQ, Aer, assemble, transpile
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.providers.ibmq import least_busy
from qiskit.visualization import plot_histogram
from tqdm import tqdm
def verification():
    weight = [[0, 0], [0, 1], [1, 0], [1, 1]]
    data = [[0, 0, 0],[0, 1, 1],[1, 0, 1], [1, 1, 0]]
    accuracy = []
    for w in weight:
        acc = 0
        for d in data:
            model_output = (w[0] * d[0]) ^ (w[1] * d[1])
            if model_output == d[2]:
                acc += 1
        accuracy.append(acc / 4.0)
    print(accuracy)
verification()
record = []
runs = 100
for p in tqdm(range(runs)):
    """
    initialize the basic structual of QC 
    """
    n = 6 # n qubits
    m = 6 # m cbits
    qc = QuantumCircuit(n,m) # create the QC
    """
    the meaning of the qubit
    """
    weights = [0, 1]
    features = [2, 3]
    label = 4
    readout = 5
    allqubits = [0, 1, 2, 3, 4, 5]
    """
    init weight
    """
    for q in weights:
        qc.h(q)
    qc.barrier()
    """
    init sample
    """
    for q in features:
        qc.h(q)
    qc.x(label)
    qc.ccx(features[0], features[1], label)
    qc.x(features[0])
    qc.x(features[1])
    qc.ccx(features[0], features[1], label)
    qc.x(features[0])
    qc.x(features[1])
    qc.barrier()

    

    
    """
    model forward
    """
    # qc.ccx(weights[0], features[0], readout)
    # qc.ccx(weights[1], features[1], readout)
    # qc.barrier()

    for _ in range(p):
            """
            oracle
            """
            qc.ccx(weights[0], features[0], readout)
            qc.ccx(weights[1], features[1], readout)
            qc.barrier()
            qc.h(weights[0])
            qc.ccx(readout, label, weights[0])
            qc.h(weights[0])
            qc.barrier()
            qc.h(weights[0])
            qc.x(readout)
            qc.x(label)
            qc.ccx(readout, label, weights[0])
            qc.x(readout)
            qc.x(label)
            qc.h(weights[0])
            qc.barrier()
            qc.ccx(weights[1], features[1], readout)
            qc.ccx(weights[0], features[0], readout)
            qc.barrier()
            """
            diffusion
            """
            for q in [0, 1, 2, 3]:
                qc.h(q)
            qc.barrier()
            qc.x(label)
            qc.ccx(features[0], features[1], label)
            qc.x(features[0])
            qc.x(features[1])
            qc.ccx(features[0], features[1], label)
            qc.x(features[0])
            qc.x(features[1])
            qc.barrier()
            for q in [0, 1, 2, 3, 4]:   
                qc.x(q)
            qc.barrier()
            qc.h(weights[0])
            qc.mct([1, 2, 3, 4], 0)
            qc.h(weights[0])
            qc.barrier()
            for q in [0, 1, 2, 3, 4]:
                qc.x(q)
            qc.x(features[1])
            qc.x(features[0])
            qc.ccx(features[0], features[1], label)
            qc.x(features[1])
            qc.x(features[0])
            qc.ccx(features[0], features[1], label)
            qc.x(label)
            qc.barrier()
            for q in [0, 1, 2, 3]:   
                qc.h(q)
    if p == 6:
       mpl = qc.draw(output='mpl')
       mpl.savefig('qc_xor_v2.jpg'.format(p))
    qc.measure([weights[0], weights[1]], [weights[0], weights[1]])
    

    """
    execute
            """
    aer_sim = Aer.get_backend('aer_simulator')
    aer_sim.set_options(device='GPU')
    transpiled_qc = transpile(qc, aer_sim)
    qobj = assemble(transpiled_qc)
    results = aer_sim.run(qobj).result()
    counts = results.get_counts()
        #print(counts)
    record.append(counts['000011'])

plt.figure(figsize=(12, 12))  
plt.plot(record)  
plt.ylabel('best weight appear times') 
plt.xlabel('grover_iterations') 
plt.title("xor_result_v2")
plt.savefig('xor_result_v2.jpg')
index = []
for i in range(len(record)):
    if record[i] >= 280:
            index.append(i)   
record.sort(reverse=True)
print(record)
print(index)


