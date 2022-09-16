from turtle import title
from qram import QRAMDataSet, RowPatternDataset, MNISTDataset
from qmodel import QModel, RowdetectModel, RowdetectModelA, RowdetectModelB, MNISTMODEL
from qoptimizer import QOptimizer
from qiskit import QuantumCircuit
import matplotlib.pyplot as plt
import torch
from qiskit.visualization import plot_histogram,plot_state_city
from qiskit import IBMQ, Aer, assemble, transpile, execute
import time
import numpy as np
def verification():
    torch.set_printoptions (profile="full") 
    W = [i for i in range(1024)]
    Weights = []
    for w in W:
        w = list(bin(w)[2:].rjust(10, '0'))
        for i in range(len(w)):
            w[i] =  int((w[i] == '1'))
        Weights.append(w)
    train_data = torch.load('./mnist_dataset/train_data6_7.pt')
    train_label = torch.load('./mnist_dataset/train_label6_7.pt')
    test_data = torch.load('./mnist_dataset/test_data6_7.pt')
    test_label = torch.load('./mnist_dataset/test_label6_7.pt')
    TRAIN_DATA = []
    TEST_DATA = []
    for i in range(train_data.shape[0]):
        x = train_data[i]
        x = list(bin(x)[2:].rjust(9, '0'))
        for j in range(len(x)):
            x[j] =  int((x[j] == '1'))
        TRAIN_DATA.append(x)
    for i in range(test_data.shape[0]):
        x = test_data[i]
        x = list(bin(x)[2:].rjust(9, '0'))
        for j in range(len(x)):
            x[j] =  int((x[j] == '1'))
        TEST_DATA.append(x)
    ACC = []
    for w in Weights:
        train_accuracy = 0
        test_accuracy = 0
        for i in range(len(TRAIN_DATA)):
            predict = 0
            for j in range(len(w) - 1):
                predict += w[j] * TRAIN_DATA[i][j]
            if w[9] == 0:
                if predict >= 1 and train_label[i] == 0:
                    train_accuracy += 1
                if predict == 0 and train_label[i] == 1:
                    train_accuracy += 1
            if w[9] == 1:
                if predict >= 1 and train_label[i] == 1:
                    train_accuracy += 1
                if predict == 0 and train_label[i] == 0:
                    train_accuracy += 1
        for i in range(len(TEST_DATA)):
            predict = 0
            for j in range(len(w) - 1):
                predict += w[j] * TEST_DATA[i][j]
            if w[9] == 0:
                if predict >= 1 and test_label[i] == 0:
                    test_accuracy += 1
                if predict == 0 and test_label[i] == 1:
                    test_accuracy += 1
            if w[9] == 1:
                if predict >= 1 and test_label[i] == 1:
                    test_accuracy += 1
                if predict == 0 and test_label[i] == 0:
                    test_accuracy += 1
        ACC.append(train_accuracy)
        print("{}:  {:.3f}, {:.3f} ".format(w,train_accuracy,test_accuracy))
    acc = torch.Tensor(ACC)
    print(acc.mean())
    print(acc.sort(descending=True))
    acc = acc.sort(descending=True)[0]
    print(acc)
    presum = torch.zeros_like(acc)
    for i in range(acc.shape[0]):
        presum[i] = acc[0:i].sum()
    presum = presum / acc.sum()
    plt.title('Accuracy Distribution of WEIGHTS')
    plt.xlabel('order of accuracy')
    plt.ylabel('prefix sum')
    plt.plot(presum)
    plt.savefig('weights_distribution.jpg')
    print(presum)

def Optimize():
    weights_bits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    address_bits = [10, 11, 12, 13, 14, 15]
    dataset_bits = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    ancillas = [21, 22, 23, 24, 25, 26, 27, 28, 29]
    output = 30
    allqubits = list(range(31))
    qc = QuantumCircuit(31, 1)
    train = torch.Tensor([1 for _ in range(64)]).long()
    train[62:64] = 0
    mnist_dataset = MNISTDataset(address_qubits=address_bits,qc=qc,dataset_qubits=dataset_bits,train=train)
    mnist_model = MNISTMODEL(qc=qc,ancilla_bits=ancillas,dataset_qubits=dataset_bits, weights=weights_bits,output=output)
    optimizer = QOptimizer(qc=qc,dataset_qubits=dataset_bits,output=output,data=mnist_dataset,model=mnist_model,allqubits=allqubits)
    mnist_dataset.encode()
    # mnist_model.forward()
    # mnist_model.de_forward()
    mnist_dataset.decode()
    #optimizer.optimize(iter=3)
    # qc.cx(dataset_bits[-2], output)
    # qc.measure(output, 0)
    qc.measure_all()
    aer_sim = Aer.get_backend('aer_simulator')
    job = execute(qc, aer_sim, shots=1000000)
    counts = job.result().get_counts()
    print(counts)
start = time.time()
Optimize()
print(time.time() - start)



