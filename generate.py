from genericpath import exists
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
import os
from tqdm import tqdm
targets = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
if not exists('./mnist_dataset'):
        os.mkdir('mnist_dataset')
for A,B in itertools.product(targets, targets):
    print("{}:{}".format(A, B))
    if A >= B:
        continue
    labels = [A , B]
    transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Resize([3, 3]),
            ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                        transform=transform)
        
    dataset2 = datasets.MNIST('../data', train=False,
                        transform=transform)
    train_index = []
    test_index = []
    for i in range(len(dataset1)):
            if dataset1[i][1] in labels:
                train_index.append(i)
    train_data = torch.cat([dataset1[x][0] for x in train_index], dim=0)
    train_label = [dataset1[x][1] for x in train_index]
    train_label = torch.Tensor(train_label).long()
    train_data = train_data - train_data.mean()
    binary_train_data = train_data.gt(0).long()
    print(train_label.shape)
    print(train_label)

    for i in range(len(dataset2)):
            if dataset2[i][1] in labels:
                test_index.append(i)
    test_data = torch.cat([dataset2[x][0] for x in test_index], dim=0)
    test_label = [dataset2[x][1] for x in test_index]
    test_label = torch.Tensor(test_label).long()
    test_data = test_data - test_data.mean()
    binary_test_data = test_data.gt(0).long()

    print(test_label.shape)
    print(test_label)


    hash_data_train = [(256 * x[0][0].item() + 128 * x[0][1].item() + 64 * x[0][2].item() + 32 * x[1][0].item() + 16 * x[1][1].item() + 8 * x[1][2].item() + 4 * x[2][0].item() + 2 * x[2][1].item() + x[2][2].item())for x in binary_train_data]
    hash_data_test = [(256 * x[0][0].item() + 128 * x[0][1].item() + 64 * x[0][2].item() + 32 * x[1][0].item() + 16 * x[1][1].item() + 8 * x[1][2].item() + 4 * x[2][0].item() + 2 * x[2][1].item() + x[2][2].item())for x in binary_test_data]


    hash_data_train_unique = list(set(hash_data_train))
    hash_data_test_unique = list(set(hash_data_test))

    print(len(hash_data_train_unique))
    print(len(hash_data_test_unique))

    train_info = torch.zeros((2, 512)).long()
    test_info = torch.zeros((2, 512)).long()

    for i in range(len(hash_data_train)):
        if train_label[i] == A:
            train_info[0][hash_data_train[i]] += 1
        elif train_label[i] == B:
            train_info[1][hash_data_train[i]] += 1
    for i in range(len(hash_data_test)):
        if test_label[i] == A:
            test_info[0][hash_data_test[i]] += 1
        elif test_label[i] == B:
            test_info[1][hash_data_test[i]] += 1
    print(train_info[0].sum())
    print(train_info[1].sum())

    print(test_info[0].sum())
    print(test_info[1].sum())

    PATH_TRAIN_LOG = './mnist_dataset/train{}_{}.log'.format(A, B)
    PATH_TEST_LOG = './mnist_dataset/test{}_{}.log'.format(A, B)
    train_log = open(PATH_TRAIN_LOG,'w')
    test_log = open(PATH_TEST_LOG,'w')
    clean_train_data = []
    clean_train_label = []
    clean_test_data = []
    clean_test_label = []
    for i in range(512):
        if (train_info[0][i] + train_info[1][i] > 1e-3) and (train_info[0][i] != train_info[1][i]):
            train_log.write("{:09b}---{} instance :{}, {} instance :{}\n".format((i), A, train_info[0][i], B, train_info[1][i]))
            clean_train_data.append(i)
            if train_info[0][i] > train_info[1][i]:
                clean_train_label.append(0)
            else :
                clean_train_label.append(1)


    for i in range(512):
        if (test_info[0][i] + test_info[1][i] > 1e-3) and (test_info[0][i] != test_info[1][i]):
            test_log.write("{:09b}---{} instance :{}, {} instance :{}\n".format((i), A, test_info[0][i], B, test_info[1][i]))
            clean_test_data.append(i)
            if test_info[0][i] > test_info[1][i]:
                clean_test_label.append(0)
            else :
                clean_test_label.append(1)
    print(len(clean_train_data))
    print(len(clean_train_label))
    print(len(clean_test_data))
    print(len(clean_test_label))

    clean_train_data = torch.Tensor(clean_train_data).long()
    clean_train_label = torch.Tensor(clean_train_label).long()
    clean_test_data = torch.Tensor(clean_test_data).long()
    clean_test_label = torch.Tensor(clean_test_label).long()


    print((clean_train_data.shape))
    print((clean_train_label.shape))
    print((clean_test_data.shape))
    print((clean_test_label.shape))

    
    torch.save(clean_train_data,'./mnist_dataset/train_data{}_{}.pt'.format(A, B))
    torch.save(clean_train_label,'./mnist_dataset/train_label{}_{}.pt'.format(A, B))

    torch.save(clean_test_data,'./mnist_dataset/test_data{}_{}.pt'.format(A, B))
    torch.save(clean_test_label,'./mnist_dataset/test_label{}_{}.pt'.format(A, B))

    accuracy = []

    for i in range(len(hash_data_test)):
        if train_info[0][hash_data_test[i]] != train_info[1][hash_data_test[i]]:
            label_i = test_label[i]
            predict_i = A
            if train_info[0][hash_data_test[i]] < train_info[1][hash_data_test[i]]:
                predict_i = B
            if predict_i == label_i:
                accuracy.append(1)
            else:
                accuracy.append(0)


        else :
            accuracy.append(1)
    print(sum(accuracy)/len(accuracy))

    t_accuracy = []
    for i in range(512):
        if (test_info[0][i] + test_info[1][i] > 1e-3) and (test_info[0][i] != test_info[1][i]):
            if test_info[0][i] > test_info[1][i]:
                if train_info[0][i] > train_info[1][i]:
                    t_accuracy.append(1)
                else:
                    t_accuracy.append(0)
                    print("{:09b}".format(i))
            else :
                if train_info[0][i] >= train_info[1][i]:
                    t_accuracy.append(0)
                    print("{:09b}".format(i))
                else:
                    t_accuracy.append(1)
    print(len(t_accuracy))
    print(sum(t_accuracy)/len(t_accuracy))
