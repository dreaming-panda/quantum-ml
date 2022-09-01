from genericpath import exists
import itertools
from random import seed, shuffle, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
import os
from tqdm import tqdm
bitstring = [i for i in range(512)]
samples = torch.zeros((512, 3, 3))
labels = torch.zeros((512))
r = random
seed(1)
shuffle(bitstring, random=r)
for i in range(len(bitstring)):
    binary_i = list(bin(bitstring[i])[2:].rjust(9 ,'0'))
    
    
    samples[i][0][0] = int(binary_i[0] == '1')
    samples[i][0][1] = int(binary_i[1] == '1')
    samples[i][0][2] = int(binary_i[2] == '1')
    samples[i][1][0] = int(binary_i[3] == '1')
    samples[i][1][1] = int(binary_i[4] == '1')
    samples[i][1][2] = int(binary_i[5] == '1')
    samples[i][2][0] = int(binary_i[6] == '1')
    samples[i][2][1] = int(binary_i[7] == '1')
    samples[i][2][2] = int(binary_i[8] == '1')

samples = samples.long()

def give_label(matrix):
    if matrix[0][0] + matrix[0][1] + matrix[0][2] == 3:
        return 1
    if matrix[1][0] + matrix[1][1] + matrix[1][2] == 3:
        return 1
    if matrix[2][0] + matrix[2][1] + matrix[2][2] == 3:
        return 1
    if matrix[0][0] + matrix[1][0] + matrix[2][0] == 3:
        return 1
    if matrix[0][1] + matrix[1][1] + matrix[2][1] == 3:
        return 1
    if matrix[0][2] + matrix[1][2] + matrix[2][2] == 3:
        return 1
    if matrix[0][0] + matrix[1][1] + matrix[2][2] == 3:
        return 1
    if matrix[2][0] + matrix[1][1] + matrix[0][2] == 3:
        return 1
    return 0
for i in range(len(bitstring)):
    labels[i] = give_label(samples[i])

labels = labels.long()
print(labels.sum())
if not exists('./queen_dataset'):
    os.mkdir('queen_dataset')
torch.save(samples, './queen_dataset/queen_full.pt')   
torch.save(labels, './queen_dataset/queen_labels_full.pt')

torch.save(samples[0:400], './queen_dataset/queen_train.pt')   
torch.save(labels[0:400], './queen_dataset/queen_labels_train.pt')

torch.save(samples[400:512], './queen_dataset/queen_test.pt')   
torch.save(labels[400:512], './queen_dataset/queen_labels_test.pt')









