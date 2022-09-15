from genericpath import exists
import itertools
from random import random, seed, shuffle
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
r = random
seed(1)
shuffle(bitstring, random=r)
samples = torch.zeros((512, 3, 3))
compress_sample = torch.zeros((512, 3))
labels = torch.zeros((512))
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

    compress_sample[i][0] = int((samples[i][0][0] * samples[i][0][1] * samples[i][0][2] + samples[i][1][0] * samples[i][1][1] * samples[i][1][2] + samples[i][2][0] * samples[i][2][1] * samples[i][2][2]) >= 1)
    compress_sample[i][1] = int((samples[i][0][0] * samples[i][1][0] * samples[i][2][0] + samples[i][0][1] * samples[i][1][1] * samples[i][2][1] + samples[i][0][2] * samples[i][1][2] * samples[i][2][2]) >= 1)
    compress_sample[i][2] = int((samples[i][0][0] * samples[i][1][1] * samples[i][2][2] + samples[i][0][2] * samples[i][1][1] * samples[i][2][0]) >= 1)
samples = samples.long()
compress_sample = compress_sample.long()
def give_label(matrix):
    x = 0
    if matrix[0][0] + matrix[0][1] + matrix[0][2] == 3:
        return 1
    if matrix[1][0] + matrix[1][1] + matrix[1][2] == 3:
        return 1
    if matrix[2][0] + matrix[2][1] + matrix[2][2] == 3:
        return 1
    return 0
for i in range(len(bitstring)):
    labels[i] = give_label(samples[i])

labels = labels.long()
print(labels.sum())
if not exists('./strong_row_pattern_dataset'):
    os.mkdir('strong_row_pattern_dataset')
torch.save(compress_sample, './strong_row_pattern_dataset/strong_row_pattern_full.pt')   
torch.save(labels, './strong_row_pattern_dataset/strong_row_pattern_labels_full.pt')   

torch.save(compress_sample[0:400], './strong_row_pattern_dataset/strong_row_pattern_train.pt')   
torch.save(labels[0:400], './strong_row_pattern_dataset/strong_row_pattern_labels_train.pt')   

torch.save(compress_sample[400 : 512], './strong_row_pattern_dataset/strong_row_pattern_test.pt')   
torch.save(labels[400 : 512], './strong_row_pattern_dataset/strong_row_pattern_labels_test.pt')

W = [[0,0,0], [0,0,1], [0,1,0], [0,1,1], [1,0,0], [1,0,1], [1,1,0], [1,1,1]]
for w in W:
    accuracy = 0
    for i in range(512):
        predict = int((w[0] * compress_sample[i][0] + w[1] * compress_sample[i][1] + w[2] * compress_sample[i][2]) >= 1)
        if predict == labels[i]:
            accuracy += 1
    print("{}{}{}:  {}".format(w[0],w[1],w[2],accuracy))








