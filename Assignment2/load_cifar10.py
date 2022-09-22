import pickle
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data 

dir = r'C:/Users/豹豹/OneDrive - 中山大学/大三下/机器学习与数据挖掘/Assignment2/data'

def load_data():
    X_train = []
    Y_train = []
    for i in range(1, 6):
        with open(dir + r'/data_batch_' + str(i), 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            X_train.append(dict[b'data'])
            Y_train += dict[b'labels']
    X_train = np.concatenate(X_train, axis=0)
    with open(dir + r'/test_batch', 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        X_test = dict[b'data']
        Y_test = dict[b'labels']

    X_train = np.array(X_train, dtype=np.float32).T
    X_test = np.array(X_test, dtype=np.float32).T
    Y_train = np.array(Y_train, dtype=np.int32)
    Y_test = np.array(Y_test, dtype=np.int32)

    return X_train, Y_train, X_test, Y_test

X_train, Y_train, X_test, Y_test = load_data()
print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

# convert np array to torch tensor, Y to long tensor, then to one-hot
def np_to_tensor(X_train, Y_train, X_test, Y_test):
    X_train = torch.from_numpy(X_train)
    Y_train = torch.from_numpy(Y_train).long()
    X_test = torch.from_numpy(X_test)
    Y_test = torch.from_numpy(Y_test).long()
    # Y_train = torch.zeros(Y_train.shape[0], 10).scatter_(1, Y_train.view(-1, 1), 1)
    # Y_test = torch.zeros(Y_test.shape[0], 10).scatter_(1, Y_test.view(-1, 1), 1)
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
    return X_train, Y_train, X_test, Y_test

