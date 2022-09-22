# train classifier on cifar10 using LeNet

from load_cifar10 import load_data
from load_cifar10 import np_to_tensor
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt

# read cifar 10 from C:\Users\豹豹\OneDrive - 中山大学\大三下\机器学习与数据挖掘\Assignment2\data folder
X_train, Y_train, X_test, Y_test = load_data()
# convert np array to torch tensor
X_train, Y_train, X_test, Y_test = np_to_tensor(X_train, Y_train, X_test, Y_test)
# convert to 4d tensor for Conv2d
# torch.Size([3072, 50000])
X_train = torch.transpose(X_train, 0, 1)
X_train = X_train.view(X_train.shape[0], 3, 32, 32)
X_test = torch.transpose(X_test, 0, 1)
X_test = X_test.view(X_test.shape[0], 3, 32, 32)

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # self.fc1 = nn.Linear(6 *  14 * 14, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        # print(x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        # x = x.view(-1, 6 * 14 * 14)
        # print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def train(self, X_train, Y_train, X_test, Y_test, epochs=4, batch_size=64, learning_rate=0.001):
        train_losses = []
        test_losses = []
        train_accuracies = []
        test_accuracies = []
        # define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        # sgd momentum optimizer
        optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum=0.9)
        # optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        # train the model
        for epoch in range(epochs):
            self.losses = []
            total = 0
            correct = 0
            for i in range(0, X_train.shape[0], batch_size):
                # get the inputs
                inputs = X_train[i:i+batch_size]
                labels = Y_train[i:i+batch_size]
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                self.losses.append(loss.item())
            print('Epoch {}: loss = {}'.format(epoch, np.mean(self.losses)))
            # print accuracy
            print('Accuracy of the network on the train images: %d %%' % (100 * correct / total))
            # test the model
            train_losses.append(np.mean(self.losses))
            train_accuracies.append(100 * correct / total)
            # test the model
            test_accuracy, test_loss = self.test(X_test, Y_test, batch_size)
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)
        return train_losses, train_accuracies, test_losses, test_accuracies
    
    def test(self, X_test, Y_test, batch_size=64):
        # test the model
        correct = 0
        total = 0
        with torch.no_grad():
            losses = []
            for i in range(0, X_test.shape[0], batch_size):
                # get the inputs
                inputs = X_test[i:i+batch_size]
                labels = Y_test[i:i+batch_size]
                # forward
                outputs = self(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                loss = F.cross_entropy(outputs, labels)
                losses.append(loss.item())
        print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
        print('Average loss of the network on the 10000 test images: {}'.format(np.mean(losses)))
        return 100 * correct / total, np.mean(losses)
    
    def predict(self, X_test, batch_size=64):
        # test the model
        correct = 0
        total = 0
        with torch.no_grad():
            for i in range(0, X_test.shape[0], batch_size):
                # get the inputs
                inputs = X_test[i:i+batch_size]
                # forward
                outputs = self(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += inputs.shape[0]
                correct += (predicted == inputs).sum().item()
        print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
    
    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        self.load_state_dict(torch.load(path))
    
    def __repr__(self):
        return 'LeNet'

def plot_loss(train_losses, test_losses, train_accuracies, test_accuracies):
    plt.figure(1)
    plt.plot(train_losses, label='train loss')
    plt.plot(test_losses, label='test loss')
    plt.legend()
    plt.figure(2)
    plt.plot(train_accuracies, label='train accuracy')
    plt.plot(test_accuracies, label='test accuracy')
    plt.legend()
    plt.show()   

if __name__ == '__main__':
    # define the model
    model = LeNet()
    # train the model
    train_losses, train_accuracies, test_losses, test_accuracies = model.train(X_train, Y_train, X_test, Y_test)
    plot_loss(train_losses, test_losses, train_accuracies, test_accuracies)


