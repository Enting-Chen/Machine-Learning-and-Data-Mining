# train and test MLP model to classify cifar 10
from load_cifar10 import load_data
from load_cifar10 import np_to_tensor
import numpy as np
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# read cifar 10 using load_data()
X_train, Y_train, X_test, Y_test = load_data()
# convert np array to torch tensor
X_train, Y_train, X_test, Y_test = np_to_tensor(X_train, Y_train, X_test, Y_test)
X_test = torch.transpose(X_test, 0, 1)
X_train = torch.transpose(X_train, 0, 1)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(3072, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def predict(self, x):
        outputs = self(x)
        _, y_pred = torch.max(outputs, 1)
        # print(y_pred)
        return y_pred

    def cross_entropy_loss(self, x, y, reg):
        scores = self.forward(x)
        # print(scores.shape, y.shape)
        loss = self.loss_func(scores, y)
        # loss += reg * (torch.sum(self.fc1.weight ** 2) + torch.sum(self.fc2.weight ** 2) + torch.sum(self.fc3.weight ** 2))
        return loss, loss.grad_fn.next_functions[0][0]

    def train(self, x, y, x_test, y_test, lr=1e-4, reg=1e-3, epochs = 20, batch_size=64):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss_func = nn.CrossEntropyLoss()
        criterion = nn.CrossEntropyLoss()
        train_losses = []
        test_losses = []
        test_accuracies = []
        train_accuracies = []
        # for it in range(num_iters):
        #     # get mini-batch 200*3072 from 50000*3072
        #     # print(x.shape, y.shape)
        #     # dsf
        #     idx = np.random.choice(x.shape[1], batch_size, replace=False)
        #     x_batch = x[:, idx]
        #     # transpose to 3072*200
        #     x_batch = torch.transpose(x_batch, 0, 1)
        #     y_batch = y[idx]
        #     # print(x_batch.shape, y_batch.shape)
            
        #     # Calculate loss and gradient for the iteration
        #     loss, grad = self.cross_entropy_loss(x_batch, y_batch, reg)

        #     # Update W
        #     self.optimizer.zero_grad()
        #     loss.backward()
        #     self.optimizer.step()

        #     # print iteration number, loss, and accuracy
        #     if it % 50 == 0 and it >= 100:
        #         self.train_losses.append(loss.item())
        #         self.train_accuracies.append(self.test(x_batch, y_batch))
        #         self.test_losses.append(self.cross_entropy_loss(x_test, y_test, reg)[0].item())
        #         self.test_accuracies.append(self.test(x_test, y_test))
        #         print('iteration %d / %d: loss %f' % (it, num_iters, loss))
        #         print('accuracy: %f' % self.test(x_batch, y_batch))

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
            train_accuracies.append(correct / total)
            # test the model
            test_accuracy, test_loss = self.test(X_test, Y_test)
            # print(test_accuracy)
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)
        return train_losses, train_accuracies, test_losses, test_accuracies

    def test(self, x, y):
        y_pred = self.predict(x)
        # print(y_pred == y)
        # print(torch.sum(y_pred == y))
        accuracy = torch.sum(y_pred == y).item() / y.shape[0]
        print(accuracy)
        # loss 
        loss = self.loss_func(self(x), y)
        return accuracy, loss.item()

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
    
# train and test MLP model to classify cifar 10
model = MLP()
train_losses, train_accuracies, test_losses, test_accuracies = model.train(X_train, Y_train, X_test, Y_test)
plot_loss(train_losses, test_losses, train_accuracies, test_accuracies)
acc, loss = model.test(X_test, Y_test)
print('Test accuracy: %.2f%%' % (acc * 100))
