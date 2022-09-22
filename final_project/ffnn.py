import pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
import pandas as pd

# 依次运行preprocess.py，(只用跑一次)
# split.py，(只用跑一次)
# tf_and_tf_idf.py
# word2vec.py
# train.py

# parameters
# feature = 'tf' or 'tf-idf' or 'word2vec'
feature = 'word2vec'
epochs = 2000
#可以调小省时间

# torch.manual_seed(1)

# Define loss function.
criterion = nn.BCELoss()
# criterion = nn.CrossEntropyLoss()

class FFNN(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size):
        super(FFNN, self).__init__()
        # 层数可以调, 网络结构可以调
        self.fc1 = nn.Linear(input_size, hidden_size_1)
        self.relu_1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.relu_2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size_2, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu_1(out)
        out = self.fc2(out)
        out = self.relu_2(out)
        out = self.fc3(out)
        return torch.squeeze(torch.sigmoid(out))

    def predict(self, x):
        self.eval()
        torch.no_grad()
        return self.forward(x)

    def evaluate(self, output, y):
        output = torch.round(output)
        correct = (output == y).float()
        return correct.sum() / len(correct)

    # evaluate with precision and recall and f score
    def evaluate_f(self, output, y):
        output = torch.round(output)
        true_positive = (output * y).sum()
        precision = true_positive / output.sum()
        recall = true_positive / y.sum()
        f_score = 2 * precision * recall / (precision + recall)
        return precision, recall, f_score

    def train_model(self, x, y, x_test, y_test, epochs):
        self.train()
        epoch_list = []
        train_loss = []
        test_loss = []
        train_acc = []
        test_acc = []
        for epoch in range(epochs):
            output = self.forward(x)
            loss = criterion(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % 20 == 0:
                epoch_list.append(epoch)
                accuracy = self.evaluate(output, y)
                train_loss.append(loss.item())
                train_acc.append(accuracy.item())
                print(
                    f'Epoch: {epoch}, Loss: {loss.item().__round__(4)}, Accuracy: {accuracy.item().__round__(4)}')

                output = self.forward(x_test)
                loss = criterion(output, y_test)
                accuracy = self.evaluate(output, y_test)
                test_loss.append(loss.item())
                test_acc.append(accuracy.item())

                print(
                    f'Test Set Loss: {loss.item().__round__(4)}, Accuracy: {accuracy.item().__round__(4)}')
        df = pd.DataFrame(list(zip(epoch_list, train_loss, train_acc, test_loss, test_acc)), columns =['epoch_list', 'train_loss', 'train_acc', 'test_loss', 'test_acc'])
        df.to_excel("training_log21.xlsx")
        return train_loss, train_acc, test_loss, test_acc, epoch_list

    # save and load checkpoint
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        return self


with open(feature+'.bin', 'rb') as f:
    X_train = pickle.load(f)
    X_test = pickle.load(f)
    X_valid = pickle.load(f)

with open("Y_splitted.npy", 'rb') as f:
    Y_train = np.load(f, allow_pickle=True)
    Y_test = np.load(f, allow_pickle=True)
    Y_valid = np.load(f, allow_pickle=True)

print("finished loading")

# Convert X datasets to tensors.
X_train = torch.tensor(X_train).to(torch.float32)
X_test = torch.tensor(X_test).to(torch.float32)
X_valid = torch.tensor(X_valid).to(torch.float32)

# Convert Y datasets to tensors.
Y_train = torch.tensor(Y_train).to(torch.float32)
Y_test = torch.tensor(Y_test).to(torch.float32)
Y_valid = torch.tensor(Y_valid).to(torch.float32)

# Dimensions of each layer and num of epochs.
# 这个是隐藏层的结构, 可以调, hidden_dim1<=word to vec length, hidden_dim2<=hidden_dim1
input_dim = X_train.shape[1]
hidden_dim_1 = 100
hidden_dim_2 = 100
output_dim = 1

# Define feed forward neural network.
model = FFNN(input_dim, hidden_dim_1, hidden_dim_2, output_dim)
print("created model")

# Define as optimizer Adam.
optimizer = optim.Adam(model.parameters(), lr=0.1e-3, weight_decay=1e-3)

print(feature)
