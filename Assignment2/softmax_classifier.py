from load_cifar10 import load_data
import numpy as np
import matplotlib.pyplot as plt

# read cifar 10 from C:\Users\豹豹\OneDrive - 中山大学\大三下\机器学习与数据挖掘\Assignment2\data folder
X_train, Y_train, X_test, Y_test = load_data()
print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

class SoftmaxClassifier:
    def __init__(self):
        # Random initialization of W
        self.W = np.random.randn(10, 3072) * 0.001

    def train(self, x, y, lr=1e-6, reg=1e-3, num_iters=2000, batch_size=200):
        print(x.shape)
        train_losses = []
        test_losses = []
        test_accuracies = []
        train_accuracies = []
        for it in range(num_iters):
            # get mini-batch
            indices = np.random.choice(x.shape[1], batch_size, replace=True)
            x_batch = x[:, indices]
            y_batch = y[indices]

            # Calculate loss and gradient for the iteration
            loss, grad = self.cross_entropy_loss(x_batch, y_batch, reg)
            # train_losses.append(loss)

            # Update W
            self.W -= lr * grad

            if it % 50 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))
                train_losses.append(loss)
                # print accuracy
                y_pred = self.predict(x_batch)
                acc = np.mean(y_pred == y_batch)
                print('accuracy: %f' % acc)
                train_accuracies.append(acc)
                # Calculate loss and accuracy for test
                y_pred = self.predict(X_test)
                test_loss = self.cross_entropy_loss(X_test, Y_test, reg)[0]
                test_losses.append(test_loss)
                acc = np.mean(y_pred == Y_test)
                test_accuracies.append(acc)
                print('test accuracy: %f' % acc)
        return train_losses, test_losses, train_accuracies, test_accuracies

    def predict(self, x):
        scores = self.W.dot(x)
        scores -= np.max(scores, axis=0)
        scores = np.exp(scores)
        scores /= np.sum(scores, axis=0)
        y_pred = np.argmax(scores, axis=0)
        return y_pred

    def cross_entropy_loss(self, x, y, reg):
        np.seterr(divide = 'ignore')
        # Calculate the scores with softmax
        scores = self.W.dot(x)
        scores -= np.max(scores, axis=0)
        scores = np.exp(scores)
        scores /= np.sum(scores, axis=0)

        # Calculate the loss
        # print(scores.shape, y.shape)
        loss = -np.sum(np.log(scores[y, np.arange(len(y))]))
        loss /= x.shape[1]
        # loss += reg * np.sum(self.W * self.W)

        # Calculate the gradient
        scores[y, range(scores.shape[1])] -= 1
        grad = scores.dot(x.T)
        grad /= x.shape[1]
        # grad += 2 * reg * self.W

        return loss, grad

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

# Test the model
model = SoftmaxClassifier()
train_losses, test_losses, train_accuracies, test_accuracies = model.train(X_train, Y_train)
plot_loss(train_losses, test_losses, train_accuracies, test_accuracies)
y_pred = model.predict(X_test)
print('Accuracy: %f' % (np.mean(y_pred == Y_test)))
