import numpy as np

# read data from mnist_train.csv and mnist_test.csv
train = np.loadtxt('mnist_train.csv', delimiter=',', skiprows=1)
test = np.loadtxt('mnist_test.csv', delimiter=',', skiprows=1)

print('train shape is', train.shape)
print('test shape is', test.shape)

np.save('mnist_train1.npy', train)
np.save('mnist_test1.npy', test)