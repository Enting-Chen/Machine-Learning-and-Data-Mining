import numpy as np

with open('X_Y.npy', 'rb') as f:
    X = np.load(f, allow_pickle=True)
    Y = np.load(f, allow_pickle=True)

X_train = X[:30000]
Y_train = Y[:30000]
X_test = X[30000:40000]
Y_test = Y[30000:40000]
X_valid = X[40000:]
Y_valid = Y[40000:]

with open('X_splitted.npy', 'wb') as f:
    np.save(f, X_train)
    np.save(f, X_test)
    np.save(f, X_valid)

with open('Y_splitted.npy', 'wb') as f:
    np.save(f, Y_train)
    np.save(f, Y_test)
    np.save(f, Y_valid)