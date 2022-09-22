import pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
from ffnn import *

model.load('word2vec20.pth')

output = model.predict(X_test)
print("test set:", model.evaluate(output, Y_test).item().__round__(4), end=" ")

precision, recall, f_score = model.evaluate_f(output, Y_test)
print("precision =", precision.item().__round__(4), end=" ")
print("recall =", recall.item().__round__(4), end=" ")
print("f_score =", f_score.item().__round__(4))

output = model.predict(X_valid)
print("valid set:", model.evaluate(output, Y_valid).item().__round__(4), end=" ")

precision, recall, f_score = model.evaluate_f(output, Y_valid)
print("precision =", precision.item().__round__(4), end=" ")
print("recall =", recall.item().__round__(4), end=" ")
print("f_score =", f_score.item().__round__(4))

