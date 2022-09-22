from ffnn import *
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from os import environ


def train_model(x_train, y_train, x_test, y_test, x_valid, y_valid, epochs):
    train_loss, train_acc, test_loss, test_acc, epoch_list = model.train_model(
        x_train, y_train, x_test, y_test, epochs)
    print("fitted model")
    print(model.evaluate(model.forward(x_test), y_test))
    return train_loss, train_acc, test_loss, test_acc, epoch_list


train_loss, train_acc, test_loss, test_acc, epoch_list = train_model(
    X_train, Y_train, X_test, Y_test, X_valid, Y_valid, epochs)

model.save(feature + '21.pth')
# 有用的话改一下文件名，accuracy > 86 or 88, 记下参数和accuracy （最后一次输出）

def suppress_qt_warnings():
    environ["QT_DEVICE_PIXEL_RATIO"] = "0"
    environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    environ["QT_SCALE_FACTOR"] = "1"


suppress_qt_warnings()

figure(figsize=(8, 8))
plt.plot(epoch_list, train_loss)
plt.plot(epoch_list, test_loss)
plt.title(feature + ' Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.show()

figure(figsize=(8, 8))
plt.plot(epoch_list, train_acc)
plt.plot(epoch_list, test_acc)
plt.title(feature + ' Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.show()
