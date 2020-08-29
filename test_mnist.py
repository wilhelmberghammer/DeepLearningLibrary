import os
os.system('clear')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from src.mlp import MLP
from src.activation import Sigmoid, Relu
from src.loss import MSE

from data.mnist_data import x_train, x_test, y_train, y_test


nn = MLP((784, 16, 16, 10), (Sigmoid, Sigmoid, Sigmoid))
nn.summary()
nn.fit(x_train, y_train, loss_func=MSE, epochs=10, batch_size=64, learning_rate=.01, report_epochs = 1)
#nn.plt_loss()

def val_loss():
    val_loss = []
    for i in range(len(y_test)):
        val_loss.append(np.mean((nn.predict(x_test[i]) - y_test[i])**2))

    np.array(val_loss)
    avg_val_loss = np.mean(val_loss)

    return avg_val_loss


print('Validation loss: ', val_loss())

