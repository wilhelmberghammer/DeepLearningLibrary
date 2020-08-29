import os
os.system('clear')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from src.mlp import MLP
from src.activation import Sigmoid, Relu
from src.loss import MSE

from data.make_moons import X, y, y_



nn = MLP((2, 4, 4, 4, 4, 2), (Relu, Relu, Relu, Relu, Sigmoid))
nn.summary()
nn.fit(X, y, loss_func=MSE, epochs=100, batch_size=1, learning_rate=0.001, report_epochs = 10)



def plt_trained_moon():
    figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

    axes = plt.gca()
    axes.set_xlim([-1.5, 2.75])
    axes.set_ylim([-1, 1.5])

    for i in np.linspace(-1.5, 2.75, 70):
        for j in np.linspace(-1, 1.5, 60):
            a = nn.predict([i, j])
            color = 'green'
            if str(np.round(a)) == '[0. 1.]':
                color = 'red'
            plt.scatter(i, j, c=color, alpha=.5)
    plt.scatter(X[:,0], X[:,1], s=10, c=y_)
    plt.show()

plt_trained_moon()