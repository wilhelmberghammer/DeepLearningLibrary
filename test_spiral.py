import os
os.system('clear')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from src.mlp import MLP
from src.activation import Sigmoid, Relu
from src.loss import MSE

from data.spiral import X, y, y_



nn = MLP((2, 64, 64, 2), (Relu, Relu, Sigmoid))
nn.summary()

nn.fit(X, y, loss_func=MSE, epochs=500, batch_size=5, learning_rate=.025)


def plt_spiral():
    figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

    axes = plt.gca()
    axes.set_xlim([-15, 15])
    axes.set_ylim([-15, 15])

    for i in np.linspace(-15, 15, 70):
        for j in np.linspace(-15, 15, 60):
            a = nn.predict([i, j])
            color = 'blue'
            if str(np.round(a)) == '[0. 1.]':
                color = 'orange'
            plt.scatter(i, j, c=color, alpha=.3)
    plt.plot(X[y_==0,0], X[y_==0,1], '.')
    plt.plot(X[y_==1,0], X[y_==1,1], '.')
    plt.show()

plt_spiral()