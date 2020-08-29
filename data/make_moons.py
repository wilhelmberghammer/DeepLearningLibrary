import numpy as np
import matplotlib.pyplot as plt
import sklearn

from sklearn import datasets

X, y_ = sklearn.datasets.make_moons(2000, noise=0.2)
#plt.scatter(X[:,0], X[:,1], s=40, c=y_)
#plt.show()

y = np.zeros((y_.size, y_.max()+1))
y[np.arange(y_.size),y_] = 1