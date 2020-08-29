import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

mnist = tf.keras.datasets.mnist
(x_tr, y_train), (x_te, y_test) = mnist.load_data()

x_tr = tf.keras.utils.normalize(x_tr, axis=1)
x_te = tf.keras.utils.normalize(x_te, axis=1)

y_train = np.eye(10)[y_train]
y_test = np.eye(10)[y_test]

np.array(x_tr)
np.array(x_te)
x_train = []
x_test = []

for i in range(len(x_tr)):
    x_train.append(x_tr[i].flatten())

for i in range(len(x_te)):
    x_test.append(x_te[i].flatten())

x_train = np.array(x_train)
x_test = np.array(x_test)