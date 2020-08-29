'''
    Loss functions

    ToDo:
        * add more loss functions
            -> change back_prop() in mlp.py
                -> last layer back_prop_error in loss functions
'''

import numpy as np

from src.activation import Sigmoid, Relu


class MSE:
    def __init__(self, activation_fn):
        self.activation_fn = activation_fn

    def activation(self, z):
        return self.activation_fn.activation(z)

    @staticmethod
    def loss(y_true, y_pred):
        return np.mean((y_pred - y_true)**2)

    @staticmethod
    def prime(y_true, y_pred):
        return y_pred - y_true