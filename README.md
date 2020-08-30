# Deep Learning Library from scratch

This is my first implementation of a deep learning library just using NumPy. This library is not intended to be used in 'real world' applications - for educational porpuses only.

At the moment (Aug. 2020) this library only enables you to make fully connected networks with a variable number of layers. You can also chose the activaiton function of each layer, except the input layer.


### test_spiral.py:

Import dependencies:
```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from src.mlp import MLP
from src.activation import Sigmoid, Relu
from src.loss import MSE
```

Get data (data/spiral.py):
```python
from data.spiral import X, y, y_
```


```python
nn = MLP((2, 32, 32, 32, 2), (Relu, Relu, Relu, Sigmoid))
nn.summary()
```

nn.sumary():
```
########################################
 Perceptron Architecture
++++++++++++++++++++++++++++++++++++++++
 Input Layer    | 2 Nodes  -  None
 Hidden Layer 1 | 32 Nodes  -  ReLU
 Hidden Layer 2 | 32 Nodes  -  ReLU
 Hidden Layer 3 | 32 Nodes  -  ReLU
 Output Layer   | 2 Nodes  -  Sigmoid
++++++++++++++++++++++++++++++++++++++++
number of biases:  98
number of weights:  2176
number of trainable parameters:  2274
########################################
```

Train network:
```python
nn.fit(X, y, loss_func=MSE, epochs=3000, batch_size=16, learning_rate=.001, report_epochs = 1000)
```

![spiral_trained.png](https://github.com/wilhelmberghammer/deeplearninglibrary/blob/master/readme_recources/spiral_trained.png?raw=true)