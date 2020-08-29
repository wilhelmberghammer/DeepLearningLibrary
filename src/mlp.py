import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from src.loss import MSE
from src.activation import Sigmoid, Relu



class MLP:
    '''
        MULTI LAYER PERCEPTON

        * __init__() .................. initialise model
        * feed_forward() .............. feed forward through model
        * predict() ................... make prediction with model
        * back_prop() ................. backpropagation algorithm
        * update_weights_biases() ..... update weights and biases
        * fit() ....................... train model
    '''

    def __init__(self, architecture, activation_functions):
        '''
            INITIALISE MODEL

            * architecture ............ tuple of architecture (2, 16, 16, 2) -> (2 input nodes, 16 hidden nodes, 16 hidden nodes, 2 output nodes)
            * n_layers ................ number of layers (lenght of the architecture tuple)
            * loss_func ............... reference to loss function (loss class, MSE for example) - loss.py
            * learning_rate ........... initialised as None - fit()

            * losses .................. list of losses to plot loss graph
            * activ_func .............. tuple ofthe activation functions used - for summary() only

            * w ....................... dict with weights -- w[1] -> weights between inout layer and hidden layer 1
            * b ....................... dict with biases

            * activation_functions .... dict with activation fuctions per layer -- activation_functions[2] -> activation function of hidden layer 1
        '''

        self.architecture = architecture
        self.n_layers = len(architecture)
        self.loss_func = None
        self.learning_rate = None

        self.losses = []

        self.activ_func = activation_functions

        self.w = {}
        self.b = {}

        self.activation_functions = {}


        for i in range(len(architecture) - 1):
            self.w[i + 1] = np.random.randn(architecture[i], architecture[i + 1]) / np.sqrt(architecture[i])
            self.b[i + 1] = np.zeros(architecture[i + 1])
            self.activation_functions[i + 2] = activation_functions[i]



    def feed_forward(self, X):
        '''
            * z ........................ dict with z of neurons in layers (z = a * w + b)
            * a ........................ dict with a (activations) of neurons in layers (a = f(z) -> f() ... activatin function)
        '''

        z = {}

        # 1: ... input layer -> inputs of input layer are the data X, not activations of the precious layer (there is no previous layer ... no shit)
        a = {1: X}

        # range(1, self.n_layers) ... (layer 1 (input layer), number of layers (last layer ... output layer))
        for i in range(1, self.n_layers):
            # calculate z and a for the next layer with the activations, weights and biases of the current layer
            z[i + 1] = np.dot(a[i], self.w[i]) + self.b[i]
            a[i + 1] = self.activation_functions[i + 1].activation(z[i + 1])

        return z, a



    def predict(self, X):
        '''
            returns activations of the last layer (output layer)
        '''

        _, a = self.feed_forward(X)
        return a[self.n_layers]


    def back_prop(self, z, a, y):
        """
            * z ........................ dict of z for the layers (keys) -> (a * w + b)
            * a ........................ dict of a for the layer (keys) -> (f(z))
            * y ........................ one hot array of the label

            * update_parameters ........ dict with the parameters used for updating the weights and biases -> dw and back_prop_error
            * back_prop_error .......... backpropagation error of a layer
        """

        # calculate partial derivative (dC_dw) and back_prop_error for the output layer
        # if architechture is (x, y, z):
        #   dC/dw for w[2]
        '''
            back_prop_error[last_layer] = (prediction - y) * f_prime[last_layer](z[last_layer])
        '''
        back_prop_error = self.loss_func.prime(y, a[self.n_layers]) * self.activ_func[-1].prime(a[self.n_layers])
        dC_dw = np.dot(a[self.n_layers - 1].T, back_prop_error)

        update_parameters = {
            self.n_layers - 1: (dC_dw, back_prop_error)
        }

        # Determine partial derivative (dC_dw) and back_prop_error for the rest of the layers
        # if architecture = (x, y, z):
        #   dC/dw[1]
        '''
            back_prop_error[n] = back_prop_error[n+1] * w[n] * f_prime[n](z[n]) 
            dC/dw[n-1] = back_prop_error[n] * a[n-1].T
        '''
        for n in reversed(range(2, self.n_layers)):
            back_prop_error = np.dot(back_prop_error, self.w[n].T) * self.activation_functions[n].prime(z[n])
            dC_dw = np.dot(a[n - 1].T, back_prop_error)
            update_parameters[n - 1] = (dC_dw, back_prop_error)
	

        # Update parameters
        '''
            dC/db[n-1] = back_prop_error[n]
        '''
        for i, update_param in update_parameters.items():
            # update_param[0] ... dC_dw
            # update_param[1] ... back_prop_error
            self.w[i] -= self.learning_rate * update_param[0]
            self.b[i] -= self.learning_rate * np.mean(update_param[1], 0)



    def fit(self, X_train, y_train, loss_func, epochs, batch_size, learning_rate=0.01, report_epochs=1):
        """
            * X_train .................. array with training data (features)
            * y_train .................. array with the training labels
            * loss_func ................ reference to loss function (loss class, MSE for example) - loss.py
            * epochs ................... number of epocchs (iterations)
            * batch_size: .............. batch size
        """
        
        # Initiate the loss object with the final activation function
        self.loss_func = loss_func(self.activation_functions[self.n_layers])
        self.learning_rate = learning_rate

        for i in range(epochs):
            # shuffle data
            seed = np.arange(X_train.shape[0])
            np.random.shuffle(seed)
            x_ex = X_train[seed]
            y_ex = y_train[seed]

            # fit
            for j in range(X_train.shape[0] // batch_size):
                m = j * batch_size
                n = (j + 1) * batch_size
                z, a = self.feed_forward(x_ex[m:n])
                self.back_prop(z, a, y_ex[m:n])


            # Reporting
            if (i + 1) % report_epochs == 0 or (i + 1) == 1:
                _, a = self.feed_forward(X_train)
                print("Epochs: ", (i+1), "- Loss:", self.loss_func.loss(y_train, a[self.n_layers]))

            if (i + 1) % 1 == 0 or (i + 1) == 1:
                _, a = self.feed_forward(X_train)
                self.losses.append(self.loss_func.loss(y_train, a[self.n_layers]))
    

    def summary(self):
        '''
            .summary() -> to get a summary of the model
        '''

        archi = []
        archi_hidden = []
        activ_func = []
        activ_func_str = []
        activ_func_hidden = []

        trainable_params = 0
        n_biases = 0
        n_weights = 0
        w_array = []

        for i in range(len(self.architecture)):
            archi.append(self.architecture[i])

        for j in range(len(self.activ_func) + 1):
            if j == 0:
                activ_func.append('None')
            else:
                activ_func.append(self.activ_func[j-1])
        
        for k in range(len(activ_func)):
            if str(activ_func[k]) == "<class 'src.activation.Relu'>":
                activ_func_str.append('ReLU')
            elif str(activ_func[k]) == "<class 'src.activation.Sigmoid'>":
                activ_func_str.append('Sigmoid')
            else:
                activ_func_str.append('None')


        activ_func_hidden = activ_func_str[1:-1]
        archi_hidden = archi[1:-1]


        n_biases = (np.sum(archi) - archi[0])

        for i in range(len(archi) - 1):
            w_array.append(archi[i] * archi[i + 1])

        n_weights = np.sum(w_array)

        trainable_params = n_biases + n_weights

        print('#' * 40)
        print(' Perceptron Architecture')
        print('+' * 40)
        print(' Input Layer    |', archi[0], 'Nodes',' - ', activ_func_str[0])

        for l in range(len(activ_func_hidden)):
            print(' Hidden Layer', l+1,'|', archi_hidden[l], 'Nodes',' - ', activ_func_hidden[l])

        print(' Output Layer   |', archi[-1], 'Nodes',' - ', activ_func_str[-1])
        print('+' * 40)
        print('number of biases: ', n_biases)
        print('number of weights: ', n_weights)
        print('number of trainable parameters: ', trainable_params)
        print('#' * 40)
        print('')


    def plt_loss(self):
        '''
            .plt_loss() -> plot loss chart
        '''
        
        figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
        plt.grid()
        plt.plot(self.losses)
        plt.show()