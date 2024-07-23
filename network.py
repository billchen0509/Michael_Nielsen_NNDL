'''
network.py

A module to implement the stochastic gradient descent learning algorithm 
for a feedforward neural network. Gradients are calculated using backpropagation. 
'''

### Load library
import random
import numpy as np

### initializa a Network object
class Network(object):
    def __init__(self,sizes):
        self.num_layers = len(sizes)
        '''
        The list 'sizes' contains the number of neurons in the respective layers of the network.
        For examples: if the list was [2, 3, 1] then it would be a three-layer network, 
        with the first layer containing 2 neurons , 
        the second layer 3 neurons , 
        and the third layer 1 neuron.
        '''
        self.sizes = sizes
        self.basis = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(x,y) for x,y in zip(sizes[:-1], sizes[1:])]


    def feedforward(self, a):
        '''
        Return the output of the neural network if 'a' in an input
        '''
        for b, w in zip(self.basis, self.weights):
            a = sigmoid(np.dot(w,a) + b)
            return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data = None):
        if test_data:
            n_test = len(test_data)
            n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n , mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch,eta)
        if test_data:
            print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test) )
        else:
            print("Epoch {0} complete".format(j))

    def updata_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.basis]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabala_w = self.backprop(x, y)

    def backprop(self, x , y):
        



def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(-z))
