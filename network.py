'''
network.py

A module to implement the stochastic gradient descent learning algorithm 
for a feedforward neural network. Gradients are calculated using backpropagation. 
'''

### Load library
import random
import numpy as np
import time

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
            time1 = time.time()
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            time2 = time.time()
            if test_data:
                print("Epoch {0}: {1} / {2}, took {3:.2f} seconds".format(
                    j, self.evaluate(test_data), n_test, time2-time1))
            else:
                print("Epoch {0} complete in {1:.2f} seconds".format(j, time2-time1))

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.basis]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [ w - (eta/len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.basis = [ b - (eta/len(mini_batch)) * nb for b, nb in zip(self.basis, nabla_b)]

    
    def backprop(self, x , y):
        '''
        Return a tuple (nabla_b, nabla_w) representing the gradient for the cost function C_x.
        '''
        nabla_b = [np.zeros(b.shape) for b in self.basis]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.basis, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        
        for l in range(2, self.num_layers):
            z = zs[-1]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return  (nabla_b, nabla_w)
    
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
    
    def cost_derivative(self, output_activations, y):
        return (output_activations - y)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(-z))
