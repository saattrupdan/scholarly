import numpy as np
import matplotlib.pyplot as plt # for plotting cost
from sklearn.base import BaseEstimator, TransformerMixin
import warnings # allows suppression of warnings
import itertools as it # enables count()
from functools import reduce # used to calculate accuracy
import os

##### ACTIVATION FUNCTIONS #####

def sigmoid(x, derivative = False):
    """
    Implement the sigmoid activation function, x -> 1/(1+e^{-x}).

    INPUT:
    x -- numpy array
    derivative -- bool, whether the derivative should be output

    OUTPUT:
    sigmoid, or derivative of sigmoid, of x
    """
    sigm = 1. / (1. + np.exp(-x))
    return sigm * (1. - sigm) if derivative else sigm

def relu(x, derivative = False):
    """
    Implement the relu activation function, x -> max(0,x).

    INPUT:
    x -- numpy array
    derivative -- bool, whether the derivative should be output

    OUTPUT:
    relu, or derivative of relu, of x
    """
    return np.where(x < 0, 0, 1) if derivative else np.maximum(x, 0)

def linear(x, derivative = False):
    """
    Implement the linear activation function, x -> x.

    INPUT:
    x -- numpy array
    derivative -- bool, whether the derivative should be output

    OUTPUT:
    linear, or derivative of linear, of x
    """
    return np.ones(x.shape) if derivative else x

def tanh(x, derivative = False):
    """
    Implement the tanh activation function, x -> tanh(x).

    INPUT:
    x -- numpy array
    derivative -- bool, whether the derivative should be output

    OUTPUT:
    tanh, or derivative of tanh, of x
    """
    return 1 - np.square(np.tanh(x)) if derivative else np.tanh(x)


##### COST FUNCTIONS #####

def binary_cross_entropy_cost(Yhat, Y):
    """
    Implement the binary cross entropy cost function. Note that this
    is also used for multilabel classification, in a "one vs rest" fashion.

    INPUT:
    Yhat -- probability vector corresponding to predictions,
            shape (output_layer, m)
    Y -- true "label" vector, shape (output_layer, m)

    OUTPUT:
    cost -- cross-entropy cost
    """
    
    assert Yhat.shape == Y.shape
    
    m = Y.shape[1]
    
    return -1. / m * np.sum(Y * np.log(Yhat) + (1. - Y) * np.log(1. - Yhat))

def multiclass_cross_entropy_cost(Yhat, Y):
    """
    Implement the cross entropy cost function in a multiclass setup.

    INPUT:
    Yhat -- probability vector corresponding to label predictions, shape (c, m)
    Y -- true "label" vector, shape (c, m)

    OUTPUT:
    cost -- multiclass cross-entropy cost
    """
    
    assert Yhat.shape == Y.shape
    
    m = Y.shape[1]
    return -1. / m * np.sum(Y * np.log(Yhat))

def l2_cost(Yhat, Y):
    """
    Implement the l2 cost function.

    INPUT:
    Yhat -- vector corresponding to label predictions, shape (1, m)
    Y -- true "label" vector, shape (1, m)

    OUTPUT:
    cost -- l2 cost
    """

    assert Yhat.shape == Y.shape
    
    m = Y.shape[1]
    return 1. / (2. * m) * np.sum(np.square(Y - Yhat))


##### PARAMETER FUNCTIONS #####

def initialise_params(layer_dims, init_method = 'he'):
    """
    Initialise the parameters of the neural network.
    
    INPUT:
    layer_dims -- numpy array containing the dimensions of each layer in
                  our network
    init_method -- string determining the type of initialisation;
                   can be 'he', 'xavier' or 'manual'
    
    OUTPUT:
    params -- python dictionary containing parameters
    """
    
    params = {}

    # the number of layers, including input layer
    L = len(layer_dims)

    for l in np.arange(1, L):
        if init_method == 'he':
            params[f'W{l}'] = np.random.randn(layer_dims[l],
                layer_dims[l-1]) * np.sqrt(2 / layer_dims[l-1])
        elif init_method == 'xavier':
            params[f'W{l}'] = np.random.randn(layer_dims[l],
                layer_dims[l-1]) * np.sqrt(1 / layer_dims[l-1])
        elif init_method == 'manual':
            params[f'W{l}'] = np.random.randn(layer_dims[l],
                layer_dims[l-1]) * 0.0001
        
        params[f'b{l}'] = np.zeros((layer_dims[l], 1))
        
    return params

def update_params(params, grads, learning_rate):
    """
    Update parameters using gradient descent.
    
    INPUT:
    params -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients, output of back_prop
    learning_rate -- learning rate of the gradient descent update rule
    
    OUTPUT:
    params -- python dictionary containing your updated parameters 
    """
    
    # the number of layers, including input layer
    L = len(params) // 2 
    
    for l in np.arange(L):
        params[f'W{l+1}'] -= learning_rate * grads[f'dW{l+1}']
        params[f'b{l+1}'] -= learning_rate * grads[f'db{l+1}']
        
    return params


##### FORWARD PROPAGATION #####

def forward_step(A_prev, W, b, activation):
    """
    Implement one step of the forward propagation.

    INPUT:
    A_prev -- activations from previous layer: (size of previous layer, m)
    W -- weights matrix: numpy array of shape (size of current layer,
        size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a
        text string

    OUTPUT:
    A -- the output of the activation function, also called the
        post-activation value 
    cache -- a python dictionary containing W, b, Z and A_prev;
        used in back propagation
    """

    Z = np.dot(W, A_prev) + b
    
    if activation == "sigmoid":
        A = sigmoid(Z)
    elif activation == "relu":
        A = relu(Z)
    elif activation == 'linear':
        A = linear(Z)
    elif activation == 'tanh':
        A = tanh(Z)
        
    cache = {
        'W' : W,
        'b' : b,
        'Z' : Z,
        'A_prev' : A_prev
    }
    
    return A, cache

def forward_prop(X, params, activations = 'default'):
    """
    Implement the forward propagation.
    
    INPUT:
    X -- data, numpy array of shape (n, m)
    params -- output of initialise_params()
    activations -- list of activation functions used; defaults to 
        ReLU + sigmoid
    
    OUTPUT:
    AL -- last post-activation value
    caches -- list of caches containing every cache of forward_step()
    """

    # the number of layers, including input layer
    L = len(params) // 2

    caches = []
    A = X
    
    if activations == 'default':
        activations = ['relu'] * (L - 1) + ['sigmoid']
    
    for l in np.arange(L):
        A_prev = np.copy(A)
        W = params[f"W{l+1}"]
        b = params[f"b{l+1}"]
        A, cache = forward_step(A_prev, W, b, activations[l])
        caches.append(cache)
                
    return A, caches


##### BACKWARD PROPAGATION #####

def back_step(dA, cache, activation, cost_function = 'binary_cross_entropy'):
    """
    Implement one step of the backward propagation.
    
    INPUT:
    dA -- post-activation gradient for current layer l 
    cache -- a python dictionary containing W, b, Z and A_prev;
        used in back propagation
    activation -- the activation to be used in this layer, stored as
        a text string
    cost_function -- a string describing what cost function is used
    
    OUTPUT:
    dA_prev -- Gradient of the cost with respect to the previous activation,
        same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l),
        same shape as W
    db -- Gradient of the cost with respect to b (current layer l),
        same shape as b
    """
    A_prev = cache['A_prev']
    W = cache['W']
    b = cache['b']
    Z = cache['Z']
    m = A_prev.shape[1]
    
    if activation == "relu":
        dAct = relu(Z, derivative = True)        
    elif activation == "sigmoid":
        dAct = sigmoid(Z, derivative = True)
    elif activation == 'linear':
        dAct = linear(Z, derivative = True)
    elif activation == 'tanh':
        dAct = tanh(Z, derivative = True)
    
    # not sure if these should be different for multiclass cross entropy
    dZ = dA * dAct
    dW = 1/m * np.dot(dZ, A_prev.T)
    db = 1/m * np.sum(dZ, keepdims = True, axis = 1)
    dA_prev = np.dot(W.T, dZ)
        
    return dA_prev, dW, db

def back_prop(AL, Y, caches, activations = 'default', 
    cost_function = 'binary_cross_entropy'):
    """
    Implement the backward propagation.
    
    INPUT:
    AL -- probability vector, output of forward_prop()
    Y -- true "label" vector
    caches -- list of dictionaries containing the W's, b's, Z's and A_prev's
    activations -- list of activation functions used; defaults to 
        ReLU + sigmoid
    
    OUTPUT:
    grads -- A dictionary with the gradients
    """
    
    # the number of layers, including input layer
    L = len(caches)

    grads = {}
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    
    if activations == 'default':
        activations = ['relu'] * (L - 1) + ['sigmoid']
    
    # Initializing the backpropagation
    if cost_function == 'binary_cross_entropy':
        grads[f"dA{L}"] = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    elif cost_function == 'multiclass_cross_entropy':
        grads[f"dA{L}"] = -np.divide(Y, AL) # not sure about this
    elif cost_function == 'l2':
        grads[f"dA{L}"] = AL - Y

    for l in np.arange(L)[::-1]:
        current_cache = caches[l]
        grads[f"dA{l}"], grads[f"dW{l+1}"], grads[f"db{l+1}"] = \
            back_step(grads[f"dA{l+1}"], current_cache, activations[l], 
            cost_function)

    return grads


##### BUILD MODEL #####

def train_nn(X, Y, layer_dims, params, activations = 'default', 
    cost_function = 'binary_cross_entropy', init_learning_rate = 1.0,
    plots = False, adaptive_learning = 0.05,
    target_accuracy = 0.95, min_learning_rate = 0.00001,
    test_set = None):
    """
    Trains a neural network.
    
    INPUT:
    X -- training data, of shape (n, m)
    Y -- true "label" vector, of shape (output_layers, m)
    layer_dims -- list with the input size and each layer size, of length
                  (number of layers + 1)
    activations -- list of activation functions used; defaults to 
                   ReLU + sigmoid
    cost_function -- a string describing what cost function is used
    init_learning_rate -- initial learning rate of the gradient descent
    plots -- if True, it plots the cost
    adaptive_learning -- gradually lower learning rate when cost starts 
                         increasing, if it's nonzero. Must be strictly 
                         between 0 and 1
    target_accuracy -- stop gradient descent when training accuracy is
                       below this number
    min_learning_rate -- minimal allowed learning rate
    test_set -- a tuple (X_test, Y_test) for calculating test accuracy
    
    OUTPUT:
    params -- parameters learnt by the model.
    """

    train_costs = np.asarray([])
    test_costs = np.asarray([])
    train_accs = np.asarray([])
    test_accs = np.asarray([])
    confidence = 0
    learning_rate = init_learning_rate

    if test_set:
        (X_test, Y_test) = test_set
        test_accuracy = 0
    
    print(f"Performing batch gradient descent...", end = "\r")
    for i in it.count():
        # save old parameters
        if i > 0:
            old_params = params
            old_train_cost = train_cost
        
        # give the model some confidence
        confidence += 1
        
        # if the model is sufficiently confident then increase
        # the learning rate
        if adaptive_learning and confidence % 100 == 0:
            learning_rate /= 1 - (adaptive_learning * (confidence / 1000))
        
        # forwardprop + backprop
        AL, caches = forward_prop(X, params, activations)
        grads = back_prop(AL, Y, caches, activations, cost_function)
        params = update_params(params, grads, learning_rate)
        
        # compute cost
        if cost_function == 'binary_cross_entropy':
            train_cost = binary_cross_entropy_cost(AL, Y)
        elif cost_function == 'multiclass_cross_entropy':
            train_cost = multiclass_cross_entropy_cost(AL, Y)
        elif cost_function == 'l2':
            train_cost = l2_cost(AL, Y)

        if i % 100 == 0:
            
            # calculate training accuracy
            Yhat, _ = forward_prop(X, params, activations)
            Yhat = np.squeeze(np.around(Yhat, decimals = 0)).astype('int')
            correct_predictions = np.sum(np.asarray(
                [reduce(lambda z, w: z and w, x) 
                for x in np.equal(Y.T, Yhat.T)]
                ))
            train_accuracy = correct_predictions / X.shape[1]
            
            # if target training accuracy is reached then stop
            if train_accuracy > target_accuracy:
                print("") # deal with \r
                print("Reached target training accuracy.")
                break
        
            if test_set:
                Yhat, _ = forward_prop(X_test, params, activations)

                # calculate test cost 
                if cost_function == 'binary_cross_entropy':
                    test_cost = binary_cross_entropy_cost(Yhat, Y_test)
                elif cost_function == 'multiclass_cross_entropy':
                    test_cost = multiclass_cross_entropy_cost(Yhat, Y_test)
                elif cost_function == 'l2':
                    test_cost = l2_cost(Yhat, Y_test)

                # calculate test accuracy
                Yhat = np.squeeze(np.around(Yhat, decimals = 0)).astype('int')
                correct_predictions = np.sum(np.asarray(
                    [reduce(lambda z, w: z and w, x)
                    for x in np.equal(Y_test.T, Yhat.T)]
                    ))
                test_accuracy = correct_predictions / X_test.shape[1]
            
                # store test costs and accuracies
                test_costs = np.append(test_costs, test_cost)
                test_accs = np.append(test_accs, test_accuracy)
            
            # store training costs and accuracies
            train_costs = np.append(train_costs, train_cost)
            train_accs = np.append(train_accs, train_accuracy)

        # if training cost starts to increase then rewind one step
        # and lower the learning rate
        if adaptive_learning and i > 0 and train_cost > old_train_cost:
            params = old_params
            train_cost = old_train_cost

            if confidence >= 100:
                learning_rate *= 1 - (adaptive_learning * (confidence / 1000))
                confidence -= 100
            else:
                learning_rate *= 1 - adaptive_learning
                confidence = 0

            if learning_rate < min_learning_rate:
                print("")
                print("Minimal learning rate reached.")
                break
        
        if (i+1) % 100 == 0:
            print(f"Performing batch gradient descent... " \
                  f"{i+1} iterations completed. " \
                  f"Learning rate: {np.around(learning_rate, 3)}. " \
                  #f"Confidence: {confidence}. " \
                  f"Training cost: {np.around(train_cost, 5)}. " \
                  f"Test accuracy: {np.around(test_accuracy * 100, 2)}%. " \
                  + " " * 25, end = "\r")

    # build and save plots
    if plots:
        plt.plot(np.squeeze(train_costs), label = 'train')
        plt.plot(np.squeeze(test_costs), label = 'test')
        plt.title('Cost')
        plt.ylabel('cost')
        plt.xlabel('iterations (hundreds)')
        plt.legend()
        plt.savefig(f'{X.shape[1]}_cost.png')
        plt.clf()

        plt.plot(np.squeeze(train_accs), label = 'train')
        plt.plot(np.squeeze(test_accs), label = 'test')
        plt.title('Accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('iterations (hundreds)')
        plt.legend()
        plt.savefig(f'{X.shape[1]}_acc.png')
        plt.clf()
    
    return params


class NeuralNetwork(TransformerMixin, BaseEstimator):

    def __init__(self, layer_dims = [1], activations = 'default', 
        init_method = 'he', cost_function = 'binary_cross_entropy',
        init_learning_rate = 1.0, plots = False,
        adaptive_learning = 0.05, target_accuracy = 0.95,
        min_learning_rate = 0.00001, test_set = None):

        self.layer_dims_ = layer_dims
        self.activations_ = activations
        self.init_method_ = init_method
        self.cost_function_ = cost_function
        self.target_accuracy_ = target_accuracy
        self.plots_ = plots
        self.init_learning_rate_ = init_learning_rate
        self.min_learning_rate_ = min_learning_rate
        self.adaptive_learning_ = adaptive_learning 
        self.test_set_ = test_set
        self.params_ = None
    
    # add input layer and initialise params
    def fit(self, X):
        self.layer_dims_ = [X.shape[0]] + self.layer_dims_
        self.params_ = initialise_params(self.layer_dims_)
        return self

    def train(self, X, Y):
        self.params_ = train_nn(
            X = X,
            Y = Y, 
            layer_dims = self.layer_dims_,
            params = self.params_,
            activations = self.activations_, 
            cost_function = self.cost_function_, 
            target_accuracy = self.target_accuracy_, 
            plots = self.plots_, 
            init_learning_rate = self.init_learning_rate_,
            min_learning_rate = self.min_learning_rate_,
            adaptive_learning = self.adaptive_learning_,
            test_set = self.test_set_
            )
        return self
    
    def predict(self, X):
        Yhat, _ = forward_prop(X, self.params_, self.activations_)
        return Yhat
