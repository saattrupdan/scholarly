import numpy as np
import sys
import os
import time
from functools import partial, reduce
from itertools import permutations, chain, product, count

# Neural network packages
import tensorflow.compat.v1 as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras import backend as K
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, precision_score, recall_score
from tensorflow.python.util import deprecation

from naturalselection.core import Genus


# Suppress deprecation warnings
deprecation._PRINT_DEPRECATION_WARNINGS = False

# Suppress tensorflow warnings and infos
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class FNN(Genus):
    ''' Feedforward neural network genus.

    INPUT:
        (iterable) number_of_hidden_layers: numbers of hidden layers
        (iterable) dropout: values for dropout
        (iterable) neurons_per_hidden_layer = neurons in hidden layers
        (iterable) optimizer: keras optimizers
        (iterable) hidden_activation: keras activation functions
        (iterable) batch_size: batch sizes
        (iterable) initializer: keras initializers
        '''
    def __init__(self,
        max_number_of_hidden_layers = 5,
        dropout = np.arange(0, 0.6, 0.1),
        neurons_per_hidden_layer = np.array([2 ** n for n in range(4, 11)]),
        optimizer = np.array(['sgd', 'rmsprop', 'adagrad', 'adadelta',
                              'adamax', 'adam', 'nadam']),
        hidden_activation = np.array(['relu', 'elu', 'softplus', 'softsign']),
        batch_size = np.array([2 ** n for n in range(4, 12)]),
        initializer = np.array(['lecun_uniform', 'lecun_normal',
                                'glorot_uniform', 'glorot_normal',
                                'he_uniform', 'he_normal'])):

        self.optimizer = np.unique(np.asarray(optimizer))
        self.hidden_activation = np.unique(np.asarray(hidden_activation))
        self.batch_size = np.unique(np.asarray(batch_size))
        self.initializer = np.unique(np.asarray(initializer))
        self.input_dropout = np.unique(np.asarray(dropout))

        layers = np.unique(np.append(neurons_per_hidden_layer, 0))
        dropout = np.around(np.unique(np.append(dropout, 0)), 2)

        layers_dropout = {}
        for layer_idx in range(max_number_of_hidden_layers):
            layers_dropout[f"layer{layer_idx}"] = layers
            layers_dropout[f"dropout{layer_idx}"] = dropout

        self.__dict__.update(layers_dropout)

class TimeStopping(Callback):
    ''' Callback to stop training when enough time has passed.

    INPUT
        (int) seconds: maximum time before stopping.
        (int) verbose: verbosity mode.
    '''
    def __init__(self, seconds = None, verbose = 0):
        super(Callback, self).__init__()
        self.start_time = 0
        self.seconds = seconds
        self.verbose = verbose

    def on_train_begin(self, logs = {}):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs = {}):
        if time.time() - self.start_time > self.seconds:
            self.model.stop_training = True
            if self.verbose:
                print(f'Stopping after {self.seconds} seconds.')

def train_fnn(fnn, train_val_sets, loss_fn = 'binary_crossentropy',
    number_of_inputs = 'infer', number_of_outputs = 'infer',
    output_activation = 'sigmoid', score = 'accuracy',
    max_epochs = 1000000, patience = 5, min_change = 1e-4,
    max_training_time = None, verbose = False):
    ''' Train a feedforward neural network and output the score.
    
    INPUT
        (FNN) fnn: a feedforward neural network genus
        (tuple) train_val_sets: a quadruple of the form
                (X_train, Y_train, X_val, Y_val)
        (string) loss_fn: keras loss function, or a custom one
        (int or string) number_of_inputs: number of input features,
                        will infer from X_train if it's set to 'infer'
        (int or string) number_of_outputs: number of output features,
                        will infer from Y_train if it's set to 'infer'
        (string) output_activation: keras activation to be used on output
        (string) score: the scoring used. Can either be a custom scoring
                 function which takes (Y_val, Y_hat) as inputs, or can be
                 set to the predefined ones: 'accuracy', 'f1', 'precision'
                 or 'recall', where the micro-average will be taken if there
                 are multiple outputs
        (int) max_epochs: maximum number of epochs to train for
        (int) patience: number of epochs with no progress above min_change
        (float) min_change: everything below this number won't count as a
                change in the score
        (int) max_training_time: maximum number of seconds to train for,
              also training the final epoch after the time has run out
        (int) verbose: verbosity mode

    OUTPUT
        (float) the score of the neural network
    '''

    X_train, Y_train, X_val, Y_val = train_val_sets

    if number_of_inputs == 'infer':
        number_of_inputs = X_train.shape[1]
    if number_of_outputs == 'infer':
        number_of_outputs = Y_train.shape[1]

    inputs = Input(shape = (number_of_inputs,))
    x = Dropout(fnn.input_dropout)(inputs)
    for i in count():
        try:
            layer = fnn.__dict__[f"layer{i}"]
            if layer:
                x = Dense(layer, activation = fnn.hidden_activation,
                    kernel_initializer = fnn.initializer)(x)
            dropout = fnn.__dict__[f"dropout{i}"]
            if dropout:
                x = Dropout(dropout)(x)
        except:
            break
    outputs = Dense(number_of_outputs, activation = output_activation,
        kernel_initializer = fnn.initializer)(x)
    nn = Model(inputs = inputs, outputs = outputs)

    nn.compile(
        loss = loss_fn,
        optimizer = fnn.optimizer
        )

    early_stopping = EarlyStopping(
        monitor = 'val_loss',
        patience = patience,
        min_delta = min_change,
        restore_best_weights = True,
        verbose = verbose
        )

    time_stopping = TimeStopping(
        seconds = max_training_time,
        verbose = verbose
        )

    H = nn.fit(
        X_train,
        Y_train,
        batch_size = fnn.batch_size,
        validation_data = (X_val, Y_val),
        epochs = max_epochs,
        callbacks = [early_stopping, time_stopping],
        verbose = verbose
        )

    if Y_val.shape[1] > 1:
        average = 'micro'
    else:
        average = None

    Y_hat = nn.predict(X_val, batch_size = 32)
    if score == 'accuracy':
        Y_hat = np.greater(Y_hat, 0.5)
        fitness = accuracy_score(Y_val, Y_hat)
    elif score == 'f1':
        Y_hat = np.greater(Y_hat, 0.5)
        fitness = f1_score(Y_val, Y_hat, average = average)
    elif score == 'precision':
        Y_hat = np.greater(Y_hat, 0.5)
        fitness = precision_score(Y_val, Y_hat, average = average)
    elif score == 'recall':
        Y_hat = np.greater(Y_hat, 0.5)
        fitness = recall_score(Y_val, Y_hat, average = average)
    elif score == 'loss':
        fitness = np.divide(1, nn.evaluate(X_val, Y_val))
    else:
        # Custom scoring function
        fitness = score(Y_val, Y_hat)
    
    # Clear tensorflow session to avoid memory leak
    K.clear_session()
        
    return fitness

def get_nn_fitness_fn(train_val_sets, loss_fn, number_of_inputs = 'infer',
    number_of_outputs = 'infer', output_activation = 'sigmoid',
    score = 'accuracy', max_epochs = 1000000, patience = 5,
    min_change = 1e-4, max_training_time = None, verbose = False):
    ''' Return a neural network fitness function.
    
    INPUT
        (tuple) train_val_sets: a quadruple of the form
                (X_train, Y_train, X_val, Y_val)
        (string) loss_fn: keras loss function
        (int or string) number_of_inputs: number of input features,
                        will infer from X_train if it's set to 'infer'
        (int or string) number_of_outputs: number of output features,
                        will infer from Y_train if it's set to 'infer'
        (string) output_activation: keras activation to be used on output
        (string) score: the scoring used. Can be 'accuracy', 'f1',
                 'precision', 'recall' or 'loss', where the micro-average
                 will be taken if possible
        (int) max_epochs: maximum number of epochs to train for
        (int) patience: number of epochs with no progress above min_change
        (float) min_change: everything below this number won't count as a
                change in the score
        (int) max_training_time: maximum number of seconds to train for,
              also training the final epoch after the time has run out
        (int) verbose: verbosity mode

    OUTPUT
        (function) fitness function, which will output 1 / score if score
                   is 'loss', and 1 / (1 - score) otherwise, to ensure 
                   that the range is unbounded
    '''

    fitness_fn = partial(
        train_fnn,
        train_val_sets      = train_val_sets,
        loss_fn             = loss_fn,
        number_of_inputs    = number_of_inputs,
        number_of_outputs   = number_of_outputs,
        output_activation   = output_activation,
        score               = score,
        max_epochs          = max_epochs,
        patience            = patience,
        min_change          = min_change,
        max_training_time   = max_training_time,
        verbose             = verbose
        )
    
    return fitness_fn


def __main__():
    pass
