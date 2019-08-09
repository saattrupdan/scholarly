import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools as it
import warnings
from functools import reduce

import os
import sys
import pickle
from pathlib import Path

# homegrown neural network
from NN import NeuralNetwork
from datetime import datetime

# keras neural network packages
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD

# scores to evaluate models
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# custom multilabel loss function
import tensorflow as tf
import keras.backend.tensorflow_backend as tfb


def weighted_binary_crossentropy(target, output):
    '''
    Weighted binary crossentropy between an output tensor 
    and a target tensor. POS_WEIGHT is used as a multiplier 
    for the positive targets.

    Combination of the following functions:
    * keras.losses.binary_crossentropy
    * keras.backend.tensorflow_backend.binary_crossentropy
    * tf.nn.weighted_cross_entropy_with_logits
    '''
    # transform back to logits
    _epsilon = tfb._to_tensor(tfb.epsilon(), output.dtype.base_dtype)
    output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
    output = tf.log(output / (1 - output))

    # compute weighted loss
    loss = tf.nn.weighted_cross_entropy_with_logits(
            targets = target,
            logits = output,
            pos_weight = POS_WEIGHT
            )

    return tf.reduce_mean(loss, axis = -1)

def multi_label_bins(prediction, threshold = 0.5):
    ''' Turn probabilities of multi-label predictions into binary values. '''
    return (prediction > (max(prediction) * threshold)).astype('int')

def multi_label_accuracy(y, yhat, leeway = 0):
    '''
    Compute accuracy of multi-label predictions, where to be counted as a
    correct prediction, there can be at most leeway many mislabellings.
    '''
    assert np.asarray(y).shape == np.asarray(yhat).shape
    leeway_bool = lambda x: (sum(x) + leeway) >= len(x)
    accuracies = np.asarray([leeway_bool(x) for x in np.equal(yhat, y)])
    return np.average(accuracies)

def threshold_f1(probabilities, threshold):
    ''' Get F1 score from probabilities and threshold. '''
    predictions = np.asarray([multi_label_bins(prob, threshold) 
        for prob in probabilities])
    return f1_score(Y_train, predictions, average = 'micro')
            

# don't run the code below if we're just importing things from this file
if __name__ == '__main__':

    # set list of file_names
    if len(sys.argv) > 1:
        file_names = sys.argv[1:]
    else:
        file_names = [f'arxiv_sample_{i}' for i in
            [1000, 5000, 10000, 25000, 50000, 100000]]

    home_dir = str(Path.home())
    data_path = os.path.join(home_dir, "pCloudDrive", "public_folder",
        "scholarly_data")


    ##### Hyperparameters #####

    # multiplier for positive targets in binary cross entropy
    # forces larger recall and smaller precision
    # relu seem to need smaller pos_weight than tanh
    POS_WEIGHT = 5
    
    # activation function for the hidden layers
    ACTIVATION = 'relu'

    # number of neurons in each hidden layer
    NEURONS = [2048, 1024, 512]

    # amount of dropout
    INPUT_DROPOUT = 0.2
    HIDDEN_DROPOUT = 0.5

    # epochs allowed with no improvement
    MONITORING = 'val_loss'
    PATIENCE = 20

    # gradient descent batch size
    BATCH_SIZE = 512

    # gradient descent optimizer
    OPTIMIZER = 'nadam'

    ###########################

    
    for file_name in file_names:
        print("------------------------------------")
        print(f"NOW PROCESSING: {file_name}")
        print("------------------------------------")

        full_path = os.path.join(data_path,
            f"{file_name}_model.pickle")
        if os.path.isfile(full_path):
            with open(full_path, 'rb') as pickle_in:
                nn = pickle.load(pickle_in)
            print("Model already trained.")
        else:
            # load validation set
            full_path = os.path.join(data_path, f"arxiv_val_set.csv")
            val_df = pd.read_csv(full_path)
            X_val = np.asarray(val_df.iloc[:, :1024])
            Y_val = np.asarray(val_df.iloc[:, 1024:])

            # fetch data
            full_path = os.path.join(data_path, f"{file_name}_1hot.csv")
            df = pd.read_csv(full_path)
            X_train = np.asarray(df.iloc[:, :1024])
            Y_train = np.asarray(df.iloc[:, 1024:])

            max_epochs = 1000000

            nn = Sequential()
            nn.add(Dropout(INPUT_DROPOUT))
            for neurons in NEURONS:
                nn.add(Dense(neurons, activation = ACTIVATION))
                nn.add(Dropout(HIDDEN_DROPOUT))
            nn.add(Dense(153, activation = 'sigmoid'))

            nn.compile(
                loss = weighted_binary_crossentropy, 
                optimizer = OPTIMIZER,
                )

            early_stopping = EarlyStopping(
                monitor = MONITORING,
                patience = PATIENCE,
                min_delta = 1e-4,
                restore_best_weights = True
                )

            past = datetime.now()
            H = nn.fit(
                    X_train, 
                    Y_train,
                    validation_data = (X_val, Y_val),
                    epochs = max_epochs,
                    batch_size = BATCH_SIZE,
                    callbacks = [early_stopping]
                    )
            duration = datetime.now() - past

            # find optimal threshold value based on training data
            print("")
            print("Finding optimal threshold value...")

            THRESHOLD = 0.00
            t_probs = np.asarray(nn.predict(X_train, batch_size = 32))
            for i in np.arange(THRESHOLD, 1.00, 0.05):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    current_f1 = threshold_f1(t_probs, THRESHOLD)
                    new_f1 = threshold_f1(t_probs, THRESHOLD + i)
                if new_f1 >= current_f1:
                    THRESHOLD = np.around(THRESHOLD + i, 2)
            print(f"Optimal threshold value is " \
                  f"{np.around(THRESHOLD * 100, 2)}%")

            # evaluate the network
            print("Calculating scores...")
            t_preds = np.asarray([multi_label_bins(prob, THRESHOLD) 
                for prob in t_probs])
            t_acc_0 = multi_label_accuracy(Y_train, t_preds, leeway = 0)
            t_acc_1 = multi_label_accuracy(Y_train, t_preds, leeway = 1)
            t_acc_2 = multi_label_accuracy(Y_train, t_preds, leeway = 2)
            t_acc_3 = multi_label_accuracy(Y_train, t_preds, leeway = 3)
            t_prec = precision_score(Y_train, t_preds, average = 'micro')
            t_rec = recall_score(Y_train, t_preds, average = 'micro')
            t_f1 = f1_score(Y_train, t_preds, average = 'micro')
            
            v_probs = np.asarray(nn.predict(X_val, batch_size = 32))
            v_preds = np.asarray([multi_label_bins(prob, THRESHOLD) 
                for prob in v_probs])
            v_acc_0 = multi_label_accuracy(Y_val, v_preds)
            v_acc_1 = multi_label_accuracy(Y_val, v_preds, leeway = 1)
            v_acc_2 = multi_label_accuracy(Y_val, v_preds, leeway = 2)
            v_acc_3 = multi_label_accuracy(Y_val, v_preds, leeway = 3)
            v_prec = precision_score(Y_val, v_preds, average = 'micro')
            v_rec = recall_score(Y_val, v_preds, average = 'micro')
            v_f1 = f1_score(Y_val, v_preds, average = 'micro')
            
            # print and store scores
            full_path = os.path.join(data_path, 'training_log.txt')
            log_text = \
f'''\n\n ~~~ {file_name} ~ {datetime.now()} ~~~
Training duration: {duration}
Neurons in each layer: {NEURONS}
Activation function: {ACTIVATION}
Input dropout: {np.around(INPUT_DROPOUT * 100, 2)}%
Hidden dropout: {np.around(HIDDEN_DROPOUT * 100, 2)}%
Pos_weight: {POS_WEIGHT}
Prediction threshold: {np.around(THRESHOLD * 100, 2)}%
Batch size: {BATCH_SIZE}
Patience: {PATIENCE}
Monitoring: {MONITORING}
Optimizer: {OPTIMIZER}

TRAINING DATA
A0 score: {np.around(t_acc_0 * 100, 2)}%
A1 score: {np.around(t_acc_1 * 100, 2)}%
A2 score: {np.around(t_acc_2 * 100, 2)}%
A3 score: {np.around(t_acc_3 * 100, 2)}%
Micro-average precision: {np.around(t_prec * 100, 2)}%
Micro-average recall: {np.around(t_rec * 100, 2)}%
Micro-average F1 score: {np.around(t_f1 * 100, 2)}%

TEST DATA
A0 score: {np.around(v_acc_0 * 100, 2)}%
A1 score: {np.around(v_acc_1 * 100, 2)}%
A2 score: {np.around(v_acc_2 * 100, 2)}%
A3 score: {np.around(v_acc_3 * 100, 2)}%
Micro-average precision: {np.around(v_prec * 100, 2)}%
Micro-average recall: {np.around(v_rec * 100, 2)}%
Micro-average F1 score: {np.around(v_f1 * 100, 2)}%\n '''

            print(log_text)
            with open(full_path, 'a+') as log_file:
                log_file.write(log_text)

            # plot the training loss and accuracy
            eff_epochs = len(H.history['loss'])
            N = np.arange(10, eff_epochs)
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(N, H.history["loss"][10:], label = "train_loss")
            plt.plot(N, H.history["val_loss"][10:], label = "val_loss")
            plt.title(file_name)
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            full_path = os.path.join(data_path, f'{file_name}_plot.png')
            plt.savefig(full_path)

            # save model
            #full_path = os.path.join(data_path, f'{file_name}_model.pickle')
            #with open(full_path, 'wb') as pickle_out:
            #    pickle.dump(nn, pickle_out)
            
