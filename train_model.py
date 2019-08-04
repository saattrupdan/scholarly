import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools as it
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
from keras.optimizers import SGD
from keras.optimizers import Adam 
from keras.callbacks import EarlyStopping

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
    ''' Transform probabilities of multi-label predictions into binary values. '''
    return (prediction > (max(prediction) * threshold)).astype('int')

def multi_label_accuracy(y, yhat):
    '''
    Compute accuracy of multi-label predictions, where to be counted as a
    correct prediction, all the labels have to be predicted correctly.
    '''
    assert y.shape == yhat.shape
    accuracies = np.asarray([reduce(lambda z, w: z and w, x) for x in np.equal(yhat, y)])
    return np.average(accuracies)


# set list of file_names
if len(sys.argv) > 1:
    file_names = sys.argv[1:]
else:
    file_names = [f'arxiv_sample_{i}' for i in
        [1000, 5000, 10000, 25000, 50000]]#, 100000]]

home_dir = str(Path.home())
data_path = os.path.join(home_dir, "pCloudDrive", "Public Folder",
    "scholarly_data")


### Hyperparameters ###

# multiplier for positive targets in binary cross entropy
# forces higher recall and lower precision
# more complex model -> larger multiplier
POS_WEIGHT = 10

for k in np.arange(4, 15):
    POS_WEIGHT = k
    for file_name in file_names:
        print("------------------------------------")
        print(f"NOW PROCESSING: {file_name}")
        print("------------------------------------")

        full_path = os.path.join(data_path,
            f"{file_name}_model.pickle")
        if os.path.isfile(full_path):
            with open(full_path, 'rb') as pickle_in:
                nn_model = pickle.load(pickle_in)
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

            nn_model = Sequential()
            nn_model.add(Dropout(0.5))
            nn_model.add(Dense(1024, activation = 'tanh'))
            nn_model.add(Dropout(0.5))
            nn_model.add(Dense(1024, activation = 'tanh'))
            nn_model.add(Dropout(0.5))
            nn_model.add(Dense(153, activation = 'sigmoid'))

            nn_model.compile(
                #loss = 'binary_crossentropy', 
                loss = weighted_binary_crossentropy, 
                optimizer = Adam(),
                )

            early_stopping = EarlyStopping(
                #monitor = 'loss',
                monitor = 'val_loss',
                patience = 50,
                min_delta = 1e-4,
                restore_best_weights = True
                )

            H = nn_model.fit(
                    X_train, 
                    Y_train,
                    validation_data = (X_val, Y_val),
                    epochs = max_epochs,
                    batch_size = 512,
                    callbacks = [early_stopping]
                    )

            # find optimal threshold value based on training data
            print("")
            print("Finding optimal threshold value... ")

            def get_acc(probabilities, threshold):
                ''' Get multi-label accuracy from probabilities and threshold. '''
                predictions = np.asarray([multi_label_bins(prob, threshold) 
                    for prob in probabilities])
                return multi_label_accuracy(Y_train, predictions)
            
            t_probs = np.asarray(nn_model.predict(X_train, batch_size = 32))
            
            THRESHOLD = 0.25
            for i in np.arange(THRESHOLD, 1.00, 0.05):
                if get_acc(t_probs, THRESHOLD + i) >= get_acc(t_probs, THRESHOLD):
                    THRESHOLD = np.around(THRESHOLD + i, 2)
            print(f"Optimal threshold value is {np.around(THRESHOLD * 100, 2)}%")

            # evaluate the network
            t_preds = np.asarray([multi_label_bins(prob, THRESHOLD) 
                for prob in t_probs])

            t_acc = multi_label_accuracy(Y_train, t_preds)
            t_prec = precision_score(Y_train, t_preds, average = 'micro')
            t_rec = recall_score(Y_train, t_preds, average = 'micro')
            t_f1 = f1_score(Y_train, t_preds, average = 'micro')
            
            v_probs = np.asarray(nn_model.predict(X_val, batch_size = 32))
            v_preds = np.asarray([multi_label_bins(prob, THRESHOLD) 
                for prob in v_probs])

            v_acc = multi_label_accuracy(Y_val, v_preds)
            v_prec = precision_score(Y_val, v_preds, average = 'micro')
            v_rec = recall_score(Y_val, v_preds, average = 'micro')
            v_f1 = f1_score(Y_val, v_preds, average = 'micro')
            
            print("")
            print("TRAINING DATA")
            print(f"Multi-label accuracy: {np.around(t_acc * 100, 2)}%")
            print(f"Micro-average precision: {np.around(t_prec * 100, 2)}%")
            print(f"Micro-average recall: {np.around(t_rec * 100, 2)}%")
            print(f"Micro-average f1 score: {np.around(t_f1 * 100, 2)}%")
            print("")
            print("TEST DATA")
            print(f"Multi-label accuracy: {np.around(v_acc * 100, 2)}%")
            print(f"Micro-average precision: {np.around(v_prec * 100, 2)}%")
            print(f"Micro-average recall: {np.around(v_rec * 100, 2)}%")
            print(f"Micro-average f1 score: {np.around(v_f1 * 100, 2)}%")

            # store the scores in a log file
            full_path = os.path.join(data_path, 'training_log.txt')
            with open(full_path, 'a+') as log_file:
                log_file.write(f"~~~ {file_name} ~ {datetime.now()} ~~~\n")
                log_file.write(f"Pos_weight value: {POS_WEIGHT}\n")
                log_file.write(f"Threshold value: {np.around(THRESHOLD * 100, 2)}%\n")
                log_file.write("\n")
                log_file.write("TRAINING DATA\n")
                log_file.write(f"Multi-label accuracy: {np.around(t_acc * 100, 2)}%\n")
                log_file.write(f"Micro-average precision: {np.around(t_prec * 100, 2)}%\n")
                log_file.write(f"Micro-average recall: {np.around(t_rec * 100, 2)}%\n")
                log_file.write(f"Micro-average f1 score: {np.around(t_f1 * 100, 2)}%\n")
                log_file.write("\n")
                log_file.write("TEST DATA\n")
                log_file.write(f"Multi-label accuracy: {np.around(v_acc * 100, 2)}%\n")
                log_file.write(f"Micro-average precision: {np.around(v_prec * 100, 2)}%\n")
                log_file.write(f"Micro-average recall: {np.around(v_rec * 100, 2)}%\n")
                log_file.write(f"Micro-average f1 score: {np.around(v_f1 * 100, 2)}%\n")
                log_file.write("\n\n")

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

 
        ### HOMEGROWN NEURAL NETWORK ###
        
        #nn_model = NeuralNetwork(
        #    layer_dims = [30, 5],
        #    activations = ['tanh', 'sigmoid'],
        #    target_accuracy = 0.65,
        #    test_set = (X_val, Y_val),
        #    plots = True 
        #    )
        
        # time training for good measure
        #start_time = datetime.now()
    
        #nn_model.fit(X)
        #nn_model.train(X, Y)

        # calculate training accuracy
        #Yhat = np.squeeze(np.around(nn_model.predict(X), decimals = 0))
        #Yhat = Yhat.astype('int')
        #correct_predictions = np.sum(np.asarray(
        #    [reduce(lambda z, w: z and w, x) for x in np.equal(Y.T, Yhat.T)]))
        #train_accuracy = correct_predictions / X.shape[1]

        ## calculate test accuracy
        #Yhat = np.squeeze(np.around(nn_model.predict(X_val), decimals = 0))
        #Yhat = Yhat.astype('int')
        #correct_predictions = np.sum(np.asarray([reduce(
        #    lambda z, w: z and w, x) for x in np.equal(Y_val.T, Yhat.T)]))
        #test_accuracy = correct_predictions / X_val.shape[1]

        #print("Training complete!")
        #print(f"Training accuracy: {np.around(train_accuracy * 100, 2)}%")
        #print(f"Test accuracy: {np.around(test_accuracy * 100, 2)}%")
        #print(f"Time spent: {datetime.now() - start_time}")


        # save model
        #full_path = os.path.join(data_path, f'{file_name}_model.pickle')
        #with open(full_path, 'wb') as pickle_out:
        #    pickle.dump(nn_model, pickle_out)
        
