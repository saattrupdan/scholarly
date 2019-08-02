import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
import sys
from pathlib import Path
import itertools as it

# homegrown neural network
from NN import NeuralNetwork
from datetime import datetime
from functools import reduce # used to calculate accuracy

# keras neural network packages
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.optimizers import SGD
from keras.optimizers import Adam 
from keras.callbacks import EarlyStopping

# custom multilabel loss function
import tensorflow as tf
import keras.backend.tensorflow_backend as tfb


# multiplier for positive targets, needs to be tuned
# POS_WEIGHT = 1 performed really well: f1 score 96.44% after 668 epochs
POS_WEIGHT = 50
def weighted_binary_crossentropy(target, output):
    """
    Weighted binary crossentropy between an output tensor 
    and a target tensor. POS_WEIGHT is used as a multiplier 
    for the positive targets.

    Combination of the following functions:
    * keras.losses.binary_crossentropy
    * keras.backend.tensorflow_backend.binary_crossentropy
    * tf.nn.weighted_cross_entropy_with_logits
    """
    # transform back to logits
    _epsilon = tfb._to_tensor(tfb.epsilon(), output.dtype.base_dtype)
    output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
    output = tf.log(output / (1 - output))
    # compute weighted loss
    loss = tf.nn.weighted_cross_entropy_with_logits(
            targets=target,
            logits=output,
            pos_weight=POS_WEIGHT
            )
    return tf.reduce_mean(loss, axis=-1)

    def multi_label_bins(prediction):
        return (prediction > max(prediction) / 2).astype('int')

    def multi_label_accuracy(y, yhat):
        assert y.shape == yhat.shape
        return sum(np.equal(y, yhat)) / len(y)


# set list of file_names
if len(sys.argv) > 1:
    file_names = sys.argv[1:]
else:
    file_names = [f'arxiv_sample_{i}' for i in
        [1000, 5000, 10000, 25000, 50000, 100000]]

home_dir = str(Path.home())
data_path = os.path.join(home_dir, "pCloudDrive", "Public Folder",
    "scholarly_data")

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
        # load test set
        full_path = os.path.join(data_path, f"arxiv_test_set_big.csv")
        test_df = pd.read_csv(full_path)
        X_test = np.asarray(test_df.iloc[:, :1024])
        Y_test = np.asarray(test_df.iloc[:, 1024:])

        # fetch data
        full_path = os.path.join(data_path, f"{file_name}_1hot.csv")
        df = pd.read_csv(full_path)
        X_train = np.asarray(df.iloc[:, :1024])
        Y_train = np.asarray(df.iloc[:, 1024:])

        max_epochs = 1000000

        nn_model = Sequential()
        nn_model.add(Dropout(0.5))
        nn_model.add(Dense(2048, activation = 'tanh'))
        nn_model.add(Dropout(0.5))
        nn_model.add(Dense(1024, activation = 'tanh'))
        nn_model.add(Dropout(0.5))
        nn_model.add(Dense(153, activation = 'sigmoid'))

        nn_model.compile(
            #loss = 'binary_crossentropy', 
            loss = weighted_binary_crossentropy, 
            #optimizer = SGD(momentum = 0.9),
            optimizer = Adam(),
            )

        early_stopping = EarlyStopping(
            monitor = 'loss',
            #monitor = 'val_loss',
            patience = 20,
            min_delta = 1e-4,
            restore_best_weights = True
            )

        H = nn_model.fit(
                X_train, 
                Y_train,
                validation_data = (X_test, Y_test),
                epochs = max_epochs,
                batch_size = 512,
                callbacks = [early_stopping]
                )

        # evaluate the network
        print("[INFO] evaluating network...")

        print("")
        print("TRAINING DATA")
        
        predictions = np.asarray(nn_model.predict(X_train, batch_size = 32))
        bin_predictions = np.asarray([multi_label_bins(prediction) 
            for prediction in predictions])
        prec = precision_score(Y_train, bin_predictions, average = 'micro')
        rec = recall_score(Y_train, bin_predictions, average = 'micro')
        f1 = f1_score(Y_train, bin_predictions, average = 'micro')

        print(f"Micro-average precision: {np.around(prec * 100, 2)}%")
        print(f"Micro-average recall: {np.around(rec * 100, 2)}%")
        print(f"Micro-average f1 score: {np.around(f1 * 100, 2)}%")
        
        print("")
        print("TEST DATA")
        
        predictions = np.asarray(nn_model.predict(X_test, batch_size = 32))
        bin_predictions = np.asarray([multi_label_bins(prediction) 
            for prediction in predictions])
        prec = precision_score(Y_test, bin_predictions, average = 'micro')
        rec = recall_score(Y_test, bin_predictions, average = 'micro')
        f1 = f1_score(Y_test, bin_predictions, average = 'micro')

        print(f"Micro-average precision: {np.around(prec * 100, 2)}%")
        print(f"Micro-average recall: {np.around(rec * 100, 2)}%")
        print(f"Micro-average f1 score: {np.around(f1 * 100, 2)}%")

        # plot the training loss and accuracy
        eff_epochs = len(H.history['val_loss'])
        N = np.arange(0, eff_epochs)
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(N, H.history["loss"], label = "train_loss")
        plt.plot(N, H.history["val_loss"], label = "val_loss")
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
        #    test_set = (X_test, Y_test),
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
        #Yhat = np.squeeze(np.around(nn_model.predict(X_test), decimals = 0))
        #Yhat = Yhat.astype('int')
        #correct_predictions = np.sum(np.asarray([reduce(
        #    lambda z, w: z and w, x) for x in np.equal(Y_test.T, Yhat.T)]))
        #test_accuracy = correct_predictions / X_test.shape[1]

        #print("Training complete!")
        #print(f"Training accuracy: {np.around(train_accuracy * 100, 2)}%")
        #print(f"Test accuracy: {np.around(test_accuracy * 100, 2)}%")
        #print(f"Time spent: {datetime.now() - start_time}")


        # save model
        #full_path = os.path.join(data_path, f'{file_name}_model.pickle')
        #with open(full_path, 'wb') as pickle_out:
        #    pickle.dump(nn_model, pickle_out)
        
