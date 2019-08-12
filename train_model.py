import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools as it
import warnings
from functools import reduce
from datetime import datetime

import os
import sys
import pickle
from pathlib import Path

# neural network packages
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.utils import Sequence

# used for custom multilabel loss function
import tensorflow.keras.backend as tfb

# scores to evaluate models
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


class BatchGenerator(Sequence):
    ''' Generator clas for training neural networks in batches. '''
    
    def __init__(self, data_path, file_name, labels, batch_size):
        self.data_path = data_path
        self.file_name = file_name
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        ''' Return number of batches per epoch. '''
        return np.ceil(self.labels.shape[0] / self.batch_size).astype('int')

    def __getitem__(self, idx):
        ''' Return batch of a given index. '''
        X_path = os.path.join(self.data_path, f"{self.file_name}_elmo.csv")
        X_batch = pd.read_csv(
                    X_path,
                    skiprows = idx * self.batch_size,
                    nrows = self.batch_size,
                    na_filter = False
                    )
        Y_batch = self.labels[idx * self.batch_size : (idx + 1) * self.batch_size, :]
        return (X_batch, Y_batch)


def weighted_binary_crossentropy(target, output, pos_weight = 1):
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
    _epsilon = tfb.epsilon()
    output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
    output = tf.log(output / (1 - output))

    # compute weighted loss
    loss = tf.nn.weighted_cross_entropy_with_logits(
            targets = target,
            logits = output,
            pos_weight = pos_weight
            )

    return tf.reduce_mean(loss, axis = -1)

def multi_label_accuracy(y, yhat, leeway = 0):
    '''
    Compute accuracy of multi-label predictions, where to be counted as a
    correct prediction, there can be at most leeway many mislabellings.
    '''
    assert np.asarray(y).shape == np.asarray(yhat).shape
    leeway_bool = lambda x: (sum(x) + leeway) >= len(x)
    accuracies = np.asarray([leeway_bool(x) for x in np.equal(yhat, y)])
    return np.average(accuracies)

def multi_label_bins(prediction, threshold = 0.5):
    ''' Turn probabilities of multi-label predictions into binary values. '''
    return (prediction > (max(prediction) * threshold)).astype('int')

def threshold_f1(probabilities, threshold, Y_train):
    ''' Get F1 score from probabilities and threshold. '''
    predictions = np.asarray([multi_label_bins(prob, threshold) 
        for prob in probabilities])
    return f1_score(Y_train, predictions, average = 'micro')
            

def train_model(file_names, info, data_path = 'data', POS_WEIGHT = 5,
    ACTIVATION = 'elu', NEURONS = [1024, 1024], INPUT_DROPOUT = 0.2,
    HIDDEN_DROPOUT = 0.5, MONITORING = 'val_loss', PATIENCE = 20,
    BATCH_SIZE = 512, OPTIMIZER = 'nadam', MAX_EPOCHS = 1000000,
    save_model = False):

    # load validation set
    X_path = os.path.join(data_path, f'arxiv_val_big_elmo.csv')
    Y_path = os.path.join(data_path, f'arxiv_val_big_{info["1hot"]}.csv')
    X_val = np.asarray(pd.read_csv(X_path))
    Y_val = np.asarray(pd.read_csv(Y_path))[:, 1:]

    print(X_val.shape)

    for file_name in file_names:
        print("------------------------------------")
        print(f"NOW PROCESSING: {file_name}")
        print("------------------------------------")

        full_path = os.path.join(data_path,
            f"{file_name}_model.pickle")
        if os.path.isfile(full_path) and save_model:
            with open(full_path, 'rb') as pickle_in:
                nn = pickle.load(pickle_in)
            print("Model already trained.")
        else:
            # initialise neural network
            inputs = Input(shape = (1024,))
            x = Dropout(INPUT_DROPOUT)(inputs)
            for neurons in NEURONS:
                x = Dense(neurons, activation = ACTIVATION)(x)
                x = Dropout(HIDDEN_DROPOUT)(x)
            outputs = Dense(info['outputs'], activation = 'sigmoid')(x)

            nn = Model(inputs = inputs, outputs = outputs)
            
            # make keras recognise custom loss when importing models
            loss_fn = lambda x, y: weighted_binary_crossentropy(x, y, POS_WEIGHT)
            get_custom_objects().update(
                {'weighted_binary_crossentropy': loss_fn}
                )

            nn.compile(
                loss = 'weighted_binary_crossentropy',
                optimizer = OPTIMIZER,
                )

            early_stopping = EarlyStopping(
                monitor = MONITORING,
                patience = PATIENCE,
                min_delta = 1e-4,
                restore_best_weights = True
                )

            # get training labels
            full_path = os.path.join(data_path, f"{file_name}_{info['1hot']}.csv")
            Y_train = np.asarray(pd.read_csv(full_path))[:, 1:]

            # train neural network
            past = datetime.now()
            batch_gen = BatchGenerator(
                            data_path = data_path,
                            file_name = file_name,
                            labels = Y_train,
                            batch_size = BATCH_SIZE,
                            )
            H = nn.fit_generator(
                generator = batch_gen,
                validation_data = (X_val, Y_val),
                epochs = MAX_EPOCHS,
                callbacks = [early_stopping],
                use_multiprocessing = True
                )
            duration = datetime.now() - past
    
            print("")
            print("Producing predictions...")
            full_path = os.path.join(data_path, f'{file_name}_elmo.csv')
            batches = pd.read_csv(full_path, chunksize = 512)
            t_probs = np.concatenate(
                [np.asarray(nn.predict(batch, batch_size = 32))
                for batch in batches]
                )

            # find optimal threshold value based on training data
            print("Finding optimal threshold value...")
            current_f1 = threshold_f1(t_probs, 0.00, Y_train)
            step_size = 0.10
            THRESHOLD = 0.00
            for threshold in np.arange(THRESHOLD, 1.00, step_size):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    new_f1 = threshold_f1(t_probs, threshold, Y_train)
                if new_f1 >= current_f1:
                    current_f1 = new_f1
                else:
                    THRESHOLD = threshold - step_size
                    break
            print(f"Optimal threshold value is {np.around(THRESHOLD * 100, 2)}%")

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
f'''\n\n~~~ {file_name}_{info['1hot']} ~~~
Datetime = {datetime.now()}
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

TRAINING SET 
A0 score: {np.around(t_acc_0 * 100, 2)}%
A1 score: {np.around(t_acc_1 * 100, 2)}%
A2 score: {np.around(t_acc_2 * 100, 2)}%
A3 score: {np.around(t_acc_3 * 100, 2)}%
Micro-average precision: {np.around(t_prec * 100, 2)}%
Micro-average recall: {np.around(t_rec * 100, 2)}%
Micro-average F1 score: {np.around(t_f1 * 100, 2)}%

VALIDATION SET 
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

            # save model
            if save_model:
                full_path = os.path.join(data_path, f'{file_name}_model.pickle')
                with open(full_path, 'wb') as pickle_out:
                    pickle.dump(nn, pickle_out)


if __name__ == '__main__':

    home_dir = str(Path.home())
    data_path = os.path.join(home_dir, "pCloudDrive", "public_folder", "scholarly_data")
    
    # set list of file_names
    if len(sys.argv) > 1:
        file_names = sys.argv[1:]
    else:
        file_names = [f'arxiv_sample_{i}' for i in
            [1000, 5000, 10000, 25000]]

    # reminder: 1hot has 153 cats, 1hot_agg has 7 cats
    train_model(
        file_names,
        info = {'val' : 'arxiv_val_big_1hot_agg', '1hot' : '1hot_agg', 'outputs' : 7},
        data_path = data_path,
        save_model = False,
        POS_WEIGHT = 1,
        ACTIVATION = 'elu',
        NEURONS = [64],
        INPUT_DROPOUT = 0.2,
        HIDDEN_DROPOUT = 0.0,
        MONITORING = 'val_loss',
        PATIENCE = 10,
        BATCH_SIZE = 512,
        OPTIMIZER = 'nadam',
        MAX_EPOCHS = 1000000
        )
