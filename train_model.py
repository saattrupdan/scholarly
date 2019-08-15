# core packages
import numpy as np
import pandas as pd
import os
import sys

# plots
import matplotlib.pyplot as plt

# dealing with .pickle and .npz files
import pickle
from scipy.sparse import save_npz, load_npz

# used to get current time
from datetime import datetime

# used to suppress warnings
import warnings

# used to get current directory
import pathlib

# neural network packages
import tensorflow.compat.v1 as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras import backend as tfb
from sklearn.metrics import precision_score, recall_score, f1_score

# disable tensorflow INFO and WARNING messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def weighted_binary_crossentropy(target, output, pos_weight = 1):
    ''' Weighted binary crossentropy between an output tensor 
    and a target tensor, where pos_weight is used as a multiplier 
    for the positive targets. '''

    # transform back to logits
    _epsilon = tfb.epsilon()
    output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
    output = tf.log(output / (1 - output))

    # compute weighted loss
    loss = tf.nn.weighted_cross_entropy_with_logits(
            labels = target,
            logits = output,
            pos_weight = pos_weight
            )

    return tf.reduce_mean(loss, axis = -1)

def multilabel_accuracy(y, yhat, leeway = 0):
    ''' Compute accuracy of multi-label predictions, where to be counted as
    a correct prediction, there can be at most leeway many mislabellings. '''

    assert np.asarray(y).shape == np.asarray(yhat).shape
    leeway_bool = lambda x: (sum(x) + leeway) >= len(x)
    accuracies = np.asarray([leeway_bool(x) for x in np.equal(yhat, y)])
    return np.average(accuracies)

def multilabel_bins(probabilities, threshold = 0.5):
    ''' Turn probabilities of multi-label predictions into binary values. '''
    return (probabilities > (max(probabilities) * threshold)).astype('int8')

def threshold_f1(probabilities, threshold, Y_train):
    ''' Get F1 score from probabilities and threshold. '''
    predictions = np.asarray([multilabel_bins(prob, threshold) 
        for prob in probabilities])
    return f1_score(Y_train, predictions, average = 'micro')

def train_model(file_names, labels_name, no_labels, data_path = 'data',
    pos_weight = 5, activation = 'elu', neurons = [1024, 1024],
    input_dropout = 0.2, hidden_dropout = 0.5, monitoring = 'val_loss',
    patience = 20, batch_size = 512, optimizer = 'nadam',
    max_epochs = 1000000, val_name = 'arxiv_val'):
    
    for file_name in file_names:
        print("------------------------------------")
        print(f"NOW PROCESSING: {file_name}")
        print("------------------------------------")

        # set up paths
        predictor_fname = f"{file_name}_{labels_name}_predictor.pickle"
        predictor_path = os.path.join(data_path, predictor_fname)
        tfidf_model_fname = f'{file_name}_tfidf_model.pickle'
        tfidf_model_path = os.path.join(data_path, tfidf_model_fname)
        val_path = os.path.join(data_path, f"{val_name}_{labels_name}.csv")
        tfidf_path = os.path.join(data_path, f"{file_name}_tfidf.npz")
        labels_path = os.path.join(data_path,f"{file_name}_{labels_name}.csv")
        log_path = os.path.join(data_path, 'training_log.txt')
        plot_path = os.path.join(data_path, f'{file_name}_plot.png')

        if os.path.isfile(predictor_path):
            print('Model already exists.')
            continue

        # load tf-idf model 
        with open(tfidf_model_path, 'rb+') as pickle_in:
            tfidf = pickle.load(pickle_in)

        # load validation set
        X_val = np.asarray(pd.read_csv(val_path))[:, 0]
        X_val = tfidf.transform(X_val)
        Y_val = np.asarray(pd.read_csv(val_path))[:, 1:].astype('int8')

        # initialise neural network
        inputs = Input(shape = (4096,))
        x = Dropout(input_dropout)(inputs)
        for n in neurons:
            x = Dense(n, activation = activation)(x)
            x = Dropout(hidden_dropout)(x)
        outputs = Dense(no_labels, activation = 'sigmoid')(x)

        nn = Model(inputs = inputs, outputs = outputs)
        
        # make keras recognise custom loss when importing models
        loss_fn = lambda x, y: weighted_binary_crossentropy(x, y,
            pos_weight)
        get_custom_objects().update(
            {'weighted_binary_crossentropy': loss_fn}
            )

        nn.compile(
            loss = 'weighted_binary_crossentropy',
            optimizer = optimizer,
            )

        early_stopping = EarlyStopping(
            monitor = monitoring,
            patience = patience,
            min_delta = 1e-4,
            restore_best_weights = True
            )

        # get data
        X_train = load_npz(tfidf_path)
        Y_train = np.asarray(pd.read_csv(labels_path, header = None))
        Y_train = Y_train[:, 1:].astype('int8')
        
        # train neural network
        past = datetime.now()
        H = nn.fit(
            X_train,
            Y_train,
            batch_size = batch_size,
            validation_data = (X_val, Y_val),
            epochs = max_epochs,
            callbacks = [early_stopping],
            )
        duration = datetime.now() - past

        print("")
        print("Producing predictions...")
        t_probs = np.asarray(nn.predict(X_train, batch_size = 32))

        # find optimal threshold value based on training data
        print("Finding optimal threshold value...")
        current_f1 = threshold_f1(t_probs, 0.00, Y_train)
        step_size = 0.10
        threshold = 0.00
        for candidate in np.arange(threshold, 1.00, step_size):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                new_f1 = threshold_f1(t_probs, candidate, Y_train)
            if new_f1 >= current_f1:
                current_f1 = new_f1
            else:
                threshold = candidate - step_size
                break
        print(f"Optimal threshold value is {np.around(threshold * 100, 2)}%")

        # evaluate the network
        print("Calculating scores...")
        t_preds = np.asarray([multilabel_bins(prob, threshold) 
            for prob in t_probs])
        t_acc_0 = multilabel_accuracy(Y_train, t_preds, leeway = 0)
        t_acc_1 = multilabel_accuracy(Y_train, t_preds, leeway = 1)
        t_acc_2 = multilabel_accuracy(Y_train, t_preds, leeway = 2)
        t_acc_3 = multilabel_accuracy(Y_train, t_preds, leeway = 3)
        t_prec = precision_score(Y_train, t_preds, average = 'micro')
        t_rec = recall_score(Y_train, t_preds, average = 'micro')
        t_f1 = f1_score(Y_train, t_preds, average = 'micro')
        
        v_probs = np.asarray(nn.predict(X_val, batch_size = 32))
        v_preds = np.asarray([multilabel_bins(prob, threshold) 
            for prob in v_probs])
        v_acc_0 = multilabel_accuracy(Y_val, v_preds)
        v_acc_1 = multilabel_accuracy(Y_val, v_preds, leeway = 1)
        v_acc_2 = multilabel_accuracy(Y_val, v_preds, leeway = 2)
        v_acc_3 = multilabel_accuracy(Y_val, v_preds, leeway = 3)
        v_prec = precision_score(Y_val, v_preds, average = 'micro')
        v_rec = recall_score(Y_val, v_preds, average = 'micro')
        v_f1 = f1_score(Y_val, v_preds, average = 'micro')
        
        # print and store scores
        log_text = \
f'''\n\n~~~ {file_name}_{labels_name} ~~~
Datetime = {datetime.now()}
Training duration: {duration}
Neurons in each layer: {neurons}
Activation function: {activation}
Input dropout: {np.around(input_dropout * 100, 2)}%
Hidden dropout: {np.around(hidden_dropout * 100, 2)}%
Pos_weight: {pos_weight}
Prediction threshold: {np.around(threshold * 100, 2)}%
Batch size: {batch_size}
Patience: {patience}
Monitoring: {monitoring}
Optimizer: {optimizer}

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
        with open(log_path, 'a+') as log_file:
            log_file.write(log_text)

        # plot the training loss and accuracy
        eff_epochs = len(H.history['loss'])
        N = range(0, eff_epochs)
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(N, H.history["loss"], label = "train_loss")
        plt.plot(N, H.history["val_loss"], label = "val_loss")
        plt.title(file_name)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(plot_path)

        # save nn
        nn_path = os.path.join(data_path, f"{file_name}_nn.h5")
        nn.save(nn_path)

        # save nn data
        nn_data_path = os.path.join(data_path, f"{file_name}_nn_data.pickle")
        nn_data = {'threshold' : threshold}
        with open(nn_data_path, 'wb+') as pickle_out:
            pickle.dump(nn_data, pickle_out)


if __name__ == '__main__':

    home_dir = str(pathlib.Path.home())
    data_path = os.path.join(home_dir, "pCloudDrive", "public_folder",
        "scholarly_data")
    
    # set list of file_names
    if len(sys.argv) > 1:
        file_names = sys.argv[1:]
    else:
        file_names = [f'arxiv_sample_{i}' for i in
            [1000, 5000, 10000, 25000, 50000, 100000, 200000,
             500000, 750000]] + ['arxiv']

    # reminder: 1hot has 153 cats, 1hot_agg has 7 cats
    #           [1024, 1024] neurons is good for 1hot
    #           [64, 64] neurons is good for 1hot_agg
    train_model(
        file_names,
        val_name = 'arxiv_val',
        labels_name = '1hot_agg',
        no_labels = 7,
        data_path = data_path,
        pos_weight = 1,
        activation = 'elu',
        neurons = [64, 64],
        input_dropout = 0.2,
        hidden_dropout = 0.5,
        monitoring = 'val_loss',
        patience = 10,
        batch_size = 1024,
        optimizer = 'nadam',
        max_epochs = 1000000
        )
