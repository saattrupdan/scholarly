import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
import sys
from pathlib import Path
from datetime import datetime
from functools import reduce # used to calculate accuracy

from NN import NeuralNetwork

# set list of file_names
if len(sys.argv) > 1:
    file_names = sys.argv[1:]
else:
    file_names = [f'arxiv_sample_{i}' for i in
        [1000, 5000, 10000, 25000, 50000, 100000]]

home_dir = str(Path.home())
data_path = os.path.join(home_dir, "pCloudDrive", "Public Folder",
    "scholarly_data")

old_weights = None

for file_name in file_names:
    print("------------------------------------")
    print(f"NOW PROCESSING: {file_name}")
    print("------------------------------------")

    full_path = os.path.join(data_path,
        f"{file_name}_model.pickle")
    if os.path.isfile(full_path):
        with open(full_path, 'rb') as pickle_in:
            nn_model = pickle.load(pickle_in)
            old_weights = nn_model.params_
        print("Model already trained.")
    else:
        # time training for good measure
        start_time = datetime.now()

        # fetch test sets
        full_path = os.path.join(data_path, f"arxiv_test_1hot_agg.csv")
        df_1hot_agg = pd.read_csv(full_path)
        X_test = np.asarray(df_1hot_agg.iloc[:, :1024]).T
        Y_test = np.asarray(df_1hot_agg.iloc[:, 1024:]).T

        # build homegrown neural network
        nn_model = NeuralNetwork(
            layer_dims = [30, 5],
            activations = ['tanh', 'sigmoid'],
            target_accuracy = 0.95,
            test_set = (X_test, Y_test)
            )
        
        # fetch data
        full_path = os.path.join(data_path, f"{file_name}_1hot_agg.csv")
        df_1hot_agg = pd.read_csv(full_path)
        X = np.asarray(df_1hot_agg.iloc[:, :1024]).T
        Y = np.asarray(df_1hot_agg.iloc[:, 1024:]).T
        
        # fit the neural network to X and initialise params
        nn_model.fit(X)

        # transfer learning
        #if old_weights:
        #    nn_model.params_ = old_weights

        # train the neural network
        nn_model.train(X, Y)
        
        # save model
        full_path = os.path.join(data_path,
            f'{file_name}_model.pickle')
        with open(full_path, 'wb') as pickle_out:
            pickle.dump(nn_model, pickle_out)

        # calculate training accuracy
        Yhat = np.squeeze(np.around(nn_model.predict(X), decimals = 0))
        Yhat = Yhat.astype('int')
        correct_predictions = np.sum(np.asarray(
            [reduce(lambda z, w: z and w, x) for x in np.equal(Y.T, Yhat.T)]))
        train_accuracy = correct_predictions / X.shape[1]

        # calculate test accuracy
        Yhat = np.squeeze(np.around(nn_model.predict(X_test), decimals = 0))
        Yhat = Yhat.astype('int')
        correct_predictions = np.sum(np.asarray([reduce(
            lambda z, w: z and w, x) for x in np.equal(Y_test.T, Yhat.T)]))
        test_accuracy = correct_predictions / X_test.shape[1]

        print("Training complete!")
        print(f"Training accuracy: {np.around(train_accuracy * 100, 2)}%")
        print(f"Test accuracy: {np.around(test_accuracy * 100, 2)}%")
        print(f"Time spent: {datetime.now() - start_time}")
