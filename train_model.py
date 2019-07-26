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

# some neural network hyperparameters
iterations = 100000
learning_rate = 0.005
old_weights = None

# load weights from a pre-trained model    
full_path = os.path.join(data_path,
    f"arxiv_sample_1000_model_25000_0.005.pickle")
with open(full_path, 'rb') as pickle_in:
    nn_model = pickle.load(pickle_in)
    old_weights = nn_model.params_
    
for file_name in file_names:
    print("------------------------------------")
    print(f"NOW PROCESSING: {file_name}")
    print("------------------------------------")

    full_path = os.path.join(data_path,
        f"{file_name}_model_{iterations}_{learning_rate}.pickle")
    if os.path.isfile(full_path):
        with open(full_path, 'rb') as pickle_in:
            nn_model = pickle.load(pickle_in)
        print("Model already trained.")
    else:
        # time training for good measure
        start_time = datetime.now()
        
        # fetch data
        full_path = os.path.join(data_path, f"{file_name}_1hot_agg.csv")
        df_1hot_agg = pd.read_csv(full_path)
        X = np.asarray(df_1hot_agg.iloc[:, :1024]).T
        y = np.asarray(df_1hot_agg.iloc[:, 1024:]).T
        
        # build homegrown neural network
        nn_model = NeuralNetwork(
            layer_dims = [750, 500, 5],
            activations = ['tanh', 'tanh', 'sigmoid'],
            early_stopping = True,
            plot_cost = False,
            num_iterations = iterations,
            learning_rate = learning_rate
            )
        
        # fit the neural network to X and initialise params
        nn_model.fit(X)

        # transfer old weights to new model, if applicable
        if old_weights:
            nn_model.params_ = old_weights

        # train the neural network
        nn_model.train(X, y)
        
        # save model
        full_path = os.path.join(data_path,
            f'{file_name}_model_{iterations}_{learning_rate}.pickle')
        with open(full_path, 'wb') as pickle_out:
            pickle.dump(nn_model, pickle_out)

        # calculate training accuracy
        yhat = np.squeeze(np.around(nn_model.predict(X), decimals = 0))
        yhat = yhat.astype('int')
        correct_predictions = np.sum(np.asarray(
            [reduce(lambda z, w: z and w, x) for x in np.equal(y.T, yhat.T)]))
        train_accuracy = correct_predictions / X.shape[1]

        print("Training complete!")
        print(f"Training accuracy: {train_accuracy}")
        print(f"Time spent: {datetime.now() - start_time}")

    # save the weights for transfer learning
    old_weights = nn_model.params_
