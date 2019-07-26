import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
import sys
from pathlib import Path

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

# neural network hyperparameters
iterations = 100000
learning_rate = 0.005
    
for file_name in file_names:
    print("------------------------------------")
    print(f"NOW PROCESSING: {file_name}")
    print("------------------------------------")

    full_path = os.path.join(data_path,
        f"{file_name}_model_{iterations}_{learning_rate}.csv")
    if os.path.isfile(full_path):
        print("Model already trained.")
    else:
        full_path = os.path.join(data_path, f"{file_name}_1hot_agg.csv")
        df_1hot_agg = pd.read_csv(full_path)

        X = np.asarray(df_1hot_agg.iloc[:, :1024]).T
        y = np.asarray(df_1hot_agg.iloc[:, 1024:]).T
        
        nn_model = NeuralNetwork(
            layer_dims = [750, 500, 5],
            activations = ['tanh', 'tanh', 'sigmoid'],
            early_stopping = True,
            plot_cost = False,
            num_iterations = iterations,
            learning_rate = learning_rate,
            )

        nn_model.fit(X, y)

        full_path = os.path.join(data_path,
            f'{file_name}_model_{iterations}_{learning_rate}.pickle')
        with open(full_path, 'wb') as pickle_out:
            pickle.dump(nn_model, pickle_out)
    
        print("Training complete!")
