import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle

from NN import NeuralNetwork


def train_model(file_name, path = 'data', num_iterations = 25000,
    plot_cost = True):
    ''' Train homegrown neural network. '''

    nn_model = NeuralNetwork(
        layer_dims = [500, 5],
        activations = ['tanh', 'sigmoid'],
        learning_rate = 0.0075,
        num_iterations = num_iterations,
        cost_function = 'cross_entropy',
        plot_cost = plot_cost,
        init_method = 'he'
    )

    full_path = os.path.join(path, f"{file_name}_1hot_agg.csv")
    df_1hot_agg = pd.read_csv(full_path)

    X = np.asarray(df_1hot_agg.iloc[:, :1024]).T
    y = np.asarray(df_1hot_agg.iloc[:, 1024:]).T

    nn_model.fit(X, y)

    print("") # deal with \r
    print("Training complete!")

    return nn_model


path = os.path.join('/home', 'dn16382', 'pCloudDrive', 'Public Folder',
    'scholarly_data')
file_names = [f'arxiv_sample_{i}' for i in
    [1000, 5000, 10000, 25000, 50000, 100000]]

for file_name in file_names:
    print("------------------------------------")
    print(f"NOW PROCESSING: {file_name}")
    print("------------------------------------")
    iterations = 1000
    model = train_model(file_name, path, num_iterations = iterations, plot_cost = False)
    full_path = os.path.join(path, f'{file_name}_model_{iterations}.pickle')
    with open(full_path, 'wb') as pickle_out:
        pickle.dump(model, pickle_out)
