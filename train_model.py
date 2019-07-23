import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from NN import NeuralNetwork

def train_model(file_name, path = 'data', num_iterations = 25000,
    plot_cost = True):
    ''' Train homegrown neural network. '''

    nn_model = NeuralNetwork(
        layer_dims = [1024, 512, 5],
        activations = ['tanh', 'tanh', 'sigmoid'],
        learning_rate = 0.0075,
        num_iterations = num_iterations,
        cost_function = 'cross_entropy',
        plot_cost = plot_cost,
        init_method = 'he'
    )

    full_path = os.path.join(path, f"{file_name}_1hot_agg.csv")
    df_1hot_agg = pd.read_csv(full_path)

    X = np.asarray(df_1hot_agg.iloc[:, :1024].T)
    y = np.asarray(df_1hot_agg.iloc[:, :1024])
    y = y.reshape(5, y.shape[0])

    nn_model.fit(X, y)

    return nn_model

if __self__ == __main__:
    file_name = 'arxiv_sample_1000'
    path = os.path.join('/home', 'dn16382', 'pCloudDrive', 'Public Folder',
        'scholarly_data')
    train_model(file_name, path)
