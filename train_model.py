import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from NN import NeuralNetwork

nn_model = NeuralNetwork(
    layer_dims = [256, 1],
    activations = ['tanh', 'tanh', 'sigmoid'],
    alpha = 0.01,
    num_iterations = 100000,
    plot_cost = True,
    init_method = 'he'
)

file_name = "arxiv_sample_25000"
data_path = os.path.join("P:/", "Public Folder", "scholarly_data")
full_path = os.path.join(data_path, f"{file_name}_1hot_agg.csv")

df_1hot_agg = pd.read_csv(full_path)

X = np.asarray(df_1hot_agg.iloc[:, :1024].T)
y = np.asarray(df_1hot_agg.loc[:, 'physics'])
y = y.reshape(1, y.size)

nn_model.fit(X, y)
