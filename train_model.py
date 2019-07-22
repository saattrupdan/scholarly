import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from NN import NeuralNetwork

nn_model = NeuralNetwork(
    layer_dims = [256, 1],
    activations = ['tanh', 'tanh', 'sigmoid'],
    learning_rate = 0.01,
    num_iterations = 100000,
    plot_cost = True,
    init_method = 'he'
)

home_dir = str(Path.home())
file_name = "arxiv_sample_1000"
data_path = os.path.join(home_dir, "pCloudDrive", "Public Folder", "scholarly_data")
full_path = os.path.join(data_path, f"{file_name}_1hot_agg.csv")

df_1hot_agg = pd.read_csv(full_path)

X = np.asarray(df_1hot_agg.iloc[:, :1024].T)
y = np.asarray(df_1hot_agg.loc[:, 'physics'])
y = y.reshape(1, y.size)

nn_model.fit(X, y)
