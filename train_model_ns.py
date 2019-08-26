# core packages
import numpy as np
import pandas as pd
import os

# hyperparameter tuning
import naturalselection as ns

# dealing with .pickle and .npz files
import pickle
from scipy.sparse import save_npz, load_npz

def train_model(file_name, data_path = 'data', val_name = 'arxiv_val',
    verbose = 0):
    
    # set up paths
    tfidf_model_fname = f'{file_name}_tfidf_model.pickle'
    tfidf_model_path = os.path.join(data_path, tfidf_model_fname)
    val_path = os.path.join(data_path, f"{val_name}_labels_agg.csv")
    tfidf_path = os.path.join(data_path, f"{file_name}_tfidf.npz")
    labels_path = os.path.join(data_path,f"{file_name}_labels_agg.csv")
    log_path = os.path.join(data_path, 'training_log.txt')
    plot_path = os.path.join(data_path, f'{file_name}_plot.png')
    nn_path = os.path.join(data_path, f"{file_name}_nn.h5")
    nn_data_path = os.path.join(data_path, f"{file_name}_nn_data.pickle")

    # get training data
    X_train = load_npz(tfidf_path)
    Y_train = np.asarray(pd.read_csv(labels_path, header = None))
    Y_train = Y_train[:, 1:].astype('int8')

    # load tf-idf model 
    with open(tfidf_model_path, 'rb+') as pickle_in:
        tfidf = pickle.load(pickle_in)

    # load validation set
    X_val = np.asarray(pd.read_csv(val_path))[:, 0]
    X_val = tfidf.transform(X_val)
    Y_val = np.asarray(pd.read_csv(val_path))[:, 1:].astype('int8')

    fitness_fn = ns.get_nn_fitness_fn(
        train_val_sets = (X_train, Y_train, X_val, Y_val),
        loss_fn = 'binary_crossentropy',
        score = 'f1',
        output_activation = 'sigmoid',
        max_training_time = 60
        )

    fnns = ns.Population(
        genus = ns.FNN(),
        fitness_fn = fitness_fn,
        size = 50
        )

    history = fnns.evolve(
        generations = 20,
        multiprocessing = False,
        verbose = verbose
        )
    
    print("Best overall genome is:")
    print(history.fittest)

    history.plot()


if __name__ == '__main__':

    # used to get current directory
    from pathlib import Path

    home_dir = str(Path.home())
    data_path = os.path.join(home_dir, "pCloudDrive", "public_folder",
        "scholarly_data")
    file_name = 'arxiv_sample_1000'

    # Reminder: there are 153 labels and 7 aggregated labels
    #
    # Settings that seem to work well for the aggregated labels:
    #     activation  = elu
    #     optimizer   = adam
    #     neurons     = around [64, 64]
    #     pos_weight  = between 1 and 2
    #     batch size  = 32
    #     dropout     = input around 20% and hidden around 25%

    train_model(
        file_name = file_name,
        val_name = 'arxiv_val',
        data_path = data_path,
        verbose = 2
        )
