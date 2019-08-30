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
    verbose = 0, pop_size = 50, generations = 20):
    
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

    fnns = ns.FNNs(
        size = pop_size,
        train_val_sets = (X_train, Y_train, X_val, Y_val),
        loss_fn = 'binary_crossentropy',
        score = 'f1',
        output_activation = 'sigmoid',
        max_training_time = 600
        )

    history = fnns.evolve(generations = generations)
    print("Best overall genome:", history.fittest)

    history.plot(
        title = "Validation accuracy by generation",
        ylabel = "Validation accuracy",
        show_plot = False,
        file_name = "scholarly_plot.png"
        )

    best_score = fnns.train_best()
    print("Best score:", best_score)


if __name__ == '__main__':

    # used to get current directory
    from pathlib import Path

    home_dir = str(Path.home())
    data_path = os.path.join(home_dir, "pCloudDrive", "public_folder",
        "scholarly_data")
    file_name = 'arxiv'

    # Reminder: there are 153 labels and 7 aggregated labels
    #
    # NaturalSelection has come up with the following, giving F1-score 86.86%:
    #   activation  = relu
    #   optimizer   = adam
    #   neurons     = [128, 512, 128, 2048]
    #   dropouts    = [0%, 20%, 40%, 0%, 10%, 0%]
    #   pos_weight  = ?? (used 1)
    #   batch size  = 512 (finished in five mins)

    train_model(
        file_name = file_name,
        val_name = 'arxiv_val',
        data_path = data_path,
        pop_size = 50,
        generations = 20
        )
