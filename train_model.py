# core packages
import numpy as np
import pandas as pd
import os

# hyperparameter tuning
import naturalselection as ns

# dealing with .pickle and .npz files
import pickle
from scipy.sparse import save_npz, load_npz

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
    
    # NaturalSelection has found the following, giving val F1-score 88.9%:
    #   activation  = relu
    #   optimizer   = adam
    #   initializer = he_normal
    #   neurons     = [512, 512, 64, 1024, 128]
    #   dropouts    = [10%, 0%, 10%, 20%, 40%, 0%]
    #   pos_weight  = ?? (used 1)
    #   batch size  = 1024 (scoring after ten mins)

    train_model(
        file_name = file_name,
        val_name = 'arxiv_val',
        data_path = data_path,
        pop_size = 50,
        generations = 20
        )
