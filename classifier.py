# core packages
import pandas as pd
import numpy as np
import os

# downloading files
import wget

# used for lemmatising
import spacy

# used to load things
import pickle
from tensorflow.keras.models import load_model

# partial functions
from functools import partial

# suppress warnings
import warnings

# local packages
from extract_features import aggregate_cat, basic_clean
from train_model import multilabel_bins, weighted_binary_crossentropy

# suppress deprecation warnings
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False


def classify(title, abstract, tfidf_model, nn, threshold, all_agg_cats):
    clean_text = pd.Series([title + abstract])
    clean_text = basic_clean(clean_text)[0]

    nlp = spacy.load('en', disable=['parser', 'ner'])
    clean_text = ' '.join(np.asarray(
        [token.lemma_ for token in nlp(clean_text) if not token.is_stop]))

    tfidf_text = tfidf_model.transform(np.asarray([clean_text]))
    predictions = multilabel_bins(nn.predict(tfidf_text), threshold).ravel()
    return all_agg_cats[np.nonzero(predictions)]

def generate_classifier(file_name, data_path = 'data'):

    # set up directory
    if not os.path.isdir(data_path):
        os.system(f"mkdir {data_path}")

    # set up paths
    cats_path = os.path.join(data_path, 'cats.csv')
    tfidf_path = os.path.join(data_path, f'{file_name}_tfidf_model.pickle')
    nn_path = os.path.join(data_path, f'{file_name}_nn.h5')
    nn_data_path = os.path.join(data_path, f'{file_name}_nn_data.pickle')

    # download files
    url_start = f"https://filedn.com/lRBwPhPxgV74tO0rDoe8SpH/scholarly_data/"
    if not os.path.isfile(cats_path):
        url = url_start + "cats.csv"
        wget.download(url, out = cats_path)
    if not os.path.isfile(tfidf_path):
        url = url_start + f"{file_name}_tfidf_model.pickle"
        wget.download(url, out = tfidf_path)
    if not os.path.isfile(nn_path):
        url = url_start + f"{file_name}_nn.h5"
        wget.download(url, out = nn_path)
    if not os.path.isfile(nn_data_path):
        url = url_start + f"{file_name}_nn_data.pickle"
        wget.download(url, out = nn_data_path)

    # load in data
    cats_df = pd.read_csv(cats_path)
    with open(nn_data_path, 'rb+') as pickle_in:
        nn_data = pickle.load(pickle_in)
        threshold = nn_data['threshold']
        pos_weight = nn_data['pos_weight']

    # load in tf-idf model
    with open(tfidf_path, 'rb+') as pickle_in:
        tfidf_model = pickle.load(pickle_in)

    # load in neural network
    custom_objects = {'weighted_binary_crossentropy' : lambda x, y: \
        weighted_binary_crossentropy(x, y, pos_weight = pos_weight)}
    nn = load_model(nn_path, custom_objects = custom_objects)
    
    # get list of all aggregated categories
    all_agg_cats = cats_df['category'].apply(aggregate_cat).unique()
    
    # stitch together the classifier
    classifier = partial(
        classify, 
        tfidf_model = tfidf_model, 
        nn = nn, 
        threshold = threshold,
        all_agg_cats = all_agg_cats
        )

    return classifier


if __name__ == '__main__':

    title = 'title'
    abstract = 'abstract'
    
    cf = generate_classifier('arxiv_sample_1000')

    print("The predicted aggregated categories are:")
    print(cf(title, abstract))
