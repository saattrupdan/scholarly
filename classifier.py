# core packages
import pandas as pd
import numpy as np

# used for lemmatising
import spacy

# used to load things
import pickle
from tensorflow.keras.models import load_model

# partial functions
from functools import partial

# local packages
from extract_features import aggregate_cat, basic_clean
from train_model import multilabel_bins


def classify(title, abstract, tfidf_model, nn, threshold, all_agg_cats):
    clean_text = pd.Series([title + abstract])
    clean_text = basic_clean(clean_text).iloc[0, 0]
    clean_text = ' '.join(np.asarray(
        [token.lemma_ for token in nlp(clean_text) if not token.is_stop]))

    tfidf_text = tfidf_model.transform(np.asarray([clean_text]))
    predictions = multilabel_bins(nn.predict(tfidf_text)[0, 0], threshold)
    return all_agg_cats[predictions]

def generate_classifier(file_name, labels_name, data_path = 'data'):

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
        wget.download(url, out = cats_path)
    if not os.path.isfile(nn_path):
        url = url_start + f"{file_name}_nn.h5"
        wget.download(url, out = nn_path)
    if not os.path.isfile(nn_data_path):
        url = url_start + f"{file_name}_nn_data.pickle"
        wget.download(url, out = nn_data_path)

    # load in files
    cats_df = pd.read_csv(cats_path)
    with open(tfidf_path, 'rb+') as pickle_in:
        tfidf_model = pickle.load(pickle_in)
    nn = load_model(nn_path)
    with open(nn_data_path, 'rb+') as pickle_in:
        nn_data = pickle.load(pickle_in)
        threshold = nn_data['threshold']
    
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
