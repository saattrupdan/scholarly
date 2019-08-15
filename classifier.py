# core packages
import pandas as pd
import numpy as np

# used for lemmatising
import spacy

# used to load models
import pickle

# partial functions
from functools import partial

# local packages
from extract_features import basic_clean
from train_model import multilabel_bins


def classify(title, abstract, tfidf_model, predictor, all_cats):
    clean_text = pd.Series([title + abstract])
    clean_text = basic_clean(clean_text).iloc[0, 0]
    clean_text = ' '.join(np.asarray(
        [token.lemma_ for token in nlp(clean_text) if not token.is_stop]))

    tfidf_text = tfidf_model.transform(np.asarray([clean_text]))
    predictions = predictor(tfidf_text)
    return all_cats[predictions]

def generate_classifier(file_name, labels_name, data_path = 'data'):

    # set up paths
    tfidf_path = os.path.join(data_path, f'{file_name}_tfidf_model.pickle')
    predictor_fname = f'{file_name}_{labels_name}_predictor.pickle'
    predictor_path = os.path.join(data_path, predictor_fname)
    cats_path = os.path.join(data_path, 'cats.csv')

    # download files
    url_start = f"https://filedn.com/lRBwPhPxgV74tO0rDoe8SpH/scholarly_data/"
    if not os.path.isfile(cats_path):
        url = url_start + "cats.csv"
        wget.download(url, out = cats_path)
    if not os.path.isfile(tfidf_path):
        url = url_start + f"{file_name}_tfidf_model.pickle"
        wget.download(url, out = cats_path)
    if not os.path.isfile(predictor_path):
        url = url_start + f"{file_name}_predictor.pickle"
        wget.download(url, out = predictor_path)

    # load in files
    cats_df = pd.read_csv(cats_path)
    with open(tfidf_path, 'rb+') as pickle_in:
        tfidf_model = pickle.load(pickle_in)
    with open(predictor_path, 'rb+') as pickle_in:
        predictor = pickle.load(pickle_in)
    
    # get list of all categories
    cat_finder = {
        '1hot' : lambda x: x,
        '1hot_agg' : aggregate_cat
        }
    all_cats = cats_df['category'].apply(cat_finder[labels_name]).unique()
    
    # stitch together the classifier
    classifier = partial(
        classify, 
        tfidf_model = tfidf_model, 
        predictor = predictor, 
        all_cats = all_cats
        )

    return classifier
