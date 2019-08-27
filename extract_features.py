# core packages
import numpy as np
import pandas as pd
import os
import sys
import re

# partial functions
from functools import partial

# dealing with iterators
import itertools

# high-level operations on files
import shutil

# dealing with .pickle and .npz files
import pickle
from scipy.sparse import save_npz, load_npz

# extracting tf-idf features
from sklearn.feature_extraction.text import TfidfVectorizer

# downloading files
import wget

# parallelising tasks
import multiprocessing

# progress bars
from tqdm import tqdm

# used to get current directory
import pathlib

# used to suppress warnings
import warnings

# nlp model used for lemmatising text
import spacy

# provides string.punctuation for a neat list of all punctuation
import string

# provides QUOTE_ALL to keep commas in lists when outputting csv files
import csv


def setup(data_path = "data"):
    ''' Create data path and download list of arXiv categories. '''

    # set up paths
    cats_path = os.path.join(data_path, "cats.csv")
    labels_agg_path = os.path.join(data_path, "arxiv_val_labels_agg.csv")
    
    # create data directory
    if not os.path.isdir(data_path):
        os.system(f"mkdir {data_path}")

    url_start = f"https://filedn.com/lRBwPhPxgV74tO0rDoe8SpH/scholarly_data/"

    # download a list of all the arXiv categories
    if not os.path.isfile(cats_path):
        wget.download(url_start + "cats.csv", out = cats_path)
    
    # download validation set
    if not os.path.isfile(labels_agg_path):
        url = url_start + "arxiv_val_labels_agg.csv"
        wget.download(url, out = labels_agg_path)

def download_papers(file_name, data_path = "data"):
    ''' Download the raw paper data. '''
    
    url_start = f"https://filedn.com/lRBwPhPxgV74tO0rDoe8SpH/scholarly_data/"
    raw_path = os.path.join(data_path, f'{file_name}.csv')
    if not os.path.isfile(raw_path):
        wget.download(url_start + f"{file_name}.csv", out = raw_path)

def clean_cats(texts, all_cats, data_path = "data"):
    ''' Convert a string-like list of cats like "[a,b,c]" into
        an array [a,b,c] with only legal categories in it. '''

    # convert string to numpy array
    arr = np.asarray(re.sub('[\' \[\]]', '', texts).split(','))

    # remove cats which isn't an ArXiv category
    cat_arr = np.intersect1d(np.asarray(arr), all_cats)

    return np.nan if cat_arr.size == 0 else cat_arr

def aggregate_cat(cat):
    ''' Convert an ArXiv category into an aggregated one. '''

    if cat[:8] == 'astro-ph':
        agg_cat = 'physics'

    elif cat[:2] == 'cs':
        agg_cat = 'compsci'

    elif cat[:5] == 'gr-qc':
        agg_cat = 'physics'

    elif cat[:3] == 'hep':
        agg_cat = 'physics'

    elif cat[:4] == 'math':
        agg_cat = 'mathematics'

    elif cat[:4] == 'nlin':
        agg_cat = 'physics'

    elif cat[:4] == 'nucl':
        agg_cat = 'physics'

    elif cat[:7] == 'physics':
        agg_cat = 'physics'

    elif cat[:8] == 'quant-ph':
        agg_cat = 'physics'

    elif cat[:4] == 'stat':
        agg_cat = 'statistics'

    elif cat[:4] == 'econ':
        agg_cat = 'statistics'

    elif cat[:4] == 'eess':
        agg_cat = 'eess'
    
    elif cat[:8] == 'cond-mat':
        agg_cat = 'physics'
    
    elif cat[:5] == 'q-bio':
        agg_cat = 'biology'
    
    elif cat[:5] == 'q-fin':
        agg_cat = 'finance'

    else:
        agg_cat = np.nan

    return agg_cat

def agg_cats_to_binary(cat_list, all_agg_cats):
    ''' Turns cats into 0-1 seq's with 1's at every agg category index. '''
    agg_cat_list = np.asarray([aggregate_cat(cat) for cat in cat_list])
    return np.in1d(all_agg_cats, agg_cat_list).astype('int8')

def cats_to_binary(cat_list, all_cats):
    ''' Turns cats into 0-1 seq's with 1's at every category index. '''
    return np.in1d(all_cats, cat_list).astype('int8')

def get_rows(file_name, data_path = 'data'):
    ''' Get number of rows in a csv file. '''
    print("Counting the number of rows in the file...")
    file_path = os.path.join(data_path, f'{file_name}.csv')
    with open(file_path, 'rb+') as file_in:
        rows = sum(1 for row in file_in
                   if 'arxiv.org/pdf' in row.decode('utf-8'))
    return rows

def basic_clean(series):
    ''' Perform basic cleaning operations to a Pandas Series. '''

    # remove newline symbols
    series = series.str.replace('\n', ' ')

    # remove equations
    dollareqn = \
        '(?<!\$)\${1,2}(?!\$)' + \
        '(.*?)' + \
        '(?<!\$)(?<!\$)\${1,2}(?!\$)(?!\$)'
    bracketeqn = \
        '(\\*[\[\(])' + \
        '(.*?)' + \
        '(\\*[\]\)])'
    dollar_fn = lambda x: re.sub(dollareqn, '', x)
    bracket_fn = lambda x: re.sub(bracketeqn, '', x)
    series = series.apply(dollar_fn)
    series = series.apply(bracket_fn)

    # convert text to lowercase
    series = series.str.lower()

    # remove numbers
    series = series.str.replace('[0-9]', '')

    # turn hyphens into spaces
    series = series.str.replace('\-', ' ')
    
    # remove punctuation
    punctuation = re.escape(string.punctuation)
    punctuation_fn = lambda x: re.sub(f'[{punctuation}]', '', x)
    series = series.apply(punctuation_fn)

    # remove whitespaces
    rm_whitespace = lambda x:' '.join(x.split())
    series = series.apply(rm_whitespace)

    return series
    
def clean_batch(enum_batch, file_name, temp_dir, nlp_model, all_agg_cats,
    cats, batch_size = 1024):
    ''' Clean a single batch. '''

    idx, batch = enum_batch

    # merge title and abstract
    batch['clean_text'] = batch['title'] + ' ' + batch['abstract']
    batch.drop(columns = ['title', 'abstract'], inplace = True)
    
    # clean texts
    batch['clean_text'] = basic_clean(batch['clean_text'])
    lemmatise = lambda text: ' '.join(np.asarray(
        [token.lemma_ for token in nlp_model(text) if not token.is_stop]))
    batch['clean_text'] = batch['clean_text'].apply(lemmatise)
    
    # remove rows with no categories or empty text
    batch['clean_text'].replace('', np.nan, inplace = True)
    batch.dropna(inplace = True)

    # reset index of batch, which will enable concatenation with
    # other dataframes
    batch.reset_index(inplace = True, drop = True)

    # one-hot encode aggregated categories
    bincat_arr = np.array([agg_cats_to_binary(cat_list, all_agg_cats)
        for cat_list in batch['category']]).transpose()
    bincat_dict = {key:value for (key,value) in
        zip(all_agg_cats, bincat_arr)}
    df_labels_agg = pd.DataFrame.from_dict(bincat_dict)
    df_labels_agg = pd.concat([batch['clean_text'], df_labels_agg], axis = 1)

    # save aggregated batch
    temp_fname = f'{file_name}_labels_agg_{idx}.csv'
    batch_path = os.path.join(temp_dir, temp_fname)
    df_labels_agg.to_csv(batch_path, index = False, header = None)

    # deal with the individual aggregated categories
    for agg_cat in all_agg_cats:
        bincat_arr = np.array([cats_to_binary(cat_list, cats[agg_cat])
            for cat_list in batch['category']]).T
        rows_to_drop = np.array([i for (i, bincat) in enumerate(bincat_arr.T)
                                 if 1 not in bincat])
        bincat_dict = {key:value for (key,value) in
            zip(cats[agg_cat], bincat_arr)}
        df_labels = pd.DataFrame.from_dict(bincat_dict)
        df_labels = pd.concat([batch['clean_text'], df_labels], axis = 1)
        df_labels.drop(rows_to_drop, axis = 0, inplace = True)

        temp_fname = f'{file_name}_labels_{agg_cat}_{idx}.csv'
        batch_path = os.path.join(temp_dir, temp_fname)
        df_labels.to_csv(batch_path, index = False, header = None)

def clean_file(file_name, batch_size = 1024, data_path = "data", rows = None):
    ''' Clean file in batches, and save to csv. '''

    # create paths
    cats_path = os.path.join(data_path, "cats.csv")
    raw_path = os.path.join(data_path, f"{file_name}.csv")
    temp_dir = os.path.join(data_path, f'{file_name}_temp')
    clean_path = os.path.join(data_path, f"{file_name}_clean.csv")
    labels_agg_path = os.path.join(data_path, f"{file_name}_labels_agg.csv")
    temp_agg_fname = lambda i: f"{file_name}_labels_agg_{i}.csv"
    temp_agg_path = lambda i: os.path.join(temp_dir, temp_agg_fname(i))

    # get all the aggregated categories
    cats_series = pd.read_csv(cats_path)['category']
    all_agg_cats = np.asarray(cats_series.apply(aggregate_cat).unique())

    # get all the categories
    all_cats = np.asarray(cats_series.unique())
    aggregate_cats = np.vectorize(aggregate_cat)
    cats = {agg_cat : all_cats[aggregate_cats(all_cats) == agg_cat]
                      for agg_cat in all_agg_cats}

    # create paths for the individual aggregated categories
    labels_fname = lambda cat: f'{file_name}_labels_{cat}.csv'
    labels_path = lambda cat: os.path.join(data_path, labels_fname(cat))
    temp_fname = lambda cat, i: f'{file_name}_labels_{cat}_{i}.csv'
    temp_path = lambda cat, i: os.path.join(temp_dir, temp_fname(cat, i))

    # load English spaCy model for lemmatising
    nlp_model = spacy.load('en_core_web_sm')

    # create directory for the temporary files
    if not os.path.isdir(temp_dir):
        os.system(f"mkdir {temp_dir}")

    # set up batches
    cat_cleaner = partial(
        clean_cats,
        data_path = data_path,
        all_cats = all_cats
        )
    batches = pd.read_csv(
        raw_path,
        usecols = ['title', 'abstract', 'category'],
        converters = {'category' : cat_cleaner},
        chunksize = batch_size,
        header = 0
        )
   
    # define cleaning function 
    cleaner = partial(
        clean_batch,
        temp_dir = temp_dir,
        file_name = file_name,
        batch_size = batch_size,
        nlp_model = nlp_model,
        all_agg_cats = all_agg_cats,
        cats = cats
        )

    # clean file in parallel batches and show progress bar
    with multiprocessing.Pool() as pool:
        clean_iter = pool.imap(cleaner, enumerate(batches))
        clean_iter = tqdm(clean_iter,
            total = np.ceil(rows / batch_size).astype(int))
        clean_iter.set_description("Cleaning file")
        for _ in clean_iter:
            pass
    
    # merge temporary files
    cats_iter = tqdm(iter(np.append(all_agg_cats, 'agg')),
                total = all_agg_cats.size + 1)
    cats_iter.set_description("Merging and removing temporary files")
    for agg_cat in cats_iter:
        if agg_cat == 'agg':
            with open(labels_agg_path, 'wb+') as file_out:
                for i in itertools.count():
                    try:
                        with open(temp_agg_path(i), "rb+") as file_in:
                            shutil.copyfileobj(file_in, file_out)
                    except IOError:
                        break
        else:
            with open(labels_path(agg_cat), 'wb+') as file_out:
                for i in itertools.count():
                    try:
                        with open(temp_path(agg_cat, i), "rb+") as file_in:
                            shutil.copyfileobj(file_in, file_out)
                    except IOError:
                        break
    
    # remove temporary files
    shutil.rmtree(temp_dir)

def extract_tfidf(file_name, rows, data_path = "data"):
    ''' Extract tf-idf features from file and store them as a npz file. '''
    
    # set up paths
    tfidf_path = os.path.join(data_path, f"{file_name}_tfidf.npz")
    tfidf_model_fname = f"{file_name}_tfidf_model.pickle"
    tfidf_model_path = os.path.join(data_path, tfidf_model_fname)
    text_path = os.path.join(data_path, f"{file_name}_labels_agg.csv")

    data = iter(pd.read_csv(
        text_path, 
        header = None, 
        encoding = 'utf-8'
        ).iloc[:, 0])
    tfidf = TfidfVectorizer(max_features = 15000)
    data_iter = tqdm(data, total = rows)
    data_iter.set_description("Extracting tf-idf features")
    tfidf_data = tfidf.fit_transform(data_iter)
    save_npz(tfidf_path, tfidf_data)
    with open(tfidf_model_path, 'wb+') as pickle_out:
        pickle.dump(tfidf, pickle_out)


if __name__ == '__main__':

    # set list of file_names
    if len(sys.argv) > 1:
        file_names = sys.argv[1:]
    else:
        file_names = [f'arxiv_sample_{i}' for i in [1000, 5000, 10000, \
            25000, 50000, 100000, 200000, 500000, 750000]] + ['arxiv']

    home_dir = str(pathlib.Path.home())
    data_path = os.path.join(home_dir, "pCloudDrive", "public_folder",
        "scholarly_data")

    setup(data_path = data_path)
    for file_name in file_names:

        print("------------------------------------")
        print(f"NOW PROCESSING: {file_name}")
        print("------------------------------------")

        file_path = os.path.join(data_path, f'{file_name}_tfidf.npz')
        if os.path.isfile(file_path):
            print(f"{file_name} already processed.")
        else:
            download_papers(file_name, data_path)
            rows = get_rows(file_name, data_path)
            clean_file(
                file_name = file_name, 
                data_path = data_path,
                rows = rows,
                batch_size = 1024
                )
            extract_tfidf(
                file_name = file_name,
                data_path = data_path,
                rows = rows
                )
            print("Feature extraction complete.")
