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

# progress bar
from tqdm import tqdm

# used to get current directory
import pathlib

# used to suppress warnings
import warnings

# nlp model used for lemmatising text
import spacy as sp

# provides string.punctuation for a neat list of all punctuation
import string

# provides QUOTE_ALL to keep commas in lists when outputting csv files
import csv


def setup(data_path = "data", onehot_names = ['1hot']):
    ''' Create data path and download list of arXiv categories. '''

    # set up paths
    cats_path = os.path.join(data_path, "cats.csv")
    onehot_paths = [os.path.join(data_path, f"arxiv_val_{onehot_name}.csv")
                    for onehot_name in onehot_names]
    
    # create data directory
    if not os.path.isdir(data_path):
        os.system(f"mkdir {path}")
        print(f"Created {path} directory")

    url_start = f"https://filedn.com/lRBwPhPxgV74tO0rDoe8SpH/scholarly_data/"

    # download a list of all the arXiv categories
    if not os.path.isfile(cats_path):
        wget.download(url_start + "cats.csv", out = cats_path)
    else:
        print("cats.csv is already downloaded.")
    
    # download validation set
    for onehot_name, onehot_path in zip(onehot_names, onehot_paths):
        if not os.path.isfile(onehot_path):
            wget.download(url_start + f"arxiv_val_{onehot_name}.csv", 
                out = onehot_path)
        else:
            print(f"arxiv_val_{onehot_name}.csv is already downloaded.")

def download_papers(file_name, data_path = "data"):
    ''' Download the raw paper data. '''
    
    url_start = f"https://filedn.com/lRBwPhPxgV74tO0rDoe8SpH/scholarly_data/"
    raw_path = os.path.join(data_path, f'{file_name}.csv')
    if not os.path.isfile(raw_path):
        wget.download(url_start + f"{file_name}.csv", out = raw_path)
    else:
        print(f"{file_name}.csv is already downloaded.")

def clean_cats(texts, all_cats, data_path = "data"):
    ''' Convert a string-like list of cats like "[a,b,c]" into
        a an array [a,b,c] with only legal categories in it. '''

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
        agg_cat = 'computer science'

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
        agg_cat = 'electrical engineering and systems science'
    
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
                   if 'arxiv.org/abs' in row.decode('utf-8'))
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
    
def clean_batch(enum_batch, file_name, temp_dir, nlp_model,
    cat_binner, all_cats, batch_size = 1024):
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

    # one-hot encode
    for (onehot_name, bin_fn) in cat_binner.items():
        bincat_arr = np.array([bin_fn(cat_list, all_cats[onehot_name])
            for cat_list in batch['category']]).transpose()
        bincat_dict = {key:value for (key,value) in
            zip(all_cats[onehot_name], bincat_arr)}
        df_1hot = pd.DataFrame.from_dict(bincat_dict)

        # merge clean text and one-hot cats
        df_1hot = pd.concat([batch['clean_text'], df_1hot], axis = 1)

        # save batch
        temp_fname = f'{file_name}_{onehot_name}_{idx}.csv'
        batch_path = os.path.join(temp_dir, temp_fname)
        df_1hot.to_csv(batch_path, index = False, header = None)

def clean_file(file_name, cat_finder, cat_binner, batch_size = 1024,
    data_path = "data", confirmation = True, rows = None):
    ''' Clean file in batches, and save to csv. '''

    # create paths
    cats_path = os.path.join(data_path, "cats.csv")
    raw_path = os.path.join(data_path, f"{file_name}.csv")
    temp_dir = os.path.join(data_path, f'{file_name}_temp')
    clean_path = os.path.join(data_path, f"{file_name}_clean.csv")
    onehot_path = {}
    temp_path = {}
    for onehot_name in cat_finder.keys():
        fname = f"{file_name}_{onehot_name}.csv"
        onehot_path[onehot_name] = os.path.join(data_path, fname)
        temp_fname = lambda i: f"{file_name}_{onehot_name}_{i}.csv"
        temp_path[onehot_name] = \
            lambda i: os.path.join(temp_dir, temp_fname(i))

    # load English spaCy model for lemmatising
    nlp_model = sp.load('en', disable=['parser', 'ner'])
        
    # create directory for the temporary files
    if not os.path.isdir(temp_dir):
        os.system(f"mkdir {temp_dir}")

    # get cats
    cats_df = pd.read_csv(cats_path)
    all_cats = {}
    for (onehot_name, finder_fn) in cat_finder.items():
        all_cats[onehot_name] = np.asarray(
            cats_df['category'].apply(finder_fn).unique())

    print("Cleaning file...")

    # set up batches
    cat_cleaner = partial(
        clean_cats,
        data_path = data_path,
        all_cats = all_cats['1hot']
        )
    batches = pd.read_csv(
        raw_path,
        usecols = ['title', 'abstract', 'category'],
        converters = {'category' : cat_cleaner},
        chunksize = batch_size,
        header = 0
        )

    # set up multiprocessing
    pool = multiprocessing.Pool()
    cleaner = partial(
        clean_batch,
        temp_dir = temp_dir,
        file_name = file_name,
        batch_size = batch_size,
        nlp_model = nlp_model,
        all_cats = all_cats,
        cat_binner = cat_binner
        )

    # clean file in parallel batches and show progress bar
    for _ in tqdm(pool.imap(cleaner, enumerate(batches)),
        total = np.ceil(rows / batch_size)):
        pass
    pool.close()
    pool.join()
    
    # ask user if they want to merge batches
    if confirmation:
        cont = None
    else:
        cont = 'y'

    while cont not in {'y','n'}:
        cont = input('Processed all batches. Merge them all and ' \
                     'delete batches? (y/n) \n > ')
        if cont not in {'y','n'}:
            print("Please answer 'y' for yes or 'n' for no.")
    
    if cont == 'y':
        print("Merging and removing temporary files...")

        # merge temporary files
        for onehot_name in cat_finder.keys():
            with open(onehot_path[onehot_name], 'wb+') as file_out:
                for i in itertools.count():
                    try:
                        with open(temp_path[onehot_name](i), "rb+") as file_in:
                            shutil.copyfileobj(file_in, file_out)
                    except IOError:
                        break
        
        # remove temporary files
        shutil.rmtree(temp_dir)

def extract_tfidf(file_name, data_path = "data"):
    ''' Extract tf-idf features from file and store them as a npz file. '''
    
    # set up paths
    tfidf_path = os.path.join(data_path, f"{file_name}_tfidf.npz")
    tmodel_path = os.path.join(data_path, f"{file_name}_tfidf_model.pickle")
    text_path = os.path.join(data_path, f"{file_name}_1hot.csv")

    print("Extracting tf-idf features...")
    data = iter(pd.read_csv(
        text_path, 
        header = None, 
        encoding = 'utf-8'
        ).iloc[:, 0])
    tfidf = TfidfVectorizer(max_features = 4096)
    tfidf_data = tfidf.fit_transform(data)
    save_npz(tfidf_path, tfidf_data)
    with open(tmodel_path, 'wb+') as pickle_out:
        pickle.dump(tfidf, pickle_out)


if __name == '__main__':

    # set list of file_names
    if len(sys.argv) > 1:
        file_names = sys.argv[1:]
    else:
        file_names = [f'arxiv_sample_{i}' for i in
            [1000, 5000, 10000, 25000, 50000, 100000, 200000,
             500000, 750000]] + ['arxiv']

    home_dir = str(pathlib.Path.home())
    data_path = os.path.join(home_dir, "pCloudDrive", "public_folder",
        "scholarly_data")

    # set function that finds cats
    cat_finder = {
        '1hot' : lambda x: x,
        '1hot_agg' : aggregate_cat
        }

    # set function that creates binary cats
    cat_binner = {
        '1hot' : cats_to_binary,
        '1hot_agg' : agg_cats_to_binary
        }

    setup(data_path = data_path, onehot_names = ['1hot', '1hot_agg'])
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
                batch_size = 1024,
                confirmation = False,
                rows = rows,
                cat_finder = cat_finder,
                cat_binner = cat_binner
                )
            extract_tfidf(file_name = file_name, data_path = data_path)
            print("Feature extraction complete.")
