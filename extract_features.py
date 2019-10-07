import os
import numpy as np
import pandas as pd
from tqdm import tqdm, trange

def identity(x):
    ''' Identity function, used for multiprocessing. '''
    return x

def setup(data_path = "data"):
    ''' Create data path and download list of arXiv categories. '''
    import wget

    # Set up paths
    cats_path = os.path.join(data_path, "cats.csv")
    agg_path = os.path.join(data_path, "arxiv_val_agg.csv")
    ft_path = os.path.join(data_path, "fasttext_model.bin")
    
    # Create data directory
    if not os.path.isdir(data_path):
        os.system(f"mkdir {data_path}")

    url_start = f"https://filedn.com/lRBwPhPxgV74tO0rDoe8SpH/scholarly_data/"

    # Download a list of all the arXiv categories
    if not os.path.isfile(cats_path):
        wget.download(url_start + "cats.csv", out = cats_path)

    # Download validation set
    if not os.path.isfile(agg_path):
        url = url_start + "arxiv_val_agg.csv"
        wget.download(url, out = agg_path)

    # Download FastText model
    if not os.path.isfile(ft_path):
        wget.download(url_start + "fasttext_model.bin", out = ft_path)

def download_papers(file_name, data_path = "data"):
    ''' Download the raw paper data. '''
    import wget
    
    url_start = f"https://filedn.com/lRBwPhPxgV74tO0rDoe8SpH/scholarly_data/"
    raw_path = os.path.join(data_path, f'{file_name}.csv')
    if not os.path.isfile(raw_path):
        wget.download(url_start + f"{file_name}.csv", out = raw_path)

def clean_cats(texts, all_cats, data_path = "data"):
    ''' Convert a string-like list of cats like "[a,b,c]" into
        an array [a,b,c] with only legal categories in it. '''
    import re

    # Convert string to numpy array
    arr = np.asarray(re.sub('[\' \[\]]', '', texts).split(','))

    # Remove cats which isn't an ArXiv category
    cat_arr = np.intersect1d(np.asarray(arr), all_cats)

    return np.nan if cat_arr.size == 0 else cat_arr

def aggregate_cat(cat):
    ''' Convert an ArXiv category into an aggregated one. '''

    # Physics categories
    if cat[:8] == 'astro-ph':
        agg_cat = 'physics'
    elif cat[:8] == 'cond-mat':
        agg_cat = 'physics'
    elif cat[:5] == 'gr-qc':
        agg_cat = 'physics'
    elif cat[:3] == 'hep':
        agg_cat = 'physics'
    elif cat[:7] == 'math-ph':
        agg_cat = 'physics'
    elif cat[:4] == 'nlin':
        agg_cat = 'physics'
    elif cat[:4] == 'nucl':
        agg_cat = 'physics'
    elif cat[:7] == 'physics':
        agg_cat = 'physics'
    elif cat[:8] == 'quant-ph':
        agg_cat = 'physics'

    # Mathematics categories
    elif cat[:4] == 'math':
        agg_cat = 'maths'

    # Computer science categories
    elif cat[:2] == 'cs':
        agg_cat = 'compsci'

    # Quantitative biology categories
    elif cat[:5] == 'q-bio':
        agg_cat = 'quantbio'

    # Quantitative finance categories
    elif cat[:5] == 'q-fin':
        agg_cat = 'quantfin'

    # Statistics categories
    elif cat[:4] == 'stat':
        agg_cat = 'stats'

    # Set to NaN otherwise
    else:
        agg_cat = np.nan

    return agg_cat

def agg_cats_to_binary(cat_list, all_agg_cats):
    ''' Turns cats into 0-1 seq's with 1's at every agg category index. '''
    # We cannot convert the below list into a numpy array, since then the
    # below 'cat in agg_cat_list' would cause a FutureWarning, as the 'in'
    # will in the future output an element-wise numpy array
    agg_cat_list = [aggregate_cat(cat) for cat in cat_list]
    bin_cats = np.array([cat in agg_cat_list for cat in all_agg_cats])
    return bin_cats.astype('int8')

def cats_to_binary(cat_list, all_cats):
    ''' Turns cats into 0-1 seq's with 1's at every category index. '''
    return np.in1d(all_cats, cat_list).astype('int8')

def get_rows(file_name, data_path = 'data', raw = False):
    ''' Get number of rows in a csv file. '''
    if raw:
        print("Counting the number of rows in the raw file...")
        file_path = os.path.join(data_path, f'{file_name}.csv')
        header = 0
    else:
        print("Counting the number of rows in the cleaned file...")
        file_path = os.path.join(data_path, f'{file_name}_agg.csv')
        header = None
    data = pd.read_csv(file_path, usecols = [0], chunksize = 10000, 
        squeeze = True, header = header)
    return sum(chunk.shape[0] for chunk in data)

def basic_clean(series):
    ''' Perform basic cleaning operations of a Pandas Series. '''
    import string
    import re

    # Remove newline symbols
    series = series.str.replace('\n', ' ')

    # Remove equations
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

    # Convert text to lowercase
    series = series.str.lower()

    # Remove numbers
    series = series.str.replace('[0-9]', '')

    # Turn hyphens into spaces
    series = series.str.replace('\-', ' ')
    
    # Remove punctuation
    punctuation = re.escape(string.punctuation)
    punctuation_fn = lambda x: re.sub(f'[{punctuation}]', '', x)
    series = series.apply(punctuation_fn)

    # Remove whitespaces
    rm_whitespace = lambda x:' '.join(x.split())
    series = series.apply(rm_whitespace)

    return series
    
def clean_batch(enum_batch, file_name, temp_dir, nlp_model, all_agg_cats,
    cats, batch_size = 1024):
    ''' Clean a single batch. '''

    idx, batch = enum_batch

    batch_path = os.path.join(temp_dir, f'{file_name}_agg_{idx}.csv')
    if not os.path.isfile(batch_path):

        # Merge title and abstract
        batch['clean_text'] = batch['title'] + ' ' + batch['abstract']
        batch.drop(columns = ['title', 'abstract'], inplace = True)

        # Clean texts
        batch['clean_text'] = basic_clean(batch['clean_text'])
        lemmatise = lambda text: ' '.join(np.asarray(
            [token.lemma_ for token in nlp_model(text) if not token.is_stop]))
        batch['clean_text'] = batch['clean_text'].apply(lemmatise)
        
        # Remove rows with no categories or empty text
        batch['clean_text'].replace('', np.nan, inplace = True)
        batch.dropna(inplace = True)

        # Reset index of batch, which will enable concatenation with
        # other dataframes
        batch.reset_index(inplace = True, drop = True)

        # One-hot encode aggregated categories
        bincat_arr = np.array([agg_cats_to_binary(cat_list, all_agg_cats)
            for cat_list in batch['category']]).transpose()
        bincat_dict = {key:value for (key,value) in
            zip(all_agg_cats, bincat_arr)}
        df_agg = pd.DataFrame.from_dict(bincat_dict)
        df_agg = pd.concat([batch['clean_text'], df_agg], axis = 1)

        # Save aggregated batch
        temp_fname = f'{file_name}_agg_{idx}.csv'
        batch_path = os.path.join(temp_dir, temp_fname)
        df_agg.to_csv(batch_path, index = False, header = None)

        # Deal with the individual aggregated categories
        for agg_cat in all_agg_cats:
            bincat_arr = np.array([cats_to_binary(cat_list, cats[agg_cat])
                for cat_list in batch['category']]).T
            rows_to_drop = np.array([i 
                for (i, bincat) in enumerate(bincat_arr.T)
                if 1 not in bincat])
            bincat_dict = {key:value for (key,value) in
                zip(cats[agg_cat], bincat_arr)}
            df_labels = pd.DataFrame.from_dict(bincat_dict)
            df_labels = pd.concat([batch['clean_text'], df_labels], axis = 1)
            df_labels.drop(rows_to_drop, axis = 0, inplace = True)

            temp_fname = f'{file_name}_{agg_cat}_{idx}.csv'
            batch_path = os.path.join(temp_dir, temp_fname)
            df_labels.to_csv(batch_path, index = False, header = None)

        del bincat_arr, bincat_dict, df_agg, df_labels, rows_to_drop

    del batch, enum_batch

def clean_file(file_name, batch_size = 1024, data_path = "data", rows = None):
    ''' Clean file in batches, and save to csv. '''
    import multiprocessing as mp
    import spacy as sp
    from itertools import count
    from functools import partial
    import shutil

    # Create paths
    cats_path = os.path.join(data_path, "cats.csv")
    raw_path = os.path.join(data_path, f"{file_name}.csv")
    temp_dir = os.path.join(data_path, f'{file_name}_temp')
    clean_path = os.path.join(data_path, f"{file_name}_clean.csv")
    agg_path = os.path.join(data_path, f"{file_name}_agg.csv")
    temp_agg_fname = lambda i: f"{file_name}_agg_{i}.csv"
    temp_agg_path = lambda i: os.path.join(temp_dir, temp_agg_fname(i))

    # Get all the aggregated categories, not including NaN
    cats_series = pd.read_csv(cats_path)['category']
    all_agg_cats = np.asarray(cats_series.apply(aggregate_cat).unique())
    all_agg_cats = np.array([agg_cat for agg_cat in all_agg_cats
                            if isinstance(agg_cat, str)])

    # Get all the categories
    all_cats = np.asarray(cats_series.unique())
    aggregate_cats = np.vectorize(aggregate_cat)
    cats = {agg_cat : all_cats[np.array([cat == agg_cat 
                for cat in aggregate_cats(all_cats)])]
                for agg_cat in all_agg_cats}

    # Create paths for the individual aggregated categories
    labels_fname = lambda cat: f'{file_name}_{cat}.csv'
    labels_path = lambda cat: os.path.join(data_path, labels_fname(cat))
    temp_fname = lambda cat, i: f'{file_name}_{cat}_{i}.csv'
    temp_path = lambda cat, i: os.path.join(temp_dir, temp_fname(cat, i))

    # Load English spaCy model for lemmatising
    nlp_model = sp.load('en_core_web_sm')

    # Create directory for the temporary files
    if not os.path.isdir(temp_dir):
        os.system(f"mkdir {temp_dir}")

    # Set up batches
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
   
    # Define cleaning function 
    cleaner = partial(
        clean_batch,
        temp_dir = temp_dir,
        file_name = file_name,
        batch_size = batch_size,
        nlp_model = nlp_model,
        all_agg_cats = all_agg_cats,
        cats = cats
        )

    # Count the number of batches
    nm_batches = np.ceil(rows / batch_size).astype(int)

    # Clean file in parallel batches and show progress bar
    pool = mp.Pool(maxtasksperchild = 10)
    clean_iter = pool.imap(cleaner, enumerate(batches))
    clean_iter = tqdm(clean_iter, total = nm_batches)
    clean_iter.set_description("Cleaning file")
    for _ in clean_iter:
        pass

    # Close and join pool
    pool.close()
    pool.join()

    # Close progress bar
    clean_iter.close()
    
    # Merge temporary files
    cats_iter = tqdm(iter(np.append(all_agg_cats, 'agg')),
                total = all_agg_cats.size + 1)
    cats_iter.set_description("Merging and removing temporary files")
    for agg_cat in cats_iter:
        if agg_cat == 'agg':
            with open(agg_path, 'wb+') as file_out:
                for i in count():
                    try:
                        with open(temp_agg_path(i), "rb+") as file_in:
                            shutil.copyfileobj(file_in, file_out)
                    except IOError:
                        break
        else:
            with open(labels_path(agg_cat), 'wb+') as file_out:
                for i in count():
                    try:
                        with open(temp_path(agg_cat, i), "rb+") as file_in:
                            shutil.copyfileobj(file_in, file_out)
                    except IOError:
                        break

    # Close progress bar
    cats_iter.close()
    
    # Remove temporary files
    shutil.rmtree(temp_dir)

    del clean_iter, cats_iter, nlp_model, pool, batches

def extract_tfidf(file_name, rows, data_path = "data"):
    ''' Extract tf-idf features from file and store them as a npz file. '''
    import pickle
    from scipy.sparse import save_npz, load_npz
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    # Set up paths
    tfidf_path = os.path.join(data_path, f"{file_name}_tfidf.npz")
    tfidf_model_fname = f"{file_name}_tfidf_model.pickle"
    tfidf_model_path = os.path.join(data_path, tfidf_model_fname)
    text_path = os.path.join(data_path, f"{file_name}_agg.csv")

    data = pd.read_csv(
        text_path, 
        header = None, 
        encoding = 'utf-8',
        usecols = [0],
        chunksize = 1
        )
    tfidf = TfidfVectorizer(max_features = 15000)
    data_iter = tqdm(data, total = rows)
    data_iter.set_description("Extracting tf-idf features")
    tfidf_data = tfidf.fit_transform(data_iter)
    save_npz(tfidf_path, tfidf_data)
    with open(tfidf_model_path, 'wb+') as pickle_out:
        pickle.dump(tfidf, pickle_out)

def extract_fasttext(file_name, rows, data_path = 'data'):
    import multiprocessing as mp
    from functools import partial
    import fasttext as ft

    print("Loading FastText model...")
    ft_model = ft.load_model('fasttext_model.bin')

    text_path = os.path.join(data_path, f'{file_name}_agg.csv')
    converters = {
        0: lambda d: ft_model.get_sentence_vector(d)
        }
    data = pd.read_csv(text_path, header = None, encoding = 'utf-8', 
        usecols = [0], converters = converters, chunksize = 1,
        squeeze = True)

    arr = np.zeros((rows, 128))
    with mp.Pool() as pool:
        pbar = pool.imap(identity, enumerate(data), chunksize = 32)
        pbar = tqdm(pbar, total = rows)
        pbar.set_description('Computing sentence vectors')
        for idx, ser in pbar:
            arr[idx, :] = np.array(ser)[0]
            
    np.save(os.path.join(data_path, f'{file_name}_agg_vec.npy'), arr)

if __name__ == '__main__':
    import sys
    import pathlib
        
    # Set list of file_names
    if len(sys.argv) > 1:
        file_names = sys.argv[1:]
    else:
        file_names = ['arxiv_val'] + [f'arxiv_sample_{i}' 
            for i in [1000, 5000, 10000, 25000, 50000, 100000, \
            200000, 500000, 750000]] + ['arxiv']

    home_dir = str(pathlib.Path.home())
    data_path = os.path.join(home_dir, "pCloudDrive", "public_folder",
        "scholarly_data")

    setup(data_path = data_path)
    for file_name in file_names:

        print("------------------------------------")
        print(f"NOW PROCESSING: {file_name}")
        print("------------------------------------")

        file_path = os.path.join(data_path, f'{file_name}_agg_vec.npy')
        if os.path.isfile(file_path):
            print(f"{file_name} already processed.")
        else:
            download_papers(file_name, data_path)
            clean_path = os.path.join(data_path, f'{file_name}_agg.csv')
            if not os.path.isfile(clean_path):
                raw_rows = get_rows(file_name, data_path, raw = True)
                clean_file(
                    file_name = file_name, 
                    data_path = data_path,
                    rows = raw_rows,
                    batch_size = 1024
                    )
            rows = get_rows(file_name, data_path)
            extract_fasttext(
                file_name = file_name,
                data_path = data_path,
                rows = rows
                )
            print("Feature extraction complete.")
