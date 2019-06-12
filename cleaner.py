import os
import pandas as pd
import numpy as np
import re # regular expressions
import spacy as sp # used for lemmatising text
import wget # downloading files

def nan_if_empty(texts):
    ''' Converts empty iterable to NaNs, making it easier to detect by pandas. '''
    arr = np.asarray(texts)
    if arr.size == 0:
        return np.nan
    else:
        return arr

def remove_non_cats(texts, cats):
    ''' Removes every string in input which does not occur in the list of arXiv categories. '''
    return np.intersect1d(np.asarray(texts), cats)

def str_to_arr(texts):
    ''' Converts a string to a numpy array. '''
    return np.asarray(re.sub('[\' \[\]]', '', texts).split(','))

def clean_cats(texts, path = "data"):
    ''' Composition of nan_if_empty, remove_non_cats and str_to_arr. '''
    full_path = os.path.join(path, "cats.csv")
    cats_df = pd.read_csv(full_path)
    cats = np.asarray(cats_df['category'].values)
    arr = str_to_arr(texts)
    cat_arr = remove_non_cats(arr, cats)
    return nan_if_empty(cat_arr)

def lemmatise(texts):
    ''' Lemmatise an iterable of strings. '''
    
    # import spacy's language model
    try:
        nlp = sp.load('en', disable=['parser', 'ner'])
    except:
        os.system("python -m spacy download en --user")
        nlp = sp.load('en', disable=['parser', 'ner']) 

    return pd.Series([' '.join(np.asarray([token.lemma_ for token in nlp(text)])) for text in texts])

def setup(path = "data"):
    # create data directory
    if not os.path.isdir(path):
        os.system(f"mkdir {path}")
        print(f"Created {path} directory")

    # download a list of all the arXiv categories
    url_start = f"https://filedn.com/lRBwPhPxgV74tO0rDoe8SpH/scholarly_data/"
    full_path = os.path.join(path, "cats.csv")
    if not os.path.isfile(full_path):
        wget.download(url_start + "cats.csv", out = full_path)
    else:
        print("it's already downloaded!")

def download_papers(file_name):
    url_start = f"https://filedn.com/lRBwPhPxgV74tO0rDoe8SpH/scholarly_data/"
    full_path = os.path.join(path, f'{file_name}.csv')
    if not os.path.isfile(full_path):
        wget.download(url_start + f"{file_name}.csv", out = full_path)

def get_clean_text(file_name, path = "data"):
    full_path = os.path.join(path, f"{file_name}.csv")
    clean_cats_with_path = lambda x: clean_cats(x, path = path)
    df = pd.read_csv(full_path, converters = {'category': clean_cats_with_path})
    df = df[['title', 'abstract', 'category']]

    # drop rows with NaNs
    df.dropna(inplace=True)

    # merge title and abstract
    df['clean_text'] = df['title'] + ' ' + df['abstract']
    df.drop(columns = ['title', 'abstract'], inplace = True)

    # remove punctuation marks
    punctuation ='\!\"\#\$\%\&\(\)\*\+\-\/\:\;\<\=\>\?\@\[\\\]\^\_\`\{\|\}\~'
    df['clean_text'] = df['clean_text'].apply(lambda x: re.sub(punctuation, '', x))

    # convert text to lowercase
    df['clean_text'] = df['clean_text'].str.lower()

    # remove numbers
    df['clean_text'] = df['clean_text'].str.replace("[0-9]", " ")

    # remove whitespaces
    df['clean_text'] = df['clean_text'].apply(lambda x:' '.join(x.split()))
    
    print(f"Done!")
    return df['clean_text']

def lemmatise_file(series, file_name, batch_size = 100, path = "data"):
    data_rows = len(series)
    batches = np.asarray([series[i:i+batch_size] for i in 
                np.arange(0, data_rows, batch_size)])
    num_batches = data_rows // batch_size
    if data_rows % batch_size:
        num_batches += 1

    status_text = f"Lemmatising {file_name}.csv..."
    for i, batch in enumerate(batches):
        status_perc = round(i / num_batches * 100, 2)
        print(f"{status_text} {status_perc}% completed.", end = "\r")
        
        full_path = os.path.join(path, f'{file_name}_clean_{i}.csv')
        if not os.path.isfile(full_path):
            batch_series = lemmatise(batch)
            batch_series.to_csv(full_path, header = False, index = False)

    print(f"{status_text} 100.0% completed.")
    print(f"Saving clean series...", end = " ")
    
    lst = [] 
    for i in range(num_batches):
        full_path = os.path.join(path, f'{file_name}_clean_{i}.csv')
        series = pd.read_csv(full_path)
        lst.append(series.values)
    
    arr_lemm = np.concatenate(lst)[:, 1]
    series_lemm = pd.Series(arr_lemm)
    
    full_path = os.path.join(path, f'{file_name}_clean.csv')
    series_lemm.to_csv(full_path, header = False, index = False)

    for i in range(num_batches):
        full_path = os.path.join(path, f'{file_name}_clean_{i}.csv')
        os.remove(full_path)

    print(f"Done!")
    full_path = os.path.join(path, f'{file_name}_clean.csv')
    print(f"Saved to {full_path}.")

    return np.asarray(series_lemm)
        

def clean(file_name, lemm_batch_size = 100, path = "data"):
    full_path = os.path.join(path, f"{file_name}_clean.csv")
    if os.path.isfile(full_path):
        print("File already cleaned! Loading in clean text...", end = " ")
        arr_lemm = np.asarray(pd.read_csv(full_path).values)[:, 1]
        print("Done!")
    else:
        download_papers(file_name)
        
        full_path = os.path.join(path, f"{file_name}.csv")
        print(f"Loading in data from {full_path}...", end = " ")
        series_clean = get_clean_text(file_name, path = path)
        print("Done!")

        arr_lemm = lemmatise_file(
            series_clean, 
            file_name = file_name, 
            batch_size = lemm_batch_size,
            path = path
        )

    return arr_lemm
