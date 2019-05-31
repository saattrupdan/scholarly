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

def clean_cats(texts):
    ''' Composition of nan_if_empty, remove_non_cats and str_to_arr. '''
    cats_df = pd.read_csv("data/cats.csv")
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

def setup():
    # create data directory
    if not os.path.isdir("data"):
        os.system("mkdir data")
        print("Created data directory")

    # download a list of all the arXiv categories
    url_start = f"https://filedn.com/lRBwPhPxgV74tO0rDoe8SpH/scholarly_data/"
    if not os.path.isfile('data/cats.csv'):
        wget.download(url_start + "cats.csv", out = "data/cats.csv")
    else:
        print("it's already downloaded!")

def download_papers(file_name):
    url_start = f"https://filedn.com/lRBwPhPxgV74tO0rDoe8SpH/scholarly_data/"
    if not os.path.isfile(f'data/{file_name}.csv'):
        wget.download(url_start + f"{file_name}.csv", out = f"data/{file_name}.csv")

def get_clean_text(file_name):
    df = pd.read_csv(f'data/{file_name}.csv', converters={'category': clean_cats})[['title', 'abstract', 'category']]

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

def lemmatise_file(series, file_name, batch_size = 100):
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
        
        if not os.path.isfile(f'data/{file_name}_clean_{i}.csv'):
            batch_series = lemmatise(batch)
            batch_series.to_csv(f"data/{file_name}_clean_{i}.csv", header = False)

    print(f"{status_text} 100.0% completed.")
    print(f"Saving clean series...", end = " ")
    
    lst = [] 
    for i in range(num_batches):
        series = pd.read_csv(f"data/{file_name}_clean_{i}.csv")
        lst.append(series.values)
    
    arr_lemm = np.concatenate(lst)[:, 1]
    series_lemm = pd.Series(arr_lemm)
    
    series_lemm.to_csv(f"data/{file_name}_clean.csv", header = False)

    for i in range(num_batches):
        os.remove(f"data/{file_name}_clean_{i}.csv")

    print(f"Done!")
    print(f"Saved to data/{file_name}_clean.csv")

    return series_lemm
        

def clean(file_name, lemm_batch_size = 100):
    if os.path.isfile(f'data/{file_name}_clean.csv'):
        print("File already cleaned! Loading in clean text...", end = " ")
        series_lemm = pd.read_csv(f"data/{file_name}_clean.csv")
        print("Done!")
    else:
        download_papers(file_name)
        
        print(f"Loading in data from data/{file_name}.csv...", end = " ")
        series_clean = get_clean_text(file_name)
        print("Done!")

        series_lemm = lemmatise_file(
            series_clean, 
            file_name = file_name, 
            batch_size = lemm_batch_size)

    return np.asarray(series_lemm)
