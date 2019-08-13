import os
import pandas as pd
import numpy as np
import re # regular expressions
import string
import spacy as sp # used for lemmatising text
import wget # downloading files
import itertools as it # handling iterators like count()
import shutil # copying data with copyfileobj() and removing with rmtree()
import warnings # allows suppression of warnings
import csv # enables QUOTE_ALL to keep commas in lists when outputting

def nan_if_empty(texts):
    ''' Converts empty iterable to NaNs, making it easier to detect
        by pandas. '''

    arr = np.asarray(texts)
    if arr.size == 0:
        return np.nan
    else:
        return list(arr)


def remove_non_cats(texts, cats):
    ''' Removes every string in input which does not occur in the
        list of arXiv categories. '''

    return np.intersect1d(np.asarray(texts), cats)


def str_to_arr(texts):
    ''' Converts a string to a numpy array. '''

    return np.asarray(re.sub('[\' \[\]]', '', texts).split(','))


def clean_cats(texts, path = "data"):
    ''' Composition of nan_if_empty, remove_non_cats and str_to_arr. '''

    full_path = os.path.join(path, "cats.csv")
    cats = np.loadtxt(
        fname = full_path,
        delimiter = ",",
        dtype = object,
        usecols = 0,
        skiprows = 1
        )

    arr = str_to_arr(texts)
    cat_arr = remove_non_cats(arr, cats)
    return nan_if_empty(cat_arr)


def setup(path = "data"):
    ''' Create data path and download list of arXiv categories. '''

    # create data directory
    if not os.path.isdir(path):
        os.system(f"mkdir {path}")
        print(f"Created {path} directory")

    url_start = f"https://filedn.com/lRBwPhPxgV74tO0rDoe8SpH/scholarly_data/"

    # download a list of all the arXiv categories
    full_path = os.path.join(path, "cats.csv")
    if not os.path.isfile(full_path):
        wget.download(url_start + "cats.csv", out = full_path)
    else:
        print("cats.csv is already downloaded.")
    
    # download validation set
    for onehot_name in ['1hot', '1hot_agg']:
        full_path = os.path.join(path, f"arxiv_val_{onehot_name}.csv")
        if not os.path.isfile(full_path):
            wget.download(url_start + f"arxiv_val_{onehot_name}.csv", 
                out = full_path)
        else:
            print(f"arxiv_val_{onehot_name}.csv is already downloaded.")


def download_papers(file_name, path = "data"):
    ''' Download the raw paper data. '''
    
    url_start = f"https://filedn.com/lRBwPhPxgV74tO0rDoe8SpH/scholarly_data/"
    full_path = os.path.join(path, f'{file_name}.csv')
    if not os.path.isfile(full_path):
        wget.download(url_start + f"{file_name}.csv", out = full_path)
    else:
        print(f"{file_name}.csv is already downloaded.")


# this function is the only place where we use pandas
# and the only place where we load in a full file
def get_preclean_text(file_name, path = "data"):
    ''' Get csv file, perform basic cleaning tasks and save it to csv. '''
    
    print("Loading in raw file...")

    full_path = os.path.join(path, f"{file_name}.csv")
    df = pd.read_csv(full_path, usecols = ['title', 'abstract', 'category'],
            header = 0)

    print("Raw file loaded!")
    print("Precleaning raw file...")

    # drop rows with NaNs
    df.dropna(inplace = True)

    # merge title and abstract
    df['clean_text'] = df['title'] + ' ' + df['abstract']
    df.drop(columns = ['title', 'abstract'], inplace = True)
    
    # remove newline symbols
    df['clean_text'] = df['clean_text'].str.replace('\n', ' ')

    # remove equations
    bracketeqn = \
        '(\\*[\[\(])' + \
        '(.*?)' + \
        '(\\*[\]\)])'
    dollareqn = \
        '(?<!\$)\${1,2}(?!\$)' + \
        '(.*?)' + \
        '(?<!\$)(?<!\$)\${1,2}(?!\$)(?!\$)'
    df['clean_text'] = df['clean_text'].apply(
        lambda x: re.sub(dollareqn, '', x))
    df['clean_text'] = df['clean_text'].apply(
        lambda x: re.sub(bracketeqn, '', x))

    # convert text to lowercase
    df['clean_text'] = df['clean_text'].str.lower()

    # remove numbers
    df['clean_text'] = df['clean_text'].str.replace('[0-9]', '')

    # turn hyphens into spaces
    df['clean_text'] = df['clean_text'].str.replace('\-', ' ')
    
    # remove punctuation
    punctuation = re.escape(string.punctuation)
    df['clean_text'] = df['clean_text'].apply(
        lambda x: re.sub(f'[{punctuation}]', '', x))

    # remove whitespaces
    df['clean_text'] = df['clean_text'].apply(
        lambda x:' '.join(x.split()))

    full_path = os.path.join(path, f'{file_name}_preclean.csv')
    df.to_csv(full_path, index = False, header = None,
        quoting = csv.QUOTE_ALL)
    
    print("Preclean complete!")
    

def lemmatise_file(file_name, batch_size = 1024, path = "data",
    confirmation = True):
    ''' Lemmatise file in batches, and save to csv. '''

    print("Cleaning precleaned file...")
    nlp = sp.load('en', disable=['parser', 'ner'])
        
    # create directory for the temporary files
    temp_dir = os.path.join(path, f'{file_name}_clean_temp')
    if not os.path.isdir(temp_dir):
        os.system(f"mkdir {temp_dir}")

    full_path = os.path.join(path, f"{file_name}_preclean.csv")
    batches = pd.read_csv(full_path, chunksize = batch_size, header = None)

    for (i, batch) in enumerate(batches):
        full_path = os.path.join(temp_dir, f"{file_name}_clean_{i}.csv")
        if not os.path.isfile(full_path):
            clean_text = np.asarray([' '.join(np.asarray(
                [token.lemma_ for token in nlp(text)
                if not token.is_stop]))
                for text in batch.iloc[:, 1]])
            batch.iloc[0 : clean_text.size, 1] = clean_text

            full_path = os.path.join(temp_dir, f'{file_name}_clean_{i}.csv')
            batch.to_csv(full_path, index = False, header = None)
            
            print(f"Cleaned {(i+1) * batch_size} papers...", end = "\r")

    print("Cleaning done!" + " " * 25)
    
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
        print("Merging files...")

        # concatenate all temporary csv files into a single csv file
        full_path = os.path.join(path, f"{file_name}_clean.csv")
        with open(full_path, 'wb+') as file_out:
            for i in it.count():
                if i % 100 == 0 and i > 0:
                    print(f"{i} files merged...", end = "\r")
                try:
                    full_path = os.path.join(temp_dir,
                        f"{file_name}_clean_{i}.csv")
                    with open(full_path, "rb") as file_in:
                        shutil.copyfileobj(file_in, file_out)
                except IOError:
                    break
        
        print("Merge complete!" + " " * 25)
        
        # remove all the temporary batch files
        print("Removing temporary files...")
        shutil.rmtree(temp_dir)
        try:
            os.rmdir(temp_dir)
        except:
            pass

        # remove precleaned file as we have the fully cleaned one
        try:
            full_path = os.path.join(path, f"{file_name}_preclean.csv")
            os.remove(full_path)
        except:
            pass

        print("Removal complete!")
    

def clean(file_name, lemm_batch_size = 1000, path = "data",
    confirmation = True):
    ''' Download and clean raw file. '''

    full_path = os.path.join(path, f"{file_name}_clean.csv")
    if os.path.isfile(full_path):
        print("File already cleaned.")

    else:
        full_path = os.path.join(path, f"{file_name}_preclean.csv")
        if not os.path.isfile(full_path):
            # download the raw file
            download_papers(file_name, path = path)
            
            # preclean and save the raw file
            get_preclean_text(file_name, path = path)
        
        # lemmatise and save the precleaned file
        lemmatise_file(
            file_name = file_name, 
            batch_size = lemm_batch_size,
            path = path,
            confirmation = confirmation
        )
