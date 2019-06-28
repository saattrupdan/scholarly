import os
import pandas as pd
import numpy as np
import re # regular expressions
import spacy as sp # used for lemmatising text
import wget # downloading files
import itertools as it # handling iterators like count()
import shutil # enables copying data without using memory with copyfileobj()

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

    # download a list of all the arXiv categories
    url_start = f"https://filedn.com/lRBwPhPxgV74tO0rDoe8SpH/scholarly_data/"
    full_path = os.path.join(path, "cats.csv")
    if not os.path.isfile(full_path):
        wget.download(url_start + "cats.csv", out = full_path)
    else:
        print("it's already downloaded!")


def download_papers(file_name, path = "data"):
    ''' Download the raw paper data. '''

    url_start = f"https://filedn.com/lRBwPhPxgV74tO0rDoe8SpH/scholarly_data/"
    full_path = os.path.join(path, f'{file_name}.csv')
    if not os.path.isfile(full_path):
        wget.download(url_start + f"{file_name}.csv", out = full_path)


# this function is the only place where we use pandas
def get_preclean_text(file_name, path = "data"):
    ''' Get csv file, perform basic cleaning tasks and save it to csv. '''
    
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

    preclean_arr = np.asarray(df['clean_text'])
    full_path = os.path.join(path, f"{file_name}_preclean.csv")
    np.savetxt(full_path, preclean_arr, fmt = '%s')
    del df, preclean_arr
    

def lemmatise_file(file_name, batch_size = 100, path = "data"):
    ''' Lemmatise file in batches, and save to csv. '''

    nlp = sp.load('en', disable=['parser', 'ner'])
    
    for i in it.count():
        full_path = os.path.join(path, f"{file_name}_clean_{i}.csv")
        if not os.path.isfile(full_path):
            try:
                full_path = os.path.join(path, f"{file_name}_preclean.csv")
                batch = np.loadtxt(
                    fname = full_path,
                    delimiter = '\n',
                    skiprows = i * batch_size,
                    max_rows = batch_size,
                    dtype = object
                    )
             
                batch_arr = np.asarray(
                    [' '.join(np.asarray([token.lemma_ for token in nlp(text)]))
                    for text in batch]
                    )
                
                full_path = os.path.join(path, f'{file_name}_clean_{i}.csv')
                np.savetxt(full_path, batch_arr, delimiter = ',', fmt = '%s')
            except:
                break
        
        print(f"Cleaned {(i+1) * batch_size} papers...", end = "\r")
    
    # ask user if they want to merge batches    
    cont = None
    while cont not in {'y','n'}:
        cont = input('Processed all batches. Merge them all and delete batches? (y/n)')
        if cont not in {'y','n'}:
            print("Please answer 'y' for yes or 'n' for no.")
    
    if cont = 'y'
        # concatenate all temporary csv files into a single csv file
        # this uses the shutil.copyfileobj() function, which doesn't
        # store the files in memory
        full_path = os.path.join(path, f"{file_name}_clean.csv")
        with open(full_path, 'wb+') as file_out:
            for i in it.count():
                try:
                    full_path = os.path.join(path, f"{file_name}_clean_{i}.csv")
                    with open(full_path, "rb") as file_in:
                        shutil.copyfileobj(file_in, file_out)
                except:
                    break
        
        # remove all the temporary batch files
        for i in it.count():
            try:
                full_path = os.path.join(path, f"{file_name}_clean_{i}.csv")
                os.remove(full_path)
            except:
                break
    
    print("All done with cleaning!" + " " * 100)


def clean(file_name, lemm_batch_size = 100, path = "data"):
    ''' Download and clean raw file. '''

    full_path = os.path.join(path, f"{file_name}_clean.csv")
    if os.path.isfile(full_path):
        print("File already cleaned.")

    else:
        full_path = os.path.join(path, f"{file_name}_preclean.csv")
        if not os.path.isfile(full_path):
            print("Fetching and precleaning raw file...", end = ' ')
            # download the raw file
            download_papers(file_name, path = path)
            
            # preclean and save the raw file
            get_preclean_text(file_name, path = path)
            print("Done!")
        
        print("Cleaning precleaned file...")
        # lemmatise and save the precleaned file
        lemmatise_file(
            file_name = file_name, 
            batch_size = lemm_batch_size,
            path = path
        )

        # remove precleaned file as we have the fully cleaned one at this point
        try:
            full_path = os.path.join(path, f"{file_name}_preclean.csv")
            os.remove(full_path)
        except:
            pass
