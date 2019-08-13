import pandas as pd
import os 
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from scipy.sparse import save_npz
from scipy.sparse import load_npz


def extract(file_name, path = "data"):
    '''
    Extract TFIDF features from file and store them as a npz file.

    INPUT:
        str file_name       =   name of file, without file extension
        str path            =   output folder
    '''
    
    full_path = os.path.join(path, f"{file_name}_tfidf.npz")
    if os.path.isfile(full_path):
        print("File already TFIDF'd.")
    else:
        print("Extracting TFIDF features...")

        full_path = os.path.join(path, f'{file_name}_1hot.csv')
        data = iter(pd.read_csv(
            full_path, 
            header = None, 
            encoding = 'utf-8'
            ).iloc[:, 0])

        tfidf = TfidfVectorizer(max_features = 4096)
        tfidf_data = tfidf.fit_transform(data)

        full_path = os.path.join(path, f"{file_name}_tfidf.npz")
        save_npz(full_path, tfidf_data)

        full_path = os.path.join(path, f'{file_name}_tfidf_model.pickle')
        with open(full_path, 'wb+') as pickle_out:
            pickle.dump(tfidf, pickle_out)

        print("TFIDF extraction complete!")
