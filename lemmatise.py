from sys import argv
# download spacy if needed
#!pip install -U spacy && python -m spacy download en

import pandas as pd
import numpy as np
import pickle # enables saving data and models locally
from spacy import load as sp


def lemmatization(texts):

    # import spaCy's language model
    nlp = sp('en', disable=['parser', 'ner'])
    
    #output = []
    #for text in texts:
    #    s = [token.lemma_ for token in nlp(text)]
    #    output.append(' '.join(s))
    
    output = np.asarray([' '.join(np.asarray([token.lemma_ for token in nlp(text)])) 
            for text in texts])
    
    return output

file_name = f'{argv[1]}'

# load dataframe    
with open(f"{file_name}_clean_text.pickle", "rb") as pickle_in:
    df = pickle.load(pickle_in)

# lemmatise text
df['clean_text'] = lemmatization(df['clean_text'])

# save dataframe to {file_name}_clean_text.pickle
with open(f"{file_name}_lemmatised.pickle","wb") as pickle_out:
    pickle.dump(df[['clean_text']], pickle_out)

print(f"Lemmatisation of {file_name}_clean_text.pickle complete. Dataframe with the clean_text column saved to {file_name}_lemmatised.pickle.")
