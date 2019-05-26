# import packages, and install them if needed
import sys
try:
    import pandas as pd
except:
    !pip install pandas --user
try:
    import numpy as np
except:
    !pip install numpy --user
try:
    import pickle # enables saving data and models locally
except:
    !pip install pickle --user
try:
    from spacy import load as sp # provides the lemmatise function
except:
    !pip install -U spacy --user && python -m spacy download en --user

def lemmatise(texts):
    nlp = sp('en', disable=['parser', 'ner']) # import spacy's language model
    return np.asarray([' '.join(np.asarray([token.lemma_ for token in nlp(text)])) for text in texts])

def lemmatise_file(file_name):
    
    # make sure that input is correct
    if type(file_name) != str:
        sys.exit("The input is not a string. Aborting.")
    elif file_name[-18:] == "_clean_text.pickle":
        file_name = file_name[:-18]
    elif file_name[-11:] == "_clean_text":
        file_name = file_name[:-11]

    # load dataframe
    try:
        with open(f"{file_name}_clean_text.pickle", "rb") as pickle_in:
            df = pickle.load(pickle_in)
    except:
        sys.exit(f"It seems like the file {file_name}_clean_text.pickle doesn't exist. Aborting.")
          
    # lemmatise text
    try:
        df['clean_text'] = lemmatise(df['clean_text'])
    except:
        sys.exit(f"It seems like the dataframe {file_name}_clean_text.pickle doesn't have a clean_text column. Aborting.")

    # save dataframe to {file_name}_clean_text.pickle
    with open(f"{file_name}_lemmatised.pickle","wb") as pickle_out:
        pickle.dump(df[['clean_text']], pickle_out)

    sys.exit(f"Lemmatisation of {file_name}_clean_text.pickle complete. Dataframe with the clean_text column saved to {file_name}_lemmatised.pickle.")

lemmatise_file(sys.argv[1])
