import sys, os

try:
    import pandas as pd
except:
    os.system("pip install pandas --user")
    import pandas as pd

try:
    import numpy as np
except:
    os.system("pip install numpy --user")
    import numpy as np

# pickle enables saving data and models locally
try:
    import pickle 
except:
    os.system("pip install pickle --user")
    import pickle 

# spacy provides the lemmatise function
try:
    import spacy as sp
except:
    os.system("pip install -U spacy --user")
    import spacy as sp

def lemmatise(texts):

    # import spacy's language model
    try:
        nlp = sp.load('en', disable=['parser', 'ner'])
    except:
        os.system("python -m spacy download en --user")
        nlp = sp.load('en', disable=['parser', 'ner']) 

    return np.asarray([' '.join(np.asarray([token.lemma_ for token in nlp(text)])) for text in texts])

def lemmatise_file(file_name):
    
    # make sure that input is correct
    if type(file_name) != str:
        sys.exit("The input is not a string. Aborting.")
    elif file_name[-10:] == "_df.pickle":
        file_name = file_name[:-10]
    elif file_name[-3:] == "_df":
        file_name = file_name[:-3]

    # load dataframe
    try:
        with open(f"{file_name}_df.pickle", "rb") as pickle_in:
            df = pickle.load(pickle_in)
    except:
        sys.exit(f"It seems like the file {file_name}_df.pickle doesn't exist. Aborting.")
          
    # lemmatise text
    #try:
        df['clean_text'] = lemmatise(df['clean_text'])
    #except:
    #    sys.exit(f"It seems like the dataframe {file_name}_df.pickle doesn't have a clean_text column. Aborting.")

    # save dataframe to {file_name}_lemmatised.pickle
    with open(f"{file_name}_clean_df.pickle","wb") as pickle_out:
        pickle.dump(df[['clean_text']], pickle_out)

    sys.exit(f"Lemmatisation of {file_name}_df.pickle complete. Dataframe with the clean_text column saved to {file_name}_clean_df.pickle.")
