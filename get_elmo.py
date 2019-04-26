import pandas as pd
import numpy as np
import pickle
import tensorflow_hub as hub
import tensorflow as tf
import spacy
from sys import argv


# download links for convenience:

# large data set
#!wget -O arxiv.csv https://filedn.com/lRBwPhPxgV74tO0rDoe8SpH/arxiv.csv

# small data set
#!wget -O arxiv_small.csv https://filedn.com/lRBwPhPxgV74tO0rDoe8SpH/arxiv_small.csv # small data

# ELMo model
#!mkdir elmo
#!curl -L "https://tfhub.dev/google/elmo/2?tf-hub-format=compressed" | tar -zxvC elmo


def lemmatization(texts):

    # import spaCy's language model
    nlp = spacy.load('en', disable=['parser', 'ner'])
    
    output = []
    for text in texts:
        s = [token.lemma_ for token in nlp(text)]
        output.append(' '.join(s))
    
    return output

def elmo_vectors(x, data_rows = 0):
    
    # the actual ELMo feature extraction
    embeddings = elmo_model(x.tolist(), signature="default", as_dict=True)["elmo"]
    
    # report progress
    index = x.keys().tolist()[-1]
    if data_rows:
        print(f"Extracting ELMo features from {file_name}.csv... " \
                f"{round(index / data_rows * 100, 1)}% completed.", end = "\r")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())

        # return average of ELMo features
        return sess.run(tf.reduce_mean(embeddings,1))


file_name = f'{argv[1]}'

print("Loading and cleaning data...")

# set up dataframe with titles and abstracts
df = pd.read_csv(f"{file_name}.csv")[['title', 'abstract']]
df = df[:100] # remove later
data_rows = df.shape[0]

# drop rows with NaNs
df.dropna(inplace=True)

# merge title and abstract
df['clean_text'] = df['title'] + ' ' + df['abstract']

# remove punctuation marks
punctuation = '!"#$%&()*+-/:;<=>?@[\\]^_`{|}~'
df['clean_text'].apply(lambda x: ''.join(ch for ch in x if ch not in set(punctuation)))

# convert text to lowercase
df['clean_text'] = df['clean_text'].str.lower()

# remove numbers
df['clean_text'] = df['clean_text'].str.replace("[0-9]", " ")

# remove whitespaces
df['clean_text'] = df['clean_text'].apply(lambda x:' '.join(x.split()))

# lemmatise text
df['clean_text'] = lemmatization(df['clean_text'])

# load the ELMo model
elmo_model = hub.Module("elmo", trainable=False)

print(f"Extracting ELMo features from {file_name}.csv...", end = "\r")

# build ELMo data
batch_size = 10
batches = [df[i:i+batch_size]['clean_text'] for i in range(0, data_rows, batch_size)]
elmo_batches = [elmo_vectors(batch, data_rows) for batch in batches]
elmo_data = np.concatenate(elmo_batches, axis = 0)

# save ELMo data
with open(f"{file_name}.pickle","wb") as pickle_out:
    pickle.dump(elmo_data, pickle_out)

print("All done! The ELMo data is saved as {file_name}.pickle.")
