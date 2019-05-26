import sys
import pandas as pd
import numpy as np
import pickle # enables saving data and models locally

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # ignore tensorflow warnings

import tensorflow_hub as hub
import tensorflow as tf

def elmo_vectors(x, sess, current_batch=0, num_batches=0):
    
    # the actual ELMo feature extraction
    embeddings = elmo_model(
            x.tolist(),
            signature="default", 
            as_dict=True)["elmo"]
    
    # report progress
    if num_batches and current_batch:
        print(f"Extracting ELMo features from the {data_rows} rows " \
                f"in {file_name}_clean_text.pickle... " \
                f"{round(current_batch / num_batches * 100, 2)}% " \
                f"completed.", end = "\r")

    # return average of ELMo features
    return sess.run(tf.reduce_mean(embeddings, 1))

file_name = argv[1]

# set up dataframe with cleaned text
with open(f"{file_name}_lemmatised.pickle", "rb") as pickle_in:
    df = pickle.load(pickle_in)

# load the ELMo model
elmo_model = hub.Module("elmo", trainable=False)

# get the amount of rows in the data set
data_rows = len(df['clean_text'])

print(f"Extracting ELMo features from the {data_rows} rows " \
        f"in {file_name}_lemmatised.pickle...", end = "\r")

# set up batches, larger batch size requires more computational resources
# but yields more accurate ELMo vectors
batch_size = 50
batches = np.asarray(
        [df[i:i+batch_size]['clean_text'] for i in 
            np.arange(0, data_rows, batch_size)]
        )
num_batches = len(batches)

# build ELMo data
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    elmo_batches = np.asarray([elmo_vectors(batch, sess, i, num_batches)
            for i, batch in enumerate(batches)])
    
elmo_data = np.concatenate(elmo_batches, axis = 0)

# save ELMo data
with open(f"{file_name}_elmo_lemmatised.pickle","wb") as pickle_out:
    pickle.dump(elmo_data, pickle_out)

print(' ' * 100, end = '\r')
print(f"All done! The ELMo data is saved as {file_name}_elmo.pickle.")
