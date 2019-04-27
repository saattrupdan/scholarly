from sys import argv
import pandas as pd
import numpy as np
import pickle # enables saving data and models locally

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # ignore tensorflow warnings

import tensorflow_hub as hub
import tensorflow as tf

def elmo_vectors(x, data_rows = 0):
    
    # the actual ELMo feature extraction
    embeddings = elmo_model(x.tolist(), signature="default", as_dict=True)["elmo"]
    
    # report progress
    index = x.keys().tolist()[-1]
    if data_rows:
        print(f"Extracting ELMo features from {file_name}.csv... " \
                f"{round(index / data_rows * 100, 2)}% completed.", end = "\r")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())

        # return average of ELMo features
        return sess.run(tf.reduce_mean(embeddings,1))


file_name = f'{argv[1]}'

# set up dataframe with cleaned text
with open(f"{file_name}_clean_text.pickle", "rb") as pickle_in:
    df = pickle.load(pickle_in)

# get the amount of rows in the dataframe
data_rows = df.shape[0]

# load the ELMo model
elmo_model = hub.Module("elmo", trainable=False)

print(f"Extracting ELMo features from {file_name}.csv...", end = "\r")

# build ELMo data
batch_size = 25
batches = np.asarray([df[i:i+batch_size]['clean_text'] for i in range(0, data_rows, batch_size)])
elmo_batches = np.asarray([elmo_vectors(batch, data_rows) for batch in batches])
elmo_data = np.concatenate(elmo_batches, axis = 0)

# save ELMo data
with open(f"{file_name}_elmo.pickle","wb") as pickle_out:
    pickle.dump(elmo_data, pickle_out)

print("All done! The ELMo data is saved as {file_name}_elmo.pickle.")
