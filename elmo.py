import pandas as pd
import numpy as np
import pickle # enables saving data and models locally
import itertools as it # handling iterators like count()

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # ignore tensorflow warnings

import tensorflow_hub as hub
import tensorflow as tf

def download_elmo_model():
    if not os.path.isdir("elmo"):
        # download ELMo model
        print("Downloading compressed ELMo model...", end = " ")
        url = "https://tfhub.dev/google/elmo/2?tf-hub-format=compressed"
        wget.download(url, out="elmo.tar.gz")
        print("Done!")

        # uncompress ELMo model
        print("Uncompressing into the 'elmo' directory...", end = " ")
        os.system("mkdir elmo") # create directory
        with tarfile.open("elmo.tar.gz") as tar:
            tar.extractall("elmo")
        os.remove("elmo.tar.gz")
        print("Done!")

def extract(file_name, path = "data", batch_size = 50):
    # load the ELMo model
    model = hub.Module("elmo", trainable = True)

    print("Extracting ELMo features...")
    
    # infinite loop
    for i in it.count():
        full_path = os.path.join(path, f"{file_name}_elmo_{i}.pickle")
        if not os.path.isfile(full_path):
            
            # open tensorflow session
            with tf.Session() as sess:
                
                # get the next batch and break loop if it does not exist 
                full_path = os.path.join(path, f"{file_name}_clean.csv")
                try:
                    batch = np.asarray(pd.read_csv(
                        full_path,
                        skiprows = i * batch_size,
                        nrows = batch_size
                        ))[:, 1]
                except:
                    break

                # initialise session
                sess.run(tf.global_variables_initializer())
                sess.run(tf.tables_initializer())
                
                # extract ELMo features
                embeddings = model(
                        batch, 
                        signature="default", 
                        as_dict=True
                        )["elmo"]

                # save the average ELMo features for every title+abstract
                elmo_data = sess.run(tf.reduce_mean(embeddings, 1))
            
                # save ELMo features for the batch into a pickle file
                temp_pickle_file_name = f"{file_name}_elmo_{i}.pickle"
                full_path = os.path.join(path, temp_pickle_file_name)
                with open(full_path, "wb") as pickle_out:
                    pickle.dump(elmo_data, pickle_out)

        print(f"Processed {(i+1) * batch_size} papers...", end = "\r")
    
    # concatenate all batches into a single array 'elmo_batches'
    elmo_batches = np.ones((1,1024))
    for i in it.count():
        full_path = os.path.join(path, f"{file_name}_elmo_{i}.pickle")
        try:
            with open(full_path, "rb") as pickle_in:
                batch = np.asanyarray(pickle.load(pickle_in))
        except:
            break
        elmo_batches = np.vstack((elmo_batches, batch))

    # save into a pickle file
    full_path = os.path.join(path, f"{file_name}_elmo.pickle")
    with open(full_path, "wb") as pickle_out:
        pickle.dump(elmo_batches[1:, :], pickle_out)
    
    # remove all the temporary batch pickle files
    for i in it.count():
        full_path = os.path.join(path, f"{file_name}_elmo_{i}.pickle")
        try:
            os.remove(full_path)
        except:
            break

    print("All done!" + " " * 100)
        
    return None
