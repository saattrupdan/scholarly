import numpy as np
import itertools as it # handling iterators like count()
import time # used for sleep()
import shutil # enables copying data without using memory with copyfileobj()
import os # manipulation of file system

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

def extract(file_name, path = "data", batch_size = 10):
    # load the ELMo model
    model = hub.Module("elmo", trainable = True)
            
    print("Extracting ELMo features...")
    
    # infinite loop
    for i in it.count():
        full_path = os.path.join(path, f"{file_name}_elmo_{i}.csv")
        if not os.path.isfile(full_path):
            # sleep for one second, which should reduce cpu load 
            time.sleep(1)

            # open tensorflow session
            with tf.Session() as sess:
                
                # get the next batch and break loop if it does not exist 
                full_path = os.path.join(path, f"{file_name}_clean.csv")
                try:
                    batch = np.loadtxt(
                        fname = full_path,
                        delimiter = '\n',
                        skiprows = i * batch_size,
                        max_rows = batch_size,
                        dtype = object
                        )
                
                    # initialise session
                    sess.run(tf.global_variables_initializer())
                    sess.run(tf.tables_initializer())
                    
                    # extract ELMo features
                    embeddings = model(
                        batch, 
                        signature = "default", 
                        as_dict = True
                        )["elmo"]

                    # save the average ELMo features for every title+abstract
                    elmo_data = sess.run(tf.reduce_mean(embeddings, 1))
                
                    # save ELMo features for the batch into a csv file
                    temp_file_name = f"{file_name}_elmo_{i}.csv"
                    full_path = os.path.join(path, temp_file_name)
                    np.savetxt(full_path, elmo_data, delimiter = ',')
                except:
                    break

        print(f"ELMo processed {(i+1) * batch_size} papers...", end = "\r")
    
    # ask user if they want to merge batches    
    cont = None
    while cont not in {'y','n'}:
        cont = input('Processed all batches. Merge them all and delete batches? (y/n)')
        if cont not in {'y','n'}:
            print("Please answer 'y' for yes or 'n' for no.")
    
    if cont == 'y':
        # concatenate all temporary csv files into a single csv file
        # this uses the shutil.copyfileobj() function, which doesn't
        # store the files in memory
        full_path = os.path.join(path, f"{file_name}_elmo.csv")
        with open(full_path, 'wb+') as file_out:
            for i in it.count():
                try:
                    full_path = os.path.join(path, f"{file_name}_elmo_{i}.csv")
                    with open(full_path, "rb") as file_in:
                        shutil.copyfileobj(file_in, file_out)
                except:
                    break

        # remove all the temporary batch files
        for i in it.count():
            try:
                full_path = os.path.join(path, f"{file_name}_elmo_{i}.csv")
                os.remove(full_path)
            except:
                break

    print("All done with ELMo feature extraction!" + " " * 100)
