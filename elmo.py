import numpy as np
import itertools as it # handling iterators like count()
import time # used for sleep()
import shutil # copying data with copyfileobj() and removing with rmtree()
import os # manipulation of file system
import sys # used for exit()
import wget # for downloading files
import tarfile # for unpacking files
import warnings # allows suppression of warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # ignore tensorflow warnings

import tensorflow_hub as hub
import tensorflow as tf


def download_elmo_model():
    ''' Download the pre-trained ELMo model and store it.'''

    if not os.path.isdir("pretrained_elmo"):
        # download ELMo model
        print("Downloading compressed ELMo model...", end = " ")
        url = "https://tfhub.dev/google/elmo/2?tf-hub-format=compressed"
        wget.download(url, out="elmo.tar.gz")
        print("Done!")

        # uncompress ELMo model
        print("Uncompressing into the 'pretrained_elmo' directory...",
            end = " ")
        os.system("mkdir pretrained_elmo")
        with tarfile.open("elmo.tar.gz") as tar:
            tar.extractall("pretrained_elmo")
        os.remove("elmo.tar.gz")
        print("Done!")
    else:
        print("ELMo model already downloaded.")


def extract(file_name, path = "data", batch_size = 10,
        doomsday_clock = np.inf, confirmation = False):
    '''
    Extract ELMo features from file and store them as a csv file.

    INPUT:
        str file_name       =   name of file, without file extension
        str path            =   output folder
        int batch_size      =   amount of texts processed at a time
        int doomsday_clock  =   stop script after this many iterations
        bool confirmation   =   prompt user before merging batches.
                                this is helpful when prototyping,
                                as otherwise it'll merge whatever
                                it has when aborting script
    '''
    
    full_path = os.path.join(path, f"{file_name}_elmo.csv")
    if os.path.isfile(full_path):
        print("File already ELMo'd.")
    else:
        # load the ELMo model
        model = hub.Module("pretrained_elmo", trainable = False)
                
        print("Extracting ELMo features...")

        # create directory for the temporary files
        temp_dir = os.path.join(path, f'{file_name}_elmo_temp')
        if not os.path.isdir(temp_dir):
            os.system(f"mkdir {temp_dir}")
        
        # infinite loop
        for i in it.count():
            # if it's doomsday then exit python
            if doomsday_clock == 0:
                print("") # deal with \r
                sys.exit('Doomsday clock ticked out.\n')

            if doomsday_clock == np.inf:
                print(f"ELMo processed {(i+1) * batch_size} " \
                      f"papers...", end = "\r")
            else:
                print(f"ELMo processed {(i+1) * batch_size} " \
                      f"papers... Doomsday clock at " \
                      f"{doomsday_clock}...", end = "\r")

            full_path = os.path.join(temp_dir, f"{file_name}_elmo_{i}.csv")
            if not os.path.isfile(full_path):
                # open tensorflow session
                with tf.compat.v1.Session() as sess:
                    
                    # get the next batch and break loop if it does not exist 
                    full_path = os.path.join(path, f"{file_name}_clean.csv")
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            batch = np.loadtxt(
                                fname = full_path,
                                delimiter = '\n',
                                skiprows = i * batch_size,
                                max_rows = batch_size,
                                dtype = object,
                                encoding = 'utf-8'
                                )
                    except StopIteration:
                        break
                    
                    # create empty file to reserve it
                    full_path = os.path.join(temp_dir,
                        f"{file_name}_elmo_{i}.csv")
                    open(full_path, 'a').close()
                
                    # initialise session
                    sess.run(tf.compat.v1.global_variables_initializer())
                    sess.run(tf.compat.v1.tables_initializer())
                    
                    # extract ELMo features
                    embeddings = model(
                        batch, 
                        signature = "default", 
                        as_dict = True
                        )["elmo"]

                    # save the average ELMo features for every title+abstract
                    elmo_data = sess.run(tf.reduce_mean(embeddings, 1))
                
                    # save ELMo features for the batch into a csv file
                    np.savetxt(full_path, elmo_data, delimiter = ',')
                    
                    # doomsday clock gets one step closer to doomsday
                    # if doomsday_clock == np.inf then this stays np.inf
                    doomsday_clock -= 1

        print("") # to deal with \r
        
        # ask user if they want to merge batches
        if confirmation:
            cont = None
        else:
            cont = 'y'

        while cont not in {'y','n'}:
            cont = input('Processed all batches. Merge them all ' \
                         'and delete batches? (y/n) \n > ')
            if cont not in {'y','n'}:
                print("Please answer 'y' for yes or 'n' for no.")
        
        if cont == 'y':
            print("Merging files...")
            
            # concatenate all temporary csv files into a single csv file
            # this uses the shutil.copyfileobj() function, which doesn't
            # store the files in memory
            full_path = os.path.join(path, f"{file_name}_elmo.csv")
            with open(full_path, 'wb+') as file_out:
                for i in it.count():
                    if i % 100 == 0:
                        print(f"{i} files merged...", end = "\r")
                    try:
                        full_path = os.path.join(temp_dir,
                            f"{file_name}_elmo_{i}.csv")
                        with open(full_path, "rb") as file_in:
                            shutil.copyfileobj(file_in, file_out)
                    except IOError:
                        break

            print("") # to deal with \r
            print("Merge complete.")

            # remove all the temporary batch files
            print("Removing temporary files...")
            shutil.rmtree(temp_dir)
            print("Removal complete.")

        print("All done with ELMo feature extraction!")
