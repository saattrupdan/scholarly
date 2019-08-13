import numpy as np
import pandas as pd
import itertools as it # handling iterators like count()
import time # used for sleep()
import shutil # copying data with copyfileobj() and removing with rmtree()
import os # manipulation of file system
import sys # used for exit()
import wget # for downloading files
import tarfile # for unpacking files
import warnings # allows suppression of warnings
import tensorflow_hub as hub
import tensorflow.compat.v1 as tf

# disable tensorflow INFO and WARNING messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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


def generate_elmo_function(module):
    '''
    From an ELMo model generate a function that can extract ELMo
    features from sentences.
    '''
    with tf.Graph().as_default():
        sentences = tf.placeholder(tf.string)
        embed = hub.Module(module, trainable = False)
        embeddings = embed(
            sentences,
            signature = "default", 
            as_dict = True
            )['elmo']
        mean_embeddings = tf.reduce_mean(embeddings, 1)
        session = tf.train.MonitoredSession()
    return lambda x: session.run(mean_embeddings, {sentences: x})


def extract(file_name, path = "data", batch_size = 16,
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
        # build a function that can extract ELMo features
        elmo = generate_elmo_function('pretrained_elmo')
                
        # create directory for the temporary files
        temp_dir = os.path.join(path, f'{file_name}_elmo_temp')
        if not os.path.isdir(temp_dir):
            os.system(f"mkdir {temp_dir}")
        
        full_path = os.path.join(path, f'{file_name}_1hot.csv')
        batches = pd.read_csv(
            full_path, 
            header = None, 
            chunksize = batch_size,
            encoding = 'utf-8'
            )

        print("Extracting ELMo features...")
        for (i, batch) in enumerate(batches):
            
            full_path = os.path.join(temp_dir, f"{file_name}_elmo_{i}.csv")
            if os.path.isfile(full_path):
                continue

            # if it's doomsday then exit python
            if doomsday_clock == 0:
                print("") # deal with \r
                sys.exit('Doomsday clock ticked out.\n')

            if doomsday_clock == np.inf:
                print(f"ELMo processed {i * batch_size} " \
                      f"papers...", end = "\r")
            else:
                print(f"ELMo processed {i * batch_size} " \
                      f"papers... Doomsday clock at " \
                      f"{doomsday_clock}...", end = "\r")

            # extract ELMo features
            elmo_data = pd.DataFrame(elmo(batch.iloc[:, 0]))

            # save ELMo features for the batch into a csv file
            elmo_data.to_csv(full_path, index = False, header = None)
            
            # doomsday clock gets one step closer to doomsday
            # if doomsday_clock == np.inf then this stays np.inf
            doomsday_clock -= 1
        
        print("ELMo extraction complete!" + " " * 25)
        
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
            full_path = os.path.join(path, f"{file_name}_elmo.csv")
            with open(full_path, 'wb+') as file_out:
                for i in it.count():
                    try:
                        full_path = os.path.join(temp_dir,
                            f"{file_name}_elmo_{i}.csv")
                        with open(full_path, "rb") as file_in:
                            shutil.copyfileobj(file_in, file_out)
                        print(f"{i+1} files merged...", end = "\r")
                    except IOError:
                        break
                print("") # deal with \r

            print("Merge complete!" + " " * 25)

            # remove all the temporary batch files
            print("Removing temporary files...")
            shutil.rmtree(temp_dir)
            try:
                os.rmdir(temp_dir)
            except:
                pass
            print("Removal complete!")
