import hpc_elmo
import os
import sys
import itertools as it

# set list of file_names
#file_names = ['arxiv_sample_{}'.format(i) for i in
#    [1000, 5000, 10000, 25000, 50000, 100000, 200000, 500000, 750000]] + ['arxiv']

file_names = ['arxiv_sample_1000']
data_path = "data"

elmo.download_elmo_model()

for file_name in file_names:
    print ("------------------------------------")
    print ("NOW PROCESSING: {}".format(file_name))
    print ("------------------------------------")
    output_path = os.path.join(data_path, "{}_elmo.csv".format(file_name))
    if not os.path.isfile(output_path):
        hpc_elmo.extract(
            file_name = file_name,
            path = data_path,
            batch_size = 1000,
            )
    else:
        print ("Already ELMo'd that one. Moving on...")
