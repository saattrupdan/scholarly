from sys import argv
import os
import pickle
import cleaner
import elmo

# set list of file_names
if len(argv) > 1:
    file_names = argv[1:]
else:
    file_names = [f'arxiv_sample_{i}' for i in
        [1000, 5000, 10000, 25000, 50000, 100000, 200000, 500000, 750000]]# + ['arxiv']

data_path = os.path.join("P:/", "Public Folder", "scholarly_data")
#data_path = os.path.join("/home", "leidem", "pCloudDrive", "Public Folder", "scholarly_data")
cleaner.setup(path = data_path)
elmo.download_elmo_model()

for file_name in file_names:
    print("--------------------------")
    print(f"NOW PROCESSING: {file_name}")
    print("--------------------------")
    output_path = os.path.join(data_path, f"{file_name}_elmo.pickle")
    if not os.path.isfile(output_path):
        clean_text = cleaner.clean(file_name, lemm_batch_size = 100, path = data_path)
        elmo_data = elmo.extract(
            clean_text, 
            batch_size = 10, 
            file_name = file_name, 
            path = data_path
        )
    else:
        print(f"Already ELMo'd that one. Moving on...")
