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
        [1000, 5000, 10000, 25000, 50000, 100000, 200000, 500000, 750000]] + ['arxiv']

data_path = os.path.join("P:/", "Public Folder", "scholarly_data")
#data_path = os.path.join("/home", "leidem", "pCloudDrive", "Public Folder", "scholarly_data")

cleaner.setup(path = data_path)
elmo.download_elmo_model()

for file_name in file_names:
    print("------------------------------------")
    print(f"NOW PROCESSING: {file_name}")
    print("--------------------------")
    output_path = os.path.join(data_path, f"{file_name}_elmo.csv")
    if not os.path.isfile(output_path):
        cleaner.clean(
            file_name = file_name, 
            lemm_batch_size = 100, 
            path = data_path
            )
        elmo.extract(
            file_name = file_name,
            path = data_path,
            batch_size = 10
            )
    else:
        print(f"Already ELMo'd that one. Moving on...")
