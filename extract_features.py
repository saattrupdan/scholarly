import cleaner
import elmo
import onehot

import os
import sys
import itertools as it # tools for iterators, enables e.g. count()
from pathlib import Path # to get home directory

# set list of file_names
if len(sys.argv) > 1:
    file_names = sys.argv[1:]
else:
    file_names = [f'arxiv_sample_{i}' for i in
        [1000, 5000, 10000, 25000, 50000, 100000, 200000, 500000, 750000]] + ['arxiv']

home_dir = str(Path.home())
data_path = os.path.join(home_dir, "pCloudDrive", "Public Folder", "scholarly_data")

cleaner.setup(path = data_path)
elmo.download_elmo_model()

for file_name in file_names:
    print("------------------------------------")
    print(f"NOW PROCESSING: {file_name}")
    print("------------------------------------")
    cleaner.clean(
        file_name = file_name, 
        path = data_path,
        lemm_batch_size = 1000,
        confirmation = False 
        )
    elmo.extract(
        file_name = file_name,
        path = data_path,
        batch_size = 20,
        doomsday_clock = 50,
        confirmation = True
        )
    onehot.one_hot(
        file_name = file_name,
        path = data_path
        )
