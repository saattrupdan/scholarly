import cleaner
import onehot
import elmo

import os
import sys
from pathlib import Path 


# set list of file_names
if len(sys.argv) > 1:
    file_names = sys.argv[1:]
else:
    file_names = [f'arxiv_sample_{i}' for i in
        [1000, 5000, 10000, 25000, 50000, 100000, 200000, 500000, 750000]] \
        + ['arxiv', 'arxiv_val']

home_dir = str(Path.home())
data_path = os.path.join(home_dir, "pCloudDrive", "public_folder",
    "scholarly_data")

#cleaner.setup(path = data_path)
elmo.download_elmo_model()

for file_name in file_names:

    print("------------------------------------")
    print(f"NOW PROCESSING: {file_name}")
    print("------------------------------------")

    cleaner.clean(
        file_name = file_name, 
        path = data_path,
        lemm_batch_size = 1024,
        confirmation = False 
        )

    onehot_names = ['1hot', '1hot_agg']
    for onehot_name in onehot_names:
        onehot.one_hot(
            file_name = file_name,
            path = data_path,
            batch_size = 1024,
            onehot_name = onehot_name
            )

    elmo.extract(
        file_name = file_name,
        path = data_path,
        batch_size = 16,
        doomsday_clock = 50,
        confirmation = True
        )
