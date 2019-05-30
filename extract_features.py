from sys import argv
import pickle
import cleaner
import elmo

# set list of file_names
if len(argv) > 1:
    file_names = argv[1:]
else:
    file_names = [f'arxiv_sample_{i}' for i in
        [1000, 5000, 10000, 25000, 50000, 100000, 200000, 500000, 750000]] + ['arxiv']

cleaner.setup()
elmo.download_elmo_model()

for file_name in file_names:
    print("--------------------------")
    print(f"NOW PROCESSING: {file_name}")
    print("--------------------------")
    series_clean = cleaner.clean(file_name)
    #elmo_data = elmo.extract(series_clean, batch_size = 10)
    #with open(f"data/{file_name}_elmo.pickle", "wb") as pickle_out:
    #    pickle.dump(elmo_data, pickle_out)
