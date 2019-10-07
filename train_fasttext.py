import fasttext as ft
import pathlib
import os
import pandas as pd
import numpy as np

home_dir = str(pathlib.Path.home())
data_path = os.path.join(home_dir, 'pCloudDrive', 'public_folder',
    'scholarly_data')

full_path = os.path.join(data_path, 'arxiv_labels_agg.csv')
data = iter(pd.read_csv(
    full_path,
    header = None,
    encoding = 'utf-8',
    ).iloc[:, 0])

text_path = os.path.join(data_path, 'arxiv.txt')
if not os.path.isfile(text_path):
    with open(text_path, 'a+') as file_out:
        for row in data:
            file_out.write(row + '\n')

model = ft.train_unsupervised(text_path, wordNgrams = 3, dim = 128,
    thread = 4, minn = 2)
model.save_model('fasttext_model.bin')

emb_mat = model.get_output_matrix()
vocab = np.array(model.words)
np.save('fasttext_matrix.npy', emb_mat)
np.save('fasttext_vocab.npy', vocab)

os.remove(text_path)
