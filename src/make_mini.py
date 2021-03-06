from utils import get_path, get_nrows, get_cats

def make_mini(from_fname: str = 'arxiv_data', name: str = 'mini', 
    size: int = 100000, data_dir: str = '.data', batch_size: int = 10000):
    ''' Make a smaller version of a given dataset, without needing to load
    the larger dataset into memory.
    
    INPUT
        from_fname: str = 'arxiv_data'
            The large dataset
        name: str = 'mini'
            The name of the smaller dataset, which will be appended to the
            file name of the larger one
        size: int = 100000
            The number of rows in the small dataset
        data_dir: str = '.data'
            The name of the data directory
        batch_size: int = 10000
            How many rows of the large dataset we are processing at a time
    '''
    import pandas as pd
    import numpy as np
    from tqdm.auto import tqdm

    from_path = get_path(data_dir) / f'{from_fname}_pp.tsv'
    to_path = get_path(data_dir) / f'{from_fname}_{name}_pp.tsv'

    df = pd.read_csv(from_path, sep = '\t', chunksize = batch_size)
    cats = get_cats(data_dir = data_dir)['id']
    nrows = get_nrows(f'{from_fname}_pp.tsv', data_dir = data_dir)

    text_path = get_path(data_dir) / 'text.tmp'
    labels_path = get_path(data_dir) / 'labels.tmp'

    text = np.memmap(
        text_path,
        dtype = object,
        mode = 'w+',
        shape = (nrows, 1)
    )

    labels = np.memmap(
        labels_path,
        dtype = int,
        mode = 'w+',
        shape = (nrows, len(cats))
    )

    with tqdm(total = nrows, desc = f'Loading {from_fname}_pp.tsv') as pbar:
        for idx, row in enumerate(df):
            text[idx * batch_size: (idx + 1) * batch_size, 0] = row['text']
            labels[idx * batch_size: (idx + 1) * batch_size, :] = row[cats]
            pbar.update(len(row))

    rnd_idxs = np.random.choice(nrows, size = size, replace = False)
    text = text[rnd_idxs, 0]
    labels = labels[rnd_idxs, :]

    mini_df = pd.DataFrame(columns = ['text'] + cats)
    mini_df['text'] = text
    mini_df[cats] = labels
    mini_df.to_csv(to_path, sep = '\t', index = False)

    text_path.unlink()
    labels_path.unlink()

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--from_fname', default = 'arxiv_data')
    parser.add_argument('--name', default = 'mini')
    parser.add_argument('--size', type = int, default = 100000)
    parser.add_argument('--data-dir', default = '.data')
    parser.add_argument('--batch-size', type = int, default = 10000)

    make_mini(**vars(parser.parse_args()))
