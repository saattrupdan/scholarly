def make_mini(from_fname: str, name: str, size: int, data_dir: str = '.data',
    batch_size: int = 10000):
    import pandas as pd
    import numpy as np
    from tqdm.auto import tqdm
    from utils import get_path, get_nrows, get_cats

    from_path = get_path(data_dir) / f'{from_fname}.tsv'
    to_path = get_path(data_dir) / f'{from_fname}_{name}.tsv'

    df = pd.read_csv(from_path, sep = '\t', chunksize = batch_size)
    cats = get_cats(f'{from_fname}.tsv', data_dir = data_dir)
    nrows = get_nrows(f'{from_fname}.tsv', data_dir = data_dir)

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

    with tqdm(total = nrows, desc = f'Loading {from_fname}.tsv') as pbar:
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

    ap = ArgumentParser()
    ap.add_argument('-f', '--from', required = True)
    ap.add_argument('-n', '--name', required = True)
    ap.add_argument('-s', '--size', required = True)
    ap.add_argument('-d', '--data-dir')
    ap.add_argument('-b', '--batch-size')
    args = vars(ap.parse_args())

    make_mini(
        from_fname = args['from'], 
        name = args['name'], 
        size = int(args['size']), 
        data_dir = args.get('data-dir', '.data'),
        batch_size = args.get('batch-size', 10000)
    )
