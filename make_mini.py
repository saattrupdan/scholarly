def make_mini(from_fname: str, name: str, size: int, data_dir: str = 'data'):
    import pandas as pd
    from pathlib import Path
    from_path = Path(data_dir) / f'{from_fname}.tsv'
    to_path = Path(data_dir) / f'{from_fname}_{name}.tsv'
    df = pd.read_csv(from_path, sep = '\t')
    mini_df = df.sample(size)
    mini_df.to_csv(to_path, sep = '\t', index = False)

if __name__ == '__main__':
    from argparse import ArgumentParser

    ap = ArgumentParser()
    ap.add_argument('-f', '--from', required = True)
    ap.add_argument('-n', '--name', required = True)
    ap.add_argument('-s', '--size', required = True)
    ap.add_argument('-d', '--data-dir')
    args = vars(ap.parse_args())

    make_mini(
        from_fname = args['from'], 
        name = args['name'], 
        size = args['size'], 
        data_dir = args.get('data-dir', 'data')
    )
