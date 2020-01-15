def main(mcat_ratio: float, epochs: int, dim: int, nlayers: int, fname: str, 
    gpu: bool, name: str, lr: float, batch_size: int, split_ratio: float, 
    vectors: str, data_dir: str, pbar_width: int, wandb: bool, boom_dim: int,
    dropout: float) -> str:
    from data import load_data
    from utils import get_path

    pp_path = get_path(data_dir) / f'{fname}_pp.tsv'
    if not pp_path.is_file():
        from data import preprocess_data
        raw_path = get_path(data_dir) / f'{fname}.tsv'
        cats_path = get_path(data_dir) / 'cats.json'
        mcat_dict_path = get_path(data_dir) / 'mcat_dict.json'

        if not (raw_path.is_file() and cats_path.is_file() and
            mcat_dict_path.is_file()):
            from db import ArXivDatabase
            db = ArXivDatabase(data_dir = data_dir)
            db.get_mcat_dict()
            db.get_cats()
            if not raw_path.is_file():
                db.get_training_df()

        preprocess_data(data_dir = data_dir)

    train_dl, val_dl, vocab = load_data(
        tsv_fname = f'{fname}_pp',
        batch_size = batch_size,
        split_ratio = split_ratio,
        vectors = vectors,
        data_dir = data_dir
    )

    model = SHARNN(dim = dim, nlayers = nlayers, data_dir = data_dir, 
        pbar_width = pbar_width, vocab = vocab, boom_dim = boom_dim,
        dropout = dropout)
    if gpu: model.cuda()

    model = model.fit(train_dl, val_dl, 
        epochs = epochs, 
        lr = lr,
        mcat_ratio = mcat_ratio,
        name = name,
        use_wandb = wandb
    )

    return model.evaluate(val_dl)

if __name__ == '__main__':
    from argparse import ArgumentParser
    from utils import boolean

    parser = ArgumentParser()
    parser.add_argument('--name', default = 'no_name')
    parser.add_argument('--batch_size', type = int, default = 32)
    parser.add_argument('--split_ratio', type = float, default = 0.95)
    parser.add_argument('--lr', type = float, default = 3e-4)
    parser.add_argument('--nlayers', type = int, default = 1)
    parser.add_argument('--epochs', type = int, default = 5)
    parser.add_argument('--dim', type = int, default = 256)
    parser.add_argument('--boom_dim', type = int, default = 512)
    parser.add_argument('--mcat_ratio', type = float,  default = 0.1)
    parser.add_argument('--fname', default = 'arxiv_data')
    parser.add_argument('--gpu', type = boolean, default = False)
    parser.add_argument('--wandb', type = boolean, default = True)
    parser.add_argument('--data_dir', default = '.data')
    parser.add_argument('--pbar_width', type = int, default = None)
    parser.add_argument('--dropout', type = float, default = 0.)
    parser.add_argument('--vectors', choices = ['fasttext', 'glove'],
        default = 'fasttext')

    print(main(**vars(parser.parse_args())))
