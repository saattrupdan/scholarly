def main(mcat_ratio: float, epochs: int, dim: int, model: str, 
    nlayers: int, fname: str, gpu: bool, name: str, lr: float,
    batch_size: int, split_ratio: float, vectors: str, data_dir: str,
    pbar_width: int, wandb: bool) -> str:
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

    train_dl, val_dl, params = load_data(
        tsv_fname = f'{fname}_pp',
        batch_size = batch_size,
        split_ratio = split_ratio,
        vectors = vectors,
        data_dir = data_dir
    )

    if model == 'logreg':
        from modules import LogisticRegression
        model = LogisticRegression
    elif model == 'mlp':
        from modules import MLP
        model = MLP
    elif model == 'cnn':
        from modules import CNN
        model = CNN
    elif model == 'sharnn':
        from modules import SHARNN
        model = SHARNN
    elif model == 'convrnn':
        from modules import ConvRNN
        model = ConvRNN
    else:
        raise RuntimeError('Invalid model name.')

    model = model(dim = dim, nlayers = nlayers, data_dir = data_dir, 
        pbar_width = pbar_width, **params)
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

    parser = ArgumentParser()
    parser.add_argument('--name', default = 'no_name')
    parser.add_argument('--batch_size', type = int, default = 32)
    parser.add_argument('--split_ratio', type = float, default = 0.95)
    parser.add_argument('--lr', type = float, default = 3e-4)
    parser.add_argument('--nlayers', type = int, default = 1)
    parser.add_argument('--epochs', type = int, default = 5)
    parser.add_argument('--dim', type = int, default = 256)
    parser.add_argument('--mcat_ratio', type = float,  default = 0.1)
    parser.add_argument('--fname', default = 'arxiv_data')
    parser.add_argument('--gpu', type = bool, default = False)
    parser.add_argument('--wandb', type = bool, default = True)
    parser.add_argument('--data_dir', default = '.data')
    parser.add_argument('--pbar_width', type = int, default = None)
    parser.add_argument('--model', default = 'sharnn',
        choices = ['sharnn', 'logreg', 'cnn', 'mlp', 'convrnn'])
    parser.add_argument('--vectors', default = 'fasttext', 
        choices = ['fasttext', 'glove'])

    print(main(**vars(parser.parse_args())))
