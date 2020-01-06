def main(mcat_ratio: float = 0.5, epochs: int = 5, dim: int = 128, 
    model: str = 'sharnn', nlayers: int = 1) -> str:
    from data import load_data, preprocess_data
    from db import ArXivDatabase
    from utils import get_path

    raw_fname = 'arxiv_data'
    pp_path = get_path('.data') / f'{raw_fname}_pp.tsv'

    if not pp_path.is_file():

        raw_path = get_path('.data') / f'{raw_fname}.tsv'
        cats_path = get_path('.data') / 'cats.json'
        mcat_dict_path = get_path('.data') / 'mcat_dict.json'

        if not (raw_path.is_file() and cats_path.is_file() and 
            mcat_dict_path.is_file()):
            db = ArXivDatabase()
            db.get_mcat_dict()
            db.get_cats()
            if not raw_path.is_file():
                db.get_training_df()

        preprocess_data()

    train_dl, val_dl, params = load_data(
        tsv_fname = f'{raw_fname}_pp',
        vectors = 'fasttext',
        batch_size = 32,
        split_ratio = 0.9
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
    else:
        raise RuntimeError('Invalid model, please choose between '\
                           '"logreg", "mlp", "cnn" and "sharnn".')

    print(f'Training {model.__name__} with dimension {dim} and {nlayers} '\
          f'layer(s), for {epochs} epoch(s) with mcat ratio {mcat_ratio}.')

    model = model(dim = dim, nlayers = nlayers, **params)
    model = model.fit(train_dl, val_dl, 
        epochs = epochs, 
        lr = 3e-4,
        mcat_ratio = mcat_ratio
    )

    return model.evaluate(val_dl)

if __name__ == '__main__':
    from argparse import ArgumentParser

    ap = ArgumentParser()
    ap.add_argument('-l', '--layers', type = int, default = [1], nargs = '*')
    ap.add_argument('-e', '--epochs', type = int, default = [5], nargs = '*')
    ap.add_argument('-d', '--dim', type = int, default = [128], nargs = '*')
    ap.add_argument('-m', '--model', default = 'sharnn', nargs = '*',
        type = str, choices = ['sharnn', 'logreg', 'cnn', 'mlp'])
    ap.add_argument('-r', '--ratio', type = float, nargs = '+', 
        choices = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    args = vars(ap.parse_args())

    for model in args['model']:
        for ratio in args['ratio']:
            for epochs in args['epochs']:
                for dim in args['dim']:
                    for layers in args['layers']:
                        scores = main(
                            model = model,
                            mcat_ratio = ratio,
                            epochs = epochs,
                            dim = dim,
                            nlayers = layers
                        )
                        print(scores)
