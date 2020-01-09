def main(mcat_ratio: float = 0.5, epochs: int = 5, dim: int = 128, 
    model: str = 'sharnn', nlayers: int = 1, fname: str = 'arxiv_data',
    gpu: bool = False) -> str:
    from data import load_data, preprocess_data
    from db import ArXivDatabase
    from utils import get_path

    pp_path = get_path('.data') / f'{fname}_pp.tsv'

    if not pp_path.is_file():

        raw_path = get_path('.data') / f'{fname}.tsv'
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
        tsv_fname = f'{fname}_pp',
        vectors = 'fasttext',
        batch_size = 32,
        split_ratio = 0.95
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
        raise RuntimeError('Invalid model, please choose between '\
                           '"logreg", "mlp", "cnn", "sharnn" and "convrnn".')

    print(f'Training {model.__name__} with dimension {dim} and {nlayers} '\
          f'layer(s), for {epochs} epoch(s) with mcat ratio {mcat_ratio}.')

    model = model(dim = dim, nlayers = nlayers, gpu = gpu, **params)
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
    ap.add_argument('-d', '--dim', type = int, default = [256], nargs = '*')
    ap.add_argument('-m', '--model', default = 'sharnn', nargs = '*',
        choices = ['sharnn', 'logreg', 'cnn', 'mlp', 'convrnn'])
    ap.add_argument('-r', '--ratio', type = float, nargs = '*', 
        choices = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        default = [0.5])
    ap.add_argument('-f', '--fname', nargs = '*', default = ['arxiv_data'])
    ap.add_argument('-g', '--gpu', type = bool, default = False)
    args = vars(ap.parse_args())

    for model in args['model']:
        for ratio in args['ratio']:
            for epochs in args['epochs']:
                for dim in args['dim']:
                    for layers in args['layers']:
                        for fname in args['fname']:
                            scores = main(
                                model = model,
                                mcat_ratio = ratio,
                                epochs = epochs,
                                dim = dim,
                                nlayers = layers,
                                fname = fname,
                                gpu = args['gpu']
                            )
                            print(scores)
