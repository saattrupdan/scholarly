def main(mcat_ratio):
    from data import load_data
    from modules import SHARNN, load_model
    from training import train_model
    from inference import get_scores

    train_dl, val_dl, params = load_data(
        tsv_fname = 'arxiv_data_cats_pp_mini',
        vectors = 'fasttext',
        batch_size = 32,
        split_ratio = 0.9
    )

    model = SHARNN(dim = 128, boom_dim = 512, **params)
    model = train_model(model, train_dl, val_dl, 
        epochs = 3, 
        lr = 3e-4,
        mcat_ratio = mcat_ratio
    )

    return get_scores(model, val_dl)

if __name__ == '__main__':
    print(main(0.0))
    print(main(0.5))
    print(main(1.0))
