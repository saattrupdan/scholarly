def train_fasttext(
    txt_fname: str = 'preprocessed_docs.txt',
    model_fname: str = 'fasttext.bin', 
    data_dir: str = 'data', 
    lr: float = 0.05, 
    emb_dim: int = 100, 
    window: int = 5, 
    epochs: int = 5, 
    min_count: int = 5, 
    min_char_ngram: int = 3, 
    max_char_ngram: int = 6, 
    neg_samples: int = 5, 
    max_word_ngram: int = 1):

    import fasttext
    from pathlib import Path

    txt_path = Path(data_dir) / txt_fname
    model_path = Path(data_dir) / model_fname

    ft = fasttext.train_unsupervised(
        str(txt_path),
        lr = lr,
        dim = emb_dim,
        ws = window,
        epoch = epochs,
        minCount = min_count,
        minn = min_char_ngram,
        maxn = max_char_ngram,
        neg = neg_samples,
        wordNgrams = max_word_ngram,
    )

    ft.save_model(str(model_path))
    del ft
    

if __name__ == '__main__':
    train_fasttext(
        txt_fname = 'preprocessed_docs.txt',
        model_fname = 'fasttext.bin',
        data_dir = 'data'
    )
