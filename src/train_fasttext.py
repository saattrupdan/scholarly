def train_fasttext(
    txt_fname: str = 'preprocessed_docs.txt',
    model_fname: str = 'fasttext.bin', 
    vec_fname: str = 'fasttext',
    data_dir: str = '.data', 
    lr: float = 0.05, 
    emb_dim: int = 100, 
    window: int = 5, 
    epochs: int = 5, 
    min_count: int = 5, 
    min_char_ngram: int = 3, 
    max_char_ngram: int = 6, 
    neg_samples: int = 5, 
    max_word_ngram: int = 1):
    ''' Train FastText vectors on a corpus. All default values are the 
        official defaults.

    INPUT
        txt_fname: str
            The name of the txt file containing the corpus, including its
            file extension
        model_fname: str = 'fasttext.bin'
            The name of the output FastText model file
        vec_fname: str = 'fasttext'
            The name of the output txt file containing the word vectors
        data_dir: str = '.data'
            The directory containing all data files
        lr: float = 0.05
            The learning rate
        emb_dim: int = 100
            The dimension of the word embeddings
        window: int = 5
            The size of the window considered at every word, where the
            model will learn to guess the word based on the words within
            the window (its 'context')
        min_count: int = 5
            The minimal number of times a word has to occur to be assigned
            a word vector
        min_char_ngram: int = 3
            The minimum number of characters in the character n-grams
        max_char_ngram: int = 6
            The maximum number of characters in the character n-grams
        neg_samples: int = 5
            How many negative samples to include for every positive sample
        max_word_ngram: int = 1
            The maximum number of words in the word n-grams
    '''
    import fasttext
    from tqdm.auto import tqdm
    from utils import get_path

    txt_path = get_path(data_dir) / txt_fname
    model_path = get_path(data_dir) / model_fname

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

    # Save model
    ft.save_model(str(model_path))

    # Save vectors
    with open(get_path(data_dir) / vec_fname, 'w') as f:
        for word in tqdm(ft.words, desc = 'Saving word vectors'):
            vec_string = ' '.join(str(x) for x in ft.get_word_vector(word))
            f.write(word + ' ' + vec_string + '\n')
    del ft
    

if __name__ == '__main__':
    from argparse import ArgumentParser

    # The fasttext model used in the scholarly project used the default
    # parameters except min_count = 2 and min_char_ngram = 2

    ap = ArgumentParser()
    ap.add_argument('--txt_fname', default = 'preprocessed_docs.txt')
    ap.add_argument('--vec_fname', default = 'fasttext')
    ap.add_argument('--data_dir', default = '.data')
    ap.add_argument('--lr', type = float, default = 0.05)
    ap.add_argument('--emb_dim', type = int, default = 100)
    ap.add_argument('--window', type = int, default = 5)
    ap.add_argument('--epochs', type = int, default = 5)
    ap.add_argument('--min_count', type = int, default = 5)
    ap.add_argument('--min_char_ngram', type = int, default = 3)
    ap.add_argument('--max_char_ngram', type = int, default = 6)
    ap.add_argument('--neg_samples', type = int, default = 5)
    ap.add_argument('--max_word_ngram', type = int, default = 1)

    train_fasttext(**vars(ap.parse_args()))
