def batch_iter(iterable: iter, batch_size: int):
    ''' Split an iterable into batches. 
    
    INPUT
        iterable: iter
            An input iterable
        batch_size: int
            The size of each batch

    OUTPUT
        A generator that iterates over the input iterable in batches.
        When there are no batches left then it will continue outputting
        empty iterators
    '''
    from itertools import islice, chain
    source_iter = iter(iterable)
    while True:
        batch_iter = islice(source_iter, batch_size)
        try:
            yield chain([next(batch_iter)], batch_iter)
        except StopIteration:
            break
        finally:
            del batch_iter

def preprocess_data(cats_fname: str = 'arxiv_data_cats', 
    mcats_fname: str = 'arxiv_data_mcats', data_dir: str = 'data', 
    batch_size: int = 1000):
    ''' 
    Preprocess text data. This merges titles and abstracts, lemmatises 
    all words, removes stop words, separates tokens by spaces and makes
    all words lower case. It also saves the resulted dataframe into

        <data_dir> / <tsv_fname>_pp.tsv.

    Note that this function uses a constant amount of memory, which is
    achieved by replacing the raw texts by the preprocessed texts in
    batches.
    
    INPUT
        cats_fname: str
            The name of the tsv file containing all the categories, 
            without file extension
        mcats_fname: str
            The name of the tsv file containing only the master categories, 
            without file extension
        data_dir: str = 'data'
            The data directory
        batch_size: int = 1000
            The amount of rows being preprocessed at a time
    '''
    import pandas as pd
    from pathlib import Path
    from tqdm import tqdm
    import spacy

    # Specify the input- and output paths
    cats_in = Path(data_dir) / (cats_fname + '.tsv')
    mcats_in = Path(data_dir) / (mcats_fname + '.tsv')
    cats_out = Path(data_dir) / (cats_fname + '_pp.tsv')
    mcats_out = Path(data_dir) / (mcats_fname + '_pp.tsv')

    # Load the English spaCy model
    nlp = spacy.load('en', disable = ['tagger', 'parser', 'ner'])
   
    # Load in the dataframe, merge titles and abstracts and batch them
    df = pd.read_csv(cats_in, sep = '\t', usecols = ['title', 'abstract'])
    df.dropna(inplace = True)
    batches = batch_iter(df['title'] + ' ' + df['abstract'], batch_size)
    nm_rows = len(df)
    del df

    # Preprocessing loop
    docs = []
    pbar = tqdm(desc = 'Preprocessing texts', total = nm_rows, leave = False)
    for batch in batches:
        docs.extend(' '.join(tok.lemma_.lower() 
            for tok in doc if not tok.is_stop)
            for doc in nlp.pipe(batch)
        )
        pbar.update(batch_size)

    # Add the preprocessed texts to the dataframe as the first column and
    # save to disk
    for (IN, OUT) in [(cats_in, cats_out), (mcats_in, mcats_out)]:
        df = pd.read_csv(IN, sep = '\t').dropna()
        df.drop(columns = ['title', 'abstract'], inplace = True)
        cats = df.columns.tolist()
        df['text'] = docs
        df = df[['text'] + cats]
        df.to_csv(OUT, sep = '\t', index = False)

def load_data(tsv_fname: str, data_dir: str = 'data', batch_size: int = 32,
    split_ratio: float = 0.99, emb_dim: int = 50, random_seed: int = 42):
    ''' 
    Loads the preprocessed data, tokenises it, builds a vocabulary,
    splits into a training- and validation set, numeralises the texts,
    batches the data into batches of similar text lengths and pads 
    every batch.

    INPUT
        tsv_fname: str
            The name of the tsv file, without file extension
        data_dir: str = 'data'
            The data directory
        batch_size: int = 32,
            The size of each batch
        split_ratio: float = 0.99
            The proportion of the dataset reserved for training
        emb_dim: {50, 100, 200, 300} = 50
            The dimension of the word vectors
        random_seed: int = 42
            A random seed to ensure that the same training/validation split
            is achieved every time

    OUTPUT
        A triple (train_iter, val_iter, TXT), with train_iter and val_iter
        being the iterators that iterates over the training- and validation
        samples, respectively, and TXT is the torchtext.Field object which
        contains the vocabulary
    '''
    from torchtext import data, vocab
    from pathlib import Path
    import pandas as pd
    import random

    # Build the tsv path
    path = Path(data_dir) / (tsv_fname + '.tsv')

    # Set up the fields in the tsv file
    TXT = data.Field()
    CAT = data.Field(sequential = False, use_vocab = False, is_target = True)
    fields = [('text', TXT)] + [(col_name, CAT) for col_name in col_names[1:]]

    # Load in the dataset and tokenise the texts
    dataset = data.TabularDataset(
        path = path,
        format = 'tsv',
        fields = fields,
        skipheader = True
    )

    # Split into a training- and validation set
    random.seed(random_seed)
    train, val = dataset.split(
        split_ratio = split_ratio, 
        random_state = random.getstate()
    )

    # Build the vocabulary of the training set
    TXT.build_vocab(train)#, vectors = vocab.GloVe('6B', dim = emb_dim)

    # Numericalise the texts, batch them into batches of similar text
    # lengths and pad the texts in each batch
    train_iter, val_iter = data.BucketIterator.splits(
        datasets = (train, val),
        batch_sizes = (batch_size, batch_size),
        sort_key = lambda sample: len(sample.text)
    )

    return train_iter, val_iter, TXT


if __name__ == '__main__':
    preprocess_data()
    #train_iter, val_iter, TXT = load_data(file_name)
