class BatchWrapper:
    ''' Wrap a torchtext data iterator. '''
    def __init__(self, data_iter, cats: list):
        self.data_iter = data_iter
        self.batch_size = data_iter.batch_size
        self.cats = cats

    def __iter__(self):
        import torch
        for batch in self.data_iter:
            x = batch.text
            y = torch.cat([getattr(batch, cat).unsqueeze(1) 
                for cat in self.cats], dim = 1)
            yield (x, y.float())

    def __len__(self):
        return len(self.data_iter)

def preprocess_data(
    cats_fname: str = 'arxiv_data_cats', 
    mcats_fname: str = 'arxiv_data_mcats', 
    txt_fname: str = 'preprocessed_docs.txt', 
    data_dir: str = 'data', 
    batch_size: int = 1000):
    ''' 
    Preprocess text data. This merges titles and abstracts and separates 
    tokens by spaces. It saves this into a text file and also saves two
    dataframes, one with all the categories and one with the master 
    categories. Note that this function uses a constant amount of memory, 
    which is achieved by working in batches and writing directly to the disk.
    
    INPUT
        cats_fname: str
            The name of the tsv file containing all the categories, 
            without file extension
        mcats_fname: str
            The name of the tsv file containing only the master categories, 
            without file extension
        txt_fname: str
            The name of the txt file containing the preprocessed texts
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
    txt_path = Path(data_dir) / 'preprocessed_docs.txt'

    # Load the English spaCy model used for tokenisation
    nlp = spacy.load('en')
    tokenizer = nlp.Defaults.create_tokenizer(nlp)
   
    # Load in the dataframe, merge titles and abstracts and batch them
    df = pd.read_csv(cats_in, sep = '\t', usecols = ['title', 'abstract'])
    df.dropna(inplace = True)
    docs = '-TITLE_START- ' + df['title'] + ' -TITLE_END- '\
           '-ABSTRACT_START- ' + df['abstract'] + ' -ABSTRACT_END-'
    del df

    # Tokenisation loop
    with tqdm(desc = 'Preprocessing texts', total = len(docs)) as pbar:
        with open(txt_path, 'w') as f:
            for doc in tokenizer.pipe(docs, batch_size = batch_size):
                f.write(' '.join(tok.text for tok in doc) + '\n')
                pbar.update()

    # Add the preprocessed texts to the dataframe as the first column and
    # save to disk
    INS_OUTS = [(cats_in, cats_out), (mcats_in, mcats_out)]
    with tqdm(INS_OUTS, desc = 'Storing the preprocessed texts') as pbar:
        for (IN, OUT) in pbar:
            df = pd.read_csv(IN, sep = '\t').dropna()
            df.drop(columns = ['title', 'abstract'], inplace = True)
            cats = df.columns.tolist()
            with open(txt_path, 'r') as f:
                df['text'] = f.readlines()
            df = df[['text'] + cats]
            df.to_csv(OUT, sep = '\t', index = False)

def load_data(tsv_fname: str, data_dir: str = 'data', batch_size: int = 32,
    split_ratio: float = 0.99, glove_emb_dim: int = 100, random_seed: int = 42,
    vectors: str = 'fasttext'):
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
        vectors: {'fasttext', 'glove'} = 'fasttext'
            The type of word vectors to use. Here the FastText vectors are
            trained on the abstracts and the GloVe vectors are pretrained
            on the 6B corpus
        glove_emb_dim: {50, 100, 200, 300} = 100
            The dimension of the GloVe word vectors. Only relevant if
            <vectors> = 'glove'.
        random_seed: int = 42
            A random seed to ensure that the same training/validation split
            is achieved every time. If set to None then no seed is used.

    OUTPUT
        A triple (train_iter, val_iter, params), with train_iter and val_iter
        being the iterators that iterates over the training- and validation
        samples, respectively, and params is a dictionary with entries:
            vocab_size
                The size of the vocabulary
            emb_dim
                The dimension of the word vectors. Will be equal to
                <glove_emb_dim> if <vectors> = 'glove', and otherwise
                set to 100
            emb_matrix
                The embedding matrix containing the word vectors
    '''
    from torchtext import data, vocab
    from pathlib import Path
    import pandas as pd
    import random

    # Build the tsv path
    path = Path(data_dir) / (tsv_fname + '.tsv')

    # Define the two types of fields in the tsv file
    TXT = data.Field()
    CAT = data.Field(sequential = False, use_vocab = False, is_target = True)

    # Set up the columns in the tsv file with their associated fields
    col_names = pd.read_csv(path, sep = '\t', nrows = 1).columns.tolist()
    fields = [('text', TXT)] + [(col_name, CAT) for col_name in col_names[1:]]

    # Load in the dataset and tokenise the texts
    dataset = data.TabularDataset(
        path = path,
        format = 'tsv',
        fields = fields,
        skip_header = True
    )

    # Split into a training- and validation set
    if random_seed is None:
        train_val = dataset.split(split_ratio = split_ratio)
    else:
        random.seed(random_seed)
        train, val = dataset.split(
            split_ratio = split_ratio, 
            random_state = random.getstate()
        )

    # Build the vocabulary of the training set
    if vectors == 'glove':
        vecs = vocab.GloVe('6B', dim = glove_emb_dim, cache = data_dir)
    elif vectors == 'fasttext':
        vecs = vocab.Vectors('fasttext', cache = data_dir)
    TXT.build_vocab(train, vectors = vecs)

    # Numericalise the texts, batch them into batches of similar text
    # lengths and pad the texts in each batch
    train_iter, val_iter = data.BucketIterator.splits(
        datasets = (train, val),
        batch_sizes = (batch_size, batch_size),
        sort_key = lambda sample: len(sample.text)
    )

    # Wrap the iterators to ensure that we output tensors
    train_dl = BatchWrapper(train_iter, cats = col_names[1:])
    val_dl = BatchWrapper(val_iter, cats = col_names[1:])

    del dataset, train, val, train_iter, val_iter

    params = {
        'vocab_size': len(TXT.vocab),
        'emb_dim': glove_emb_dim if vectors == 'glove' else 100,
        'emb_matrix': TXT.vocab.vectors
    }
    return train_dl, val_dl, params


if __name__ == '__main__':
    preprocess_data()
