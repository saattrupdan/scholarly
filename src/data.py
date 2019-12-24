import torch
import numpy as np
import pandas as pd
from torch.utils import data
from pathlib import Path
from tqdm.auto import tqdm
from utils import get_path

class BertDataset(data.Dataset):
    def __init__(self, fname: str, data_dir: str = '.data'):

        nsents = sum(len(x) for x in pd.read_csv(
            get_path(data_dir) / f'{fname}.tsv',
            sep = '\t', 
            usecols = [0], 
            squeeze = True, 
            chunksize = 50000
            )
        )

        self.memmap = np.memmap(
            get_path(data_dir) / f'{fname}.npy', 
            dtype = np.float, 
            mode = 'r',
            shape = (nsents, 768)
        )

        self.labels = pd.read_csv(
            get_path(data_dir) / f'{fname}.tsv', 
            sep = '\t', 
            usecols = lambda x: x not in ['text']
        ).values

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.memmap[idx, :])
        y = torch.FloatTensor(self.labels[idx, :])
        return x, y

    def __len__(self):
        return self.memmap.shape[0]

    @classmethod
    def splits(cls, fname: str, split_ratio: float = 0.99,
        data_dir: str = '.data'):

        x_train_path = get_path(data_dir) / f'{fname}_train.npy'
        x_val_path = get_path(data_dir) / f'{fname}_val.npy'
        y_train_path = get_path(data_dir) / f'{fname}_train.tsv'
        y_val_path = get_path(data_dir) / f'{fname}_val.tsv'

        if not x_train_path.is_file() or not x_val_path.is_file() or \
           not y_train_path.is_file() or not y_val_path.is_file():
            split_bert_data(
                fname = fname, 
                split_ratio = split_ratio, 
                data_dir = data_dir
            )

        train = cls(fname = f'{fname}_train', data_dir = data_dir)
        val = cls(fname = f'{fname}_val', data_dir = data_dir)

        return train, val

def get_bert_sentence_embeddings(fname: str, data_dir: str = '.data'):
    from transformers import DistilBertModel, DistilBertTokenizer
    
    tsv_path = get_path(data_dir) / f'{fname}.tsv'
    nsents = sum(len(x) for x in pd.read_csv(tsv_path, '\t', 
        usecols = [2], squeeze = True, chunksize = 50000))

    model_name = 'distilbert-base-uncased'
    bert = DistilBertModel.from_pretrained(model_name)
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)

    sentences = pd.read_csv(get_path(data_dir) / f'{fname}.tsv', '\t', 
        usecols = ['text'], squeeze = True, chunksize = 1)
    sentences = (list(sentence)[0].strip() for sentence in sentences)
    tokenized = (tokenizer.tokenize(sentence) for sentence in sentences)
    bert_tokens = (['[CLS]'] + tokens[:510] + ['SEP'] for tokens in tokenized)
    numericalized = (tokenizer.convert_tokens_to_ids(tokens)
                     for tokens in bert_tokens)

    sentence_embeddings = np.memmap(get_path(data_dir) / f'{fname}.npy', 
        dtype = np.float, mode = 'w+', shape = (nsents, 768))

    with tqdm(total = nsents, desc = 'Generating sentence embeddings') as pbar:
        with torch.no_grad():
            for idx, nums in enumerate(numericalized):
                inputs = torch.LongTensor([nums])
                sentence_embeddings[idx, :] = bert(inputs)[0][:, 0, :]
                pbar.update()

    del sentence_embeddings, bert, tokenizer

def split_bert_data(fname: str, split_ratio: float = 0.99, 
    data_dir: str = '.data'):

    nsents = sum(len(x) for x in pd.read_csv(
        get_path(data_dir) / f'{fname}.tsv',
        sep = '\t', 
        usecols = [2], 
        squeeze = True, 
        chunksize = 50000
        )
    )

    embeds = np.memmap(get_path(data_dir) / f'{fname}.npy', 
        dtype = np.float, mode = 'r', shape = (nsents, 768))

    ntrains = int(nsents * split_ratio)
    x_train = np.memmap(get_path(data_dir) / f'{fname}_train.npy', 
        dtype = np.float, mode = 'w+', shape = (ntrains, 768))
    x_val = np.memmap(get_path(data_dir) / f'{fname}_val.npy', 
        dtype = np.float, mode = 'w+', shape = (nsents - ntrains, 768))

    train_indices = np.random.choice(range(nsents), size = ntrains, 
        replace = False)
    train_indices = np.isin(range(nsents), train_indices, assume_unique = True)
    x_train = embeds[train_indices, :]
    x_val = embeds[~train_indices, :]

    labels = pd.read_csv(get_path(data_dir) / f'{fname}.tsv', sep = '\t')
    y_train = labels.iloc[train_indices, :]
    y_val = labels.iloc[~train_indices, :]

    y_train.to_csv(get_path(data_dir) / f'{fname}_train.tsv', sep = '\t', 
        index = False)
    y_val.to_csv(get_path(data_dir) / f'{fname}_val.tsv', sep = '\t', 
        index = False)

    del embeds, x_train, x_val, labels, y_train, y_val

def load_bert_data(fname: str, split_ratio: float = 0.99, 
    batch_size: int = 32, data_dir: str = '.data'):

    train, val = BertDataset.splits(
        fname = fname, 
        data_dir = data_dir, 
        split_ratio = split_ratio
    )

    train_dl = data.DataLoader(train, batch_size = batch_size, shuffle = True)
    val_dl = data.DataLoader(val, batch_size = batch_size, shuffle = True)

    del train, val
    return train_dl, val_dl

class BatchWrapper:
    ''' Wrap a torchtext data iterator. '''
    def __init__(self, data_iter, cats: list):
        self.data_iter = data_iter
        self.batch_size = data_iter.batch_size
        self.cats = cats

    def __iter__(self):
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
    data_dir: str = '.data', 
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
        data_dir: str = '.data'
            The data directory
        batch_size: int = 1000
            The amount of rows being preprocessed at a time
    '''
    import spacy

    # Specify the input- and output paths
    cats_in = get_path(data_dir) / (cats_fname + '.tsv')
    mcats_in = get_path(data_dir) / (mcats_fname + '.tsv')
    cats_out = get_path(data_dir) / (cats_fname + '_pp.tsv')
    mcats_out = get_path(data_dir) / (mcats_fname + '_pp.tsv')
    txt_path = get_path(data_dir) / 'preprocessed_docs.txt'

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

def load_embed_data(tsv_fname: str, data_dir: str = '.data', 
    batch_size: int = 32, split_ratio: float = 0.99, glove_emb_dim: int = 100,
    random_seed: int = 42, vectors: str = 'fasttext'):
    ''' 
    Loads the preprocessed data, tokenises it, builds a vocabulary,
    splits into a training- and validation set, numeralises the texts,
    batches the data into batches of similar text lengths and pads 
    every batch.

    INPUT
        tsv_fname: str
            The name of the tsv file, without file extension
        data_dir: str = '.data'
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
    import random

    # Build the tsv path
    path = get_path(data_dir) / f'{tsv_fname}.tsv'

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
    vector_cache = get_path(data_dir)
    if vectors == 'glove':
        vecs = vocab.GloVe('6B', dim = glove_emb_dim, cache = vector_cache)
    elif vectors == 'fasttext':
        vecs = vocab.Vectors('fasttext', cache = vector_cache)
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

    params = {
        'vocab_size': len(TXT.vocab),
        'emb_dim': glove_emb_dim if vectors == 'glove' else 100,
        'emb_matrix': TXT.vocab.vectors
    }

    del dataset, train, val, train_iter, val_iter
    return train_dl, val_dl, params


if __name__ == '__main__':
    train_dl, val_dl = load_bert_data(
        fname = 'arxiv_data_mcats_pp_mini', 
        split_ratio = 0.9,
        batch_size = 32
    )

    for x, y in train_dl:
        print('Train dataloader:', x.shape, y.shape)
        break
    for x, y in val_dl:
        print('Val dataloader:', x.shape, y.shape)
        break

    #get_bert_sentence_vectors('arxiv_data_mcats_pp_mini')
