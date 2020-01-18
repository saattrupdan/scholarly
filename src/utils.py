from pathlib import Path
import torch

def get_root_path() -> Path:
    ''' Returns project root folder. '''
    return Path.cwd().parent

def get_path(path_name: str) -> Path:
    ''' Returns data folder. '''
    return get_root_path() / path_name

def get_cats(data_dir: str = '.data') -> list:
    ''' Load the list of arXiv categories. This loads cats.json if present
    and otherwise creates it using the arXiv SQLite database arxiv_data.db.
    
    INPUT
        data_dir: str = '.data'
            The data directory
    '''
    import json
    cats_path = get_path(data_dir) / 'cats.json'
    if not cats_path.is_file():
        try:
            from db import ArXivDatabase
        except ImportError:
            from .db import ArXivDatabase
        db = ArXivDatabase(data_dir = data_dir)
        db.get_cats()
    with open(cats_path, 'r') as f:
        return json.load(f)

def get_mcat_dict(data_dir: str = '.data') -> list:
    ''' Load the dictionary translating between categories and master
    categories. This loads mcat_dict.json if present and otherwise creates 
    it using the arXiv SQLite database arxiv_data.db.
    
    INPUT
        data_dir: str = '.data'
            The data directory
    '''
    import json
    mcat_dict_path = get_path(data_dir) / 'mcat_dict.json'
    if not mcat_dict_path.is_file():
        try:
            from db import ArXivDatabase
        except ImportError:
            from .db import ArXivDatabase
        db = ArXivDatabase(data_dir = data_dir)
        db.get_mcat_dict()
    with open(mcat_dict_path, 'r') as f:
        return json.load(f)

def get_nrows(fname: str, data_dir: str = '.data') -> int:
    ''' Count the number of rows in a tsv file by streaming it, and thus
    without loading it into memory.
    
    INPUT
        fname: str
            The tsv file whose rows are to be counted, with file extension
        data_dir: str = '.data'
            The data directory
    '''
    import pandas as pd
    path = get_path(data_dir) / fname
    df = pd.read_csv(path, sep = '\t', usecols = [0], chunksize = 10000)
    return sum(len(x) for x in df)

def get_mcats(data_dir: str = '.data') -> list:
    ''' A convenience function that gets the list of master categories,
    by using the list of categories and the master category dictionary.
    
    INPUT
        data_dir: str = '.data'
            The data directory

    OUTPUT
        A list of master categories
    '''
    cats = get_cats(data_dir = data_dir)
    mcat_dict = get_mcat_dict(data_dir = data_dir)
    duplicate_mcats = [mcat_dict[cat] for cat in cats]

    # Get unique master categories while preserving the order
    mcats = list(dict.fromkeys(duplicate_mcats).keys())

    return mcats

def get_mcat_masks(data_dir: str = '.data') -> torch.FloatTensor:
    ''' Create master category masks.
    
    INPUT
        data_dir: str = '.data'
            The data directory

    OUTPUT
        A two-dimensional torch.FloatTensor of shape (num_mcats, num_cats),
        where num_cats and num_cats are the number of master categories and
        categories, respectively. Every slice contains a mask for a given
        master category.
    '''
    cats = get_cats(data_dir = data_dir)
    mcats = get_mcats(data_dir = data_dir)
    mcat_dict = get_mcat_dict(data_dir = data_dir)
    mcat2idx = {mcat: idx for idx, mcat in enumerate(mcats)}
    mcat_idxs = [mcat2idx[mcat] for mcat in mcats]
    dup_cats = torch.FloatTensor([mcat2idx[mcat_dict[cat]] for cat in cats])
    masks = torch.stack([(dup_cats == mcat_idx).float() 
        for mcat_idx in mcat_idxs])
    return masks

def apply_mask(x: torch.FloatTensor, masks: torch.FloatTensor):
    ''' Apply a mask to a tensor. 
    
    INPUT
        x: torch.FloatTensor
            A tensor of shape (*, num_cats)

    OUTPUT
        A tensor of shape (num_mcats, *, num_cats), where each slice along
        the first dimension now has as last dimension the mask for the given
        master category.
    '''
    stacked = torch.stack([x for _ in range(masks.shape[0])], dim = 0)
    return masks.unsqueeze(1) * stacked

def mix_logits(x, y):
    ''' A numerically stable version of
            1 - \sigma^{-1}([1 - \sigma(x)][1 - \sigma(y)])

        INPUT
            x: torch.FloatTensor
                A tensor containing logits
            y: torch.FloatTensor
                A tensor containing logits, of the same shape as x

        OUTPUT
            A torch.FloatTensor of the same shape as x and y, calculated as
                x + y + log(1 + exp(-x) + exp(-y))
    '''
    return x + y + torch.log(1 + torch.exp(-x) + torch.exp(-y))

def cats2mcats(pred: torch.FloatTensor, target: torch.FloatTensor, 
    masks: torch.FloatTensor = None, data_dir: str = '.data'):
    ''' Convert category logits to master category logits.
    
    INPUT
        pred: torch.FloatTensor
            A tensor containing predictions, of size 
            (seq_len, batch_size, num_cats)
        target: torch.FloatTensor
            A tensor containing true values, of size 
            (seq_len, batch_size, num_cats)
        masks: torch.FloatTensor = None
            The master category masks, defaults to computing new masks
            using the get_mcat_masks function
        data_dir: str = '.data'
            The data directory

    OUTPUT
        A pair (mpred, mtarget), both of which are torch.FloatTensor objects
        of size (seq_len, batch_size, num_mcats)
    '''
    if masks is None: masks = get_mcat_masks(data_dir = data_dir)

    shifted_logits = pred + torch.abs(torch.min(pred))
    masked_logits = apply_mask(shifted_logits, masks = masks)
    masked_logits -= torch.abs(torch.min(pred))
    sorted_logits = torch.sort(masked_logits, dim = -1)[0]
    first, second = sorted_logits[:, :, -1], sorted_logits[:, :, -2]
    mpred = mix_logits(first, second).permute(1, 0)

    masked_target = apply_mask(target, masks = masks)
    mtarget = torch.max(masked_target, dim = 2).values.permute(1, 0)
    return mpred, mtarget

def get_class_weights(dl, pbar_width: int = None, data_dir: str = '.data'):
    ''' Compute the category- and master category class weights from a dataset.

    INPUT
        dl: torch.utils.data.DataLoader
            The training dataset
        pbar_width: int = None
            The width of the progress bar. If you are using a Jupyter notebook
            then set this to ~1000
        data_dir: str = '.data'
            The data directory

    OUTPUT
        A dictionary containing
            cat_weights: torch.FloatTensor
                A one-dimensional tensor containing the category class weights
            mcat_weights: torch.FloatTensor
                A one-dimensional tensor containing the master category 
                class weights
    '''
    from tqdm.auto import tqdm
    with tqdm(desc = 'Calculating class weights', ncols = pbar_width,
        total = len(dl) * dl.batch_size) as pbar:
        counts = None
        for _, y in dl:
            if counts is None:
                counts = torch.sum(y, dim = 0) 
            else:
                counts += torch.sum(y, dim = 0)
            pbar.update(dl.batch_size)

        # Adding 1 to avoid zero division
        cat_weights = torch.max(counts) / (counts + 1)

    mcat_masks = get_mcat_masks(data_dir = data_dir)
    mcat_counts = [torch.sum(counts * mask) for mask in mcat_masks]
    mcat_counts = torch.FloatTensor(mcat_counts)

    # Adding 1 to avoid zero division
    mcat_weights = torch.max(mcat_counts) / (mcat_counts + 1)
    return {'cat_weights': cat_weights, 'mcat_weights': mcat_weights}

def boolean(input):
    ''' Convert strings 'true'/'false' into boolean True/False.

    INPUT
        input: str or bool

    OUTPUT
        A bool object which is True if input is 'true' and False 
        if input is 'false' (not case sensitive). If input is already
        of type bool then nothing happens, and if none of the above
        conditions are true then a None object is returned.
    '''
    if isinstance(input, bool): return input
    if isinstance(input, str) and input.lower() == 'true': return True
    if isinstance(input, str) and input.lower() == 'false': return False

def clean(doc: str):
    ''' Clean a document. This removes newline symbols, scare quotes,
        superfluous whitespace and replaces equations with -EQN-. 
        
    INPUT
        doc: str
            A document

    OUTPUT
        The cleaned version of the document
    '''
    import re

    # Remove newline symbols
    doc = re.sub('\n', ' ', doc)

    # Convert LaTeX equations of the form $...$, $$...$$, \[...\]
    # or \(...\) to -EQN-
    dollareqn = '(?<!\$)\${1,2}(?!\$).*?(?<!\$)\${1,2}(?!\$)'
    bracketeqn = '\\[\[\(].*?\\[\]\)]'
    eqn = f'({dollareqn}|{bracketeqn})'
    doc = re.sub(eqn, ' -EQN- ', doc)

    # Remove scare quotes, both as " and \\"
    doc = re.sub('(\\"|")', '', doc)

    # Merge multiple spaces
    doc = re.sub(r' +', ' ', doc)

    return doc.strip()


if __name__ == '__main__':
    pass
