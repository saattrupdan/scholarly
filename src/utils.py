from pathlib import Path
import torch

def get_root_path() -> Path:
    ''' Returns project root folder. '''
    return Path.cwd().parent

def get_path(path_name: str) -> Path:
    ''' Returns data folder. '''
    return get_root_path() / path_name

def get_cats(data_dir: str = '.data') -> list:
    import json
    cats_path = get_path(data_dir) / 'cats.json'
    if not cats_path.is_file():
        from db import ArXivDatabase
        db = ArXivDatabase(data_dir = data_dir)
        db.get_cats()
    with open(cats_path, 'r') as f:
        return json.load(f)

def get_mcat_dict(data_dir: str = '.data') -> list:
    import json
    mcat_dict_path = get_path(data_dir) / 'mcat_dict.json'
    if not mcat_dict_path.is_file():
        from db import ArXivDatabase
        db = ArXivDatabase(data_dir = data_dir)
        db.get_mcat_dict()
    with open(mcat_dict_path, 'r') as f:
        return json.load(f)

def get_nrows(fname: str, data_dir: str = '.data') -> int:
    import pandas as pd
    path = get_path(data_dir) / fname
    df = pd.read_csv(path, sep = '\t', usecols = [0], chunksize = 10000)
    return sum(len(x) for x in df)

def get_mcats(data_dir: str = '.data') -> list:
    cats = get_cats(data_dir = data_dir)
    mcat_dict = get_mcat_dict(data_dir = data_dir)
    duplicate_mcats = [mcat_dict[cat] for cat in cats]

    # Get unique master categories while preserving the order
    mcats = list(dict.fromkeys(duplicate_mcats).keys())

    return mcats

def get_mcat_masks(data_dir: str = '.data') -> torch.FloatTensor:
    cats = get_cats(data_dir = data_dir)
    mcats = get_mcats(data_dir = data_dir)
    mcat_dict = get_mcat_dict(data_dir = data_dir)
    mcat2idx = {mcat:idx for idx, mcat in enumerate(mcats)}
    mcat_idxs = [mcat2idx[mcat] for mcat in mcats]
    dup_cats = torch.FloatTensor([mcat2idx[mcat_dict[cat]] for cat in cats])
    masks = torch.stack([(dup_cats == mcat_idx).float() 
        for mcat_idx in mcat_idxs])
    return masks

def apply_mask(x, masks: torch.FloatTensor = None):
    stacked = torch.stack([x for _ in range(masks.shape[0])], dim = 0)
    return masks.unsqueeze(1) * stacked

def inverse_sigmoid(y, epsilon: float = 1e-7):
    return torch.log(y / (1. - y + epsilon))

def mix_logits(x, y):
    ''' Numerically stable version of 
            1 - \sigma^{-1}([1 - \sigma(x)][1 - \sigma(y)])
    '''
    return x + y + torch.log(1 + torch.exp(-x) + torch.exp(-y))

def cats2mcats(pred: torch.FloatTensor, target: torch.FloatTensor, 
    masks: torch.FloatTensor = None, data_dir: str = '.data'):
    from functools import partial

    if masks is None: masks = get_mcat_masks(data_dir = data_dir)

    shifted_logits = pred + torch.abs(torch.min(pred))
    masked_logits = apply_mask(shifted_logits, masks = masks)
    masked_logits -= torch.abs(torch.min(pred))
    sorted_logits = torch.sort(masked_logits, dim = -1)[0]
    first, second = sorted_logits[:, :, -1], sorted_logits[:, :, -2]
    mpred = mix_logits(first, second).permute(1, 0)

    #probs = torch.sigmoid(pred)
    #masked_probs = apply_mask(probs, masks = masks)
    #top3_probs = torch.sort(masked_probs)[0][:, :, -3:]
    #prod_probs = 1 - torch.prod(1 - top3_probs, dim = 2)
    #mpred = inverse_sigmoid(prod_probs).permute(1, 0)

    masked_target = apply_mask(target, masks = masks)
    mtarget = torch.max(masked_target, dim = 2).values.permute(1, 0)
    return mpred, mtarget

def get_class_weights(dl, pbar_width: int = None, data_dir: str = '.data'):
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


if __name__ == '__main__':
    pass
