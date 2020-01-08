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
    with open(get_path(data_dir) / 'cats.json', 'r') as f:
        return json.load(f)

def get_mcat_dict(data_dir: str = '.data') -> list:
    import json
    with open(get_path(data_dir) / 'mcat_dict.json', 'r') as f:
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
    cats, mcats, mcat_dict = get_cats(), get_mcats(), get_mcat_dict()
    mcat2idx = {mcat:idx for idx, mcat in enumerate(mcats)}
    mcat_idxs = [mcat2idx[mcat] for mcat in mcats]
    dup_cats = torch.FloatTensor([mcat2idx[mcat_dict[cat]] for cat in cats])
    masks = torch.stack([(dup_cats == mcat_idx).float() for mcat_idx in mcat_idxs])
    return masks

def apply_mask(x, masks: torch.FloatTensor = None):
    stacked = torch.stack([x for _ in range(masks.shape[0])])
    masked = masks.unsqueeze(1) * stacked
    return masked

def inverse_sigmoid(y, epsilon: float = 1e-12):
    return -torch.log(1. / (y + epsilon) - 1.)

def cats2mcats(pred: torch.FloatTensor, target: torch.FloatTensor, 
    masks: torch.FloatTensor = None, data_dir: str = '.data'):
    from functools import partial

    if masks is None: masks = get_mcat_masks(data_dir = data_dir)

    probs = torch.sigmoid(pred)
    masked_probs = apply_mask(probs, masks = masks)
    tops_removed = masked_probs.where(masked_probs < 0.9, torch.full_like(masked_probs, 0.9))
    prod_probs = 1 - torch.prod(1 - torch.sort(tops_removed)[0][:, :, -2:], 
        dim = 2)
    mpred = inverse_sigmoid(prod_probs).permute(1, 0)

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
        cat_weights = torch.max(counts) / counts

    mcat_masks = get_mcat_masks()
    mcat_counts = [torch.sum(counts * mask) for mask in mcat_masks]
    mcat_counts = torch.FloatTensor(mcat_counts)
    mcat_weights = torch.max(mcat_counts) / mcat_counts
    return {'cat_weights': cat_weights, 'mcat_weights': mcat_weights}


if __name__ == '__main__':
    print(get_mcat_masks())
