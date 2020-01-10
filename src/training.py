import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from tqdm.auto import tqdm
from utils import get_path

class NestedBCELoss(nn.Module):
    def __init__(self, cat_weights: torch.FloatTensor, 
        mcat_weights: torch.FloatTensor, mcat_ratio: float = 0.5,
        data_dir: str = '.data'):
        super().__init__()
        from utils import get_mcat_masks
        self.masks = get_mcat_masks(data_dir = data_dir)
        self.mcat_ratio = mcat_ratio
        self.cat_weights = cat_weights
        self.mcat_weights = mcat_weights
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        from utils import cats2mcats
        mpred, mtarget = cats2mcats(pred, target, masks = self.masks)
        cat_loss = F.binary_cross_entropy_with_logits(pred, target,
            pos_weight = self.cat_weights)
        mcat_loss = F.binary_cross_entropy_with_logits(mpred, mtarget,
            pos_weight = self.mcat_weights)

        if torch.isnan(cat_loss).any() or torch.isnan(mcat_loss).any():
            print('cat_loss =', cat_loss)
            print('mcat_loss =', mcat_loss)
            print('mpred =', mpred)
            print('mtarget =', mtarget)
            print('pred =', pred)
            print('target =', target)

        return (1 - self.mcat_ratio) * cat_loss + self.mcat_ratio * mcat_loss

    def cuda(self):
        self.masks = self.masks.cuda()
        self.cat_weights = self.cat_weights.cuda()
        self.mcat_weights = self.mcat_weights.cuda()
        return self

def train_model(model, train_dl, val_dl, epochs: int = 10, lr: float = 3e-4,
    mcat_ratio: float = 0.5, data_dir: str = '.data', pbar_width: int = None):
    from sklearn.metrics import f1_score
    import warnings
    from utils import get_mcat_masks, cats2mcats, get_class_weights

    print(f'Training on {len(train_dl) * train_dl.batch_size:,d} samples '\
          f'and validating on {len(val_dl) * val_dl.batch_size:,d} samples.')
    print(f'Number of trainable parameters: {model.trainable_params():,d}')

    weights = get_class_weights(train_dl, pbar_width = pbar_width)
    criterion = NestedBCELoss(**weights, mcat_ratio = mcat_ratio, 
        data_dir = data_dir)
    optimizer = optim.Adam(model.parameters(), lr = lr)
    mcat_masks = get_mcat_masks(data_dir = data_dir)

    if model.is_cuda():
        mcat_masks = mcat_masks.cuda()
        criterion = criterion.cuda()

    best_score = 0
    for epoch in range(epochs):
        with tqdm(total = len(train_dl) * train_dl.batch_size, 
            ncols = pbar_width) as pbar:
            model.train()

            tot_loss, avg_loss = 0, 0
            tot_cat_f1, avg_cat_f1, tot_mcat_f1, avg_mcat_f1 = 0, 0, 0, 0
            for idx, (x_train, y_train) in enumerate(train_dl):
                optimizer.zero_grad()

                if model.is_cuda():
                    x_train = x_train.cuda()
                    y_train = y_train.cuda()

                y_hat = model(x_train)
                preds = torch.sigmoid(y_hat)

                loss = criterion(y_hat, y_train)
                loss.backward()
                optimizer.step()

                tot_loss += float(loss)
                avg_loss = tot_loss / (idx + 1)

                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')

                    tot_cat_f1 += f1_score(preds.cpu() > 0.5, y_train.cpu(), 
                        average = 'samples')
                    avg_cat_f1 = tot_cat_f1 / (idx + 1)

                    my_hat, my_train = cats2mcats(y_hat, y_train, 
                        masks = mcat_masks)
                    mpreds = torch.sigmoid(my_hat)
                    tot_mcat_f1 += f1_score(mpreds.cpu() > 0.5, my_train.cpu(),
                        average = 'samples')
                    avg_mcat_f1 = tot_mcat_f1 / (idx + 1)

                desc = f'Epoch {epoch:2d} - '\
                       f'loss {avg_loss:.4f} - '\
                       f'cat f1 {avg_cat_f1:.4f} - '\
                       f'mcat f1 {avg_mcat_f1:.4f}'
                pbar.set_description(desc)
                pbar.update(train_dl.batch_size)

            with torch.no_grad():
                model.eval()

                val_loss, val_cat_f1, val_mcat_f1 = 0, 0, 0
                y_vals, y_hats = [], []
                for x_val, y_val in val_dl:

                    if model.is_cuda():
                        x_val = x_val.cuda()
                        y_val = y_val.cuda()

                    y_hat = model(x_val)
                    preds = torch.sigmoid(y_hat)

                    val_loss += float(criterion(y_hat, y_val))

                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        val_cat_f1 += f1_score(preds.cpu() > 0.5, y_val.cpu(), 
                            average = 'samples')
                        my_hat, my_val = cats2mcats(y_hat, y_val, 
                            masks = mcat_masks)
                        mpreds = torch.sigmoid(my_hat)
                        val_mcat_f1 += f1_score(mpreds.cpu() > 0.5, 
                            my_val.cpu(), average = 'samples')

                    y_vals.append(y_val)
                    y_hats.append(preds > 0.5)

                y_val = torch.cat(y_vals, dim = 0)
                y_hat = torch.cat(y_hats, dim = 0)

                val_loss /= len(val_dl)
                val_cat_f1 /= len(val_dl)
                val_mcat_f1 /= len(val_dl)

                if val_cat_f1 > best_score:
                    best_score = val_cat_f1
                    model_type = type(model).__name__
                    for f in get_path(data_dir).glob(f'{model_type}*.pt'):
                        f.unlink()

                    data = {
                        'model_type': type(model),
                        'params': model.params,
                        'state_dict': model.state_dict(),
                        'scores': model.evaluate(val_dl, output_dict = True)
                    }
                    model_fname = f'{model_type}_{val_cat_f1 * 100:.2f}_'\
                                  f'{epoch}.pt' 

                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        torch.save(data, get_path(data_dir) / model_fname)

                desc = f'Epoch {epoch:2d} - '\
                       f'loss {avg_loss:.4f} - '\
                       f'cat f1 {avg_cat_f1:.4f} - '\
                       f'mcat f1 {avg_mcat_f1:.4f} - '\
                       f'val_loss {val_loss:.4f} - '\
                       f'val cat f1 {val_cat_f1:.4f} - '\
                       f'val mcat f1 {val_mcat_f1:.4f}'
                pbar.set_description(desc)
                
    return model


if __name__ == '__main__':
    pass
