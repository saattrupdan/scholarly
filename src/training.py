import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from tqdm.auto import tqdm
from utils import get_path

class NestedBCELoss(nn.Module):
    ''' A nested form of binary cross entropy.

    From the category predictions it pulls out the master category
    predictions, using the utils.cats2mcats function, which enables
    a positive master category prediction even though all individual
    category predictions within that master category have sigmoid values
    less than 0.50.

    It then computes the binary cross entropy of the category- and master
    category predictions, with the given class weights, and scales the
    two losses in accordance with mcat_ratio.

    INPUT
        cat_weights: torch.FloatTensor
            The class weights for the categories
        mcat_weights: torch.FloatTensor
            The class weights for the master categories
        mcat_ratio: float = 0.1
            The ratio between the category loss and the master category loss
        data_dir: str = '.data'
            The path to the data directory
    '''
    def __init__(self, cat_weights, mcat_weights, mcat_ratio: float = 0.1,
        data_dir: str = '.data'):
        super().__init__()
        from utils import get_mcat_masks
        self.masks = get_mcat_masks(data_dir = data_dir)
        self.cat_weights = cat_weights
        self.mcat_weights = mcat_weights
        self.mcat_ratio = mcat_ratio
        self.data_dir = data_dir
    
    def forward(self, pred, target, weighted: bool = True):
        from utils import cats2mcats
        mpred, mtarget = cats2mcats(pred, target, masks = self.masks,
            data_dir = self.data_dir)

        cat_loss = F.binary_cross_entropy_with_logits(pred, target,
            pos_weight = self.cat_weights if weighted else None)
        mcat_loss = F.binary_cross_entropy_with_logits(mpred, mtarget,
            pos_weight = self.mcat_weights if weighted else None)
        
        cat_loss *= 1 - self.mcat_ratio
        mcat_loss *= self.mcat_ratio

        return cat_loss + mcat_loss

    def cuda(self):
        self.masks = self.masks.cuda()
        self.cat_weights = self.cat_weights.cuda()
        self.mcat_weights = self.mcat_weights.cuda()
        return self

def train_model(model, train_dl, val_dl, epochs: int = 10, lr: float = 3e-4,
    name: str = 'no_name', mcat_ratio: float = 0.1, ema: float = 0.99, 
    pbar_width: int = None, use_wandb: bool = True):
    ''' Train a given model. 
    
    INPUT
        model: torch.nn.Module
            The model we would like to train
        train_dl: torch.utils.data.DataLoader
            A dataloader containing the training set
        val_dl : torch.utils.data.DataLoader
            A dataloader containing the validation set
        epochs: int = 10
            The amount of epochs to train
        lr: float = 3e-4
            The learning rate used
        name: str = 'no_name'
            The name of the training run, used for wandb purposes
        mcat_ratio: float = 0.1
            How much the master category loss is prioritised over the
            category loss
        ema: float = 0.99
            The fact used in computing the exponential moving averages of
            the loss and sample-average F1 scores. Roughly corresponds to
            taking the average of the previous 1 / (1 - ema) many batches
        pbar_width: int = None
            The width of the progress bar. If running in a Jupyter notebook
            then this should be set to ~1000
        use_wandb: bool = True
            Whether to use the Weights & Biases online performance recording

    OUTPUT
        The trained model
    '''
    from sklearn.metrics import f1_score
    import warnings
    from pathlib import Path
    from utils import get_mcat_masks, cats2mcats, get_class_weights

    print(f'Training on {len(train_dl) * train_dl.batch_size:,d} samples '\
          f'and validating on {len(val_dl) * val_dl.batch_size:,d} samples.')
    print(f'Number of trainable parameters: {model.trainable_params():,d}')

    # Sign into wandb and log metrics from model
    if use_wandb:
        import wandb
        config = {
            'name': name,
            'mcat_ratio': mcat_ratio, 
            'epochs': epochs, 
            'lr': lr,
            'batch_size': train_dl.batch_size,
            'ema': ema,
            'vectors': train_dl.vectors,
            'dropout': model.params['dropout'],
            'nlayers': model.params['nlayers'],
            'dim': model.params['dim'],
            'boom_dim': model.params['boom_dim'],
            'emb_dim': model.params['vocab'].vectors.shape[1],
        }
        wandb.init(project = 'scholarly', config = config)
        wandb.watch(model)

    weights = get_class_weights(train_dl, pbar_width = model.pbar_width, 
        data_dir = model.data_dir)
    criterion = NestedBCELoss(**weights, mcat_ratio = mcat_ratio,
        data_dir = model.data_dir)
    optimizer = optim.Adam(model.parameters(), lr = lr)
    mcat_masks = get_mcat_masks(data_dir = model.data_dir)

    if model.is_cuda():
        mcat_masks = mcat_masks.cuda()
        criterion = criterion.cuda()

    avg_loss, avg_cat_f1, avg_mcat_f1, best_score = 0, 0, 0, 0
    for epoch in range(epochs):
        with tqdm(total = len(train_dl) * train_dl.batch_size, 
            ncols = model.pbar_width) as pbar:
            model.train()

            for idx, (x_train, y_train) in enumerate(train_dl):
                optimizer.zero_grad()

                if model.is_cuda():
                    x_train = x_train.cuda()
                    y_train = y_train.cuda()

                # Get cat predictions
                y_hat = model(x_train)
                preds = torch.sigmoid(y_hat)

                # Get master cat predictions
                my_hat, my_train = cats2mcats(y_hat, y_train, 
                    masks = mcat_masks, data_dir = model.data_dir)
                mpreds = torch.sigmoid(my_hat)

                # Calculate loss and perform backprop
                loss = criterion(y_hat, y_train)
                loss.backward()
                optimizer.step()

                # Compute f1 scores
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    cat_f1 = f1_score(preds.cpu() > 0.5, y_train.cpu(), 
                        average = 'samples')
                    mcat_f1 = f1_score(mpreds.cpu() > 0.5, my_train.cpu(),
                        average = 'samples')

                # Keep track of the current iteration index
                iteration = epoch * len(train_dl) * train_dl.batch_size
                iteration += idx * train_dl.batch_size

                # Exponentially moving average of loss and f1 scores
                avg_loss = ema * avg_loss + (1 - ema) * float(loss)
                avg_loss /= 1 - ema ** (iteration / (1 - ema) + 1)
                avg_cat_f1 = ema * avg_cat_f1 + (1 - ema) * float(cat_f1)
                avg_cat_f1 /= 1 - ema ** (iteration / (1 - ema) + 1)
                avg_mcat_f1 = ema * avg_mcat_f1 + (1 - ema) * float(mcat_f1)
                avg_mcat_f1 /= 1 - ema ** (iteration / (1 - ema) + 1)

                # Log wandb
                if use_wandb:
                    wandb.log({
                        'loss': avg_loss, 
                        'cat f1': avg_cat_f1,
                        'mcat f1': avg_mcat_f1
                    })

                # Update the progress bar
                desc = f'Epoch {epoch:2d} - '\
                       f'loss {avg_loss:.4f} - '\
                       f'cat f1 {avg_cat_f1:.4f} - '\
                       f'mcat f1 {avg_mcat_f1:.4f}'
                pbar.set_description(desc)
                pbar.update(train_dl.batch_size)

            # Compute validation scores
            with torch.no_grad():
                model.eval()

                val_loss, val_cat_f1, val_mcat_f1 = 0, 0, 0
                y_vals, y_hats = [], []
                for x_val, y_val in val_dl:

                    if model.is_cuda():
                        x_val = x_val.cuda()
                        y_val = y_val.cuda()

                    # Get cat predictions
                    y_hat = model(x_val)
                    preds = torch.sigmoid(y_hat)

                    # Get mcat predictions
                    my_hat, my_val = cats2mcats(y_hat, y_val, 
                        masks = mcat_masks, data_dir = model.data_dir)
                    mpreds = torch.sigmoid(my_hat)

                    # Collect the true and predicted labels
                    y_vals.append(y_val)
                    y_hats.append(preds > 0.5)

                    # Accumulate loss
                    val_loss += float(criterion(y_hat,y_val, weighted = False))

                    # Accumulate f1 scores
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        val_cat_f1 += f1_score(preds.cpu() > 0.5, y_val.cpu(), 
                            average = 'samples')
                        val_mcat_f1 += f1_score(mpreds.cpu() > 0.5, 
                            my_val.cpu(), average = 'samples')

                # Concatenate the true and predicted labels
                y_val = torch.cat(y_vals, dim = 0)
                y_hat = torch.cat(y_hats, dim = 0)

                # Compute the average loss and f1 scores
                val_loss /= len(val_dl)
                val_cat_f1 /= len(val_dl)
                val_mcat_f1 /= len(val_dl)

                # Log wandb
                if use_wandb:
                    wandb.log({
                        'val loss': val_loss, 
                        'val cat f1': val_cat_f1,
                        'val mcat f1': val_mcat_f1
                    })

                # If the current cat f1 score is the best so far, then
                # replace the stored model with the current one
                if val_cat_f1 > best_score:
                    best_score = val_cat_f1
                    model_type = type(model).__name__
                    glob = get_path(model.data_dir).glob(f'{model_type}*.pt')
                    for f in glob:
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
                        path = get_path(model.data_dir) / model_fname
                        torch.save(data, path)

                    # Save the model's state dict to wandb directory
                    if use_wandb:
                        wandb_path = Path(wandb.run.dir) / model_fname
                        torch.save(model.state_dict(), wandb_path)
                        wandb.save(model_fname)

                # Update progress bar
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
