import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

def calculate_class_weights(train_dl, pbar_width: int = None):
    from tqdm.auto import tqdm

    counts = None
    for _, y_train in tqdm(train_dl, desc = 'Calculating class weights',
        ncols = pbar_width):
        if counts is None:
            nsamples = y_train.shape[0]
            counts = torch.sum(y_train, dim = 0)
        else:
            nsamples += y_train.shape[0]
            counts += torch.sum(y_train, dim = 0)

    return torch.max(counts) / counts

def train_model(model, train_dl, val_dl, epochs: int = 10, lr: float = 3e-4,
    data_dir: str = 'data', pbar_width: int = None):
    from tqdm.auto import tqdm
    from sklearn.metrics import f1_score
    from pathlib import Path
    import warnings

    print(f'Training on {len(train_dl) * train_dl.batch_size:,d} samples '\
          f'and validating on {len(val_dl) * val_dl.batch_size:,d} samples.')
    print(f'Number of trainable parameters: {model.trainable_params():,d}')

    class_weights = calculate_class_weights(train_dl, pbar_width = pbar_width)
    criterion = nn.BCEWithLogitsLoss(pos_weight = class_weights)
    optimizer = optim.Adam(model.parameters(), lr = lr)

    if model.is_cuda():
        criterion = criterion.cuda()

    best_score = 0
    for epoch in range(epochs):
        with tqdm(total = len(train_dl) * train_dl.batch_size, 
            ncols = pbar_width) as pbar:
            model.train()

            tot_loss, avg_loss, tot_f1, avg_f1 = 0, 0, 0, 0
            for idx, (x_train, y_train) in enumerate(train_dl):
                optimizer.zero_grad()

                if model.is_cuda():
                    x_train = x_train.cuda()
                    y_train = y_train.cuda()

                y_hat = model(x_train)

                loss = criterion(y_hat, y_train)
                loss.backward()
                optimizer.step()

                tot_loss += float(loss)
                avg_loss = tot_loss / (idx + 1)

                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    tot_f1 += f1_score(y_hat.cpu() > 0.5, y_train.cpu(), 
                        average = 'samples')
                    avg_f1 = tot_f1 / (idx + 1)

                desc = f'Epoch {epoch:2d} - loss {avg_loss:.4f} - '\
                       f'sample f1 {avg_f1:.4f}'
                pbar.set_description(desc)
                pbar.update(train_dl.batch_size)

            with torch.no_grad():
                model.eval()

                val_loss, val_sample_f1, val_macro_f1 = 0, 0, 0
                y_vals, y_hats = [], []
                for x_val, y_val in val_dl:
                    if model.is_cuda():
                        x_val = x_val.cuda()
                        y_val = y_val.cuda()

                    y_hat = model(x_val)
                    val_loss += criterion(y_hat, y_val)
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        val_sample_f1 += f1_score(y_hat.cpu() > 0.5, 
                            y_val.cpu(), average = 'samples')

                    y_vals.append(y_val)
                    y_hats.append(y_hat > 0.5)

                y_val = torch.cat(y_vals, dim = 0)
                y_hat = torch.cat(y_hats, dim = 0)

                val_loss /= len(val_dl)
                val_sample_f1 /= len(val_dl)

                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    val_macro_f1 = f1_score(y_val.cpu(), y_hat.cpu(), 
                        average = 'macro')

                if val_macro_f1 > best_score:
                    best_score = val_macro_f1
                    model_type = type(model).__name__
                    for f in Path(data_dir).glob(f'{model_type}*.pt'):
                        f.unlink()

                    data = {
                        'model_type': type(model),
                        'params': model.params,
                        'state_dict': model.state_dict(),
                        'scores': model.report(val_dl, output_dict = True)
                    }
                    model_fname = f'{model_type}_{val_macro_f1 * 100:.2f}_'\
                                  f'{epoch}.pt' 

                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        torch.save(data, Path(data_dir) / model_fname)

                desc = f'Epoch {epoch:2d} - loss {avg_loss:.4f} - '\
                       f'sample f1 {avg_f1:.4f} - val_loss {val_loss:.4f} - '\
                       f'val sample f1 {val_sample_f1:.4f} - '\
                       f'val macro f1 {val_macro_f1:.4f}'
                pbar.set_description(desc)
                
    return model

def load_model(path: str):
    checkpoint = torch.load(path, map_location = lambda storage, log: storage)
    model = checkpoint['model_type'](**checkpoint['params'])
    model.load_state_dict(checkpoint['state_dict'])
    return model, checkpoint['scores']


if __name__ == '__main__':
    pass
