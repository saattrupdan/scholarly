import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

class BaseRNN(nn.Module):
    def __init__(self, dim, vocab_size, emb_dim, emb_matrix):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.GRU(emb_dim, dim)
        self.fc = nn.Linear(dim, 6)

    def forward(self, x):
        x = self.embed(x)
        _, h = self.rnn(x)
        return torch.sigmoid(self.fc(h)).squeeze()

def train(model, train_dl, val_dl, epochs = 10, lr = 3e-4, ema = 0.999):
    from tqdm.auto import tqdm

    optimizer = optim.Adam(model.parameters(), lr = lr)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        pbar = tqdm(total = len(train_dl) * train_dl.batch_size, leave = True)

        model.train()
        ema_loss = 0
        for x_train, y_train in train_dl:
            optimizer.zero_grad()
            y_hat = model.forward(x_train)

            loss = criterion(y_hat, y_train)
            loss.backward()
            optimizer.step()

            ema_loss = ema * ema_loss + (1 - ema) * float(loss)
            ema_loss /= 1 - ema ** ((1 + epoch) * train_dl.batch_size * 50)

            pbar.set_description(f'Epoch {epoch:2d} - loss {ema_loss:.4f}')
            pbar.update(train_dl.batch_size)

        with torch.no_grad():
            model.eval()
            val_loss = 0
            for x_val, y_val in val_dl:
                y_hat = model.forward(x_val)
                val_loss += criterion(y_hat, y_val)
            val_loss /= len(val_dl)

            desc = f'Epoch {epoch:2d} - loss {ema_loss:.4f} - '\
                   f'val_loss {val_loss:.4f}'
            pbar.set_description(desc)

if __name__ == '__main__':
    from data import load_data
    train_dl, val_dl, params = load_data('arxiv_data_mcats_pp')
    model = BaseRNN(dim = 100, **params)
    train(model, train_dl, val_dl, epochs = 10, lr = 3e-4)
