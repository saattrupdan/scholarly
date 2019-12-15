import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

class BaseMLP(nn.Module):
    def __init__(self, dim, vocab_size, emb_dim, emb_matrix):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.embed.weight.data.copy_(emb_matrix)
        self.embed.weight.requires_grad = False
        self.fc = nn.Linear(emb_dim, dim)
        self.out = nn.Linear(dim, 6)

    def trainable_params(self):
        ''' Get the number of trainable parameters in the model. '''
        return sum(param.numel() for param in self.parameters() 
                if param.requires_grad)

    def forward(self, x):
        x = self.embed(x)
        x = torch.mean(x, dim = 0)
        x = F.elu(self.fc(x))
        return torch.sigmoid(self.out(x)).squeeze()

def train(model, train_dl, val_dl, epochs = 10, lr = 3e-4):
    from tqdm.auto import tqdm

    print(f'Training on {len(train_dl) * train_dl.batch_size:,d} samples '\
          f'and validating on {len(val_dl) * val_dl.batch_size:,d} samples.')
    print(f'Number of trainable parameters: {model.trainable_params():,d}')

    optimizer = optim.Adam(model.parameters(), lr = lr)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        with tqdm(total = len(train_dl) * train_dl.batch_size) as pbar:
            model.train()

            tot_loss, avg_loss = 0, 0
            for idx, (x_train, y_train) in enumerate(train_dl):
                optimizer.zero_grad()

                y_hat = model.forward(x_train)
                loss = criterion(y_hat, y_train)
                loss.backward()
                optimizer.step()

                tot_loss += loss
                avg_loss = tot_loss / (idx + 1)

                pbar.set_description(f'Epoch {epoch:2d} - loss {avg_loss:.4f}')
                pbar.update(train_dl.batch_size)

            with torch.no_grad():
                model.eval()
                val_loss = 0
                for x_val, y_val in val_dl:
                    y_hat = model.forward(x_val)
                    val_loss += criterion(y_hat, y_val)
                val_loss /= len(val_dl)

                desc = f'Epoch {epoch:2d} - loss {avg_loss:.4f} - '\
                       f'val_loss {val_loss:.4f}'
                pbar.set_description(desc)

if __name__ == '__main__':
    from data import load_data
    train_dl, val_dl, params = load_data('arxiv_data_mcats_pp')
    model = BaseMLP(dim = 500, **params)
    train(model, train_dl, val_dl, epochs = 100, lr = 3e-4)
