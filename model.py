import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

class Base(nn.Module):
    def trainable_params(self):
        train_params = (p for p in self.parameters() if p.requires_grad)
        return sum(param.numel() for param in train_params)

class MLP(Base):
    def __init__(self, dim, vocab_size, emb_dim, emb_matrix):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.embed.weight = nn.Parameter(emb_matrix, requires_grad = False)
        self.fc = nn.Linear(emb_dim, dim)
        self.out = nn.Linear(dim, 6)

    def forward(self, x):
        x = self.embed(x)
        x = x / (torch.norm(x, dim = 2,) + 1e-12)
        x = torch.mean(x, dim = 0)
        x = F.elu(self.fc(x))
        return torch.sigmoid(self.out(x)).squeeze()

def train_model(model, train_dl, val_dl, epochs: int = 10, lr: float = 3e-4):
    from tqdm.auto import tqdm
    from sklearn.metrics import f1_score

    print(f'Training on {len(train_dl) * train_dl.batch_size:,d} samples '\
          f'and validating on {len(val_dl) * val_dl.batch_size:,d} samples.')
    print(f'Number of trainable parameters: {model.trainable_params():,d}')

    optimizer = optim.Adam(model.parameters(), lr = lr)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        with tqdm(total = len(train_dl) * train_dl.batch_size) as pbar:
            model.train()

            tot_loss, avg_loss, tot_f1, avg_f1 = 0, 0, 0, 0
            for idx, (x_train, y_train) in enumerate(train_dl):
                optimizer.zero_grad()

                y_hat = model.forward(x_train)
                loss = criterion(y_hat, y_train)
                loss.backward()
                optimizer.step()

                tot_loss += loss
                avg_loss = tot_loss / (idx + 1)

                tot_f1 += f1_score(y_hat > 0.5, y_train, average = 'samples')
                avg_f1 = tot_f1 / (idx + 1)

                desc = f'Epoch {epoch:2d} - loss {avg_loss:.4f} - '\
                       f'f1 {avg_f1:.4f}'
                pbar.set_description(desc)
                pbar.update(train_dl.batch_size)

            with torch.no_grad():
                model.eval()
                val_loss, val_f1 = 0, 0
                for x_val, y_val in val_dl:
                    y_hat = model.forward(x_val)
                    val_loss += criterion(y_hat, y_val)
                    val_f1 += f1_score(y_hat > 0.5, y_val, average = 'samples')
                val_loss /= len(val_dl)
                val_f1 /= len(val_dl)

                desc = f'Epoch {epoch:2d} - loss {avg_loss:.4f} - '\
                       f'f1 {avg_f1:.4f} - val_loss {val_loss:.4f} - '\
                       f'val_f1 {val_f1:.4f}'
                pbar.set_description(desc)

if __name__ == '__main__':
    from data import load_data

    train_dl, val_dl, params = load_data(
        tsv_fname = 'arxiv_data_mcats_pp',
        vectors = 'fasttext',
        batch_size = 32,
        split_ratio = 0.98
    )

    mlp = MLP(dim = 100, **params)
    train_model(mlp, train_dl, val_dl, epochs = 5, lr = 3e-4)
