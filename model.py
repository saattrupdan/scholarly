import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

class Base(nn.Module):
    def trainable_params(self):
        train_params = (param in self.parameters() if param.requires_grad)
        return sum(param.numel() for param in train_params)

class Embedding(Base):
    def __init__(self, vocab_size, emb_dim, emb_matrix):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.embed.weight.data.copy_(emb_matrix)
        self.embed.weight.requires_grad = False

    def forward(self, x):
        return self.embed(x)

class MLP(Base):
    def __init__(self, dim, vocab_size, emb_dim, emb_matrix):
        super().__init__()
        self.embed = Embedding(vocab_size, emb_dim, emb_matrix)
        self.fc = nn.Linear(emb_dim, dim)
        self.out = nn.Linear(dim, 6)

    def forward(self, x):
        x = self.embed(x)
        x = torch.mean(x, dim = 0)
        x = F.elu(self.fc(x))
        return torch.sigmoid(self.out(x)).squeeze()

class RandomForest(Base):
    def __init__(self, vocab_size, emb_dim, emb_matrix, trees_per_batch = 100):
        super().__init__()
        from sklearn.ensemble import RandomForestClassifier
        self.embed = Embedding(vocab_size, emb_dim, emb_matrix)
        self.trees_per_batch = trees_per_batch
        self.forest = RandomForestClassifier(
            n_estimators = self.trees_per_batch
        )

    @staticmethod
    def combine_forests(forest1, forest2):
        forest1.estimators_ += forest2.estimators_
        forest1.n_estimators = len(forest1.estimators_)
        return forest1
    
    def fit(self, train_dl):
        from sklearn.ensemble import RandomForestClassifier
        from functools import reduce
        forests = []
        for x_train, y_train in train_dl:
            rf = RandomForestClassifier(n_estimators = self.trees_per_batch)
            rf.fit(x_train, y_train)
            forests.append(rf)
        self.forest = reduce(combine_forests, forests)
        return self

    def score(self, val_dl):
        from sklearn.metrics import f1_score
        f1s = []
        for x_val, y_val in val_dl:
            f1s.append(f1_score(y_val, y_hat, average = 'samples'))
        return sum(f1s) / len(f1s)

    def predict(self, x):
        return torch.LongTensor(self.forest.predict(x))

    def predict_proba(self, x):
        return torch.FloatTensor(self.forest.predict_proba(x))

def train_nn(nn, train_dl, val_dl, epochs = 10, lr = 3e-4):
    from tqdm.auto import tqdm

    print(f'Training on {len(train_dl) * train_dl.batch_size:,d} samples '\
          f'and validating on {len(val_dl) * val_dl.batch_size:,d} samples.')
    print(f'Number of trainable parameters: {nn.trainable_params():,d}')

    optimizer = optim.Adam(nn.parameters(), lr = lr)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        with tqdm(total = len(train_dl) * train_dl.batch_size) as pbar:
            nn.train()

            tot_loss, avg_loss = 0, 0
            for idx, (x_train, y_train) in enumerate(train_dl):
                optimizer.zero_grad()

                y_hat = nn.forward(x_train)
                loss = criterion(y_hat, y_train)
                loss.backward()
                optimizer.step()

                tot_loss += loss
                avg_loss = tot_loss / (idx + 1)

                pbar.set_description(f'Epoch {epoch:2d} - loss {avg_loss:.4f}')
                pbar.update(train_dl.batch_size)

            with torch.no_grad():
                nn.eval()
                val_loss = 0
                for x_val, y_val in val_dl:
                    y_hat = nn.forward(x_val)
                    val_loss += criterion(y_hat, y_val)
                val_loss /= len(val_dl)

                desc = f'Epoch {epoch:2d} - loss {avg_loss:.4f} - '\
                       f'val_loss {val_loss:.4f}'
                pbar.set_description(desc)

if __name__ == '__main__':
    from data import load_data

    train_dl, val_dl, params = load_data(
        'arxiv_data_mcats_pp',
        vectors = 'fasttext',
        batch_size = 10000
    )

    rf = RandomForest(trees_per_batch = 100, **params)
    rf.fit(train_dl)
    print(rf.score(val_dl))

    #mlp = BaseMLP(dim = 500, **params)
    #train_nn(mlp, train_dl, val_dl, epochs = 50, lr = 3e-4)
