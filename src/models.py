import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

class Base(nn.Module):
    def __init__(self, **params):
        super().__init__()
        self.params = params

    def trainable_params(self):
        train_params = (p for p in self.parameters() if p.requires_grad)
        return sum(param.numel() for param in train_params)

    def is_cuda(self):
        return next(self.parameters()).is_cuda

    def predict(self, title: str, abstract: str):
        pass

    def report(self, val_dl, mcats: bool = True, output_dict: bool = False):
        from sklearn.metrics import classification_report
        import warnings

        with torch.no_grad():
            self.eval()

            y_vals, y_hats = [], []
            for x_val, y_val in val_dl:
                if self.is_cuda():
                    x_val = x_val.cuda()
                    y_val = y_val.cuda()

                y_vals.append(y_val.int())
                y_hats.append((self.forward(x_val) > 0.5).int())
            y_val = torch.cat(y_vals, dim = 0)
            y_hat = torch.cat(y_hats, dim = 0)

            if mcats: 
                cats = ['physics', 'cs', 'math', 'q-bio', 'q-fin', 'stats']
            else:
                cats = []
            
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                report = classification_report(
                    y_true = y_val.cpu(), 
                    y_pred = y_hat.cpu(), 
                    target_names = cats,
                    output_dict = output_dict
                )
            return report

class BertMLP(Base):
    def __init__(self, **params):
        super().__init__(**params)
        self.fcs = nn.ModuleList(
            [nn.Linear(768, params['dim'])] + \
            [nn.Linear(params['dim'], params['dim']) 
                for _ in range(params['nlayers'] - 1)]
        )
        self.out = nn.Linear(params['dim'], 6)

    def forward(self, x):
        for fc in self.fcs:
            x = F.elu(fc(x))
        return self.out(x)

class EmbedLogReg(Base):
    def __init__(self, **params):
        super().__init__(**params)
        self.embed = nn.Embedding(params['vocab_size'], params['emb_dim'])
        self.embed.weight = nn.Parameter(params['emb_matrix'], 
            requires_grad = False)
        self.out = nn.Linear(params['emb_dim'], 6)

    def forward(self, x):
        x = self.embed(x)
        x = x / (torch.norm(x, dim = 2) + 1e-12).unsqueeze(2)
        x = torch.mean(x, dim = 0)
        return self.out(x).squeeze()

class EmbedMLP(Base):
    def __init__(self, **params):
        super().__init__(**params)
        self.embed = nn.Embedding(params['vocab_size'], params['emb_dim'])
        self.embed.weight = nn.Parameter(params['emb_matrix'], 
            requires_grad = False)
        self.fcs = nn.ModuleList(
            [nn.Linear(params['emb_dim'], params['dim'])] + \
            [nn.Linear(params['dim'], params['dim']) 
                for _ in range(params['nlayers'] - 1)]
        )
        self.out = nn.Linear(params['dim'], 6)

    def forward(self, x):
        x = self.embed(x)
        x = x / (torch.norm(x, dim = 2) + 1e-12).unsqueeze(2)
        x = torch.mean(x, dim = 0)
        for fc in self.fcs:
            x = F.elu(fc(x))
        return self.out(x)

class SelfAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        import numpy as np
        self.sqrt_dim = float(np.sqrt(dim))

    def forward(self, inputs):
        inputs = inputs.permute(1, 0, 2)
        x = torch.bmm(inputs, inputs.permute(0, 2, 1))
        x /= self.sqrt_dim
        x = torch.softmax(x, dim = 2)
        x = torch.bmm(x, inputs)
        return x.permute(1, 0, 2)

class EmbedAttnRNN(Base):
    def __init__(self, **params):
        super().__init__(**params)
        self.embed = nn.Embedding(params['vocab_size'], params['emb_dim'])
        self.embed.weight = nn.Parameter(params['emb_matrix'], 
            requires_grad = False)
        self.encoder = nn.GRU(params['emb_dim'], params['dim'],
            bidirectional = True)
        self.attn = SelfAttention(2 * params['dim'])
        self.fc = nn.Linear(2 * params['dim'], params['dim'])
        self.out = nn.Linear(params['dim'], 6)

    def forward(self, x):
        x = self.embed(x)
        x, _ = self.encoder(x)
        x = self.attn(x)
        x = torch.mean(x, dim = 0)
        x = F.elu(self.fc(x))
        return self.out(x)


if __name__ == '__main__':
    from data import load_embed_data, load_bert_data
    from training import train_model

    #train_dl, val_dl = load_bert_data(
    #    fname = 'arxiv_data_mcats_pp_mini',
    #    batch_size = 32,
    #    split_ratio = 0.9
    #)

    train_dl, val_dl, params = load_embed_data(
        tsv_fname = 'arxiv_data_mcats_pp_mini',
        vectors = 'fasttext',
        batch_size = 32,
        split_ratio = 0.9
    )

    model = EmbedMLP(dim = 500, nlayers = 2, **params)
    model = train_model(model, train_dl, val_dl, epochs = 50, lr = 3e-4)
    print(model.report(val_dl, mcats = True))

