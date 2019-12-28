import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

def load_model(path: str):
    checkpoint = torch.load(path, map_location = lambda storage, log: storage)
    model = checkpoint['model_type'](**checkpoint['params'])
    model.load_state_dict(checkpoint['state_dict'])
    return model, checkpoint['scores']

class Base(nn.Module):
    def __init__(self, **params):
        super().__init__()
        from utils import get_cats
        self.params = params
        self.ntargets = len(get_cats())
        self.embed = nn.Embedding(params['vocab_size'], params['emb_dim'])
        self.embed.weight = nn.Parameter(params['emb_matrix'], 
            requires_grad = False)

    def trainable_params(self):
        train_params = (p for p in self.parameters() if p.requires_grad)
        return sum(param.numel() for param in train_params)

    def report(self, val_dl, output_dict: bool = False, 
        data_dir: str = '.data'):
        from inference import get_scores
        return get_scores(self, val_dl, output_dict, data_dir)

    def predict(self, title: str, abstract: str):
        from inference import predict
        return predict(self, title, abstract)

class BoomBlock(nn.Module):
    def __init__(self, dim: int, boom_dim: int):
        super().__init__()
        self.boom_up = nn.Linear(dim, boom_dim)
        self.boom_down = nn.Linear(boom_dim, dim)

    def forward(self, inputs):
        x = F.elu(self.boom_up(inputs))
        return inputs + self.boom_down(x)

class FCBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, normalise: bool = True,
        nlayers: int = 1):
        super().__init__()
        self.fcs = nn.ModuleList(
            [nn.Linear(in_dim, out_dim)] + \
            [nn.Linear(out_dim, out_dim) for _ in range(nlayers - 1)]
        )
        self.norm = nn.LayerNorm(out_dim) if normalise else None
    
    def forward(self, x):
        for idx, fc in enumerate(self.fcs):
            x = F.elu(fc(x)) + x if idx > 0 else F.elu(fc(x))
        if self.norm is not None:
            x = self.norm(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, pool: bool = True,
        normalise: bool = True, nlayers: int = 1):
        super().__init__()
        self.convs = nn.ModuleList(
            [nn.Conv1d(in_dim, out_dim, kernel_size = 3)] + \
            [nn.Conv1d(out_dim, out_dim, kernel_size = 3) 
            for _ in range(nlayers - 1)]
        )
        self.pool = nn.MaxPool1d(kernel_size = 3) if pool else None
        self.norm = nn.LayerNorm(out_dim) if normalise else None

    def forward(self, x):
        x = x.permute(1, 2, 0)
        for conv in self.convs:
            x = F.elu(conv(x))
        if self.pool is not None:
            x = self.pool(x)
        x.permute(2, 0, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x

class RNNBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, normalise: bool = True):
        super().__init__()
        self.rnn = nn.GRU(in_dim, out_dim, bidirectional = True)
        self.norm = nn.LayerNorm(2 * out_dim) if normalise else None

    def forward(self, x, in_h = None):
        x, out_h = self.rnn(x, in_h)
        if self.norm is not None:
            x = self.norm(x)
        return x, out_h

class SelfAttentionBlock(nn.Module):
    def __init__(self, dim: int, normalise: bool = True):
        super().__init__()
        import math
        self.sqrt_dim = math.sqrt(dim)
        self.norm = nn.LayerNorm(dim) if normalise else None

    def forward(self, inputs):
        if len(inputs.shape) == 2:
            # Special 2d case with shape (batch_size, dim)
            # Treat dim as the sequence length to make sense of the
            # matrix multiplications, and set dim = 1
            # (batch_size, seq_len) -> (batch_size, seq_len, dim)
            inputs = inputs.unsqueeze(2)
        else:
            # (seq_len, batch_size, dim) -> (batch_size, seq_len, dim)
            inputs = inputs.permute(1, 0, 2)

        # (batch_size, seq_len, dim) -> (batch_size, seq_len, seq_len)
        scores = torch.bmm(inputs, inputs.permute(0, 2, 1)) / self.sqrt_dim
        weights = F.softmax(scores, dim = -1)

        # (batch_size, seq_len, seq_len) x (batch_size, seq_len, dim)
        # -> (batch_size, seq_len, dim)
        mix = torch.bmm(weights, inputs)

        # (batch_size, seq_len, dim) + (batch_size, seq_len, dim)
        # -> (batch_size, seq_len, dim)
        summed = mix + inputs

        if inputs.shape[2] == 1:
            # (batch_size, seq_len, dim) -> (batch_size, seq_len)
            out = summed.squeeze()
        else:
            # (batch_size, seq_len, dim) -> (seq_len, batch_size, dim)
            out = summed.permute(1, 0, 2)

        if self.norm is not None:
            out = self.norm(out)

        return out

class SHARNN(Base):
    def __init__(self, **params):
        super().__init__(**params)
        self.rnn = RNNBlock(params['emb_dim'], params['dim'])
        self.seq_attn = SelfAttentionBlock(2 * params['dim'])
        self.proj = FCBlock(2 * params['dim'], self.ntargets)
        self.cat_attn = SelfAttentionBlock(self.ntargets)
        #self.boom = BoomBlock(self.ntargets, params.get('boom_dim', 512))

    def forward(self, x):
        x = self.embed(x)
        x, _ = self.rnn(x)
        x = self.seq_attn(x)
        x = torch.mean(x, dim = 0)
        x = self.proj(x)
        x = self.cat_attn(x)
        #x = self.boom(x)
        return x

class LogisticRegression(Base):
    def __init__(self, **params):
        super().__init__(**params)
        self.out = nn.Linear(params['emb_dim'], self.ntargets)
        
    def forward(self, x):
        x = self.embed(x)
        x = torch.mean(x, dim = 0)
        return self.out(x)

class MLP(Base):
    def __init__(self, **params):
        super().__init__(**params)
        self.fc = FCBlock(params['emb_dim'], params['dim'],
            nlayers = params.get('nlayers', 1), normalise = True)
        self.out = nn.Linear(params['emb_dim'], self.ntargets)
        
    def forward(self, x):
        x = self.embed(x)
        x = torch.mean(x, dim = 0)
        x = self.fc(x)
        return self.out(x)

class CNN(Base):
    def __init__(self, **params):
        super().__init__(**params)
        self.conv = ConvBlock(params['emb_dim'], params['dim'], nlayers = 2)
        self.fc = FCBlock(params['dim'], params['dim'])
        self.out = nn.Linear(params['dim'], self.ntargets)

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return self.out(x)


if __name__ == '__main__':
    pass
