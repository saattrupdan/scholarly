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

    def is_cuda(self):
        return next(self.parameters()).is_cuda

    def evaluate(self, val_dl, output_dict: bool = False, 
        data_dir: str = '.data'):
        from inference import evaluate
        return evaluate(self, val_dl, output_dict, data_dir)

    def predict(self, title: str, abstract: str):
        from inference import predict
        return predict(self, title, abstract)

    def fit(self, train_dl, val_dl, epochs: int = 10, lr: float = 3e-4,
        mcat_ratio: float = 0.5, data_dir: str = '.data', 
        pbar_width: int = None):
        from training import train_model
        params = {
            'train_dl': train_dl,
            'val_dl': val_dl,
            'epochs': epochs,
            'lr': lr,
            'mcat_ratio': mcat_ratio,
            'data_dir': data_dir,
            'pbar_width': pbar_width,
            'gpu': self.is_cuda()
        }
        return train_model(self, **params)

class BoomBlock(nn.Module):
    def __init__(self, dim: int, boom_dim: int, normalise: bool = True):
        super().__init__()
        self.boom_up = nn.Linear(dim, boom_dim)
        self.norm = nn.LayerNorm(boom_dim) if normalise else None
        self.boom_down = nn.Linear(boom_dim, dim)

    def forward(self, inputs):
        x = F.gelu(self.boom_up(inputs))
        if self.norm is not None:
            x = self.norm(x)
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
            x = F.gelu(fc(x)) + x if idx > 0 else F.gelu(fc(x))
        if self.norm is not None:
            x = self.norm(x)
        return x

class BiRNNBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, normalise: bool = True,
        nlayers: int = 1):
        super().__init__()
        self.rnn = nn.GRU(in_dim, out_dim, bidirectional = True, 
            num_layers = nlayers)
        self.norm = nn.LayerNorm(2 * out_dim) if normalise else None

    def forward(self, x, h = None):
        x, h = self.rnn(x, h)
        if self.norm is not None:
            x = self.norm(x)
        return x, h

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
            x = F.gelu(conv(x))
        if self.pool is not None:
            x = self.pool(x)
        x = x.permute(2, 0, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x

class LayerNormGRUCell(nn.GRUCell):
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super(LayerNormGRUCell, self).__init__(input_size, hidden_size, bias)
        self.ln_ih = nn.LayerNorm(3 * hidden_size)
        self.ln_hh = nn.LayerNorm(3 * hidden_size)

    def forward(self, input, hx = None):
        self.check_forward_input(input)
        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, 
                dtype = input.dtype, device = input.device)
        self.check_forward_hidden(input, hx, '')

        gi = self.ln_ih(torch.mm(input, self.weight_ih.t()) + self.bias_ih)
        gh = self.ln_hh(torch.mm(hx, self.weight_hh.t()) + self.bias_hh)
        i_r, i_i, i_n = gi.chunk(3, 1)
        h_r, h_i, h_n = gh.chunk(3, 1)

        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hx - newgate)
        return hy

class LayerNormGRU(nn.Module):
    # Does not work at the moment, fails at backprop.
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__()
        torch.autograd.set_detect_anomaly(True)
        self.hidden_size = hidden_size
        self.grucell_f = LayerNormGRUCell(input_size, hidden_size, bias)
        self.grucell_b = LayerNormGRUCell(input_size, hidden_size, bias)

    def forward(self, input, hx = None):
        if hx is None:
            hx = torch.zeros(input.size(1), self.hidden_size, 
                dtype = input.dtype, device = input.device)

        hs_f = torch.zeros(input.size(0), input.size(1), self.hidden_size,
            dtype = input.dtype, device = input.device)
        hs_b = torch.zeros(input.size(0), input.size(1), self.hidden_size,
            dtype = input.dtype, device = input.device)

        seq_len = input.size(0)
        for t in range(seq_len):
            xt = input[t, :, :]
            hf_prev = hs_f[t - 1, :, :] if t != 0 else hx
            hb_prev = hs_b[seq_len - t, :, :] if t != 0 else hx
            hs_f[t, :, :] = self.grucell_f(xt, hf_prev)
            hs_b[seq_len - t - 1, :, :] = self.grucell_f(xt, hb_prev)

        last_f = hs_f[seq_len - 1, :, :]
        last_b = hs_b[seq_len - 1, :, :]
        h_last = torch.cat([last_f, last_b], dim = 1)
        h_all = torch.cat([hs_f, hs_b], dim = 2)

        return h_all, h_last

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

        if inputs.shape[2] == 1:
            # (batch_size, seq_len, dim) -> (batch_size, seq_len)
            out = mix.squeeze()
        else:
            # (batch_size, seq_len, dim) -> (seq_len, batch_size, dim)
            out = mix.permute(1, 0, 2)

        if self.norm is not None:
            out = self.norm(out)

        return out

class SHARNN(Base):
    def __init__(self, **params):
        super().__init__(**params)
        self.rnn = BiRNNBlock(params['emb_dim'], params['dim'],
            normalise = True, nlayers = params['nlayers'])
        self.seq_attn = SelfAttentionBlock(2 * params['dim'], normalise = True)
        self.proj = FCBlock(2 * params['dim'], self.ntargets, normalise = True)
        self.cat_attn = SelfAttentionBlock(self.ntargets, normalise = False)
        self.boom = BoomBlock(self.ntargets, params.get('boom_dim', 512),
            normalise = True)

    def forward(self, x):
        x = self.embed(x)
        x, _ = self.rnn(x)
        x = self.seq_attn(x)
        x = torch.sum(x, dim = 0)
        x = self.proj(x)
        x = x + self.cat_attn(x)
        x = self.boom(x)
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
        self.out = nn.Linear(params['dim'], self.ntargets)
        
    def forward(self, x):
        x = self.embed(x)
        x = torch.mean(x, dim = 0)
        x = self.fc(x)
        return self.out(x)

class CNN(Base):
    def __init__(self, **params):
        super().__init__(**params)
        self.conv = ConvBlock(params['emb_dim'], params['dim'], 
            nlayers = params.get('nlayers', 2), normalise = True)
        self.fc = FCBlock(params['dim'], params['dim'], normalise = True)
        self.out = nn.Linear(params['dim'], self.ntargets)

    def forward(self, x):
        x = self.embed(x)
        x = self.conv(x)
        x = self.fc(x)
        x = torch.mean(x, dim = 0)
        return self.out(x)

class ConvRNN(Base):
    def __init__(self, **params):
        super().__init__(**params)
        self.conv = ConvBlock(params['emb_dim'], params['dim'], 
            nlayers = params.get('nlayers', 2), normalise = True)
        self.rnn = BiRNNBlock(params['dim'], params['dim'],  normalise = True)
        self.seq_attn = SelfAttentionBlock(2 * params['dim'], normalise = True)
        self.proj = FCBlock(2 * params['dim'], self.ntargets, normalise = True)
        self.cat_attn = SelfAttentionBlock(self.ntargets, normalise = False)

    def forward(self, x):
        x = self.embed(x)
        x = self.conv(x)
        x, _ = self.rnn(x)
        x = self.seq_attn(x)
        x = torch.mean(x, dim = 0)
        x = self.proj(x)
        x = self.cat_attn(x)
        return x


if __name__ == '__main__':
    pass
