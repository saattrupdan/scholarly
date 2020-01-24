import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from utils import get_cats, get_path, clean

def load_model(path: str):
    checkpoint = torch.load(path, map_location = lambda storage, log: storage)
    model = SHARNN(**checkpoint['params'])
    model.load_state_dict(checkpoint['state_dict'])
    return model, checkpoint['scores']

class Base(nn.Module):
    ''' A base model, with a frozen word embedding layer.

    INPUT
        data_dir: str = '.data'
            The name of the data directory
        pbar_width: str = None
            The width of the progress bar when training. If you are using
            a Jupyter notebook then you should set this to ~1000
        vocab: torchtext.vocab.Vocab
            The vocabulary of the training dataset, containing the word 
            vectors and the conversion dictionary from tokens to indices
    '''
    def __init__(self, **params):
        super().__init__()
        self.data_dir = params.get('data_dir', '.data')
        self.pbar_width = params.get('pbar_width')
        self.params = params
        self.ntargets = len(get_cats(data_dir = self.data_dir)['id'])
        self.stoi = params['vocab'].stoi

        # Embedding layer
        emb_matrix = params['vocab'].vectors
        vocab_size = len(params['vocab'])
        self.emb_dim = params['vocab'].vectors.shape[1]
        self.embed = nn.Embedding(vocab_size, self.emb_dim)
        self.embed.weight = nn.Parameter(emb_matrix, requires_grad = False)

    def trainable_params(self):
        ''' Get the number of trainable parameters of the model. '''
        train_params = (p for p in self.parameters() if p.requires_grad)
        return sum(param.numel() for param in train_params)

    def is_cuda(self):
        ''' Check if the model is stored on the GPU. '''
        return next(self.parameters()).is_cuda

    def evaluate(self, *args, **kwargs):
        ''' Evaluate the performance of the model. See inference.evaluate
            for more details. '''
        from inference import evaluate
        return evaluate(self, *args, **kwargs)

    def predict(self, *args, **kwargs):
        ''' Perform predictions. See inference.predict for more details. '''
        from inference import predict
        return predict(self, *args, **kwargs)

    def fit(self, *args, **kwargs):
        ''' Train the model. See training.train_model for more details. '''
        from training import train_model
        return train_model(self, *args, **kwargs)

class LogReg(Base):
    def __init__(self, **params):
        super().__init__(**params)
        self.fc = nn.Linear(self.emb_dim, self.ntargets)

    def forward(self, x):
        x = self.embed(x)
        x = torch.mean(x, dim = 0)
        x = self.fc(x)
        return x

class BoomBlock(nn.Module):
    ''' A block consisting of two dense layers, one embedding into a high
        dimensional space, and the other projecting back into the dimension
        we started with. A GeLU activation is applied after the first
        embedding, but no activation is applied after the projection.

        INPUT
            dim: int
                The dimension of the input and output
            boom_dim: int
                The dimension of the intermediate space
            boom_normalise: bool = True
                Whether to apply a layer normalisation after embedding into
                the larger space
            boom_dropout: float = 0.
                The amount of dropout to apply after embedding into the
                larger space
            normalise: bool = True
                Whether to apply a layer normalisation after the projection
            dropout: float = 0.
                The amount of dropout to apply after the projection
    '''
    def __init__(self, dim: int, boom_dim: int, boom_normalise: bool = True,
        boom_dropout: float = 0., normalise: bool = True, dropout: float = 0.):
        super().__init__()
        self.boom_up = nn.Linear(dim, boom_dim)
        self.boom_norm = nn.LayerNorm(boom_dim) if boom_normalise else None
        self.boom_drop = nn.Dropout(boom_dropout) if boom_dropout > 0 else None
        self.boom_down = nn.Linear(boom_dim, dim)
        self.norm = nn.LayerNorm(boom_dim) if normalise else None
        self.drop = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, inputs):
        x = F.gelu(self.boom_up(inputs))
        if self.boom_norm is not None:
            x = self.boom_norm(x)
        if self.boom_drop is not None:
            x = self.boom_drop(x)

        x = inputs + self.boom_down(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.drop is not None:
            x = self.drop(x)
        return x

class FCBlock(nn.Module):
    ''' A block of fully connected layers.

    INPUT
        in_dim: int
            The dimension of the input space
        out_dim: int
            The dimension of the output space
        normalise: bool = True
            Whether to apply layer normalisation after the fully connected 
            layers has been applied
        nlayers: int = 1
            The number of fully connected layers
        dropout: float = 0.
            The amount of dropout to apply after the fully connected layers
    '''
    def __init__(self, in_dim: int, out_dim: int, normalise: bool = True,
        nlayers: int = 1, dropout: float = 0.):
        super().__init__()
        self.fcs = nn.ModuleList(
            [nn.Linear(in_dim, out_dim)] + \
            [nn.Linear(out_dim, out_dim) for _ in range(nlayers - 1)]
        )
        self.norm = nn.LayerNorm(out_dim) if normalise else None
        self.drop = nn.Dropout(dropout) if dropout > 0 else None
    
    def forward(self, x):
        for idx, fc in enumerate(self.fcs):
            x = F.gelu(fc(x)) + x if idx > 0 else F.gelu(fc(x))
        if self.norm is not None:
            x = self.norm(x)
        if self.drop is not None:
            x = self.drop(x)
        return x

class BiRNNBlock(nn.Module):
    ''' A block of bidirectional Gated Recurrent Units.

    INPUT
        in_dim: int
            The dimension of the input space
        out_dim: int
            *Half* of the dimension of the output space. The actual
            output dimension will be 2 * out_dim, as the GRUs are 
            bidirectional
        normalise: bool = True
            Whether to apply layer normalisation after the GRU layers
        nlayers: int = 1
            The number of GRU layers
        dropout: float = 0.
            The amount of dropout to apply after the GRU layers
    '''
    def __init__(self, in_dim: int, out_dim: int, normalise: bool = True,
        nlayers: int = 1, dropout: float = 0.):
        super().__init__()
        self.rnn = nn.GRU(in_dim, out_dim, bidirectional = True, 
            num_layers = nlayers)
        self.norm = nn.LayerNorm(2 * out_dim) if normalise else None
        self.drop = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x, h = None):
        x, h = self.rnn(x, h)
        if self.norm is not None:
            x = self.norm(x)
        if self.drop is not None:
            x = self.drop(x)
        return x, h

class SelfAttentionBlock(nn.Module):
    ''' A block of self-attention. The attention used here is the scaled
    dot product attention, which allows inputs to be either two- or three-
    dimensional. Note that this layer has no trainable parameters.

    INPUT
        dim: int
            The dimension of the input- and output space
        normalise: bool = True
            Whether apply layer normalisation to the output
        dropout: float = 0.
            The amount of dropout to apply after the self-attention
    '''
    def __init__(self, dim: int, normalise: bool = True, dropout: float = 0.):
        super().__init__()
        self.sqrt_dim = nn.Parameter(torch.sqrt(torch.FloatTensor([dim])), 
            requires_grad = False)
        self.norm = nn.LayerNorm(dim) if normalise else None
        self.drop = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, inputs):
        if len(inputs.shape) == 2:
            # Special 2d case with shape (batch_size, dim)
            # Treat dim as the sequence length to make sense of the
            # matrix multiplications, and set dim = 1
            # (batch_size, seq_len) -> (batch_size, seq_len, dim)
            reshaped_inputs = inputs.unsqueeze(2)
        else:
            # (seq_len, batch_size, dim) -> (batch_size, seq_len, dim)
            reshaped_inputs = inputs.permute(1, 0, 2)

        # (batch_size, seq_len, dim) -> (batch_size, seq_len, seq_len)
        scores = torch.bmm(reshaped_inputs, reshaped_inputs.permute(0, 2, 1))
        scores /= self.sqrt_dim
        weights = F.softmax(scores, dim = -1)

        # (batch_size, seq_len, seq_len) x (batch_size, seq_len, dim)
        # -> (batch_size, seq_len, dim)
        mix = torch.bmm(weights, reshaped_inputs)

        if len(inputs.shape) == 2:
            # (batch_size, seq_len, dim) -> (batch_size, seq_len)
            out = mix.squeeze()
        else:
            # (batch_size, seq_len, dim) -> (seq_len, batch_size, dim)
            out = mix.permute(1, 0, 2)

        if self.norm is not None:
            out = self.norm(out)
        if self.drop is not None:
            out = self.drop(out)
        return out, weights

class SHARNN(Base):
    ''' A single-block approximation to the SHA-RNN. The inputs are passed
    through an embedding layer and a bidirection GRU, we then attend to the
    three-dimensional outputs of the GRU, then project down to the target
    space, perform another attention (now two-dimensional) and finish off
    with a boom layer. Layer normalisation is applied everywhere.
    '''
    def __init__(self, **params):
        super().__init__(**params)
        self.rnn = BiRNNBlock(self.emb_dim, params['dim'],
            nlayers = params['nlayers'])
        self.seq_attn = SelfAttentionBlock(2 * params['dim'], 
            dropout = params['dropout'])
        self.proj = FCBlock(2 * params['dim'], self.ntargets)
        self.cat_attn = SelfAttentionBlock(self.ntargets,
            dropout = params['dropout'])
        self.boom = BoomBlock(self.ntargets, params['boom_dim'],
            boom_dropout = params['dropout'], normalise = False)

    def forward(self, x):
        x = self.embed(x)
        x, _ = self.rnn(x)
        x, _ = self.seq_attn(x)
        x = torch.sum(x, dim = 0)
        x = self.proj(x)
        x, _ = self.cat_attn(x)
        return self.boom(x)

# This layer is not used at the moment
class LayerNormGRUCell(nn.GRUCell):
    ''' A GRU cell with layer normalisation. '''
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

# This layer does not work at the moment, fails at backprop.
class LayerNormGRU(nn.Module):
    ''' A GRU in which layer normalisation is applied at every time step. '''
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

if __name__ == '__main__':
    pass
