import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange


'''
baseline model:
2-layer single-directional encoder-decoder GRU
fusion with linear layer
'''


class Baseline(nn.Module):
    def __init__(self, vocab, n_dim, image_dim, layers, dropout):
        super().__init__()

        self.vocab = vocab
        V = len(vocab)
        D = n_dim
        self.text_embedder = nn.Embedding(V, D)
        self.image_encoder = MLP(image_dim)
        self.fuser = Fuser(D, image_dim, D)
        self.question_encoder = Encoder(D, layers, dropout)
        self.decoder = Decoder(D, layers, dropout)

    def out(self, o):
        return F.linear(o, self.text_embedder.weight)

    @classmethod
    def resolve_args(cls, args, vocab):
        return cls(vocab, args.n_dim, args.image_dim, args.layers, args.dropout)

    def forward(self, que, images, shifted_target):
        q = self.text_embedder(que)
        t = self.text_embedder(shifted_target)
        h = self.image_encoder(images)
        h = self.fuser(q, h)
        _, h = self.question_encoder(h)
        o = self.decoder(h, t)
        o = self.out(o)

        return o


class Encoder(nn.Module):
    def __init__(self, n_dim, layers, dropout=0):
        super().__init__()

        self.rnn = nn.GRU(n_dim, n_dim, layers, batch_first=True, dropout=dropout)

    def forward(self, x):
        output, hn = self.rnn(x)
        return output, hn


class MLP(nn.Module):
    def __init__(self, n_dim):
        super().__init__()

        self.linear = nn.Linear(n_dim, n_dim)
        self.layer_norm = nn.LayerNorm(n_dim)

    def delta(self, x):
        x = F.relu(x)
        return self.linear(x)

    def forward(self, x):
        return x + self.layer_norm(self.delta(x))


class Fuser(nn.Module):
    def __init__(self, in_dim1, in_dim2, out_dim):
        super().__init__()

        self.linear = nn.Linear(in_dim1 + in_dim2, out_dim)

    def forward(self, x1, x2):
        # BLC, B(L)C
        if x2.dim() < 3:
            x2 = x2.unsqueeze(1).repeat(1, x1.shape[1], 1).contiguous()
        x = torch.cat((x1, x2), dim=-1)

        return self.linear(x)


class Decoder(nn.Module):
    def __init__(self, n_dim, layers, dropout=0):
        super().__init__()

        self.rnn = nn.GRU(n_dim, n_dim, layers, batch_first=True, dropout=dropout)

    def forward(self, h, target_shifted):
        output, h = self.rnn(target_shifted, h)
        return output
