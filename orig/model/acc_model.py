import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange


'''
baseline model:
accuracy variant
'''


class AccModel(nn.Module):
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

        self.out = nn.Linear(D, 1)

    @classmethod
    def resolve_args(cls, args, vocab):
        return cls(vocab, args.n_dim, args.image_dim, args.layers, args.dropout)

    def forward(self, que, images, answers):
        q = self.text_embedder(que)
        a = self.text_embedder(answers)
        images = images.mean(dim=1)
        h = self.image_encoder(images)
        h = self.fuser(q, h)
        _, h = self.question_encoder(h)
        h = self.decoder(h, a)
        o = self.out(h).squeeze(-1)  # batch answers

        return o


class Encoder(nn.Module):
    def __init__(self, n_dim, layers, dropout=0):
        super().__init__()

        self.bidirectional = True

        self.layers = layers
        self.rnn = nn.GRU(n_dim, n_dim, layers, batch_first=True,
                          bidirectional=self.bidirectional, dropout=dropout)

    def forward(self, x):
        output, hn = self.rnn(x)
        if self.bidirectional:
            output = output.view(*x.shape, -1)
            output = output.mean(dim=-1)
            hn = hn.view(self.layers, -1, *hn.shape[1:])
            hn = hn.mean(1)
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

    def forward(self, h, a):
        shape = a.shape
        a = a.view(-1, *a.shape[2:])
        h = h.view(*h.shape[:2], 1, h.shape[-1]).expand(-1, -1, shape[1], -1).contiguous()
        shape = h.shape
        h = h.view(h.shape[0], -1, h.shape[-1]).contiguous()
        output, h = self.rnn(a, h)
        h = h.view(*shape)  # num_layers, batch, answers, C
        h = h.permute(1, 2, 0, 3).mean(dim=2)  # batch, answers, C
        return h
