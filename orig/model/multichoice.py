import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange


'''
multichoice version
baseline model:
2-layer single-directional encoder-decoder GRU
fusion with linear layer
'''


class MultiChoice(nn.Module):
    def __init__(self, args, vocab, n_dim, image_dim, layers, dropout, num_choice=5):
        super().__init__()

        self.text_feature_names = args.text_feature_names
        self.feature_names = args.use_inputs

        self.vocab = vocab
        V = len(vocab)
        D = n_dim
        self.text_embedder = nn.Embedding(V, D)
        self.question_encoder = Encoder(D, layers, dropout)
        self.decoder = Decoder(D, layers, dropout)
        self.answer_fuser = Fuser(D, D, D)

        self.feature_encoders = nn.ModuleDict()
        self.feature_fusers = nn.ModuleDict()
        for name in self.feature_names:
            if name == 'images':
                encoder = MLP(image_dim)
            else:
                encoder = Encoder(D, layers, dropout)
            self.feature_encoders[name] = encoder

            if name == 'images':
                fuser = Fuser(D, image_dim, D)
            else:
                fuser = Fuser(D, D, D)
            self.feature_fusers[name] = fuser

    @classmethod
    def resolve_args(cls, args, vocab):
        return cls(args, vocab, args.n_dim, args.image_dim, args.layers, args.dropout)

    def process_feature(self, q, name, feature):
        if name in self.text_feature_names:
            feature = self.text_embedder(feature)
        feature = self.feature_encoders[name](feature)
        feature = feature.mean(dim=1)
        q = self.feature_fusers[name](q, feature)
        return q

    def forward(self, que, answers, **features):
        q = self.text_embedder(que)
        t = self.text_embedder(answers)
        # pool
        q = self.question_encoder(q)

        for name, feature in sorted(features.items()):
            q = self.process_feature(q, name, feature)

        q = q.mean(dim=1)
        o = self.answer_fuser(t, q)
        # BALC
        o = o.mean(dim=-1).mean(dim=-1)
        # BA
        return o


class Encoder(nn.Module):
    def __init__(self, n_dim, layers, dropout=0):
        super().__init__()

        self.rnn = nn.GRU(n_dim, n_dim, layers, batch_first=True, dropout=dropout)

    def run(self, x):
        output, hn = self.rnn(x)
        hn = hn.transpose(0, 1)
        return output, hn

    def forward(self, x):
        return self.run(x)[1]


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
        while x2.dim() < x1.dim():
            x2 = x2.unsqueeze(1).repeat(1,
                                        x1.shape[x1.dim() - x2.dim()],
                                        *[1 for i in range(x2.dim() - 2)],
                                        1).contiguous()
        x = torch.cat((x1, x2), dim=-1)

        return self.linear(x)


class Decoder(nn.Module):
    def __init__(self, n_dim, layers, dropout=0):
        super().__init__()

        self.rnn = nn.GRU(n_dim, n_dim, layers, batch_first=True, dropout=dropout)

    def forward(self, h, target_shifted):
        output, h = self.rnn(target_shifted, h)
        return output
