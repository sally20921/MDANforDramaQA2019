import torch
from torch import nn
import torch.nn.functional as F

from .modules import Conv1dIn, MultiHeadAttention
from .acc_model import Encoder, Decoder


class TemporalGraph(nn.Module):
    def __init__(self, vocab, n_dim, image_dim, layers, dropout):
        super().__init__()

        self.vocab = vocab
        V = len(vocab)
        D = n_dim
        self.layers = layers
        self.text_embedder = nn.Embedding(V, D)
        self.image_encoder = ImageEncoder(image_dim, D)
        #self.fusers = nn.Sequential(*[AttFuser(D) for i in range(1)])
        self.question_encoder = Encoder(D, layers, dropout)
        self.answer_encoder = Encoder(D, layers, dropout)
        self.decoder = Decoder(D, layers, dropout)

        self.out = nn.Linear(D, 1)

    @classmethod
    def resolve_args(cls, args, vocab):
        return cls(vocab, args.n_dim, args.image_dim, args.layers, args.dropout)

    def forward(self, que, images, answers):
        q = self.text_embedder(que)
        a = self.text_embedder(answers)
        q, h = self.question_encoder(q)
        a_shape = a.shape
        a = a.view(-1, *a_shape[2:]).contiguous()
        a, _ = self.answer_encoder(a)
        h = self.image_encoder(images, h)

        #_, _, a = self.fusers((q, h, a))
        #a = a.mean(dim=-2)
        #a = a.view(*a_shape[:-2], a.shape[-1]).contiguous()
        a = a.view(*a_shape).contiguous()
        h = self.decoder(h.mean(1).unsqueeze(0).repeat(self.layers, 1, 1), a)
        o = self.out(h).squeeze(-1)  # batch answers

        return o


class ImageEncoder(nn.Module):
    def __init__(self, image_dim, D):
        super(ImageEncoder, self).__init__()

        self.in_linear = nn.Linear(image_dim, D)
        self.out_linear = nn.Linear(D * 2, D)
        self.conv1 = Conv1dIn(D * 2, D * 2, 5)

    def forward(self, image, h):
        image = self.in_linear(image)
        # BIC, 2BC
        h = h.mean(dim=0)
        image = torch.cat((image, h.unsqueeze(1).expand(-1, image.shape[1], -1)), dim=-1)
        # BI (C*2)
        image = self.conv1(image)
        image = self.out_linear(image)

        return image


class AttFuser(nn.Module):
    def __init__(self, D):
        super(AttFuser, self).__init__()

        self.layer_norm = nn.LayerNorm(D)
        self.att_x_a = MultiHeadAttention(D, D, D, heads=4)
        self.att_y_a = MultiHeadAttention(D, D, D, heads=4)

    def forward(self, args):
        x, y, a = args
        B = a.shape[0]
        num = B // x.shape[0]
        a_new = self.layer_norm(a)
        a_new = self.att_x_a(a_new, x.repeat(num, 1, 1).contiguous())
        a_new = F.relu(a_new)
        a += a_new

        num = B // y.shape[0]
        a_new = self.layer_norm(a)
        a_new = self.att_y_a(a_new, y.repeat(num, 1, 1).contiguous())
        a_new = F.relu(a_new)
        a += a_new
        return (x, y, a)


