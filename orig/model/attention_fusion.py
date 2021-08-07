import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import math 
import numpy as np

""" multi-head self-attention mechanism + position-wise fully connected feed-forward"""

class AttentionFusion(nn.Module):
    def __init__(self, vocab, n_dim, image_dim, layers, dropout, num_choice=5):
        super().__init__()

        self.text_feature_names = args.text_feature_names
        self.feature_names = args.use_inputs

        self.vocab = vocab
        V = len(vocab) #vocab size 
        D = n_dim 
        self.layers = layers

        self.text_embedder = nn.Embedding(V, D) #word embedding
        self.question_encoder = EncoderMain(D, layers, dropout) #attention with q
        self.answer_encoder = Encoder(D, layers, dropout)
        self.image_encoder = ImageEncoder(image_dim, D)
        self.feature_encoder = EncoderMain(D, layers, dropout) #subtitles 
        
        self.decoder = Decoder(D, layers, dropout) #answer selection
        self.fusers = nn.Sequential(*[AttFuser(D) for i in range(1)]) #head smaller than i

        self.out = nn.Linear(D, 1)


    @classmethod
    def resolve_args(cls, args, vocab):
        return cls(args, vocab, args.n_dim, args.image_dim, args.layers, args.dropout)


    def forward(self, que, answers, images, subtitle):
        q = self.text_embedder(que)
        a = self.text_embedder(answers)
          
        h = self.feature_encoder(images) #encode question with image 
        h = self.image_encoder(images, h)
        s = self.feature_encoder(subtitle)
        s = s.mean(dim=1)
        q, h = self.question_encoder(q) #encode question 

        a_shape = a.shape
        a = a.view(-1, *a_shape[2:]).contiguous()
        a, _ = self.answer_encoder(a) 

        _, _, a = self.fusers((q, h, s, a)) #fusion question, answers, images
        a = a.mean(dim=-2)
        a = a.view(*a_shape[:-2], a.shape[-1]).contiguous()
        a = a.view(*a_shape).contiguous()
        h = self.decoder(h.mean(1).unsqueeze(0).repeat(self.layers, 1, 1), a) #choose answer
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
        image = self.multihead(image)
        h = h.mean(dim=0)
        image = torch.cat((image, h.unsqueeze(1).expand(-1, image.shape[1], -1)), dim=-1)

        image = self.conv1(image)
        image = self.out_linear(image)

        return image


class EncoderMain(nn.Module):
    def __init__(self, n_dim, layers, dropout=0):
        super().__init__()

        D = n_dim
        #self.layer_norm = nn.LayerNorm(D)
        self.multihead = MultiHeadAttention(D, D, D,heads=4) #q,k,v
        #self.bidirectional = True
        self.feedforward = FeedForward(D)
        self.layers = layers
        #self.rnn = nn.GRU(n_dim, n_dim, layers, batch_first=True,
        #                  bidirectional=self.bidirectional, dropout=dropout)

    def forward(self, x): 
        output, hn = self.multihead(x)
        #output = self.layer_norm(output)
        output = self.feedforward(output)
        #ouput = self.layer_norm(output+x)
        return output, hn

#position wise feed forward networks
class FeedForward(nn.Module):
    def __init__(self, n_dim):
        super().__init__()

        self.linear = nn.Linear(n_dim, n_dim)
        self.layer_norm = nn.LayerNorm(n_dim)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(n_dim)

    def delta(self, x):
        x = F.relu(x)
        return self.linear(x)

    def forward(self, x):
        out = self.linear(self.relu(self.linear(x)))
        out = self.layer_norm(x+out)
        return out
        #return x + self.layer_norm(self.delta(x))


class Residual_Block(nn.Module):
    def __init__(self, n_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_dim, n_dim),
            nn.Tanh(),
            nn.Linear(n_dim, n_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)+x

class Fuser2(nn.Module):
    def __init__(self, in_dim1, in_dim2, out_dim):
        super().__init__()

        self.first = Fuser1(in_dim1)
        self.second = Fuser1(in_dim2)
        self.linear = nn.Linear(in_dim1, in_dim1)

    def forward(self, x1, x2):
        x1 = self.first(x1)
        x2 =  self.second(self.second(x2))
        out = torch.mul(x1, x2)
        out += self.linear(x1)
        
        # BLC, B(L)C
        #if x2.dim() < 3:
        #    x2 = x2.unsqueeze(1).repeat(1, x1.shape[1], 1).contiguous()
        #x = torch.cat((x1, x2), dim=-1)

        return out


class ResidualFuser(nn.Module):
    def __init__(self, q_dim, v_dim, c_dim, out_dim):
        super().__init__()

        self.tanh = nn.Tanh()

        self.linear_q = nn.Linear(q_dim, q_dim)
        self.linear_v= nn.Linear(v_dim, v_dim)
        self.linear_c = nn.Linear(c_dim, c_dim)

        self.fuse2 = self.Fuse2

        self.Q_v = nn.Sequential(
            fuse2(q_dim, v_dim),
            fuse2(q_dim, v_dim),
        )

        self.Q_c = nn.Sequential(
            fuse2(q_dim, c_dim),
            fuse2(q_dim, c_dim),
        )
       

    def forward(self, args):
        q, v, c, a = args
        q1  = self.Q_v(q, v)
        q2 = self.Q_c(q, c)
        a = torch.cat((q1,q2),dim=-1)

        return a

class Conv1dIn(nn.Module):
    def __init__(self, in_dim, out_dim, num_nodes):
        super().__init__()

        self.project_k = Conv(1, in_dim, num_nodes, 1)
        self.project_v = Conv(1, in_dim, out_dim, 1)

    def forward(self, m):
        # (batch, num_rois, v_dim(+4))
        k = F.softmax(self.project_k(m).transpose(-1, -2), dim=-1)
        v = self.project_v(m)
        return torch.einsum('bnr,brd->bnd', k, v)

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


class Conv1dIn(nn.Module):
    def __init__(self, in_dim, out_dim, num_nodes):
        super().__init__()

        self.project_k = Conv(1, in_dim, num_nodes, 1)
        self.project_v = Conv(1, in_dim, out_dim, 1)

    def forward(self, m):
        # (batch, num_rois, v_dim(+4))
        k = F.softmax(self.project_k(m).transpose(-1, -2), dim=-1)
        v = self.project_v(m)
        return torch.einsum('bnr,brd->bnd', k, v)


class Conv(nn.Module):
    '''
    main purpose of this variation
    is as a wrapper to exchange channel dimension to the last
    (BC*) -> (B*C)
    plus n-d conv option
    '''
    def __init__(self, d, *args, **kwargs):
        super().__init__()

        assert d in [1, 2, 3]
        self.d = d
        self.conv = getattr(nn, "Conv{}d".format(self.d))(*args, **kwargs)

    def forward(self, x):
        x = torch.einsum('b...c->bc...', x)
        x = self.conv(x)
        x = torch.einsum('bc...->b...c', x)

        return x

class AttEncoder(nn.Module):
    def __init__(self, n_dim, layers, dropout=0):
        super().__init__()
        D = n_dim
        self.layer_norm = nn.LayerNorm(D)
        self.multihead = MultiHeadAttention(D, D, D,heads=4)
        self.bidirectional = True

        self.layers = layers
        self.rnn = nn.GRU(n_dim, n_dim, layers, batch_first=True,
                          bidirectional=self.bidirectional, dropout=dropout)

    def forward(self, x):
        output, hn = self.multihead(x)
        output = self.layer_norm(output)
        output = self.multihead(output)
        ouput = self.layer_norm(output+x)
        return output, hn



class MultiHeadAttention(nn.Module):
    def __init__(self, q_dim, k_dim=None, v_dim=None, m_dim=None, heads=1):
        super().__init__()
        #q queries, k key, v value, m mask 
        if k_dim is None:
            k_dim = q_dim
        if v_dim is None:
            v_dim = k_dim
        if m_dim is None:
            m_dim = q_dim

        heads = 1 if q_dim < heads else heads
        heads = 1 if k_dim < heads else heads
        heads = 1 if v_dim < heads else heads

        for name, dim in zip(['q_dim', 'k_dim', 'v_dim', 'm_dim'], [q_dim, k_dim, v_dim, m_dim]):
            assert dim % heads == 0, "{}: {} / n_heads: {} must be divisible".format(name, dim, heads)

        self.q = nn.Linear(q_dim // heads, m_dim // heads)
        self.k = nn.Linear(k_dim // heads, m_dim // heads)
        self.v = nn.Linear(v_dim // heads, m_dim // heads)
        self.heads = heads

    def forward(self, q, k=None, v=None, bidirectional=False):
        if k is None:
            k = q.clone()
        if v is None:
            v = k.clone()
        # BLC

        q = rearrange(q, 'b q (h c) -> b h q c', h=self.heads)
        k = rearrange(k, 'b k (h c) -> b h k c', h=self.heads)
        v = rearrange(v, 'b k (h c) -> b h k c', h=self.heads)

        q = self.q(q)
        k = self.k(k)
        v = self.v(v)

        a = torch.einsum('bhqc,bhkc->bhqk', q, k)
        a = a / math.sqrt(k.shape[-1])
        a_q = F.softmax(a, dim=-1)  # bhqk
        q_new = torch.einsum('bhqk,bhkc->bhqc', a_q, v)
        q_new = rearrange(q_new, 'b h q c -> b q (h c)')

        if bidirectional:
            a_v = F.softmax(a, dim=-2)  # bhqk
            v = torch.einsum('bhqk,bhqc->bhkc', a_v, q)
            v = rearrange(v, 'b h k c -> b k (h c)')
            return q_new, v
        else:
            return q_new


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

