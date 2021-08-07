import math

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange


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


class MultiHeadAttention(nn.Module):
    def __init__(self, q_dim, k_dim=None, v_dim=None, m_dim=None, heads=1):
        super().__init__()

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
