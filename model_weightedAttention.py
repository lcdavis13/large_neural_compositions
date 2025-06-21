import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model_maskedSoftmax import MaskedSoftmax
import model_skipgates as skips


class WeightedAttention(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, x):
        # Q, K, V shape: (B, H, L, D)
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # (B, H, L, L)
        scores = self.dropout(scores) # Apply dropout in same place as vanilla attention, except there's no softmax before it.

        # attn_weights = F.softmax(scores, dim=-1)
        V_scaled = V * x # row-wise scaling of V by x
        output = torch.matmul(scores, V_scaled)  # (B, H, L, D)

        return output


class MultiheadWeightedAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim_factor, attn_dropout, mlp_dropout, dim_qk=None, dim_v=None):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # default and validation of qk & v dimensions
        if dim_qk is None:
            dim_qk = embed_dim
        if dim_v is None:
            dim_v = embed_dim
        assert dim_qk % num_heads == 0, "dim_qk must be divisible by num_heads"
        assert dim_v % num_heads == 0, "dim_v must be divisible by num_heads"

        self.head_qk_dim = dim_qk // num_heads
        self.head_v_dim = dim_v // num_heads

        self.q_proj = nn.Linear(embed_dim, dim_qk)
        self.k_proj = nn.Linear(embed_dim, dim_qk)
        self.v_proj = nn.Linear(embed_dim, dim_v) 

        self.attention = WeightedAttention(attn_dropout)

        # MLP
        self.fcn1 = nn.Linear(dim_v, mlp_dim_factor * dim_v)
        self.dropout = nn.Dropout(mlp_dropout)
        self.gelu = nn.GELU()
        self.fcn2 = nn.Linear(mlp_dim_factor * dim_v, 1)  
        
        # # Output
        # self.masked_softmax = MaskedSoftmax()
        # if learnable_skip:
        #     self.blendskip = skips.BlendSkip()
        # else:
        #     self.blendskip = skips.StaticBlendSkip()

    def forward(self, x, ids):
        B, L, _ = ids.size()

        Q = self.q_proj(ids).view(B, L, self.num_heads, self.head_qk_dim).transpose(1, 2)
        K = self.k_proj(ids).view(B, L, self.num_heads, self.head_qk_dim).transpose(1, 2)
        V = self.v_proj(ids).view(B, L, self.num_heads, self.head_v_dim).transpose(1, 2)

        attn_output = self.attention(Q, K, V, x)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, self.head_v_dim * self.num_heads)

        # Apply MLP
        # Layernorm before MLP? Not sure since it might corrupt the weighted scales.
        y = self.fcn1(attn_output)
        y = self.dropout(y)
        y = self.gelu(y)
        y = self.fcn2(y)

        # Not including for now. Skip followed by softmax wouldn't work well because the skip is already on simplex. Softmax followed by skip would have a hard time reaching target values without extensive incremental iteration. Skip without softmax would be crippling itself because it would have no way to stay on the simplex without knowing the current abundance. Normalization by doing x+f(x)*x could work, but then we're already closing in on incorporating the Replicator Equation into the architecture. Although one benefit of that is we could allow it to choose its own timescale for ODE integration...
        # # Softmax and skip
        # y = self.masked_softmax(y, x)
        # y = self.blendskip(y, x)

        return y


# Note: pytorch and original AIAYN paper do dropout after the softmax inside the attention module, before multiplying against V. This makes the weights no longer sum to 1 interestingly. 
# Important note: Dropout before softmax would not set the weights to zero, it would set them to some "average" value, but they wanted to entirely drop certain attention weights. They would still alter the softmax output though.
# Getting dropped weights and a valid softmax output would require using my masked softmax using Dropout(Ones()) as the mask to achieve that. It's unclear what the computational hit of my masked softmax is, so I'm a bit hesitant to abandon nn.MultiheadAttention for a custom version with masked dropout in my other transformer blocks. Especially because I would sacrifice their efficient C(?) implementation. But it's an area for future exploration.

