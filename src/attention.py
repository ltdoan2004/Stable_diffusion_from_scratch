import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, n_heads: int , d_embed: int, in_bias = True, out_bias=True):
        super().__init__()
        self.in_proj = nn.Linear(d_embed, d_embed*3, bias = in_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias = out_bias)
        self.n_heads = n_heads
        self.h_dim = d_embed // n_heads

    def forward(self, x: torch.Tensor, causal_mask = False) -> torch.Tensor:
        x_shape = x.shape

        bs , seg_length, embed_dim = x_shape
        intermim_shape = (bs, seg_length, self.n_heads, self.h_dim)

        q , k , v = self.in_proj(x).chunk(3, dim =-1) #(bs, seg_length, embed_dim)

        q = q.view(intermim_shape).transpose(1,2)     #(bs, n_heads, seg_length, h_dim)
        k = k.view(intermim_shape).transpose(1,2)
        v = v.view(intermim_shape).transpose(1,2)
            
        weight = q @ k.transpose(-1,-2)               #(bs, n_heads, seg_length, seg_length)

        if causal_mask:
            # mask where the upper triangle (above the principal diagonal) is made of 1
            mask = torch.ones_like(weight, dtype = torch.bool).triu(1)
            weight.masked_fill(mask, -torch.inf)

        weight = weight / math.sqrt(self.h_dim)
        weight = F.softmax(weight, dim= -1)

        output = weight @ v         #(bs, n_heads, seg_length, h_dim)

        output = output.transpose(1,2).reshape(x_shape)

        output = self.out_proj(output)

        return output
    
class CrossAttention(nn.Module):
        def __init__(self, n_heads: int, d_embed: int, d_cross: int, in_proj_bias = True, out_proj_bias = True):
            super().__init__()
            self.q_proj = nn.Linear(d_embed, d_embed, bias = in_proj_bias)
            self.k_proj = nn.Linear(d_cross, d_embed, bias = in_proj_bias)
            self.v_proj = nn.Linear(d_cross, d_embed, bias = in_proj_bias)
            self.out_proj = nn.Linear(d_embed, d_embed, bias = out_proj_bias)

            self.n_heads = n_heads
            self.d_head = d_embed // n_heads

        def forward(self, x, y) -> torch.Tensor:
            x_shape = x.shape

            bs , seg_length, embed_dim = x_shape
            intermim_shape = (bs, -1, self.n_heads, self.d_head)

            q = self.q_proj(x)
            k = self.k_proj(y)
            v = self.v_proj(y)

            q = q.view(intermim_shape).transpose(1,2)
            k = k.view(intermim_shape).transpose(1,2)
            v = v.view(intermim_shape).transpose(1,2)

            weight = q @ k.transpose(-1,-2)   
            
            weight = weight / math.sqrt(self.d_head)

            weight = F.softmax(weight, dim =-1)

            output = weight @ v
            
            output = output.transpose(1,2).reshape(x_shape)

            output = self.out_proj(output)

            return output




            

