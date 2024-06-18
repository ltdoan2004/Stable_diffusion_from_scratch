import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention
class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocabs : int, n_embed : int, n_tokens: int):
        super().__init__()
        self.tokens_embedding = nn.Embedding(n_vocabs, n_embed)
        self.position_embedding = nn.Parameter(torch.zeros(n_tokens, n_embed))

    def forward(self, tokens):
        x = self.tokens_embedding(x)
        x += self.position_embedding(x)
        return x
    
class CLIPLayer(nn.Module):
    def __init__(self, n_head: int, n_embed: int):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(n_embed)
        self.layernorm2 = nn.LayerNorm(n_embed)
        self.linear1 = nn.Linear(n_embed, n_embed * 4)
        self.linear2 = nn.Linear(n_embed * 4, n_embed)
        self.attention = SelfAttention(n_heads=n_head, d_embed=n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residue_1 = x
        x = self.layernorm1(x)
        x = self.attention(x, causal_mask = True)
        x = residue_1 + x

        residue_2 = x
        x = self.layernorm2(x)
        x = self.linear1(x)
        x = x * torch.sigmoid(1.702 * x) #quickGELU activation function
        x = self.linear2(x)
        x = residue_2 + x
        return x



class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = CLIPEmbedding(49408, 768, 77)

        self.layers = nn.Module([
            CLIPLayer(12,768) for i in range(12)
        ])
        
        self.layernorm = nn.LayerNorm(768)

        def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
            tokens = tokens.type(torch.long)
            state =  self.embedding(tokens)
            for layer in self.layers:
                state = layer(state)

            output = self.layernorm(state)

            return output


