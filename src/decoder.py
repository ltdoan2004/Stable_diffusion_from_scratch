import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention


class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)
    def forward(self, x:torch.tensor) -> torch.tensor:
        residue = x                 # shape: bs, c, h, w
        bs, c, h, w = x.shape

        x = x.view(bs, c, h*w)      # shape: bs, c, h*w
        x = x.transpose(-1, -2)     # shape: bs, h*w, c

        x = self.attention(x)       # shape: bs, h*w, c 

        x = x.transpose(-1, -2)     # shape: bs, c, h*w
        x = x.view((bs , c, h, w))  # shape: bs, c, h, w 

        x += residue                # shape: bs, c, h, w 

        return x                    # shape: bs, c, h, w 



class VAE_ResidualBlock(nn.Module):
    def __init__ (self ,in_channels , out_channels):
        super().__init__()
        self.group_norm1 = nn.GroupNorm(32,in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1)

        self.group_norm2 = nn.GroupNorm(32,out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels,out_channels,kernel_size =1, padding =0)

    def forward(self, x: torch.tensor) -> torch.tensor:
        residue = x
        x = self.group_norm1(x)
        x = F.silu(x)
        x = self.conv1(x)
        x = self.group_norm2(x)
        x = F.silu(x)
        x = self.conv2(x)

        x = x + self.residual_layer(residue)
        return x

if __name__ == "__main__":
    x = torch.rand(3,128,512,512)
    model = VAE_ResidualBlock(128,128)
    output = model(x)
    print(output.shape)
