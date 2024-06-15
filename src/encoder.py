import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
        nn.Conv2d(3, 128, kernel_size = 3, padding =1 ),
        VAE_ResidualBlock(128, 128),
        VAE_ResidualBlock(128, 128),

        nn.Conv2d(3, 128, kernel_size =3, stride = 2, padding =0),
        VAE_ResidualBlock(128, 256),
        VAE_ResidualBlock(256, 256),
    
        nn.Conv2d(3, 128, kernel_size =3, stride = 2, padding = 0),
        VAE_ResidualBlock(256, 512),
        VAE_ResidualBlock(512, 512),
        VAE_ResidualBlock(512, 512),

        VAE_AttentionBlock(512),
        VAE_ResidualBlock(512, 512),
        nn.GroupNorm(32, 512),
        nn.Silu(),

        nn.Conv2d(512,8, kernel_size =3, padding =1),
        nn.Conv2d(8,8,kernel_size =1, padding =0)
        )
        def forward (self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
            for module in self:
                if getattr(module, 'stride', None) == (2, 2):
                    x = F.pad(x, (0,1,0,1))
                x = module(x)
            #devide tensor x of shape(bs, 8, height, width) -> two tensor of shape (bs, 4, height, width)
            mean, log_varience = torch.chunk(x, 2, dim=1)

            log_varience = torch.clamp(log_varience, min = -20, max =30)

            varience = log_varience.exp()

            stdev = varience.sprt()
            # Z =N(0,1) - > N(mean, variance) =X? 
            x = mean + stdev * noise

            x= x * 0.18215

            return x




