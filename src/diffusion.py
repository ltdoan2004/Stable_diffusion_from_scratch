import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention

class TimeEmbedding(nn.Module):
    def __init__(self, d_embed: int):
        super().__init__()
        self.linear_1 = nn.Linear(d_embed, d_embed * 4)
        self.linear_2 = nn.Linear(d_embed * 4, d_embed * 4)

    def forward(self, time: torch.tensor) -> torch.Tensor:
        time = self.linear_1(time)
        time = F.silu(time)
        time = self.linear_2(time)
        return time
    
class SwitchSequential(nn.Sequential):
    def forward(self, x, context, time):
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x

class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2D(channels, channels, kernel_size = 3, padding =1)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x , scale_factor = 2, mode = 'nearest')

        output = self.conv(x)
        
        return output
    
class UNET_OutputLayer(nn.Module):
    def __init__(self, in_channels : int, out_channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding =1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.groupnorm(x)

        x = F.silu(x)

        output = self.conv(x)

        return output
    
class UNET_ResidualBlock(nn.Module):
    def __init__(self, in_channels : int, out_channels: int, n_time = 1280):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding =1)
        self.linear_time = nn.Linear(n_time, out_channels)

        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels,kernel_size = 3, padding =1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size = 1, padding = 0)
        
    def forward(self, feature, time):
        residue = feature

        feature = self.group_norm(feature)

        feature = F.silu(feature)

        feature = self.conv1(feature)

        time = F.silu(time)

        time = self.linear_time(time)

        merged = feature + time.unsqueeze(-1).unsqueeze(-1)

        merged = self.groupnorm_merged(merged)

        merged = F.silu(merged)

        merged = self.conv2(merged)

        output = merged + self.residual_layer(residue)

        return output

class UNET_AttentionBlock(nn.Module):
    def __init__(self, n_heads: int, n_embed: int, d_context = 768):
        super().__init__()
        channels = n_heads * n_embed

        self.group_norm = nn.GroupNorm(32, channels)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size =1, padding =0)

        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_heads, n_embed, in_bias = False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_heads, n_embed, d_context, in_proj_bias = False)
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu1 = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu2 = nn.Linear(2*channels, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size =1, padding = 0)

    def forward(self, x, context) -> torch.Tensor:
        residue_long = x

        x = self.group_norm(x)

        x = self.conv_input(x)

        bs , c , h , w = x.shape

        x =  x.view((bs , c, h*w))

        x = x.transpose((-1, -2))

        residue_short = x

        x = self.layernorm_1(x)
        x = self.attention_1(x)
        x +=  residue_short


        residue_short = x
        x = self.layernorm_2(x)
        x = self.attention_2(x, context)
        x += residue_short

        residue_short = x
        x = self.layernorm_3(x)
        x, gate = self.linear_geglu1(x).chunk(2, dim = -1)
        x = x * F.gelu(gate)
        x = self.linear_geglu2(x)
        x = x + residue_short

        x.transpose(-1,-2)
        x = x.view((bs, c , h, w))

        output = self.conv_output(x) + residue_long

        return output

        


class UNET(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleList([
            #shape: (bs, 4 , h/8, w/8)
            nn.Conv2d(4, 320, kernel_size =3, padding = 1),
            SwitchSequential(UNET_ResidualBlock(320,320), UNET_AttentionBlock(8,40)),
            SwitchSequential(UNET_ResidualBlock(320,320), UNET_AttentionBlock(8,40)),

            nn.Conv2d(320,320, kernel_size = 3, stride =2, padding =1),
            SwitchSequential(UNET_ResidualBlock(320,640), UNET_AttentionBlock(8,80)),
            SwitchSequential(UNET_ResidualBlock(640,640), UNET_AttentionBlock(8,80)),

            nn.Conv2d(640,640, kernel_size = 3, stride =2, padding =1),
            SwitchSequential(UNET_ResidualBlock(640,1280), UNET_AttentionBlock(8,160)),
            SwitchSequential(UNET_ResidualBlock(1280,1280), UNET_AttentionBlock(8,160)),

            nn.Conv2d(1280,1280, kernel_size = 3, stride =2, padding =1),  
            SwitchSequential(UNET_ResidualBlock(1280,1280)),
            SwitchSequential(UNET_ResidualBlock(1280,1280))           

        ])
        self.bottle_neck = SwitchSequential(
            UNET_ResidualBlock(1280,1280),
            UNET_AttentionBlock(8,160),
            UNET_AttentionBlock(1280,1280)
        )
        self.decoders = nn.ModuleList([
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280), Upsample(1280)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8,160)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8,160)),
            SwitchSequential(UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(8,160), Upsample(1280)),
            SwitchSequential(UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(8,80)),
            SwitchSequential(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8,80)),
            SwitchSequential(UNET_ResidualBlock(960, 640), UNET_AttentionBlock(8,80), Upsample(640)),
            SwitchSequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8,40)),
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8,40)),
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8,40)),
        ])
    def forward(self, x, context, time):
        # x: (Batch_Size, 4, Height / 8, Width / 8)
        # context: (Batch_Size, Seq_Len, Dim) 
        # time: (1, 1280)
        skip_connections = []
        for layers in self.encoders:
            x = layers(x, context, time)
            skip_connections.append(x)

        x = self.bottleneck(x, context, time)

        for layers in self.decoders:
            # Since we always concat with the skip connection of the encoder, the number of features increases 
            # before being sent to the decoder's layer
            x = torch.cat((x, skip_connections.pop()), dim=1) 
            x = layers(x, context, time)
        
        return x





class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.unet_out = UNET_OutputLayer(320,4)


    def forward(self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        time_emd = self.time_embedding(time)

        output = self.unet(latent, context, time_emd)

        output = self.unet_out(output)

        return output
        

if __name__ == "__main__":
    x = torch.rand(4,1,512,512)
    y = torch.rand(4,768,32,32)
    z = torch.rand(4, 320)
    model = Diffusion()
    output = model(x)
    print(output.shape)
