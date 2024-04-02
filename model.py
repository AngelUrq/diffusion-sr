import torch
import torch.nn as nn
import math

class DownBlock(nn.Module):

    def __init__(self, in_channels, out_channels, timestep_embedding_dim=256):
        super().__init__()

        self.pool = nn.AvgPool2d(2)

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.embedding_proj = nn.Sequential(
            nn.Linear(timestep_embedding_dim, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, X, t):
        X = self.pool(X)
        X = self.double_conv(X)
        
        t = self.embedding_proj(t)

        return X + t[:, :, None, None]


class UpBlock(nn.Module):

    def __init__(self, in_channels, out_channels, timestep_embedding_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.embedding_proj = nn.Sequential(
            nn.Linear(timestep_embedding_dim, out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, X, skip, t):
        X = self.up(X)
        X = torch.cat([X, skip], dim=1)
        X = self.double_conv(X)

        t = self.embedding_proj(t)

        return X + t[:, :, None, None]


class SinusoidalPositionEmbeddings(nn.Module):
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

        return embeddings

    
class SelfAttention(nn.Module):
    
    # https://github.com/dome272/Diffusion-Models-pytorch/blob/main/modules.py
    
    def __init__(self, channels, head_size=4):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, head_size, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        size = x.shape[-1]
        x = x.view(-1, self.channels, size * size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, size, size)

    
class UNet(nn.Module):
    
    def __init__(self, filters, in_channels=3, out_channels=3, timestep_embedding_dim=256):
        super().__init__()

        self.position_embeddings = SinusoidalPositionEmbeddings(timestep_embedding_dim)

        self.input_conv = nn.Conv2d(in_channels, filters[0], kernel_size=3, padding=1)

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs_attention = nn.ModuleList()
        self.ups_attention = nn.ModuleList()

        in_channels = filters[0]
        
        for filter in filters[1:]:
            self.downs.append(DownBlock(in_channels, filter))
            self.downs_attention.append(SelfAttention(filter))
            in_channels = filter

        self.downs.append(DownBlock(filters[-1], filters[-1]))
        self.downs_attention.append(SelfAttention(filters[-1]))
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(filters[-1], filters[-1] * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(filters[-1] * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters[-1] * 2, filters[-1] * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(filters[-1] * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters[-1] * 2, filters[-1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(filters[-1]),
        )

        in_channels = filters[-1] * 2
        filters = filters[::-1]

        for filter in filters[1:]:
            self.ups.append(UpBlock(in_channels, filter))
            self.ups_attention.append(SelfAttention(filter))
            in_channels = filter * 2

        self.ups.append(UpBlock(in_channels, filters[-1]))
        self.ups_attention.append(SelfAttention(filters[-1]))

        self.output_conv = nn.Conv2d(filters[-1], out_channels, kernel_size=1)

    def forward(self, X, t):
        t = self.position_embeddings(t)
        X = self.input_conv(X)
        
        skip_connections = []

        # Encoder
        for down, attention in zip(self.downs, self.downs_attention):
            skip_connections.append(X)
            X = down(X, t)
            X = attention(X)

        # Bottleneck
        X = self.bottleneck(X)
        
        # Decoder
        for up, attention in zip(self.ups, self.ups_attention):
            skip = skip_connections.pop()
            X = up(X, skip, t)
            X = attention(X)

        X = self.output_conv(X)

        return X
    
    
class ExponentialMovingAverage:

    def __init__(self, model, beta=0.99):
        self.model = model
        self.beta = beta

    def update(self, model):
        for p1, p2 in zip(self.model.parameters(), model.parameters()):
            p1.data = self.beta * p1.data + (1 - self.beta) * p2.data
