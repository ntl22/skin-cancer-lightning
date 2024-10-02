import torch
import torch.nn as nn

from .channel_attention import ChannelAttention
from .spatial_attention import SpatialAttention


class CBAM(nn.Module):
    def __init__(self, channel_in: int, channel_out: int, reduction_ratio: int = 16):
        super().__init__()

        self.feature_extraction = nn.Sequential(
            nn.Conv2d(channel_in, channel_out, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(channel_out),
            nn.Hardswish(),
            nn.Conv2d(channel_out, channel_out, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(channel_out),
        )

        self.channel_attention = ChannelAttention(channel_out, reduction_ratio)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        out = self.feature_extraction(x)
        out = out * self.channel_attention(out)
        out = out * self.spatial_attention(out)
        return out
