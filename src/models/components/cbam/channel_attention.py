import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, channel: int, reduction_ratio):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction_ratio, channel, kernel_size=1, bias=False),
        )

    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        out = avg_out + max_out
        return torch.sigmoid(out)
