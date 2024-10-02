from typing import List
import torch.nn as nn
import torch.nn.functional as F


class ResNet(nn.Module):
    def __init__(self, channels: List[int], stride: int):
        super().__init__()

        layers = []

        for i in range(len(channels) - 1):
            layers.append(_Bottleneck(channels[i], channels[i + 1], stride))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class _Bottleneck(nn.Module):
    def __init__(self, channel_in: int, channel_out: int, stride: int):
        super().__init__()

        if channel_in != channel_out:
            self.conv0 = nn.Conv2d(
                channel_in, channel_out, kernel_size=1, stride=stride, bias=False
            )

        self.conv1 = nn.Conv2d(
            channel_in, channel_out, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(channel_out)

        self.conv2 = nn.Conv2d(
            channel_out, channel_out, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(channel_out)

    def forward(self, x):
        identity = x

        if hasattr(self, "conv0"):
            identity = self.conv0(identity)

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x += identity
        x = F.relu(x)

        return x
