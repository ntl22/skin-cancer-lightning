from typing import Tuple
import torch.nn as nn

from .primary import PrimaryCapsule
from .digits import DigitsCapsule
from .utils import size_after_conv


class CapsNet(nn.Module):
    def __init__(
        self,
        size_in: Tuple[int, int, int],
        primary_reduce_fraction: int,
        primary_dim: int,
        digit_channels: int,
        digit_dim: int,
        kernel_size: int,
        stride: int,
    ):
        super().__init__()

        size, _, channels_in = size_in

        primary_capules = channels_in // primary_reduce_fraction
        self.primary = PrimaryCapsule(channels_in, primary_capules, primary_dim)

        primary_channels = size_after_conv(size, kernel_size, stride=stride) ** 2 * primary_capules

        self.digits = DigitsCapsule(
            primary_channels, primary_dim, digit_channels, digit_dim
        )
    
    def forward(self, x):
        x = self.primary(x)
        x = self.digits(x)
        return x
