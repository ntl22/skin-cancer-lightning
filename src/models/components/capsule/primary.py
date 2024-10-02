import torch
import torch.nn as nn

from .utils import squash


class PrimaryCapsule(nn.Module):
    def __init__(self, channel_in, num_capsules, capsule_dim, kernel_size=9, stride=2):
        super().__init__()

        self.capsule_dim = capsule_dim

        self.conv_list = [
            nn.Conv2d(
                channel_in,
                num_capsules,
                kernel_size,
                stride=stride,
                groups=num_capsules,
                bias=False,
            )
            for _ in range(self.capsule_dim)
        ]

        self.conv_list = nn.ModuleList(self.conv_list)

    def forward(self, x):
        capsule = [self.conv_list[i](x) for i in range(self.capsule_dim)]
        capsule = torch.cat(capsule, dim=1)
        capsule = capsule.reshape(capsule.size(0), self.capsule_dim, -1)
        return squash(capsule).transpose(1, 2)
