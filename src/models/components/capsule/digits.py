import torch
import torch.nn as nn

from .utils import squash


class DigitsCapsule(nn.Module):
    def __init__(self, primary_channels, primary_dim, digit_channels, digit_dim):
        super().__init__()

        self.primary_channels = primary_channels
        self.digit_dim = digit_dim

        self.w = nn.parameter.Parameter(
            torch.randn(primary_channels, primary_dim, digit_dim, digit_channels)
        )

    def forward(self, x):
        """
        B: batch
        I: primary_channels
        k: primary_dim
        O: output_channel
        l: output_dim
        """

        u_hat = torch.einsum("BIk, IklO -> BIlO", x, self.w)

        b_ij = torch.zeros(self.primary_channels, self.digit_dim, device=u_hat.device)

        num_iters = 3
        for _ in range(num_iters):
            c_ij = b_ij.softmax(dim=0)
            s_j = torch.einsum("Il, BIlO -> BlO", c_ij, u_hat)

            v_j = squash(s_j).transpose(1, 2)

            u_vj = torch.einsum("BOl, BIlO -> BIl", v_j, u_hat).mean(dim=0)
            b_ij = b_ij + u_vj

        return v_j
