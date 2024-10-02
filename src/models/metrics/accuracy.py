import torch
import torch.nn as nn


class CapsNetAccuracy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        batch_size = y.size(0)

        v_mag = torch.sqrt(torch.sum(y_hat**2, dim=2))
        _, y_hat = v_mag.max(dim=1)

        correct = (y_hat == y).sum().float()
        return correct / batch_size
