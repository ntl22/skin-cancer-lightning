import torch
import torch.nn as nn
import torch.nn.functional as F


class CapsNetLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor, weight: torch.Tensor=None) -> torch.Tensor:
        batch_size = y.size(0)

        v_mag = torch.sqrt(torch.sum(y_hat**2, dim=2))

        m_plus, m_minus = 0.9, 0.1

        plus_loss = F.relu(m_plus - v_mag).view(batch_size, -1) ** 2
        minus_loss = F.relu(v_mag - m_minus).view(batch_size, -1) ** 2

        loss = y * plus_loss + 0.5 * (1.0 - y) * minus_loss

        if weight is not None:
            if not isinstance(weight, torch.Tensor):
                weight = torch.tensor(weight, device=y.device)
            
            weight = torch.stack([weight] * batch_size, dim=0)
            loss = loss * weight

        return loss.sum(dim=1).mean()
