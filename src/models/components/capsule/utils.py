import torch


def squash(s: torch.Tensor):
    sj2 = torch.sum(s**2, dim=2, keepdim=True)
    sj = torch.sqrt(sj2)
    return (sj2 / (1.0 + sj2)) * (s / (sj + 1e-7))


def size_after_conv(size, kernel_size, padding=0, stride=1):
    return (size - kernel_size + 2 * padding) // stride + 1
