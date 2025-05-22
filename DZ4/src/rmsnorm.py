import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-8):
        super().__init__()
        self.dims = dims
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dims))

    def _rms(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms_val = self._rms(x)
        normalized_x = x / rms_val
        return self.scale * normalized_x