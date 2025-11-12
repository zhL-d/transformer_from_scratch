import torch
import torch.nn as nn
import math

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        """Construct a linear transformation module.

        Args:
            in_features(int): final dimension of the input
            out_features(int): final dimension of the output
            device(torch.device | None): Device to store the parameters on
            dtype(torch.dtype | None): Data type of the parameters
        """
        super().__init__()
        self.W = nn.Parameter(torch.randn(out_features, in_features, dtype=dtype, device=device))
        std_variance = math.sqrt(2/(in_features + out_features))
        nn.init.trunc_normal_(self.W, mean=0, std=std_variance, a=-3*std_variance, b=3*std_variance)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the linear transformation to the input.
        """
        return x @ self.W.T