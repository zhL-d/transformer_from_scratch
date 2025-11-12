import torch
import torch.nn as nn
import einx

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None):
        """ Construct the RMSNorm module.
        
        Args:
            d_model(int): Hidden dimension of the model
            eps(float): Epsilon value for numerical stability
            device(torch.device | None = None): Device to store the parameters on
            dtype(torch.dtype | None = None): Data type of the parameters
        """
        super().__init__()
        self.g = nn.Parameter(torch.ones(d_model, dtype=dtype, device=device))
        self.eps = eps
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process an input tensor of shape

        Args:
            x (torch.Tensor): shape (batch_size, sequence_length, d_model)
        Returns:
            torch.Tensor: shape (batch_size, sequence_length, d_model)
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)

        mean_square = einx.mean('... d -> ... 1', x * x)
        inv_rms = torch.rsqrt(mean_square + self.eps)

        x_norm = einx.multiply('... d, ... 1 -> ... d', x, inv_rms)
        result = einx.multiply('... d, d -> ... d', x_norm, self.g)
        
        return result.to(in_dtype)
